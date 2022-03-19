import os
import torch
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner
from pytorch_metric_learning.utils.inference import CustomKNN
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_metric_learning.distances import CosineSimilarity

from happywhale.utils import get_mlp
from happywhale.models.base_model import BaseModel
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm.auto import tqdm
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float,
        m: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class MetricLearner(BaseModel):
    def __init__(self,
                 *args,
                 arc_s=30.0,
                 arc_m=0.3,
                 arc_easy_margin=False,
                 arc_ls_eps=0.0,
                 mlp=None,
                 dropout=0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp = mlp or get_mlp(
            input_dim=self.backbone_embedding_dim,
            output_dim=self.embedding_size,
            dropout_rate=dropout,
        )

        self.arc = ArcMarginProduct(
            in_features=self.embedding_size,
            out_features=self.num_labels,
            s=arc_s,
            m=arc_m,
            easy_margin=arc_easy_margin,
            ls_eps=arc_ls_eps,
        )

        self.acc_calculator = AccuracyCalculator(
            include=(
                'mean_average_precision',
                'precision_at_1',
                'mean_average_precision_at_r',
            ),
            k="max_bin_count",
            knn_func=CustomKNN(distance=CosineSimilarity()),
        )

        self.save_hyperparameters()

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(
            path,
            trainer=self.trainer,
            backbone=self.backbone,
            mlp=self.mlp,
            lr=self.lr,
            dropout=self.dropout,
            num_training_steps=self.num_training_steps,
            class_weights=self.class_weights,
            backbone_embedding_dim=self.backbone_embedding_dim,
            embedding_size=self.embedding_size,
            wandb_logger=self.wandb_logger,
            strict=False,
        )

    def forward(self, images):
        backbone_embeddings = self.backbone(images)
        embeddings = self.mlp(backbone_embeddings)
        return embeddings

    def _process_batch(self, batch, batch_idx, **kwargs):
        images, labels = batch
        embeddings = self(images)
        if self.training:
            embeddings = self.arc(embeddings, labels, self.device)
            loss = self.loss(embeddings, labels)
            return embeddings, loss
        else:
            return embeddings

    def training_step(self, batch, batch_idx):
        embeddings, loss = self._process_batch(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return None

    @torch.no_grad()
    def get_embeddings(self, dl):
        embeddings = []
        for batch in tqdm(dl):
            images, labels = batch
            images = images.to(self.device)
            batch_embeddings = self(images)
            embeddings.append(batch_embeddings.cpu())
        embeddings = torch.cat(embeddings)
        return embeddings

    def validation_epoch_end(self, validation_step_outputs):
        val_embeddings = self.get_embeddings(self.trainer.datamodule.val_dataloader())
        val_labels = self.trainer.datamodule.val.labels

        train_embeddings = self.get_embeddings(self.trainer.datamodule.train_dataloader(sampler=False, shuffle=False))
        train_labels = self.trainer.datamodule.train.labels

        scores = self.acc_calculator.get_accuracy(query=val_embeddings,
                                                  query_labels=val_labels,
                                                  reference=train_embeddings,
                                                  reference_labels=train_labels,
                                                  embeddings_come_from_same_source=False)
        self.log_dict(scores, prog_bar=True, on_epoch=True, logger=True)
        return scores

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits = self._process_batch(batch, batch_idx)
        return logits
