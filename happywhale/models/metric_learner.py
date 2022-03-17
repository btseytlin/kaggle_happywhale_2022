import os
import torch
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner
from pytorch_metric_learning.utils.inference import CustomKNN
from torch import nn
import pytorch_lightning as pl
from pytorch_metric_learning.distances import CosineSimilarity
from happywhale.models.base_model import BaseModel
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm.auto import tqdm


class MetricLearner(BaseModel):
    def __init__(self,
                 *args,
                 pos_margin=1,
                 neg_margin=0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.loss = ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin, distance=CosineSimilarity())
        self.miner = BatchEasyHardMiner(
            pos_strategy=BatchEasyHardMiner.EASY,
            neg_strategy=BatchEasyHardMiner.SEMIHARD,
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

    def forward(self, *args, **kwargs):
        backbone_embeddings = self.backbone(*args, **kwargs)
        embeddings = self.mlp(backbone_embeddings)
        return embeddings

    def _process_batch(self, batch, batch_idx, **kwargs):
        images, labels = batch
        embeddings = self(images)
        miner_out = self.miner(embeddings, labels)
        loss = self.loss(embeddings, labels, miner_out)
        return embeddings, loss

    def training_step(self, batch, batch_idx):
        embeddings, loss = self._process_batch(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, loss = self._process_batch(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("hp_metric", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return embeddings, loss

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
        # val_embeddings = torch.cat([out[0] for out in validation_step_outputs])
        val_embeddings = self.get_embeddings(self.trainer.datamodule.val_dataloader())
        val_labels = self.trainer.datamodule.val.labels

        train_embeddings = self.get_embeddings(self.trainer.datamodule.train_dataloader(sampler=False, shuffle=False))
        train_labels = self.trainer.datamodule.train.labels

        print(val_embeddings.shape, val_labels.shape, train_embeddings.shape, train_labels.shape)
        scores = self.acc_calculator.get_accuracy(query=val_embeddings,
                                                  query_labels=val_labels,
                                                  reference=train_embeddings,
                                                  reference_labels=train_labels,
                                                  embeddings_come_from_same_source=False)
        self.log_dict(scores, prog_bar=True, logger=True)
        return scores

    def test_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits, loss = self._process_batch(batch, batch_idx)
        return logits
