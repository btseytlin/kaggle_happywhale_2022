import os
import torch
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner
from torch import nn
import pytorch_lightning as pl
from pytorch_metric_learning.distances import CosineSimilarity

from happywhale.models.base_model import BaseModel


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
        logits, loss = self._process_batch(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("hp_metric", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits, loss = self._process_batch(batch, batch_idx)
        return logits
