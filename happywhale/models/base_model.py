import os
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.functional.classification import accuracy
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.loggers import WandbLogger
from happywhale.utils import get_mlp


class BaseModel(pl.LightningModule):
    def __init__(self,
                 backbone=None,
                 mlp=None,
                 trainer=None,
                 num_training_steps=None,
                 lr=1e-4,
                 backbone_embedding_dim=None,
                 num_labels=None,
                 embedding_size=None,
                 dropout=0.1,
                 class_weights=None,
                 trainer_kwargs=None,
                 offline=False,
                 **kwargs):
        super().__init__()
        self.lr = lr
        self.num_training_steps = num_training_steps
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.class_weights = class_weights
        self.backbone_embedding_dim = backbone_embedding_dim
        self.backbone = backbone

        self.mlp = mlp or get_mlp(
            input_dim=self.backbone_embedding_dim,
            output_dim=self.embedding_size,
            dropout_rate=dropout,
        )

        weight = torch.FloatTensor(self.class_weights) if self.class_weights is not None else None
        self.loss = nn.CrossEntropyLoss(weight=weight)

        if offline:
            self.wandb_logger = WandbLogger(offline=offline)
        else:
            self.wandb_logger = WandbLogger(log_model = True)

        trainer_kwargs = trainer_kwargs or {}
        trainer_kwargs.update(kwargs)
        self.trainer = trainer or self.get_trainer(self.wandb_logger, **trainer_kwargs)

        self.save_hyperparameters()

    @classmethod
    def get_trainer(cls, logger, **kwargs):
        trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=logger,
            **kwargs,
        )

        trainer.callbacks.append(
            LearningRateMonitor(logging_interval='step')
        )
        return trainer

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(
            path,
            trainer=self.trainer,
            backbone=self.backbone,
            mlp=self.mlp,
            lr=self.lr,
            dropout=self.dropout,
            num_training_steps=self.num_training_steps,
            embedding_size=self.embedding_size,
            class_weights=self.class_weights,
            backbone_embedding_dim=self.backbone_embedding_dim,
            wandb_logger=self.wandb_logger,
            strict=False,
        )

    def load_best_checkopoint(self):
        checkpoint_callbacks = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
        if len(checkpoint_callbacks) != 1:
            raise Exception(
                f'Expected to find one checkpoint callback to load weights from, found: {len(checkpoint_callbacks)}'
                )
        best_model_path = checkpoint_callbacks[0].best_model_path
        new_model = self.load_checkpoint(best_model_path)
        self.load_state_dict(new_model.state_dict(), strict=True)
        return self

    def fit(self, datamodule):
        self.trainer.fit(self, datamodule)
        try:
            return self.load_best_checkopoint()
        except Exception as e:
            print(e)
            print('WARNING: Couldn\'t load best checkpoint')
            return self

    def test(self, datamodule):
        test_results = self.trainer.test(self, datamodule=datamodule)
        return test_results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # num_warmup_steps = self.num_training_steps//2
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=num_warmup_steps,
        #                                             num_training_steps=self.num_training_steps)

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'step',
            # }
        }

    def forward(self, *args, **kwargs):
        embeddings = self.backbone(*args, **kwargs)
        logits = self.mlp(embeddings)
        return logits

    def _process_batch(self, batch, batch_idx, **kwargs):
        images, labels = batch
        logits = self(images)
        loss = self.loss(logits, labels)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, loss = self._process_batch(batch, batch_idx)
        acc = accuracy(logits, labels, average='weighted', num_classes=self.num_labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("hp_metric", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits, loss = self._process_batch(batch, batch_idx)
        acc = accuracy(logits, labels, average='weighted', num_classes=self.num_labels)
        self.log("test_acc", acc, on_step=False, on_epoch=True, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits, loss = self._process_batch(batch, batch_idx)
        return logits

