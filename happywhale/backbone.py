import timm
import os
import pytorch_lightning as pl


class ImageBackbone(pl.LightningModule):
    def __init__(self, trunk=None, model_name=None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.trunk = trunk or timm.create_model(model_name, pretrained=True)

    def forward(self, *args, **kwargs):
        embeddings = self.trunk(*args, **kwargs)
        return embeddings
