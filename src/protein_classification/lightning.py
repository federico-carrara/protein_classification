import pytorch_lightning as pl
import torch

import torch.nn as nn
import torch.nn.functional as F

from protein_classification.config import AlgorithmConfig
from protein_classification.config.losses import loss_factory
from protein_classification.model import DenseNet


class BioStructClassifier(pl.LightningModule):
    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__()
        self.config = config
        self.model = DenseNet(**config.architecture_config.model_dump())
        self.loss_fn = loss_factory(config.loss_config)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> float:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        # TODO: get params from config instead of hardcoding
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[25, 30, 35, 40],
            gamma=0.5
        )
        return {
            'optimizer': optimizer,
            'scheduler': lr_scheduler,
            'monitor': 'val_loss',
        }