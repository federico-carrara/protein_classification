import pytorch_lightning as pl
import torch

import torch.nn as nn
import torch.nn.functional as F

from protein_classification.config import DenseNetConfig, LossConfig


class BioStructClassifier(pl.LightningModule):
    def __init__(
        self, model_config: DenseNetConfig, loss_config: LossConfig
    ) -> None:
        super().__init__()
        self.model = 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)