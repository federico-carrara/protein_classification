import pytorch_lightning as pl
import torch

from torchmetrics.classification import BinaryF1Score, MulticlassF1Score

from protein_classification.config import AlgorithmConfig
from protein_classification.config.losses import loss_factory
from protein_classification.model.densenet import DenseNet
from protein_classification.model.resnet import ResNet
from protein_classification.config.architectures import DenseNetConfig, ResNetConfig


def _build_model(config):
    """Instantiate the right backbone from an architecture config."""
    if isinstance(config, DenseNetConfig):
        return DenseNet(**config.model_dump())
    if isinstance(config, ResNetConfig):
        return ResNet(**config.model_dump())
    raise ValueError(f"Unknown architecture config type: {type(config)}")


class BioStructClassifier(pl.LightningModule):
    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.architecture_config.num_classes
        self.is_binary = self.num_classes == 1
        self.model = _build_model(config.architecture_config)
        self.loss_fn = loss_factory(config.loss_config)

        # metrics
        if self.is_binary:
            self.f1_metric_train = BinaryF1Score()
            self.f1_metric_val = BinaryF1Score()
        else:
            self.f1_metric_train = MulticlassF1Score(
                num_classes=self.num_classes, average='macro'
            )
            self.f1_metric_val = MulticlassF1Score(
                num_classes=self.num_classes, average='macro'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared forward + loss computation for training and validation."""
        x, y = batch
        logits: torch.Tensor = self.model(x)

        if self.is_binary:
            logits = logits.squeeze(1)  # [B, 1] -> [B]
            loss = self.loss_fn(logits, y.float())
            preds = (logits > 0).long()
        else:
            loss = self.loss_fn(logits, y)
            preds = logits.argmax(dim=1)

        return logits, preds, y, loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> float:
        logits, preds, y, loss = self._compute_step(batch)
        bs = logits.size(0)
        self.log(
            'train_loss', loss, prog_bar=True,
            on_epoch=True, batch_size=bs, logger=True
        )

        # calculate and log metrics
        acc = (preds == y).float().mean()
        if self.is_binary:
            f1 = self.f1_metric_train(logits, y)
        else:
            f1 = self.f1_metric_train(logits, y)
        self.log(
            'train_accuracy', acc, prog_bar=True,
            on_epoch=True, batch_size=bs, logger=True
        )
        self.log(
            'train_f1', f1, prog_bar=True,
            on_epoch=True, batch_size=bs, logger=True
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        # TODO: add tta-like augmentation for validation
        logits, preds, y, loss = self._compute_step(batch)
        bs = logits.size(0)
        self.log(
            'val_loss', loss, prog_bar=True,
            on_epoch=True, batch_size=bs, logger=True
        )

        # calculate and log metrics
        acc = (preds == y).float().mean()
        if self.is_binary:
            f1 = self.f1_metric_val(logits, y)
        else:
            f1 = self.f1_metric_val(logits, y)
        self.log(
            'val_accuracy', acc, prog_bar=True,
            on_epoch=True, batch_size=bs, logger=True
        )
        self.log(
            'val_f1', f1, prog_bar=True,
            on_epoch=True, batch_size=bs, logger=True
        )

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        if self.is_binary:
            logits = logits.squeeze(1)  # [B, 1] -> [B]
            preds = (logits > 0).long()
        else:
            preds = logits.argmax(dim=1)
        return preds, logits, y

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training_config.lr)
        # TODO: get params from config instead of hardcoding
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[25, 30, 35, 40],
            gamma=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss',
        }

    def on_train_epoch_start(self) -> None:
        """Set the current epoch in the dataset class for curriculum learning."""
        dloader = self.trainer.train_dataloader
        if hasattr(dloader, 'dataset') and hasattr(dloader.dataset, 'set_epoch'):
            dloader.dataset.set_epoch(self.current_epoch)
