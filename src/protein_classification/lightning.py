import pytorch_lightning as pl
import torch

from torchmetrics.classification import MulticlassF1Score

from protein_classification.config import AlgorithmConfig
from protein_classification.config.losses import loss_factory
from protein_classification.model import DenseNet


class BioStructClassifier(pl.LightningModule):
    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__()
        self.config = config
        self.model = DenseNet(**config.architecture_config.model_dump())
        self.loss_fn = loss_factory(config.loss_config)
        
        # metrics
        self.f1_metric = MulticlassF1Score(
            num_classes=config.architecture_config.num_classes,
            average='macro',
            compute_on_step=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> float:
        x, y = batch
        logits: torch.Tensor = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        f1 = self.f1_metric(logits, y)
        self.log(
            'train_loss', loss, prog_bar=True,
            on_step=True, on_batch=True, batch_size=x.size(0)
        )
        self.log(
            'train_accuracy', acc, prog_bar=True,
            on_step=True, on_batch=True, batch_size=x.size(0)
        )
        self.log(
            'train_f1', f1, prog_bar=True,
            on_step=True, on_batch=True, batch_size=x.size(0)
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits: torch.Tensor = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        f1 = self.f1_metric(logits, y)
        self.log(
            'val_loss', loss, prog_bar=True,
            on_step=True, on_batch=True, batch_size=x.size(0)
        )
        self.log(
            'val_accuracy', acc, prog_bar=True,
            on_step=True, on_batch=True, batch_size=x.size(0)
        )
        self.log(
            'val_f1', f1, prog_bar=True,
            on_step=True, on_batch=True, batch_size=x.size(0)
        )

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