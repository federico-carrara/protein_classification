from pathlib import Path
from typing import Optional, Union

from careamics.config import TrainingConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

Callback = Union[EarlyStopping, LearningRateMonitor, ModelCheckpoint]


def get_callbacks(
    logdir: Optional[Union[str, Path]], training_config: TrainingConfig
) -> list[Callback]:
    return [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=training_config.earlystop_patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=logdir,
            filename="best-{epoch}",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]