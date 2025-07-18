from pathlib import Path
from typing import Optional, Union

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from protein_classification.config import TrainingConfig

Callback = Union[EarlyStopping, LearningRateMonitor, ModelCheckpoint]


def get_callbacks(
    logdir: Optional[Union[str, Path]], training_config: TrainingConfig
) -> list[Callback]:
    callbacks: list[Callback] = []
    if training_config.earlystop_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-6,
                patience=training_config.earlystop_patience,
                mode="min",
                verbose=True,
            )
        )
    callbacks.append(
        ModelCheckpoint(
            dirpath=logdir,
            filename="best-{epoch}",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
        )
    )
    callbacks.append(
        LearningRateMonitor(logging_interval="epoch"),
    )
    return callbacks