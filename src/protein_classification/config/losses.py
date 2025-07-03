from enum import Enum
from typing import Union

from pydantic import BaseModel, ConfigDict

from protein_classification.losses import BinaryFocalLoss, MulticlassFocalLoss
AnyLoss = Union[BinaryFocalLoss, MulticlassFocalLoss]


class SupportedLosses(Enum):
    """Enum for supported loss functions."""
    BINARY_FOCAL_LOSS = "binary_focal_loss"
    MULTICLASS_FOCAL_LOSS = "multiclass_focal_loss"


class LossConfig(BaseModel):
    """Configuration model for loss functions."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
    )
    
    loss_type: Union[str, SupportedLosses]
    """Type of the loss function to use."""


def loss_factory(config: LossConfig) -> AnyLoss:
    """Factory function to create loss instances based on the provided name."""
    if config.loss_type == SupportedLosses.BINARY_FOCAL_LOSS:
        return BinaryFocalLoss(**config.model_dump())
    elif config.loss_type == SupportedLosses.MULTICLASS_FOCAL_LOSS:
        return MulticlassFocalLoss(**config.model_dump())
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_type}")