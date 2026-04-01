from enum import Enum
from typing import Union

from pydantic import BaseModel, ConfigDict, field_validator
from torch import nn

from torch import nn

from protein_classification.losses import BinaryFocalLoss, MulticlassFocalLoss

AnyLoss = Union[nn.Module, BinaryFocalLoss, MulticlassFocalLoss]


class SupportedLosses(Enum):
    """Enum for supported loss functions."""
    BINARY_FOCAL_LOSS = "binary_focal"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    MULTICLASS_FOCAL_LOSS = "multiclass_focal"
    MULTICLASS_CROSS_ENTROPY = "multiclass_cross_entropy"


class LossConfig(BaseModel):
    """Configuration model for loss functions."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
    )

    loss_type: Union[str, SupportedLosses]
    """Type of the loss function to use."""

    @field_validator("loss_type")
    def validate_loss_type(cls, value: Union[str, SupportedLosses]) -> SupportedLosses:
        """Validate and convert the loss type to SupportedLosses enum."""
        if isinstance(value, SupportedLosses):
            return value
        try:
            return SupportedLosses(value)
        except ValueError:
            raise ValueError(f"Unsupported loss function: {value}")


def loss_factory(config: LossConfig) -> AnyLoss:
    """Factory function to create loss instances based on the provided name."""
    if config.loss_type == SupportedLosses.BINARY_FOCAL_LOSS:
        return BinaryFocalLoss()
    elif config.loss_type == SupportedLosses.BINARY_CROSS_ENTROPY:
        return nn.BCEWithLogitsLoss()
    elif config.loss_type == SupportedLosses.MULTICLASS_FOCAL_LOSS:
        return MulticlassFocalLoss()
    elif config.loss_type == SupportedLosses.MULTICLASS_CROSS_ENTROPY:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_type}")