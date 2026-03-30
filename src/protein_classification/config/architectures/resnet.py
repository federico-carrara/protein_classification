from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


RESNET18 = {
    "block_config": [2, 2, 2, 2],
    "channels": [64, 128, 256, 512],
}

RESNET34 = {
    "block_config": [3, 4, 6, 3],
    "channels": [64, 128, 256, 512],
}

_PRESETS = {
    "resnet18": RESNET18,
    "resnet34": RESNET34,
}


class ResNetConfig(BaseModel):
    """Configuration for ResNet architecture."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, validate_default=True,
    )

    model_type: Literal["resnet"] = "resnet"
    """Discriminator field used to distinguish from DenseNetConfig."""

    architecture: Literal["resnet18", "resnet34"] = "resnet18"
    """Pre-defined ResNet variant."""

    num_classes: int
    """Number of output classes for classification."""

    block_config: list[int] = [2, 2, 2, 2]
    """Number of BasicBlocks in each of the 4 layer groups."""

    channels: list[int] = [64, 128, 256, 512]
    """Output channels for each of the 4 layer groups."""

    num_init_features: int = 64
    """Number of filters in the stem convolution."""

    dropout_p: float = 0.0
    """Dropout probability before the classifier head."""

    @model_validator(mode="after")
    def set_architecture(self):
        """Override fields from the chosen preset."""
        preset = _PRESETS.get(self.architecture)
        if preset is not None:
            for key, value in preset.items():
                object.__setattr__(self, key, value)
        return self
