from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


DENSENET121 = {
    "num_init_features": 64,
    "growth_rate": 32,
    "block_config": [6, 12, 24, 16],
}

DENSENET161 = {
    "num_init_features": 96,
    "growth_rate": 48,
    "block_config": [6, 12, 36, 24],
}

DENSENET169 = {
    "num_init_features": 64,
    "growth_rate": 32,
    "block_config": [6, 12, 32, 32],
}

DENSENET201 = {
    "num_init_features": 64,
    "growth_rate": 32,
    "block_config": [6, 12, 48, 32],
}

class DenseNetConfig(BaseModel):
    """
    Configuration for DenseNet architecture.
    """
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, validate_default=True
    )
    
    architecture: Literal[
        "custom", "densenet121", "densenet161", "densenet169", "densenet201"
    ] = "custom"
    """Architecture type, either 'custom' or a pre-defined DenseNet among the provided
    ones."""
    
    num_classes: int
    """Number of output classes for classification."""
    
    num_init_features: int
    """Number of initial features after the first convolution layer."""
    
    growth_rate: int = 32
    """Growth rate for the DenseNet blocks."""
    
    block_config: list[int] = [6, 12, 24, 16]
    """Number of layers in each DenseNet block."""
    
    bn_size: int = 4
    """Bottleneck size for the DenseNet blocks. This number is multiplied by the growth
    factor to get the number of features in the bottleneck of each dense block."""
    
    dropout_block: bool = True
    """Whether to use the so-called dropout block before the classification head."""
    
    dropout_p: float = 0.5
    """Dropout probability for the model's dropout layers."""
    
    @model_validator(mode="after")
    def set_architecture(self):
        """Set the architecture based on the provided type."""
        if self.architecture == "densenet121":
            for key, value in DENSENET121.items():
                setattr(self, key, value)
        elif self.architecture == "densenet161":
            for key, value in DENSENET161.items():
                setattr(self, key, value)
        elif self.architecture == "densenet169":
            for key, value in DENSENET169.items():
                setattr(self, key, value)
        elif self.architecture == "densenet201":
            for key, value in DENSENET201.items():
                setattr(self, key, value)
        return self