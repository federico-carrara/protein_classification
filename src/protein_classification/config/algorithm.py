from pydantic import BaseModel, ConfigDict

from protein_classification.config.architecture import DenseNetConfig
from protein_classification.config.losses import LossConfig


class AlgorithmConfig(BaseModel):
    """
    Configuration for the algorithm used in the protein classification task.
    This includes model architecture and loss function configurations.
    """
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, validate_default=True
    )

    architecture_config: DenseNetConfig
    """Configuration for the DenseNet model architecture."""
    
    loss_config: LossConfig
    """Configuration for the loss function used in training."""
    
    lr: float = 3e-4
    """Learning rate for the optimizer."""
    
    # TODO: add parameters for the optimizer and scheduler