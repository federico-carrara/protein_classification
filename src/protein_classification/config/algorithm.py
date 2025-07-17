from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from protein_classification.config.architecture import DenseNetConfig
from protein_classification.config.losses import LossConfig
from protein_classification.config.training import TrainingConfig

PathLike = Path | str


class AlgorithmConfig(BaseModel):
    """
    Configuration for the algorithm used in the protein classification task.
    This includes model architecture and loss function configurations.
    """
    model_config = ConfigDict(
        extra="allow", validate_assignment=True, validate_default=True
    )

    architecture_config: DenseNetConfig
    """Configuration for the DenseNet model architecture."""
    
    loss_config: LossConfig
    """Configuration for the loss function used in training."""
    
    training_config: TrainingConfig
    """Configuration for the training process, e.g., epochs, optimizer, and scheduler
    settings."""
    
    mode: Literal["train", "eval"]
    """Mode of operation, either 'train' for training or 'eval' for evaluation."""
    
    log_dir: Optional[PathLike]
    """Directory where training logs and checkpoints will be saved."""
    
    wandb_project: str = "protein_classification"
    """Name of the Weights & Biases project for logging."""
    
    # TODO: add parameters for the optimizer and scheduler