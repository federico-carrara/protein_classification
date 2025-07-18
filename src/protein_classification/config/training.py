from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """
    Parameters related to the training.

    Mandatory parameters are:
        - num_epochs: number of epochs, greater than 0.
        - batch_size: batch size, greater than 0.
        - augmentation: whether to use data augmentation or not (True or False).

    Attributes
    ----------
    num_epochs : int
        Number of epochs, greater than 0.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    max_epochs: int = Field(default=20, ge=1)
    """Maximum number of epochs, greater than 0."""
    
    lr: float = Field(default=3e-4, gt=1e-8)
    """Learning rate."""
    
    batch_size: int = Field(default=32, gt=0)
    """Batch size, greater than 0."""
    
    earlystop_patience: Optional[int] = None
    """Patience for the early stopping callback."""
    
    precision: Literal["64", "32", "16-mixed", "bf16-mixed"] = Field(default="32")
    """Numerical precision"""
    
    max_steps: int = Field(default=-1, ge=-1)
    """Maximum number of steps to train for. -1 means no limit."""
    
    check_val_every_n_epoch: int = Field(default=1, ge=1)
    """Validation step frequency in epochs."""
    
    accumulate_grad_batches: int = Field(default=1, ge=1)
    """Number of batches to accumulate gradients over before stepping the optimizer."""
    
    gradient_clip_algorithm: Literal["value", "norm"] = "norm"
    """The algorithm to use for gradient clipping (see lightning `Trainer`)."""
    
    gradient_clip_val: Optional[Union[int, float]] = None
    """The value to which to clip the gradient"""