from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union
from typing_extensions import Self

import tifffile as tiff
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

PathLike = Union[Path, str]


class DataAugmentationConfig(BaseModel):
    """Configuration for data augmentation."""
    
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        validate_default=True,
    )
    
    transform: Optional[Literal['geometric', 'intensity', 'noise', 'all']] = None
    """The name of the augmentation/transform used at training time.
    Currently, the available ones are:
    - "geometric": applies only random geometric augmentations.
    - "intensity": applies only mild intensity scaling.
    - "noise": applies only random noise augmentations.
    - "all": applies geometric, intensity, and noise augmentations.
    By default `None`, which means no transformation is applied."""
    
    crop_size: Optional[int] = None
    """The size of the crops used for training. If `None`, no cropping is applied."""
    
    random_crop: bool = False
    """Whether to apply random cropping to the images. If `False`, center cropping is
    applied."""
    
    strategy: Optional[Literal["curriculum", "overlap", "background"]] = None
    """The cropping strategy to use. If `None`, simple cropping is applied."""
    
    crop_overlap: Optional[int] = None
    """The overlap between crops at test time. If `None`, no overlap is applied.
    This is used for "overlap" strategy."""
    
    metrics: list[Literal["std"]] = ["std"]
    """A list of metrics to combine in order to compute the difficulty score.
    By default ["std"]."""

    total_epochs: Optional[int] = Field(None, ge=0)
    """Total number of epochs on which curriculum learning is applied."""

    beta_max_alpha: float = 5.0
    """Initial Beta(α, 1) skew; α anneals from `beta_max_alpha` to 1."""

    sampling_patience: int = 10
    """Maximum number of crops to sample before giving up on finding a suitable crop."""
    
    bg_threshold: Optional[float] = None
    """Threshold for `metrics` values for identification of background crops.
    If `None`, the threshold is inferred from the metric distribution."""

    @model_validator(mode='after')
    def validate_config(self: Self) -> Self:
        """Validate the configuration."""
        if (
            self.strategy == "curriculum" or
            self.strategy == "background" and
            not self.random_crop
        ):
            print(
                "Warning: `curriculum` or `background` strategy is enabled, "
                "so `random_crop` will be forced to `True`."
            )
            self.random_crop = True
        return self


class DataConfig(BaseModel):
    """Configuration for data modules."""

    model_config = ConfigDict(
        extra='allow',
        validate_assignment=True,
        validate_default=True,
    )

    data_dir: PathLike
    """Path to the dataset directory."""
    
    # TODO: add paths to labels.json, train_labels.csv
    
    labels: Sequence[str]
    """List of labels to pick. This is used to map the integer labels to their
    string names."""
    
    img_size: int
    """Size to which images will be resized."""
    
    imreader: Callable[[PathLike], Union[NDArray, Tensor]] = Field(tiff.imread, exclude=True)
    """Function to read images from filepaths as `NDArray` arrays. By default `tiff.imread`."""
    
    bit_depth: Optional[int] = None
    """The bit depth of the input images. If specified, the images will be normalized
    to the range [0, 1] based on the bit depth. If `None`, no range normalization
    is applied."""
    
    normalize: Optional[Union[str, Literal['minmax', 'std']]] = None
    """The normalization method to apply to the images.
    - 'minmax': scales images to [0, 1] based on the min and max values.
    - 'std': standardizes images to have zero mean and unit variance.
    By default `None`, which means no normalization is applied."""

    normalization_scope: Literal["dataset", "image"] = "dataset"
    """Scope used to compute normalization statistics.
    - 'dataset': use precomputed dataset-level statistics from `dataset_stats`.
    - 'image': compute statistics independently for each image/patch."""
    
    dataset_stats: Optional[tuple[float, float]] = None
    """Pre-computed dataset statistics (mean, std) or (min, max) for normalization.
    Required when `normalize` is specified and `normalization_scope='dataset'`."""
    
    train_augmentation_config: Optional[DataAugmentationConfig] = None
    """Configuration for data augmentation, including cropping and transformations."""
    
    val_augmentation_config: Optional[DataAugmentationConfig] = None
    """Configuration for validation data augmentation. If `None`, no augmentation is applied."""

    test_augmentation_config: Optional[DataAugmentationConfig] = None
    """Configuration for test data augmentation. If `None`, no augmentation is applied."""

    @model_validator(mode='after')
    def validate_normalization(self: Self) -> Self:
        """Validate normalization settings."""
        if self.normalize is None:
            return self

        if self.normalization_scope == "dataset" and self.dataset_stats is None:
            raise ValueError(
                "`dataset_stats` must be provided when using dataset normalization."
            )

        if self.normalization_scope == "image" and self.dataset_stats is not None:
            print(
                "Warning: `dataset_stats` were provided but will be ignored because "
                "`normalization_scope='image'`."
            )

        return self
