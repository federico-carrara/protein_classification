from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import tifffile as tiff
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

PathLike = Union[Path, str]


class DataConfig(BaseModel):
    """Configuration for data modules."""

    model_config = ConfigDict(
        extra='forbid',
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
    
    crop_size: Optional[int] = None
    """The size of the crops used for training. If `None`, no cropping is applied."""
    
    random_crop: bool = False
    """Whether to apply random cropping to the images. If `False`, center cropping is
    applied."""
    
    imreader: Callable[[PathLike], Union[NDArray, Tensor]] = Field(tiff.imread, exclude=True)
    """Function to read images from filepaths as `NDArray` arrays. By default `tiff.imread`."""
    
    transform: Optional[Callable[[NDArray], NDArray]] = Field(None, exclude=True)
    """A function/transform that takes in an image and returns a transformed version.
    Currently, the available transforms are:
    - `train_augmentation`: applies random noise and geometric augmentations.
    - `geometric_augmentation`: applies only random geometric augmentations.
    - `noise_augmentation`: applies only random noise augmentations.
    By default `None`, which means no transformation is applied."""
    
    bit_depth: Optional[int] = None
    """The bit depth of the input images. If specified, the images will be normalized
    to the range [0, 1] based on the bit depth. If `None`, no range normalization
    is applied."""
    
    normalize: Optional[Union[str, Literal['minmax', 'std']]] = None
    """The normalization method to apply to the images.
    - 'minmax': scales images to [0, 1] based on the min and max values.
    - 'std': standardizes images to have zero mean and unit variance.
    By default `None`, which means no normalization is applied. If specified, the
    `dataset_stats` must also be provided."""
    
    dataset_stats: Optional[tuple[float, float]] = None
    """Pre-computed dataset statistics (mean, std) or (min, max) for normalization.
    If `normalize` is specified, this must also be provided."""