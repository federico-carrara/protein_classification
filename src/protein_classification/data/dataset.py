from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import tifffile as tiff
import torch
from numpy.typing import NDArray
from torch.utils.data.dataset import Dataset

from protein_classification.data.utils import (
    crop_img, normalize_img, normalize_range, resize_img
)

PathLike = Union[Path, str]


# TODO: deal with stratified/balanced sampling of the dataset
class PreTrainingDataset(Dataset):
    """Dataset for pre-training of protein classification model.
    
    Parameters
    ----------
    inputs : Sequence[tuple[PathLike, int]]
        Sequence of tuples of image filename and label index (optional for test set)
        for each sample.
    split : Literal['train', 'test']
        The split of the dataset, either 'train' or 'test'.
    img_size : int, optional
        The size of the input images, by default 2048. If the input images are not of this size,
        they will be resized to this size.
    crop_size : int, optional
        The size of the crops used for training. If `None`, no cropping is applied.
    random_crop : bool, optional
        Whether to apply random cropping to the images. If `True`, crop size will be
        randomly sampled between `crop_size` and `img_size` and applied to the images.
        By default `False`.
    imreader : Callable, optional
        Function to read images from filepaths as `NDArray` arrays. 
        By default `tiff.imread`.
    transform : Optional[Callable], optional
        A function/transform that takes in an image and returns a transformed version.
        Currently, the available transforms are:
        - `train_augmentation`: applies random noise and geometric augmentations.
        - `geometric_augmentation`: applies only random geometric augmentations.
        - `noise_augmentation`: applies only random noise augmentations.
        By default `None`, which means no transformation is applied.
    bit_depth : Optional[int], optional
        The bit depth of the input images. If specified, the images will be normalized
        to the range [0, 1] based on the bit depth. If `None`, no range normalization
        is applied. By default `None`.
    normalize : Literal['range', 'minmax', 'std'], optional
        The normalization method to apply to the images.
        - 'minmax': scales images to [0, 1] based on the min and max values.
        - 'std': standardizes images to have zero mean and unit variance.
        By default 'range'.
    dataset_stats : Optional[tuple[float, float]], optional
        Pre-computed dataset statistics (mean, std) or (min, max) for normalization.
    """
    def __init__(
        self,
        inputs: Sequence[tuple[PathLike, int]],
        split: Literal['train', 'test'],
        img_size: int = 768,
        crop_size: Optional[int] = None,
        random_crop: bool = False,
        imreader: Callable = tiff.imread,
        transform: Optional[Callable] = None,
        bit_depth: Optional[int] = None,
        normalize: Optional[Literal['minmax', 'std']] = None,
        dataset_stats: Optional[tuple[float, float]] = None,
        return_label: bool = True,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.inputs= inputs
        self.split = split
        self.img_size = img_size
        self.crop_size = crop_size
        self.transform = transform
        self.bit_depth = bit_depth
        self.normalize = normalize
        self.dataset_stats = dataset_stats
        self.imreader = imreader
        self.return_label = return_label
        self.random_crop = random_crop
        
        # Force transform to be None for test split
        if self.split == 'test':
            self.transform = None
            self.random_crop = False
    
    def read_file(self, fpath: PathLike) -> torch.Tensor:
        """Read an image file and preprocess it."""
        img: NDArray = self.imreader(fpath)
        
        # resize to img_size if necessary
        if img.shape != (self.img_size, self.img_size):
            img = resize_img(img, self.img_size)
        
        # crop to crop_size if necessary
        if self.img_size != self.crop_size:
            img = crop_img(img, self.crop_size, self.random_crop)
        
        # normalize the image range into [0, 1]
        if self.bit_depth is not None:
            img = normalize_range(img, self.bit_depth)
            self.dataset_stats = [
                stat / (2**self.bit_depth - 1) for stat in self.dataset_stats
            ]
            
        # normalize the image using the specified method
        if self.normalize is not None:
            img = normalize_img(
                img, self.normalize, self.dataset_stats
            )
        
        return torch.tensor(img, dtype=torch.float32)[None, ...] # add channel dim
    
    def __getitem__(self, idx: int):
        fpath, label = self.inputs[idx]
        # TODO: replace with function that processes chunks of images at once (?)
        image = self.read_file(fpath)
        
        if self.transform is not None:
            image = self.transform(image)
   
        if self.return_label:
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.inputs)