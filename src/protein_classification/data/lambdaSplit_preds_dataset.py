from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import tifffile as tiff
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

from protein_classification.config.data import DataAugmentationConfig
from protein_classification.data.utils import (
    crop_img, normalize_img, resize_img
)

PathLike = Union[Path, str]


class LambdaSplitPredsDataset(Dataset):
    """Dataset for protein classification model where inputs are loaded from memory.
        
    Parameters
    ----------
    data_path: PathLike
        Path to the directory containing the `.npz` file of λSplit predictions.
    ch_to_labels: dict[int, int]
        Dictionary mapping channel indices to label indices.
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
        data_path: PathLike,
        ch_to_labels: dict[int, int],
        split: Literal['train', 'test'],
        img_size: int,
        augmentation_config: DataAugmentationConfig,
        bit_depth: Optional[int] = None,
        normalize: Optional[Literal['minmax', 'std']] = None,
        dataset_stats: Optional[tuple[float, float]] = None,
        return_label: bool = True,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.data_path = data_path
        self.ch_to_labels = ch_to_labels
        self.split = split
        self.img_size = img_size
        self.bit_depth = bit_depth
        self.normalize = normalize
        self.return_label = return_label
        self.augmentation_config = augmentation_config
        
        # Force test_time_crop to be False for train split
        if self.split == 'train':
            self.test_time_crop = False
        
        # Force transform to be None for test split
        if self.split == 'test':
            self.transform = None
            self.random_crop = False
        
        # Read, preprocess and store the images and labels in memory
        self.images, self.labels = self.read_data()
        
        # Get dataset statistics for normalization
        if dataset_stats is None:
            self.dataset_stats = self._compute_img_stats()
        else:
            self.dataset_stats = dataset_stats
    
    def read_data(self) -> tuple[list[torch.Tensor], list[int]]:
        """Read data and preprocess them."""
        # Load predictions from the .npz file
        preds_data = np.load(self.data_path)
        preds_data = [img for img in preds_data.values()]
        
        # Get single-channel images and labels
        preds_data = [
            (
                torch.tensor(img[i], dtype=torch.float32).unsqueeze(0),
                self.ch_to_labels[i]
            ) 
            for img in preds_data
            for i in range(img.shape[0])
        ]
        images, labels = zip(*preds_data)
        return list(images), list(labels)
    
    def _compute_img_stats(self) -> tuple[float, float]:
        """Compute image statistics for normalization."""
        assert hasattr(self, "images"), "Images not loaded. Call `read_data()` first."
        
        if self.normalize is None:
            return None, None
        elif self.normalize == "minmax":
            all_images = torch.cat(self.images, dim=0)
            min_val = all_images.min().item()
            max_val = all_images.max().item()
            return min_val, max_val
        elif self.normalize == "std":
            all_images = torch.cat(self.images, dim=0)
            mean_val = all_images.mean().item()
            std_val = all_images.std().item()
            return mean_val, std_val
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize}")
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple[torch.Tensor, int]]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # apply cropping augmentation
        image = crop_augmentation(image, self.augmentation_config)
         
        # apply data augmentation
        if self.transform is not None:
            image = self.transform(image, bit_depth=self.bit_depth)
            
        # normalize image
        if self.normalize is not None:
            image = normalize_img(image, self.normalize, self.dataset_stats)
   
        if self.return_label:
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.labels)