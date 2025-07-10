from pathlib import Path
from typing import Callable, Literal, Optional, Union

import torch
import zarr
from torch.utils.data.dataset import Dataset

from protein_classification.data.utils import crop_img

PathLike = Union[Path, str]


class ZarrDataset(Dataset):
    """Dataset for protein classification model where inputs come from a Zarr file.
    
    Parameters
    ----------
    path_to_zarr : PathLike
        Path to the Zarr file containing the preprocessed images and labels.
    split : Literal['train', 'test']
        The split of the dataset, either 'train' or 'test'.
    crop_size : int, optional
        The size of the crops used for training. If `None`, no cropping is applied.
    random_crop : bool, optional
        Whether to apply random cropping to the images. If `True`, crop size will be
        randomly sampled between `crop_size` and `img_size` and applied to the images.
        By default `False`.
    transform : Optional[Callable], optional
        A function/transform that takes in an image and returns a transformed version.
        Currently, the available transforms are:
        - `train_augmentation`: applies random noise and geometric augmentations.
        - `geometric_augmentation`: applies only random geometric augmentations.
        - `noise_augmentation`: applies only random noise augmentations.
        By default `None`, which means no transformation is applied.
    return_label : bool, optional
        Whether to return the label along with the image. If `False`, only the image is
        returned. By default `True`.
    """
    def __init__(
        self,
        path_to_zarr: PathLike,
        split: Literal['train', 'test'],
        crop_size: Optional[int] = None,
        random_crop: bool = False,
        transform: Optional[Callable] = None,
        return_label: bool = True,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.path_to_zarr = Path(path_to_zarr)
        self.split = split
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.transform = transform
        self.return_label = return_label
        
        # Force transform to be None for test split
        if self.split == 'test':
            self.transform = None
            self.random_crop = False
            
        # Load the Zarr dataset
        self.zarr_group = zarr.open_group(self.path_to_zarr, mode="r")
        self.images = self.zarr_group["images"]
        self.labels = self.zarr_group["labels"]
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple[torch.Tensor, int]]:
        """Get an item from the dataset."""
        image = torch.from_numpy(self.images[idx], dtype=torch.float32)
        label = int(self.labels[idx])
        
        if self.crop_size is not None and self.crop_size < image.shape[-1]:
            image = crop_img(image, self.crop_size, self.random_crop)
        
        if self.transform is not None:
            image = self.transform(image)
   
        if self.return_label:
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.labels)