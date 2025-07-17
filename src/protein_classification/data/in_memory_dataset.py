from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import tifffile as tiff
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch import Tensor

from protein_classification.config.data import DataAugmentationConfig
from protein_classification.data.augmentations import transforms_factory
from protein_classification.data.utils import (
    compute_difficulty_score, crop_img, crop_augmentation, normalize_img, resize_img
)

PathLike = Union[Path, str]


# TODO: deal with stratified/balanced sampling of the dataset
class InMemoryDataset(Dataset):
    """Dataset for protein classification model where inputs are loaded from memory.
        
    Parameters
    ----------
    inputs : Sequence[tuple[PathLike, int]]
        Sequence of tuples of image filename and label index (optional for test set)
        for each sample.
    split : Literal['train', 'test']
        The split of the dataset, either 'train' or 'test'.
    img_size : int
        The size of the input images. If the input images are not of this size, they
        will be resized to this size.
    augmentation_config : DataAugmentationConfig
        Configuration for data augmentation. If `None`, no augmentation is applied.
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
    return_label : bool, optional
        Whether to return the label along with the image. If `False`, only the image is
        returned. By default `True`.
    """
    def __init__(
        self,
        inputs: Sequence[tuple[PathLike, int]],
        split: Literal['train', 'test'],
        img_size: int,
        augmentation_config: DataAugmentationConfig,
        imreader: Callable[[PathLike], Union[NDArray, Tensor]] = tiff.imread,
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
        self.bit_depth = bit_depth
        self.normalize = normalize
        self.dataset_stats = dataset_stats
        self.imreader = imreader
        self.return_label = return_label
        self.augmentation_config = augmentation_config 
        self.current_epoch = 0 # used for curriculum learning
        
        # Get transformation function for augmentation
        self.transform = transforms_factory(self.augmentation_config.transform)
        
        # FIXME: checks these
        # Force test_time_crop to be False for train split
        if self.split == 'train':
            self.test_time_crop = False
        
        # Force transform to be None for test split
        if self.split == 'test':
            self.transform = None
            self.random_crop = False
        
        # Read, preprocess and store the images and labels in memory
        self.images, self.labels = self.read_data()
        
        # Get the difficulty distribution of the dataset for curriculum learning
        if self.augmentation_config.curriculum_learning:
            self.difficulty_distribution = self._get_difficulty_score_distribution()
        else:
            self.difficulty_distribution = None
            
    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for curriculum learning."""
        self.current_epoch = epoch
    
    def read_data(self) -> tuple[list[torch.Tensor], list[int]]:
        """Read data and preprocess them."""
        images: list[torch.Tensor] = []
        labels: list[int] = []
        for fpath, label in tqdm(self.inputs, desc="Reading inputs"):
            img: NDArray = self.imreader(fpath)
            
            # resize to img_size if necessary
            if self.img_size is not None and img.shape != (self.img_size, self.img_size):
                img = resize_img(img, self.img_size)
            
            images.append(
                torch.tensor(img, dtype=torch.float32)[None, ...] # add channel dim
            )
            labels.append(int(label))
            
        return images, labels
    
    def _get_difficulty_score_distribution(
        self, k: int = 10, metrics: list[Literal["std"]] = ["std"], bins: int = 100
    ) -> torch.Tensor:
        """Get the distribution of "difficulty" scores of crops.

        The difficulty of a crop is assumed to be inversely related to the amount of
        signal present in it. Indeed, we assume that foreground crops with more signal
        are easier to classify with respect to background crops.
        The amount of signal can be measured by a mix of texture and variability
        metrics, like edge detection, standard deviation, entropy, etc.

        In order to compute the difficulty distribution, for each image we randomly
        sample `k` crops of size `crop_size` and compute their difficulty score.

        The returned tensor contains the quantiles of the difficulty scores
        computed from the sampled crops. Larger values indicate easier crops.
        
        Parameters
        ----------
        k : int, optional
            The number of crops to sample from each image, by default 10.
        metrics : list[Literal["std"]], optional
            A list of metrics to combine in order to compute the difficulty score.
            By default ["std"].
        bins : int, optional
            The number of bins to use for the quantization of the difficulty distribution,
            by default 100.
            
        Returns
        -------
        torch.Tensor
            A tensor of shape (bins + 1,) containing the quantiles of the difficulty scores
            computed from the sampled crops.
        """
        difficulty_scores: list[float] = []
        for img in tqdm(self.images, desc="Computing difficulty distribution"):
            for _ in range(k):
                crop = crop_img(
                    img, self.augmentation_config.crop_size, self.augmentation_config.random_crop
                )
                difficulty_scores.append(compute_difficulty_score(crop, metrics))

        difficulty_scores = torch.tensor(difficulty_scores, dtype=torch.float32)
        return torch.quantile(
            difficulty_scores, torch.linspace(0, 1, bins + 1), interpolation='linear'
        )

    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple[torch.Tensor, int]]:
        image = self.images[idx]
        label = self.labels[idx]

        # apply cropping augmentation
        image = crop_augmentation(
            image,
            self.augmentation_config,
            self.current_epoch,
            self.difficulty_distribution
        )
         
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
        return len(self.inputs)