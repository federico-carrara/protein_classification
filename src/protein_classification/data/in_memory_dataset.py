from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import tifffile as tiff
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

from protein_classification.data.utils import (
    crop_img, normalize_img, resize_img, test_time_crop_img, train_time_crop_img
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
        test_time_crop: bool = False,
        curriculum_learning: bool = False,
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
        self.test_time_crop = test_time_crop
        self.curriculum_learning = curriculum_learning 
        
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
        if curriculum_learning:
            self.difficulty_distribution = self._get_difficulty_distribution()
        else:
            self.difficulty_distribution = None
    
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
        """Get the distribution of "difficulty" of crops from images in the dataset.

        The difficulty of a crop is assumed to be inversely related to the amount of
        signal present in it. Indeed, we assume that foreground crops with more signal
        are easier to classify with respect to background crops.
        The amount of signal can be measured by a mix of texture and variability
        metrics, like edge detection, standard deviation, entropy, etc.

        In order to compute the difficulty distribution, for each image we randomly
        sample `k` crops of size `crop_size` and compute their difficulty metric.
        
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
                    img, self.crop_size, random_crop=self.random_crop
                )
                curr_score = 0.0
                if "std" in metrics:
                    curr_score += crop.std().item()
                # TODO: add more metrics here
                difficulty_scores.append(curr_score)

        difficulty_scores = torch.tensor(difficulty_scores, dtype=torch.float32)
        return torch.quantile(
            difficulty_scores, torch.linspace(0, 1, bins + 1), interpolation='linear'
        )

    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple[torch.Tensor, int]]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # crop to crop_size if necessary
        if self.crop_size is not None and self.img_size != self.crop_size:
            if self.split == 'train':
                image = train_time_crop_img(
                    image,
                    crop_size=self.crop_size,
                    random_crop=self.random_crop,
                    difficulty_distrib=self.difficulty_distribution
                )
            elif self.split == 'test' and self.test_time_crop:
                image = test_time_crop_img(image, self.crop_size)
         
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