import os
import json
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from numpy.typing import NDArray
from torch.utils.data.dataset import Dataset

from protein_classification.data.utils import (
    crop_img, normalize_img, resize_img
)
from protein_classification.utils.typing import PathLike


# TODO: deal with stratified/balanced sampling of the dataset

class PreTrainingDataset(Dataset):
    """Dataset for pre-training of protein classification model.
    
    Parameters
    ----------
    data_dir : PathLike
        Path to the directory containing the dataset.
    inputs : Union[PathLike, Sequence[tuple[PathLike, int]]]
        Path to a csv file containing the input images filenames and labels, or tuples
        of image filename and label index (optional for test set) for each sample.
    labels : PathLike
        Path to a json file containing the all the labels for the classification task as
        "ID: label_name".
    split : Literal['train', 'test']
        The split of the dataset, either 'train' or 'test'.
    img_size : int, optional
        The size of the input images, by default 2048. If the input images are not of this size,
        they will be resized to this size.
    crop_size : int, optional
        The size of the crops on which the model is trained, by default 256.
    imreader : Callable, optional
        Function to read images from filepaths as `NDArray` arrays. 
        By default `tiff.imread`.
    transform : Optional[Callable], optional
        ???
    normalize : Literal['range', 'minmax', 'std'], optional
        The normalization method to apply to the images.
        - 'range': scales unsigned integer images into [0, 1] by dividing by the range.
        - 'minmax': scales images to [0, 1] based on the min and max values.
        - 'std': standardizes images to have zero mean and unit variance.
        By default 'range'.
    random_crop : bool, optional
        Whether to apply random cropping to the images. If `True`, crop size will be
        randomly sampled between `crop_size` and `img_size` and applied to the images.
        By default `False`.
    """
    def __init__(
        self,
        data_dir: PathLike,
        inputs: Union[PathLike, list[tuple[PathLike, Optional[int]]]],
        labels: Union[PathLike, dict[int, str]],
        split: Literal['train', 'test'],
        img_size: int = 2048,
        crop_size: Optional[int] = None,
        imreader: Callable = tiff.imread,
        transform: Optional[Callable] = None,
        normalize: Literal['range', 'minmax', 'std'] = 'range',
        random_crop: bool = False,
        return_label: bool = True,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.crop_size = crop_size
        self.transform = transform
        self.normalize = normalize
        self.imreader = imreader
        self.return_label = return_label
        self.random_crop = random_crop
        
        # extract image paths and labels
        if isinstance(inputs, (list, tuple)):
            self.input_fpaths = [
                os.path.join(self.data_dir, img_fname) for img_fname, _ in inputs
            ]
            self.input_labels = [label for _, label in inputs]
        elif isinstance(inputs, (str, Path)):
            assert str(inputs).endswith('.csv'), "Input file must be a CSV file."
            inputs = pd.read_csv(inputs, header=None)
            self.input_fpaths = [
                os.path.join(self.data_dir, fname)
                for fname in inputs.iloc[:, 0].tolist()
            ]
            self.input_labels = inputs.iloc[:, 1].tolist()
            
        # extract labels
        if isinstance(labels, dict):
            self.labels = labels
        elif isinstance(labels, (str, Path)):
            assert str(labels).endswith('.json'), "Labels file must be a JSON file."
            with open(labels, 'r') as f:
                self.labels = json.load(f)
    
    def read_file(self, fpath: PathLike) -> torch.Tensor:
        """Read an image file and preprocess it."""
        img: NDArray = self.imreader(fpath)
        
        # resize to img_size if necessary
        if img.shape != (self.img_size, self.img_size):
            img = resize_img(img, self.img_size)
        
        # crop to crop_size if necessary
        if self.img_size != self.crop_size:
            img = crop_img(img)
        
        # normalize the image
        img = normalize_img(img)
        
        return img 

    def __getitem__(self, idx: int):
        # TODO: replace with function that processes chunks of images at once
        image = self.read_file(self.input_fpaths[idx])
        if self.transform is not None:
            image = self.transform(image)
   
        if self.return_label:
            label = self.labels[idx]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.input_fpaths)
