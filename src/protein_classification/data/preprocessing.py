from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import tifffile as tiff
import torch
import zarr
from numcodecs import Blosc
from numpy.typing import NDArray
from tqdm import tqdm

from protein_classification.data.utils import normalize_img, resize_img

PathLike = Union[Path, str]


class ZarrPreprocessor:
    """Dataset for pre-training of protein classification model.
    
    Parameters
    ----------
    inputs : Sequence[tuple[PathLike, int]]
        Sequence of tuples of image filename and label index (optional for test set)
        for each sample.
    output_path: PathLike
        Path to the temporary output Zarr file where the preprocessed data will be stored.
    img_size : int, optional
        The size of the input images, by default 768. If the input images are not of this size,
        they will be resized to this size.
    imreader : Callable, optional
        Function to read images from filepaths as `NDArray` arrays. 
        By default `tiff.imread`.
    normalize : Literal['range', 'minmax', 'std'], optional
        The normalization method to apply to the images.
        - 'minmax': scales images to [0, 1] based on the min and max values.
        - 'std': standardizes images to have zero mean and unit variance.
        By default 'range'.
    dataset_stats : Optional[tuple[float, float]], optional
        Pre-computed dataset statistics (mean, std) or (min, max) for normalization.
    chunk_size : int, optional
        The size of the chunks to use for the Zarr dataset. Defaults to 1.
    """
    def __init__(
        self,
        inputs: Sequence[tuple[PathLike, int]],
        output_path: PathLike,
        img_size: int = 768,
        imreader: Callable = tiff.imread,
        normalize: Optional[Literal['minmax', 'std']] = None,
        dataset_stats: Optional[tuple[float, float]] = None,
        chunk_size: int = 1,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.inputs = inputs
        self.output_path = Path(output_path)
        self.img_size = img_size
        self.normalize = normalize
        self.dataset_stats = dataset_stats
        self.imreader = imreader
        self.chunk_size = chunk_size
    
    def preprocess_file(self, fpath: PathLike) -> torch.Tensor:
        """Read an image file and preprocess it."""
        img: NDArray = self.imreader(fpath)
        
        # normalize the image using the specified method
        if self.normalize is not None:
            img = normalize_img(
                img, self.normalize, self.dataset_stats
            )
        
        # resize to img_size if necessary
        if img.shape != (self.img_size, self.img_size):
            img = resize_img(img, self.img_size)
        
        return img.astype(np.float32)
    
    def run(self) -> Path:
        num_samples = len(self.inputs)
        h, w = self.img_size, self.img_size
        shape = (num_samples, h, w)

        compressor = Blosc(cname="zstd", clevel=3)
        chunks = (self.chunk_size, h, w)

        # Create a group
        # TODO: add LRUCache for chunks
        zarr_group = zarr.open_group(self.output_path, mode="w")

        # Create image array
        image_array = zarr_group.create_dataset(
            name="images",
            shape=shape,
            dtype="float32",
            chunks=chunks,
            compressor=compressor,
        )

        # Create label array (int64 to match PyTorch conventions)
        label_array = zarr_group.create_dataset(
            name="labels",
            shape=(num_samples,),
            dtype="int64",
            compressor=None,  # Labels are tiny — no need for compression
        )

        for i, (fpath, label) in enumerate(tqdm(self.inputs, desc="Preprocessing data")):
            try:
                img_tensor = self.preprocess_file(fpath)
                image_array[i] = img_tensor
                label_array[i] = label
            except Exception as e:
                print(f"[WARN] Failed to preprocess {fpath}: {e}")
                raise

        return self.output_path