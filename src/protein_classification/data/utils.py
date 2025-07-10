from typing import Literal

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize


def normalize_range(
    img: NDArray, bit_depth: int = 8
) -> NDArray:
    """Normalize the intensity range of an `uint` image between [0, 1]."""
    max_val = 2**bit_depth - 1
    assert np.max(img) <= max_val, (
        "Image values exceed the maximum for the specified bit depth."
        f" Max value: {np.max(img)}, expected <= {max_val}."
    )
    return img / max_val


def normalize_img(
    img: NDArray, method: Literal["minmax", "std"], dataset_stats: tuple[float, float]
) -> NDArray:
    """Normalize an image using the specified method."""
    assert dataset_stats is not None, (
        "Dataset statistics must be provided for normalization."
    )
    if method == 'minmax':
        min_val, max_val = dataset_stats
        return _minmax_normalize(img, min_val, max_val)
    elif method == 'std':
        mean, std = dataset_stats
        return _std_normalize(img, mean, std)
    else:
        raise ValueError(f"Unavailable normalization method: {method}")


def _minmax_normalize(
    img: NDArray, min_val: float, max_val: float
) -> NDArray:
    """Apply min-max normalization to an image using dataset statistics."""
    return (img - min_val) / (max_val - min_val)


def _std_normalize(
    img: NDArray, mean: float, std: float
) -> NDArray:
    """Apply standard normalization to an image using dataset statistics."""
    return (img - mean) / std


def crop_img(img: NDArray, crop_size: int, random_crop: bool) -> NDArray:
    """Crop a squared image to a square of size `crop_size`."""
    assert img.shape[0] == img.shape[1], "Image must be square."
    
    img_size = img.shape[0]
    if random_crop:
        x = np.random.randint(0, img_size - crop_size + 1)
        y = np.random.randint(0, img_size - crop_size + 1)
    else:
        x = (img_size - crop_size) // 2
        y = (img_size - crop_size) // 2
    return img[y:y + crop_size, x:x + crop_size]


def resize_img(img: NDArray, size: int) -> NDArray:
    """Resize an image to a square of size `size`."""
    return resize(
        img, (size, size),
        order=1,
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True
    )
    

def train_test_split(
    inputs: list[tuple[str, int]],
    train_ratio: float = 0.8,
    deterministic: bool = False
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split the dataset into training and testing sets."""
    n_train = int(train_ratio * len(inputs))
    if not deterministic:
        inputs = np.random.permutation(len(inputs))
    train_data = inputs[:n_train]
    test_data = inputs[n_train:]
    return train_data, test_data