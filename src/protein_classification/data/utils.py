from typing import Literal, Union

import numpy as np
import torch
from numpy.typing import NDArray
from skimage.transform import resize
from torch import Tensor


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


def crop_img(img: NDArray | Tensor, crop_size: int, random_crop: bool) -> NDArray | Tensor:
    """Crop a squared image to a square of size `crop_size`.
    
    Parameters
    ----------
    img : NDArray | Tensor
        The input image to crop, shaped as (C, Y, X).
    crop_size : int
        The size of the square crop to extract from the image.
    random_crop : bool
        If `True`, a random crop is taken from the image.
        If `False`, the center crop is taken.
        
    Returns
    -------
    NDArray | Tensor
        The cropped image, shaped as (C, crop_size, crop_size).
    """
    assert img.shape[-1] == img.shape[-2], "Image must be square."
    
    img_size = img.shape[-1]
    if random_crop:
        x = np.random.randint(0, img_size - crop_size + 1)
        y = np.random.randint(0, img_size - crop_size + 1)
    else:
        x = (img_size - crop_size) // 2
        y = (img_size - crop_size) // 2
    return img[:, y:y + crop_size, x:x + crop_size]



def compute_difficulty_score(
    image: Tensor,
    metrics: list[Literal["std"]] = ["std"]
) -> float:
    """Compute the difficulty score of an image as the combination of the specified metrics."""
    # TODO: add more metrics
    score = 0.0
    if "std" in metrics:
        score += image.std().item()
    return score


def curriculum_learning_sampling(
    image: Tensor,
    crop_size: int,
    difficulty_distrib: Union[None, list[float]] = None,
    metrics: list[Literal["std"]] = ["std"],
    epoch: int = 0,
    total_epochs: int = 100,
    beta_max_alpha: float = 5.0,
    sampling_patience: int = 10
) -> Tensor:
    """Sample a crop from an image for curriculum learning using Beta-distributed
    threshold rejection sampling.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape (C, Y, X).
    crop_size : int
        Square crop size.
    difficulty_distrib : list[float], optional
        Sorted array of difficulty scores = empirical CDF (quantile function).
    metrics : list[Literal["std"]], optional
        A list of metrics to combine in order to compute the difficulty score.
        By default ["std"].
    epoch : int
        Current epoch (0-indexed).
    total_epochs : int
        Total number of epochs.
    beta_max_alpha : float
        Initial Beta(α, 1) skew; α anneals from `beta_max_alpha` to 1.
    sampling_patience : int
        Maximum number of crops to sample before giving up on finding a suitable crop.

    Returns
    -------
    torch.Tensor
        Cropped tensor of shape (C, crop_size, crop_size).
    """
    # Training progress
    p = min(epoch / total_epochs, 1.0)
    alpha = 1.0 + (beta_max_alpha - 1.0) * (1.0 - p)

    found = False
    crops: list[Tensor] = [crop]
    scores: list[float] = [compute_difficulty_score(crop, metrics)]
    while not found and len(crops) < sampling_patience:
        # Sample quantile from Beta(α, 1)
        q = np.random.beta(alpha, 1.0)

        # Use quantile to get threshold (difficulty_distrib is sorted CDF)
        idx = int(q * (len(difficulty_distrib) - 1))
        threshold = difficulty_distrib[idx]

        if scores[-1] >= threshold:
            found = True
        else:
            crops.append(crop_img(image, crop_size, random_crop=True))
            scores.append(compute_difficulty_score(crops[-1], metrics))

    if not found:
        crop = crops[np.argmax(scores)]   
    else:
        crop = crops[-1]
        
    return crop


def train_time_crop_img(
    image: Tensor,
    crop_size: int,
    random_crop: bool = True,
    curriculum_learning: bool = True,
    difficulty_distrib: Union[None, list[float]] = None,
    metrics: list[Literal["std"]] = ["std"],
    epoch: int = 0,
    total_epochs: int = 100,
    beta_max_alpha: float = 5.0,
    sampling_patience: int = 10
) -> Tensor:
    """
    Train-time cropping with soft curriculum using Beta-distributed threshold sampling.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape (C, Y, X).
    crop_size : int
        Square crop size.
    random_crop : bool
        If True, use random cropping; otherwise, center crop.
    curriculum_learning : bool
        If True, use curriculum learning sampling; otherwise, use standard cropping.
    difficulty_distrib : list[float], optional
        Sorted array of difficulty scores = empirical CDF (quantile function).
    metrics : list[Literal["std"]], optional
        A list of metrics to combine in order to compute the difficulty score.
        By default ["std"].
    epoch : int
        Current epoch (0-indexed).
    total_epochs : int
        Total number of epochs.
    beta_max_alpha : float
        Initial Beta(α, 1) skew; α anneals from `beta_max_alpha` to 1.
    sampling_patience : int
        Maximum number of crops to sample before giving up on finding a suitable crop.

    Returns
    -------
    torch.Tensor
        Cropped tensor of shape (C, crop_size, crop_size).
    """
    crop = crop_img(image, crop_size, random_crop)
    
    if curriculum_learning:
        if difficulty_distrib is None:
            raise ValueError("Difficulty distribution must be provided for curriculum learning.")
        crop = curriculum_learning_sampling(
            crop,
            crop_size,
            difficulty_distrib,
            metrics,
            epoch,
            total_epochs,
            beta_max_alpha,
            sampling_patience
        )
    
    return crop


def test_time_crop_img(image: Tensor, crop_size: int, overlap: int = None) -> Tensor:
    """Test time cropping, consiting in the extraction of overlapping crops from the
    input image tensor.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape (C, Y, X).
    crop_size : int
        Height and width of square crops.
    overlap : int, optional
        Overlap between crops (default: crop_size // 4).

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, C, crop_size, crop_size), where N is the number of crops
    """
    if overlap is None:
        overlap = crop_size // 4

    C, H, W = image.shape
    stride = crop_size - overlap

    # Calculate number of steps in each dimension
    h_steps = max(1, (H - overlap) // stride)
    w_steps = max(1, (W - overlap) // stride)

    crops: list[Tensor] = []
    for i in range(h_steps + 1):
        for j in range(w_steps + 1):
            top = min(i * stride, H - crop_size)
            left = min(j * stride, W - crop_size)

            crop = image[:, top:top + crop_size, left:left + crop_size]
            crops.append(crop)

    return torch.stack(crops)  # Shape: (N, C, crop_size, crop_size)


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
        random_idxs = np.random.permutation(len(inputs))
        inputs = [inputs[i] for i in random_idxs]
    train_data = inputs[:n_train]
    test_data = inputs[n_train:]
    return train_data, test_data


def collate_test_time_crops(
    batch: list[tuple[list[Tensor], int]]
) -> tuple[Tensor, Tensor]:
    """Collate function for overlapping crops used at test time.

    Parameters
    ----------
    batch : tuple[list[Tensor], int]
        A batch, where each item is a tuple including the list of overlapping crops as
        a tensors of shape (N_i, C, crop_size, crop_size) and an integer label.

    Returns
    -------
    tuple[Tensor, Tensor]
        Concatenated tensor of crops of shape (B * N_i, C, crop_size, crop_size) and
        a tensor of labels of shape (B * N_i,).
    """
    crops, labels = zip(*batch)
    labels = torch.tensor(labels).repeat_interleave(
        torch.tensor([len(crop) for crop in crops])
    )
    crops = torch.cat(crops, dim=0)
    return crops, labels