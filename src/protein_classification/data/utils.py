import random as rnd
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import tifffile as tiff
import torch
from numpy.typing import NDArray
from skimage.transform import resize
from torch import Tensor
from tqdm import tqdm

from protein_classification.config.data import DataAugmentationConfig


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
    img: NDArray | Tensor,
    method: Literal["minmax", "std"],
    scope: Literal["dataset", "image"],
    dataset_stats: Optional[tuple[float, float]] = None,
) -> NDArray | Tensor:
    """Normalize an image using dataset-level or per-image statistics."""
    stats = _get_normalization_stats(img, method, dataset_stats, scope)
    if method == 'minmax':
        min_val, max_val = stats
        return _minmax_normalize(img, min_val, max_val)
    elif method == 'std':
        mean, std = stats
        return _std_normalize(img, mean, std)
    else:
        raise ValueError(f"Unavailable normalization method: {method}")


def _get_normalization_stats(
    img: NDArray | Tensor,
    method: Literal["minmax", "std"],
    scope: Literal["dataset", "image"],
    dataset_stats: Optional[tuple[float, float]],
) -> tuple[float, float]:
    """Get the statistics used for normalization."""
    if scope == "dataset":
        if dataset_stats is None:
            raise ValueError(
                "Dataset statistics must be provided for dataset normalization."
            )
        return dataset_stats

    if scope != "image":
        raise ValueError(f"Unknown normalization scope: {scope}")

    if isinstance(img, torch.Tensor):
        if method == "minmax":
            return img.min().item(), img.max().item()
        return img.mean().item(), img.std().item()

    if method == "minmax":
        return float(np.min(img)), float(np.max(img))
    return float(np.mean(img)), float(np.std(img))


def _minmax_normalize(
    img: NDArray | Tensor, min_val: float, max_val: float
) -> NDArray | Tensor:
    """Apply min-max normalization to an image using dataset statistics."""
    denom = max(max_val - min_val, 1e-8)
    return (img - min_val) / denom


def _std_normalize(
    img: NDArray | Tensor, mean: float, std: float
) -> NDArray | Tensor:
    """Apply standard normalization to an image using dataset statistics."""
    return (img - mean) / max(std, 1e-8)


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
    
    if img.shape[-2:] == (crop_size, crop_size):
        return img
    
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
    metrics: list[Literal["std", "entropy"]] = ["std"]
) -> float:
    """Compute the difficulty score of an image as the combination of the specified metrics."""
    image = normalize_img(image, "minmax", "image")
    score = 0.0
    if "std" in metrics:
        score += image.std().item()
    if "entropy" in metrics:
        score += compute_shannon_entropy(image)
    return score


def compute_shannon_entropy(
    image: Tensor,
    num_bins: int = 32,
    eps: float = 1e-8,
) -> float:
    """Compute Shannon entropy on a min-max normalized patch."""
    image = normalize_img(image, "minmax", "image")
    hist = torch.histc(image, bins=num_bins, min=0.0, max=1.0)
    probs = hist / hist.sum().clamp_min(eps)
    probs = probs[probs > 0]
    entropy = -(probs * torch.log2(probs.clamp_min(eps))).sum()
    return float(entropy.item())

def get_difficulty_score_distribution(
        images: list[Tensor],
        labels: list[int],
        crop_size: int,
        random_crop: bool,
        k: int = 10,
        metrics: list[Literal["std", "entropy"]] = ["std"],
        bins: int = 100
    ) -> dict[int, torch.Tensor]:
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
    images : list[torch.Tensor]
    labels : list[int]
    crop_size : int
    random_crop : bool
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
    dict[int, torch.Tensor]
        A dictionary of tensors of shape (bins + 1,) representing the quantiles of the
        difficulty scores for each label computed over the sampled crops.
    """
    difficulty_scores: dict[int, list[float]] = {
        label: [] for label in set(labels)
    }
    for img, label in tqdm(
        zip(images, labels),
        desc="Computing difficulty distribution",
        total=len(images)
    ):
        for _ in range(k):
            crop = crop_img(img, crop_size, random_crop)
            difficulty_scores[label].append(compute_difficulty_score(crop, metrics))

    difficulty_scores = {
        label: torch.tensor(scores, dtype=torch.float32)
        for label, scores in difficulty_scores.items()
    }
    return {
        label: torch.quantile(scores, torch.linspace(0, 1, bins + 1), interpolation='linear')
        for label, scores in difficulty_scores.items()
    }


def get_curriculum_learning_crops(
    image: Tensor,
    crop_size: int,
    difficulty_distrib: Union[None, list[float]] = None,
    metrics: list[Literal["std", "entropy"]] = ["std"],
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
    crop = crop_img(image, crop_size, random_crop=True)
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


def get_overlapping_crops(
    image: Tensor, crop_size: int, overlap: int = None
) -> Tensor:
    """Extract multiple overlapping crops from the input image tensor.
    
    Used at test time for ensemble predictions.

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


def identify_background_crops(
    image: Tensor,
    label: int,
    crop_size: int,
    metrics: list[Literal["std", "entropy"]] = ["std"],
    threshold: Optional[float] = None,
    thresholds_by_label: Optional[dict[int, float]] = None,
    difficulty_distribution: Optional[list[float]] = None,
    bg_label: int = -1
) -> tuple[Tensor, int]:
    """Apply cropping to an image. If the crop doesn't contain enough signal
    then it is assigned the background label.
    
    The signal threshold is either provided as input to the function or
    computed from the difficulty distribution of the dataset.
    """
    if thresholds_by_label is not None:
        threshold = thresholds_by_label.get(int(label), threshold)

    if threshold is None:
        if difficulty_distribution is None:
            raise ValueError(
                "Either `threshold` or `difficulty_distribution` must be provided."
            )
        threshold = difficulty_distribution[10] # 10% quantile
        
    crop = crop_img(image, crop_size, random_crop=True)
    score = compute_difficulty_score(crop, metrics)
    if score < threshold:
        return crop, bg_label
    else:
        return crop, label


def compute_background_thresholds(
    inputs: Sequence[tuple[Union[str, Path], int]],
    img_size: int,
    crop_size: int,
    random_crop: bool,
    metrics: list[Literal["std", "entropy"]],
    quantile: float = 0.1,
    samples_per_image: int = 4,
    max_images: Optional[int] = None,
    imreader: Callable[[Union[str, Path]], Union[NDArray, Tensor]] = tiff.imread,
) -> dict[int, float]:
    """Precompute per-label background thresholds from training data.

    Thresholds are computed as score quantiles over random crops from the train split.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("`quantile` must be in [0, 1].")

    score_by_label: dict[int, list[float]] = defaultdict(list)
    if max_images is not None:
        selected_inputs = list(rnd.shuffle(inputs)[:max_images])
    else:
        selected_inputs = list(inputs)

    for fpath, label in tqdm(selected_inputs, desc="Computing background thresholds"):
        image = imreader(fpath)
        if isinstance(image, torch.Tensor):
            image_tensor = image.to(torch.float32)
        else:
            if img_size is not None and image.shape != (img_size, img_size):
                image = resize_img(image, img_size)
            image_tensor = torch.tensor(image, dtype=torch.float32)

        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D image tensor, got shape {tuple(image_tensor.shape)}"
            )

        if img_size is not None and image_tensor.shape[-2:] != (img_size, img_size):
            resized = resize_img(image_tensor.squeeze(0).cpu().numpy(), img_size)
            image_tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)

        for _ in range(samples_per_image):
            crop = crop_img(image_tensor, crop_size, random_crop)
            score_by_label[int(label)].append(compute_difficulty_score(crop, metrics))

    return {
        label: float(np.quantile(scores, quantile))
        for label, scores in score_by_label.items()
        if scores
    }


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


def collate_multi_crop_batches(
    batch: list[tuple[Tensor, Tensor]]
) -> tuple[Tensor, Tensor]:
    """Flatten a batch of multi-crop samples into a standard image batch.

    Each dataset item contains a tensor of crops with shape ``(K, C, H, W)`` and
    the corresponding labels with shape ``(K,)``. This collate function concatenates
    all crops and labels across the batch into ``(N, C, H, W)`` and ``(N,)``.
    """
    crops, labels = zip(*batch)
    return torch.cat(crops, dim=0), torch.cat(labels, dim=0)
