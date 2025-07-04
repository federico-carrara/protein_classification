import torch
from torch import Tensor
from typing import Optional, Union
import random


def train_augmentation(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Apply a random augmentation to image (and optional mask).

    Parameters
    ----------
    image : Tensor
        Input image tensor of shape (C, H, W).
    mask : Optional[Tensor]
        Optional mask tensor of shape (H, W) or (1, H, W).

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor]]
        Augmented image, and mask if provided.
    """
    image = noise_augmentation(image, mask)
    image = geometric_augmentation(image, mask)
    return (image, mask) if mask is not None else image


def geometric_augmentation(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Apply a random geometric augmentation to image (and optional mask).

    Parameters
    ----------
    image : Tensor
        Input image tensor of shape (C, H, W).
    mask : Optional[Tensor]
        Optional mask tensor of shape (H, W) or (1, H, W).

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor]]
        Augmented image, and mask if provided.
    """
    augment_func_list = [
        augment_default,
        augment_flipud,
        augment_fliplr,
        augment_transpose,
        augment_flipud_lr,
        augment_flipud_transpose,
        augment_fliplr_transpose,
        augment_flipud_lr_transpose,
        augment_rotate,
    ]
    aug_func = random.choice(augment_func_list)
    return aug_func(image, mask)


def augment_default(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """No augmentation applied."""
    return (image, mask) if mask is not None else image

def augment_flipud(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Flip image (and mask) vertically (up-down)."""
    image = torch.flip(image, dims=[1])
    if mask is not None:
        mask = torch.flip(mask, dims=[0])
        return image, mask
    return image

def augment_fliplr(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Flip image (and mask) horizontally (left-right)."""
    image = torch.flip(image, dims=[2])
    if mask is not None:
        mask = torch.flip(mask, dims=[1])
        return image, mask
    return image

def augment_transpose(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Transpose image (and mask) along height and width axes."""
    image = image.transpose(1, 2)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.transpose(0, 1)
        else:
            mask = mask.transpose(1, 2)
        return image, mask
    return image

def augment_flipud_lr(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Flip image (and mask) vertically and horizontally."""
    image = torch.flip(image, dims=[1, 2])
    if mask is not None:
        mask = torch.flip(mask, dims=[0, 1])
        return image, mask
    return image

def augment_flipud_transpose(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Flip vertically then transpose image (and mask)."""
    out = augment_flipud(image, mask)
    return augment_transpose(*out) if mask is not None else augment_transpose(out)

def augment_fliplr_transpose(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Flip horizontally then transpose image (and mask)."""
    out = augment_fliplr(image, mask)
    return augment_transpose(*out) if mask is not None else augment_transpose(out)

def augment_flipud_lr_transpose(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Flip both axes then transpose image (and mask)."""
    out = augment_flipud_lr(image, mask)
    return augment_transpose(*out) if mask is not None else augment_transpose(out)

def augment_rotate(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Rotate image (and mask) by 90, 180, or 270 degrees."""
    k = random.choice([1, 2, 3])  # number of 90° rotations
    image = torch.rot90(image, k=k, dims=[1, 2])
    if mask is not None:
        mask = torch.rot90(mask, k=k, dims=[0, 1])
        return image, mask
    return image


def noise_augmentation(
    image: Tensor, mask: Optional[Tensor] = None
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Add random noise to the image."""
    # TODO: consider swapping order of background and poisson
    image = add_background(image)
    image = add_poisson_noise(image)
    image = add_gaussian_noise(image)
    if mask is not None:
        return image, mask
    return image

def add_background(
    image: Tensor, intensity_range: tuple[float, float] = (1e-3, 1e-2)
) -> Tensor:
    """Simulate uneven background by adding constant or low-frequency bias.

    Parameters
    ----------
    image : Tensor
        Input image tensor, values in [0, 1].
    intensity_range : Tuple[float, float]
        Range of background intensity to sample from.

    Returns
    -------
    Tensor
        Image with added background intensity.
    """
    intensity = random.uniform(*intensity_range)
    background = torch.full_like(image, intensity)
    return torch.clamp(image + background, 0.0, 1.0)

def add_poisson_noise(
    image: Tensor, scale_range: tuple[float, float] = (30.0, 100.0)
) -> Tensor:
    """Add Poisson noise to simulate photon noise in microscopy.

    Parameters
    ----------
    image : Tensor
        Input image tensor, values in [0, 1].
    scale_range : Tuple[float, float]
        Range of peak count (signal scale) to simulate photon flux.

    Returns
    -------
    Tensor
        Image with Poisson noise applied.
    """
    scale = random.uniform(*scale_range)
    image_scaled = image * scale
    noisy = torch.poisson(image_scaled)
    return torch.clamp(noisy / scale, 0.0, 1.0)

def add_gaussian_noise(
    image: Tensor,
    std_range: tuple[float, float] = (1e-5, 2e-4)
) -> Tensor:
    """Add Gaussian noise to simulate read noise.

    Parameters
    ----------
    image : Tensor
        Input image tensor, values in [0, 1].
    std_range : Tuple[float, float]
        Range of standard deviations for Gaussian noise.

    Returns
    -------
    Tensor
        Image with Gaussian noise applied.
    """
    std = random.uniform(*std_range)
    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, 0.0, 1.0)