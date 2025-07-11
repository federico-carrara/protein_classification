import random
from typing import Callable, Optional, Union

import torch
from torch import Tensor


def transforms_factory(
    name: Optional[str] = None
) -> Optional[Callable[[Tensor, Optional[Tensor]], Union[Tensor, tuple[Tensor, Tensor]]]]:
    """Factory function to get the appropriate transformation function."""
    if name is None:
        return None
    elif name == 'geometric':
        return geometric_augmentation
    elif name == 'noise':
        return noise_augmentation
    elif name == 'all':
        return train_augmentation
    else:
        raise ValueError(f"Unknown transformation: {name}")


def train_augmentation(
    image: Tensor, mask: Optional[Tensor] = None, bit_depth: int = 8
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Apply a random augmentation to image (and optional mask).

    Parameters
    ----------
    image : Tensor
        Input image tensor of shape (C, H, W).
    bit_depth : int
        Bit depth of the image, used to scale noise.
    mask : Optional[Tensor]
        Optional mask tensor of shape (H, W) or (1, H, W).

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor]]
        Augmented image, and mask if provided.
    """
    image = noise_augmentation(image, mask, bit_depth)
    image = geometric_augmentation(image, mask)
    return (image, mask) if mask is not None else image


def geometric_augmentation(
    image: Tensor, mask: Optional[Tensor] = None, **kwargs
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
    image: Tensor, mask: Optional[Tensor] = None, bit_depth: int = 8
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Add random noise to the image."""
    image = add_poisson_noise(image, bit_depth)
    image = add_background(image, bit_depth)
    image = add_gaussian_noise(image, bit_depth)
    if mask is not None:
        return image, mask
    return image

def add_background(
    image: Tensor,
    bit_depth: int = 8,
    intensity_range: tuple[float, float] = (1e-3, 1e-1),
) -> Tensor:
    """Simulate uneven background by adding constant or low-frequency bias.

    Parameters
    ----------
    image : Tensor
        Input image tensor, unnormalized.
    bit_depth : int
        Bit depth of the image, used to scale the background intensity.
    intensity_range : Tuple[float, float]
        Range of background intensity to sample from.

    Returns
    -------
    Tensor
        Image with added background intensity.
    """
    roof = 2 ** bit_depth - 1
    intensity = random.uniform(*intensity_range) * roof
    background = torch.full_like(image, intensity)
    return torch.clamp(image + background, 0.0, roof)

def add_poisson_noise(
    image: Tensor,
    bit_depth: int = 8,
    scale_range: tuple[float, float] = (0.0, 100.0)
) -> Tensor:
    """Add Poisson noise to simulate photon noise in microscopy.
    
    NOTE: the closer the scale is to 0, the more noise is added.

    Parameters
    ----------
    image : Tensor
        Input image tensor, unnormalized.
    bit_depth : int
        Bit depth of the image, used to scale the noise.
    scale_range : Tuple[float, float]
        Range of peak count (signal scale) to simulate photon flux.

    Returns
    -------
    Tensor
        Image with Poisson noise applied.
    """
    roof = 2 ** bit_depth - 1
    # TODO: sample from a distribution that gives more prob to lower values?
    scale = random.uniform(*scale_range)
    image_scaled = image * scale
    noisy = torch.poisson(image_scaled)
    return torch.clamp(noisy / scale, 0.0, roof)

def add_gaussian_noise(
    image: Tensor,
    bit_depth: int = 8,
    std_range: tuple[float, float] = (1e-3, 1e-1)
) -> Tensor:
    """Add Gaussian noise to simulate read noise.

    Parameters
    ----------
    image : Tensor
        Input image tensor, unnormalized.
    bit_depth : int
        Bit depth of the image, used to scale the noise.
    std_range : Tuple[float, float]
        Range of standard deviations for Gaussian noise.

    Returns
    -------
    Tensor
        Image with Gaussian noise applied.
    """
    roof = 2 ** bit_depth - 1
    std = random.uniform(*std_range) * roof
    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, 0.0, roof)