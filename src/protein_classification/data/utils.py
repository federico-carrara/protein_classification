import os
from typing import Literal

import numpy as np
import torch
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


def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    if len(image.shape) == 3 :
        image = np.transpose(image, (1, 2, 0))
    image = image*std + mean
    image = image.astype(dtype=np.uint8)
    return image

def tensor_to_label(tensor):
    label = tensor.numpy()
    label = label.astype(dtype=np.uint8)
    return label

## transform (input is numpy array, read in by cv2)
def image_to_tensor(image, mean=0, std=1.):
    image = image.astype(np.float32)
    image = (image-mean)/std
    if len(image.shape) == 3:
        image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image)

    return tensor

def label_to_tensor(label, threshold=0.5):
    label_ret  = (label>threshold).astype(np.float32)
    label_ret[label<0]=-1.0
    tensor = torch.from_numpy(label_ret).type(torch.FloatTensor)
    return tensor


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


