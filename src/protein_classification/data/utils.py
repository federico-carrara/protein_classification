import os
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from skimage.transform import resize


def normalize_img(
    img: NDArray, norm_type: Literal['range', 'minmax', 'std']
) -> NDArray:
    """Normalize an image based on the specified normalization method."""
    if norm_type == 'range':
        if img.dtype == np.uint8:
            img = img / 2**8 - 1
        elif img.dtype == np.uint16:
            img = img / 2**16 - 1
        elif img.dtype == np.uint32:
            img = img / 2**32 - 1
        else:
            raise ValueError(
                f"""Range normalization is only supported for unsigned integer dtype.
                Found dtype: {img.dtype}"""
            )
    elif norm_type == 'minmax':
        # FIXME: requires dataset statistics for robustness
        img = (img - img.min()) / (img.max() - img.min())
    elif norm_type == 'std':
        # FIXME: requires dataset statistics for robustness
        img = (img - img.mean()) / img.std()
    return img


def crop_img(img: NDArray, crop_size: int, random_crop: bool) -> NDArray:
    """Crop a squared image to a square of size `crop_size`."""
    assert img.shape[0] == img.shape[1], "Image must be square."
    
    img_size = img.shape[0]
    if random_crop:
        crop_size = int(np.random.uniform(crop_size, img_size))
    
    x = np.random.randint(0, img_size - crop_size + 1)
    y = np.random.randint(0, img_size - crop_size + 1)
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


