from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import tifffile as tiff
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data.dataset import Dataset

from protein_classification.config.data import DataAugmentationConfig
from protein_classification.data.augmentations import (
    geometric_augmentation,
    intensity_augmentation,
    noise_augmentation,
)
from protein_classification.data.utils import (
    crop_img,
    get_overlapping_crops,
    identify_background_crops,
    normalize_img,
    resize_img,
)

PathLike = Union[Path, str]


class BaseTiffDataset(Dataset):
    """Lazy TIFF-backed dataset with multi-crop sampling.

    Each dataset item corresponds to one source image. The image is loaded on demand,
    optionally resized, and used to extract a stack of crops.
    """

    def __init__(
        self,
        inputs: Sequence[tuple[PathLike, int]],
        split: Literal["train", "test"],
        img_size: int,
        augmentation_config: DataAugmentationConfig,
        num_crops_per_image: int,
        imreader: Callable[[PathLike], Union[NDArray, Tensor]] = tiff.imread,
        bit_depth: Optional[int] = None,
        normalize: Optional[Literal["minmax", "std"]] = None,
        normalization_scope: Literal["dataset", "image"] = "dataset",
        dataset_stats: Optional[tuple[float, float]] = None,
        return_label: bool = True,
    ) -> None:
        super().__init__()
        if num_crops_per_image < 1:
            raise ValueError("`num_crops_per_image` must be >= 1.")

        self.inputs = list(inputs)
        self.split = split
        self.img_size = img_size
        self.bit_depth = bit_depth
        self.normalize = normalize
        self.normalization_scope = normalization_scope
        self.dataset_stats = dataset_stats
        self.imreader = imreader
        self.return_label = return_label
        self.augmentation_config = augmentation_config
        self.num_crops_per_image = num_crops_per_image

    def _transform_label(self, label: int) -> int:
        """Map the source label to the task-specific label."""
        return int(label)

    def _load_image(self, idx: int) -> Tensor:
        """Load one TIFF image lazily and return it as ``(C, H, W)`` float tensor."""
        fpath, _ = self.inputs[idx]
        image = self.imreader(fpath)
        if isinstance(image, torch.Tensor):
            image_tensor = image.to(torch.float32)
        else:
            if self.img_size is not None and image.shape != (self.img_size, self.img_size):
                image = resize_img(image, self.img_size)
            image_tensor = torch.tensor(image, dtype=torch.float32)

        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim != 3:
            raise ValueError(f"Expected 2D or 3D image tensor, got shape {tuple(image_tensor.shape)}")

        if self.img_size is not None and image_tensor.shape[-2:] != (self.img_size, self.img_size):
            resized = resize_img(image_tensor.squeeze(0).cpu().numpy(), self.img_size)
            image_tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)

        return image_tensor

    def _sample_crops(self, image: Tensor, label: int) -> tuple[Tensor, Tensor]:
        """Extract a stack of crops and the corresponding repeated labels."""
        crop_size = self.augmentation_config.crop_size
        if crop_size is None:
            crops = image.unsqueeze(0)
            labels = torch.tensor(label, dtype=torch.long).unsqueeze(0)
            return crops, labels

        # TODO: 
        strategy = self.augmentation_config.strategy
        if strategy == "overlap": # used for inference
            crops = get_overlapping_crops(
                image,
                crop_size,
                self.augmentation_config.crop_overlap,
            )
            labels = torch.full((crops.shape[0],), label, dtype=torch.long)
            return crops, labels

        crops: list[Tensor] = []
        labels: list[int] = []
        for _ in range(self.num_crops_per_image):
            if strategy == "background":
                crop, crop_label = identify_background_crops(
                    image,
                    label,
                    crop_size=crop_size,
                    metrics=self.augmentation_config.metrics,
                    threshold=self.augmentation_config.bg_threshold,
                    difficulty_distribution=None,
                    bg_label=-1,
                )
            elif strategy == "curriculum":
                raise NotImplementedError(
                    "Curriculum cropping is not supported in the lazy TIFF dataset yet."
                )
            else:
                crop = crop_img(
                    image,
                    crop_size,
                    self.augmentation_config.random_crop,
                )
                crop_label = label
            crops.append(crop)
            labels.append(crop_label)

        return torch.stack(crops), torch.tensor(labels, dtype=torch.long)

    def _apply_per_crop_processing(self, crops: Tensor) -> Tensor:
        """Apply transforms and normalization independently to each crop."""
        processed_crops: list[Tensor] = []
        for crop in crops:
            transform_name = None if self.split == "test" else self.augmentation_config.transform

            if transform_name in {"geometric", "all"}:
                crop = geometric_augmentation(crop)

            if transform_name in {"intensity", "noise", "all"}:
                crop = normalize_img(crop, "minmax", "image")
                if transform_name in {"intensity", "all"}:
                    crop = intensity_augmentation(crop)
                if transform_name in {"noise", "all"}:
                    crop = noise_augmentation(crop)

            if self.normalize is not None:
                crop = normalize_img(
                    crop,
                    self.normalize,
                    self.normalization_scope,
                    self.dataset_stats,
                )
            processed_crops.append(crop.to(torch.float32))
        return torch.stack(processed_crops)

    def __getitem__(self, idx: int) -> Union[Tensor, tuple[Tensor, Tensor]]:
        image = self._load_image(idx)
        _, source_label = self.inputs[idx]
        label = self._transform_label(source_label)

        crops, labels = self._sample_crops(image, label)
        crops = self._apply_per_crop_processing(crops)

        if self.return_label:
            return crops, labels
        return crops

    def __len__(self) -> int:
        return len(self.inputs)


class MultiClassDataset(BaseTiffDataset):
    """Lazy TIFF dataset for multiclass classification."""


class BinaryDataset(BaseTiffDataset):
    """Lazy TIFF dataset for one-vs-rest binary classification."""

    def __init__(
        self,
        inputs: Sequence[tuple[PathLike, int]],
        split: Literal["train", "test"],
        img_size: int,
        augmentation_config: DataAugmentationConfig,
        num_crops_per_image: int,
        target_label: int,
        imreader: Callable[[PathLike], Union[NDArray, Tensor]] = tiff.imread,
        bit_depth: Optional[int] = None,
        normalize: Optional[Literal["minmax", "std"]] = None,
        normalization_scope: Literal["dataset", "image"] = "dataset",
        dataset_stats: Optional[tuple[float, float]] = None,
        return_label: bool = True,
    ) -> None:
        self.target_label = int(target_label)
        super().__init__(
            inputs=inputs,
            split=split,
            img_size=img_size,
            augmentation_config=augmentation_config,
            num_crops_per_image=num_crops_per_image,
            imreader=imreader,
            bit_depth=bit_depth,
            normalize=normalize,
            normalization_scope=normalization_scope,
            dataset_stats=dataset_stats,
            return_label=return_label,
        )

    def _transform_label(self, label: int) -> int:
        return int(label == self.target_label)
