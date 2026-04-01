import random
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
    compute_background_score,
    crop_img,
    normalize_img,
    resize_img,
)

PathLike = Union[Path, str]


class MultiClassDataset(Dataset):
    """Lazy TIFF-backed dataset with multi-crop sampling for multiclass classification.

    Each dataset item corresponds to one source image. The image is loaded on demand,
    optionally resized, and used to extract a stack of crops.
    """

    _MAX_BACKGROUND_REJECTION_RETRIES = 10

    # TODO: cleanup args by simply passing the data config
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
        background_rejection_prob: float = 1.0,
        background_threshold_quantile: float = 0.1,
        background_threshold_quantiles_by_label: Optional[dict[int, float]] = None,
        background_threshold_samples_per_image: int = 4,
        background_threshold_max_images: Optional[int] = 50,
        background_metrics: Optional[list[Literal["std", "entropy"]]] = None,
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
        self.background_rejection_prob = background_rejection_prob
        self.background_threshold_quantile = background_threshold_quantile
        self.background_threshold_quantiles_by_label = background_threshold_quantiles_by_label
        self.background_threshold_samples_per_image = background_threshold_samples_per_image
        self.background_threshold_max_images = background_threshold_max_images
        self.background_metrics = background_metrics or ["std"]
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

        crops: list[Tensor] = []
        labels: list[int] = []
        for _ in range(self.num_crops_per_image):
            crop, crop_label = self._sample_single_crop(image, label)
            crops.append(crop)
            labels.append(crop_label)

        return torch.stack(crops), torch.tensor(labels, dtype=torch.long)

    def _sample_single_crop(
        self,
        image: Tensor,
        label: int,
        source_label: Optional[int] = None,
    ) -> tuple[Tensor, int]:
        """Extract a single crop from an image."""
        crop_size = self.augmentation_config.crop_size
        if crop_size is None:
            return image, label

        background_thresholds: dict[int, float] = getattr(
            self, "background_thresholds_by_label", None
        )
        if (
            source_label is None or
            background_thresholds is None or
            self.background_rejection_prob <= 0.0
        ):
            crop = crop_img(
                image,
                crop_size,
                self.augmentation_config.random_crop,
            )
            return crop, label

        threshold = background_thresholds.get(int(source_label))
        if threshold is None:
            crop = crop_img(
                image,
                crop_size,
                self.augmentation_config.random_crop,
            )
            return crop, label

        last_crop: Optional[Tensor] = None
        for _ in range(self._MAX_BACKGROUND_REJECTION_RETRIES):
            crop = crop_img(
                image,
                crop_size,
                self.augmentation_config.random_crop,
            )
            last_crop = crop
            score = compute_background_score(
                crop,
                self.background_metrics,
            )
            if score >= threshold:
                return crop, label
            if random.random() >= self.background_rejection_prob:
                return crop, label

        assert last_crop is not None
        return last_crop, label

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
