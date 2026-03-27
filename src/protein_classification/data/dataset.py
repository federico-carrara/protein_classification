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
    compute_background_thresholds,
    compute_background_score,
    crop_img,
    normalize_img,
    resize_img,
)

PathLike = Union[Path, str]


class BaseTiffDataset(Dataset):
    """Lazy TIFF-backed dataset with multi-crop sampling.

    Each dataset item corresponds to one source image. The image is loaded on demand,
    optionally resized, and used to extract a stack of crops.
    """

    _MAX_BACKGROUND_REJECTION_RETRIES = 10

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

        background_thresholds = getattr(self, "background_thresholds_by_label", None)
        if (
            source_label is None or
            background_thresholds is None or
            self.augmentation_config.background_rejection_prob <= 0.0
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
                self.augmentation_config.background_metrics,
            )
            if score >= threshold:
                return crop, label
            if random.random() >= self.augmentation_config.background_rejection_prob:
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
        positive_probability: float = 0.5,
        negative_family_weights: Optional[dict[str, float]] = None,
        mixed_alpha_range: tuple[float, float] = (0.3, 0.7),
        mixed_num_sources: int = 2,
        imreader: Callable[[PathLike], Union[NDArray, Tensor]] = tiff.imread,
        bit_depth: Optional[int] = None,
        normalize: Optional[Literal["minmax", "std"]] = None,
        normalization_scope: Literal["dataset", "image"] = "dataset",
        dataset_stats: Optional[tuple[float, float]] = None,
        return_label: bool = True,
    ) -> None:
        self.target_label = int(target_label)
        self.positive_probability = float(positive_probability)
        self.negative_family_weights = negative_family_weights or {
            "easy": 1.0,
            "mixed": 1.0,
            "inverted": 1.0,
        }
        self.mixed_alpha_range = mixed_alpha_range
        self.mixed_num_sources = mixed_num_sources # TODO: sample variable number of sources up to this max
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
        self._validate_binary_sampling_config()
        self.target_indices = [
            idx for idx, (_, label) in enumerate(self.inputs)
            if int(label) == self.target_label
        ]
        self.non_target_indices = [
            idx for idx, (_, label) in enumerate(self.inputs)
            if int(label) != self.target_label
        ]
        if not self.target_indices:
            raise ValueError("BinaryDataset requires at least one target-class sample.")
        if not self.non_target_indices:
            raise ValueError("BinaryDataset requires at least one non-target sample.")
        self.background_thresholds_by_label = self._compute_background_thresholds()

    def _transform_label(self, label: int) -> int:
        return int(label == self.target_label)

    def _validate_binary_sampling_config(self) -> None:
        """Validate binary negative-sampling configuration."""
        valid_families = {"easy", "mixed", "inverted"}
        if not 0.0 <= self.positive_probability <= 1.0:
            raise ValueError("`positive_probability` must be in [0, 1].")
        if self.mixed_num_sources < 2:
            raise ValueError("`mixed_num_sources` must be >= 2.")
        if len(self.mixed_alpha_range) != 2:
            raise ValueError("`mixed_alpha_range` must be a tuple of length 2.")
        alpha_min, alpha_max = self.mixed_alpha_range
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError("`mixed_alpha_range` values must satisfy 0 <= min <= max <= 1.")
        unknown_families = set(self.negative_family_weights) - valid_families
        if unknown_families:
            raise ValueError(
                f"Unknown negative families: {sorted(unknown_families)}."
            )
        if any(weight < 0 for weight in self.negative_family_weights.values()):
            raise ValueError("Negative family weights must be non-negative.")
        if sum(self.negative_family_weights.values()) <= 0:
            raise ValueError("At least one negative family weight must be positive.")

    def _sample_index(self, indices: list[int]) -> int:
        """Sample one dataset index from a pool."""
        return random.choice(indices)

    def _sample_negative_family(self) -> str:
        """Sample which negative family to generate."""
        families = list(self.negative_family_weights.keys())
        weights = list(self.negative_family_weights.values())
        return random.choices(families, weights=weights, k=1)[0]

    def _sample_positive_crop(self) -> tuple[Tensor, int]:
        """Sample one positive crop from the target class."""
        idx = self._sample_index(self.target_indices)
        image = self._load_image(idx)
        source_label = int(self.inputs[idx][1])
        crop, _ = self._sample_single_crop(image, label=1, source_label=source_label)
        return crop, 1

    def _sample_easy_negative_crop(self) -> tuple[Tensor, int]:
        """Sample one easy negative crop from a non-target class."""
        idx = self._sample_index(self.non_target_indices)
        image = self._load_image(idx)
        source_label = int(self.inputs[idx][1])
        crop, _ = self._sample_single_crop(image, label=0, source_label=source_label)
        return crop, 0

    def _sample_inverted_negative_crop(self) -> tuple[Tensor, int]:
        """Sample one inverted negative crop from the target class."""
        crop, _ = self._sample_positive_crop()
        crop = normalize_img(crop, "minmax", "image")
        crop = 1.0 - crop
        return crop, 0

    def _sample_mixed_negative_crop(self) -> tuple[Tensor, int]:
        """Sample one mixed negative crop from cropped patches."""
        target_crop, _ = self._sample_positive_crop()
        target_crop = normalize_img(target_crop, "minmax", "image")

        aux_crops: list[Tensor] = []
        num_sources = random.randint(1, self.mixed_num_sources - 1)
        for _ in range(num_sources):
            idx = self._sample_index(self.non_target_indices)
            image = self._load_image(idx)
            source_label = int(self.inputs[idx][1])
            aux_crop, _ = self._sample_single_crop(image, label=0, source_label=source_label)
            aux_crops.append(normalize_img(aux_crop, "minmax", "image"))

        alpha = random.uniform(*self.mixed_alpha_range)
        aux_mean = torch.stack(aux_crops).mean(dim=0)
        mixed_crop = alpha * target_crop + (1.0 - alpha) * aux_mean
        return mixed_crop, 0

    def _sample_binary_crop(self) -> tuple[Tensor, int]:
        """Sample one crop according to the binary sample policy."""
        if random.random() < self.positive_probability:
            return self._sample_positive_crop()

        family = self._sample_negative_family()
        if family == "easy":
            return self._sample_easy_negative_crop()
        if family == "mixed":
            return self._sample_mixed_negative_crop()
        if family == "inverted":
            return self._sample_inverted_negative_crop()
        raise ValueError(f"Unknown negative family: {family}")

    def __getitem__(self, idx: int) -> Union[Tensor, tuple[Tensor, Tensor]]:
        crops: list[Tensor] = []
        labels: list[int] = []
        for _ in range(self.num_crops_per_image):
            crop, label = self._sample_binary_crop()
            crops.append(crop)
            labels.append(label)

        processed_crops = self._apply_per_crop_processing(torch.stack(crops))
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if self.return_label:
            return processed_crops, labels_tensor
        return processed_crops

    def _compute_background_thresholds(self) -> dict[int, float]:
        """Precompute per-original-label thresholds used for crop rejection."""
        if self.augmentation_config.crop_size is None:
            return {}

        return compute_background_thresholds(
            inputs=self.inputs,
            img_size=self.img_size,
            crop_size=self.augmentation_config.crop_size,
            random_crop=self.augmentation_config.random_crop,
            metrics=self.augmentation_config.background_metrics,
            quantile=self.augmentation_config.background_threshold_quantile,
            samples_per_image=self.augmentation_config.background_threshold_samples_per_image,
            max_images=self.augmentation_config.background_threshold_max_images,
            imreader=self.imreader,
        )
