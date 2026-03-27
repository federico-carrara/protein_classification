import math
import random
from collections import defaultdict
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
            crop, crop_label = self._sample_single_crop(image, label)
            crops.append(crop)
            labels.append(crop_label)

        return torch.stack(crops), torch.tensor(labels, dtype=torch.long)

    def _sample_single_crop(self, image: Tensor, label: int) -> tuple[Tensor, int]:
        """Extract a single crop from an image."""
        crop_size = self.augmentation_config.crop_size
        if crop_size is None:
            return image, label

        strategy = self.augmentation_config.strategy
        if strategy == "background":
            return identify_background_crops(
                image,
                label,
                crop_size=crop_size,
                metrics=self.augmentation_config.metrics,
                threshold=self.augmentation_config.bg_threshold,
                thresholds_by_label=self.augmentation_config.bg_thresholds,
                difficulty_distribution=None,
                bg_label=-1,
            )
        if strategy == "curriculum":
            raise NotImplementedError(
                "Curriculum cropping is not supported in the lazy TIFF dataset yet."
            )
        crop = crop_img(
            image,
            crop_size,
            self.augmentation_config.random_crop,
        )
        return crop, label

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
        num_source_files_per_group_target: int = 4,
        num_source_files_per_group_non_target: int = 8,
        num_crops_per_source_image: int = 4,
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
        self.mixed_num_sources = mixed_num_sources
        self.num_source_files_per_group_target = num_source_files_per_group_target
        self.num_source_files_per_group_non_target = num_source_files_per_group_non_target
        self.num_crops_per_source_image = num_crops_per_source_image
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
        
        # Split indices into target (positive class) and non-target (others)
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
        
        # Maintain a list of indices of source files for each label for stratified sampling
        self.indices_by_label: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(self.inputs):
            self.indices_by_label[int(label)].append(idx)
        
        self._reset_source_pools()
        self._clear_reservoir()

    def _transform_label(self, label: int) -> int:
        return int(label == self.target_label)

    def _validate_binary_sampling_config(self) -> None:
        """Validate binary negative-sampling configuration."""
        valid_families = {"easy", "mixed", "inverted"}
        if not 0.0 <= self.positive_probability <= 1.0:
            raise ValueError("`positive_probability` must be in [0, 1].")
        if self.mixed_num_sources < 2:
            raise ValueError("`mixed_num_sources` must be >= 2.")
        if self.num_source_files_per_group_target < 1:
            raise ValueError("`num_source_files_per_group_target` must be >= 1.")
        if self.num_source_files_per_group_non_target < 1:
            raise ValueError("`num_source_files_per_group_non_target` must be >= 1.")
        if self.num_crops_per_source_image < 1:
            raise ValueError("`num_crops_per_source_image` must be >= 1.")
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

    def _reset_source_pools(self) -> None:
        """Reset source-file pools for a new pass."""
        self.remaining_target_indices = self.target_indices.copy()
        random.shuffle(self.remaining_target_indices)
        self.remaining_non_target_indices_by_label = {
            label: indices.copy()
            for label, indices in self.indices_by_label.items()
            if label != self.target_label
        }
        for indices in self.remaining_non_target_indices_by_label.values():
            random.shuffle(indices)
        self.current_group_id = 0

    def _clear_reservoir(self) -> None:
        """Clear the current in-memory crop reservoir."""
        self.positive_pool: list[Tensor] = []
        self.negative_pool: list[Tensor] = []
        self.negative_pool_by_label: dict[int, list[Tensor]] = defaultdict(list)

    def _pop_without_replacement(self, pool: list) -> Optional[object]:
        """Pop one random element from a list without replacement."""
        if not pool:
            return None
        idx = random.randrange(len(pool))
        pool[idx], pool[-1] = pool[-1], pool[idx]
        return pool.pop()

    def _draw_target_group_indices(self, quota: int) -> list[int]:
        """Draw target indices without replacement within the current pass."""
        selected: list[int] = []
        while len(selected) < quota:
            if not self.remaining_target_indices:
                if selected:
                    break
                # Reset only the target pool to avoid corrupting the non-target pass.
                self.remaining_target_indices = self.target_indices.copy()
                random.shuffle(self.remaining_target_indices)
            draw = min(quota - len(selected), len(self.remaining_target_indices))
            selected.extend(self.remaining_target_indices[:draw])
            del self.remaining_target_indices[:draw]
        return selected

    def _draw_non_target_group_indices(self, quota: int) -> list[int]:
        """Draw non-target indices approximately stratified by label."""
        selected: list[int] = []
        while len(selected) < quota:
            available_labels = [
                label for label, indices in self.remaining_non_target_indices_by_label.items()
                if indices
            ]
            if not available_labels:
                if selected:
                    break
                # Reset only the non-target pool to avoid corrupting the target pass.
                self.remaining_non_target_indices_by_label = {
                    label: indices.copy()
                    for label, indices in self.indices_by_label.items()
                    if label != self.target_label
                }
                for indices in self.remaining_non_target_indices_by_label.values():
                    random.shuffle(indices)
                available_labels = [
                    label for label, indices in self.remaining_non_target_indices_by_label.items()
                    if indices
                ]
                if not available_labels:
                    break

            random.shuffle(available_labels)
            progressed = False
            for label in available_labels:
                if len(selected) >= quota:
                    break
                indices = self.remaining_non_target_indices_by_label[label]
                if not indices:
                    continue
                selected.append(indices.pop())
                progressed = True
            if not progressed:
                break
        return selected

    def _sample_source_group(self) -> tuple[list[int], list[int]]:
        """Sample the next stratified group of source files."""
        target_group = self._draw_target_group_indices(
            self.num_source_files_per_group_target
        )
        non_target_group = self._draw_non_target_group_indices(
            self.num_source_files_per_group_non_target
        )
        if not target_group or not non_target_group:
            self._reset_source_pools()
            target_group = self._draw_target_group_indices(
                self.num_source_files_per_group_target
            )
            non_target_group = self._draw_non_target_group_indices(
                self.num_source_files_per_group_non_target
            )
        self.current_group_id += 1
        return target_group, non_target_group

    def _materialize_reservoir(self) -> None:
        """Load one source group and append crops to the existing reservoir pools."""
        target_group, non_target_group = self._sample_source_group()

        for idx in target_group:
            image = self._load_image(idx)
            for _ in range(self.num_crops_per_source_image):
                crop, _ = self._sample_single_crop(image, label=1)
                self.positive_pool.append(crop)

        for idx in non_target_group:
            image = self._load_image(idx)
            source_label = int(self.inputs[idx][1])
            for _ in range(self.num_crops_per_source_image):
                crop, _ = self._sample_single_crop(image, label=0)
                self.negative_pool.append(crop)
                self.negative_pool_by_label[source_label].append(crop)

    def _sample_negative_family(self) -> str:
        """Sample which negative family to generate."""
        families = list(self.negative_family_weights.keys())
        weights = list(self.negative_family_weights.values())
        return random.choices(families, weights=weights, k=1)[0]

    def _can_sample_family(self, family: str) -> bool:
        """Check whether the current reservoir can satisfy a sample family."""
        if family == "positive":
            return len(self.positive_pool) >= 1
        if family == "easy":
            return len(self.negative_pool) >= 1
        if family == "mixed":
            return (
                len(self.positive_pool) >= 1 and
                len(self.negative_pool) >= (self.mixed_num_sources - 1)
            )
        if family == "inverted":
            return len(self.positive_pool) >= 1
        raise ValueError(f"Unknown sample family: {family}")

    def _ensure_reservoir_for_family(self, family: str) -> None:
        """Ensure the current reservoir can satisfy the requested family."""
        if self._can_sample_family(family):
            return
        self._materialize_reservoir()
        if not self._can_sample_family(family):
            raise RuntimeError(
                f"Unable to satisfy sample family `{family}` from the current reservoir."
            )

    def _pop_positive_crop(self) -> Tensor:
        """Pop one positive crop from the reservoir."""
        self._ensure_reservoir_for_family("positive")
        crop = self._pop_without_replacement(self.positive_pool)
        assert isinstance(crop, torch.Tensor)
        return crop

    def _pop_negative_crop(self) -> Tensor:
        """Pop one negative crop from the reservoir."""
        self._ensure_reservoir_for_family("easy")
        crop = self._pop_without_replacement(self.negative_pool)
        assert isinstance(crop, torch.Tensor)
        return crop

    def _sample_positive_crop(self) -> tuple[Tensor, int]:
        """Sample one positive crop from the reservoir."""
        return self._pop_positive_crop(), 1

    def _sample_easy_negative_crop(self) -> tuple[Tensor, int]:
        """Sample one easy negative crop from the reservoir."""
        return self._pop_negative_crop(), 0

    def _sample_inverted_negative_crop(self) -> tuple[Tensor, int]:
        """Sample one inverted negative crop from the reservoir."""
        self._ensure_reservoir_for_family("inverted")
        crop = self._pop_positive_crop()
        crop = normalize_img(crop, "minmax", "image")
        crop = 1.0 - crop
        return crop, 0

    def _sample_mixed_negative_crop(self) -> tuple[Tensor, int]:
        """Sample one mixed negative crop from reservoir crops."""
        self._ensure_reservoir_for_family("mixed")
        target_crop = normalize_img(self._pop_positive_crop(), "minmax", "image")

        aux_crops: list[Tensor] = []
        for _ in range(self.mixed_num_sources - 1):
            aux_crop = self._pop_negative_crop()
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
        if self.augmentation_config.strategy == "overlap": # TODO: create a different dataset for inference for simplicity and maintainability
            return super().__getitem__(idx)

        if not self.positive_pool and not self.negative_pool:
            self._materialize_reservoir()

        crops: list[Tensor] = []
        labels: list[int] = []
        for _ in range(self.num_crops_per_image): # TODO: drop, as we can now sample from the reservoir
            crop, label = self._sample_binary_crop()
            crops.append(crop)
            labels.append(label)

        processed_crops = self._apply_per_crop_processing(torch.stack(crops))
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if self.return_label:
            return processed_crops, labels_tensor
        return processed_crops

    def __len__(self) -> int:
        """Approximate one dataset pass as one pass over reservoir source groups."""
        n_target_groups = math.ceil(
            len(self.target_indices) / self.num_source_files_per_group_target
        )
        n_non_target_groups = math.ceil(
            len(self.non_target_indices) / self.num_source_files_per_group_non_target
        )
        return max(n_target_groups, n_non_target_groups)
