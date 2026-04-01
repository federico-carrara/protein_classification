import random
from dataclasses import dataclass
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
from protein_classification.data.background import BackgroundAnalyzer
from protein_classification.data.utils import (
    crop_img,
    normalize_img,
    resize_img,
)

PathLike = Union[Path, str]


@dataclass(frozen=True, slots=True)
class CropRecipe:
    """Pre-computed recipe for a single crop."""

    family: str  # "positive", "trivial", "mixed", "inverted"
    source_indices: tuple[int, ...]  # dataset indices to load
    alpha: Optional[float] = None  # blending weight, only for "mixed"


class BinaryDataset(Dataset):
    """Lazy TIFF dataset for one-vs-rest binary classification."""

    # TODO: cleanup args by simply passing the data config
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
        background_analyzer: Optional[BackgroundAnalyzer] = None,
        background_metrics: Optional[list[Literal["std", "entropy"]]] = None,
        background_threshold_quantile: float = 0.05,
        background_threshold_quantiles_by_label: Optional[dict[int, float]] = None,
        background_threshold_max_images: Optional[int] = 50,
        background_stride: int = 16,
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
        self.background_metrics = background_metrics or ["std"]
        self.imreader = imreader
        self.return_label = return_label
        self.augmentation_config = augmentation_config
        self.num_crops_per_image = num_crops_per_image
        self.background_stride = background_stride

        self.target_label = int(target_label)
        self.positive_probability = float(positive_probability)
        self.negative_family_weights = negative_family_weights or {
            "trivial": 1.0,
            "mixed": 1.0,
            "inverted": 1.0,
        }
        self.mixed_alpha_range = mixed_alpha_range
        self.mixed_num_sources = mixed_num_sources  # TODO: sample variable number of sources up to this max

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

        # Background analysis: compute thresholds + valid crop positions
        if background_analyzer is not None:
            self.background_analyzer = background_analyzer
        elif self.augmentation_config.crop_size is not None:
            self.background_analyzer = BackgroundAnalyzer(
                inputs=self.inputs,
                crop_size=self.augmentation_config.crop_size,
                stride=self.background_stride,
                img_size=self.img_size,
                metrics=self.background_metrics,
                quantile=background_threshold_quantile,
                quantiles_by_label=background_threshold_quantiles_by_label,
                max_images_for_thresholds=background_threshold_max_images,
                imreader=self.imreader,
            )
        else:
            self.background_analyzer = None

        self._epoch = 0
        self._plan = self._build_epoch_plan()

    def _transform_label(self, label: int) -> int:
        return int(label == self.target_label)

    # TODO: move in pydantic config
    def _validate_binary_sampling_config(self) -> None:
        """Validate binary negative-sampling configuration."""
        valid_families = {"trivial", "mixed", "inverted"}
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

    def _sample_single_crop(
        self,
        image: Tensor,
        label: int,
        source_idx: Optional[int] = None,
    ) -> tuple[Tensor, int]:
        """Extract a single crop, preferring precomputed foreground positions.

        If *source_idx* is provided and a :class:`BackgroundAnalyzer` is
        available, a crop position is sampled from the precomputed valid
        positions (with jitter).  Otherwise falls back to a plain random or
        center crop.
        """
        crop_size = self.augmentation_config.crop_size
        if crop_size is None:
            return image, label

        _, h, w = image.shape

        # Try to use precomputed valid positions
        if (
            source_idx is not None
            and self.background_analyzer is not None
            and self.augmentation_config.random_crop
        ):
            valid_positions = self.background_analyzer.valid_positions_by_index.get(
                source_idx, []
            )
            if valid_positions:
                y, x = random.choice(valid_positions)
                # Add jitter within half a stride, clamped to image bounds
                half_stride = self.background_stride // 2
                if half_stride > 0:
                    y += random.randint(-half_stride, half_stride)
                    x += random.randint(-half_stride, half_stride)
                    y = max(0, min(y, h - crop_size))
                    x = max(0, min(x, w - crop_size))
                crop = image[:, y : y + crop_size, x : x + crop_size]
                return crop, label

        # Fallback: plain random or center crop
        crop = crop_img(image, crop_size, self.augmentation_config.random_crop)
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

    def _make_positive_crop(self, idx: int) -> tuple[Tensor, int]:
        """Make one positive crop from the target class at *idx*."""
        image = self._load_image(idx)
        crop, _ = self._sample_single_crop(image, label=1, source_idx=idx)
        return crop, 1

    def _make_trivial_negative_crop(self, idx: int) -> tuple[Tensor, int]:
        """Make one trivial negative crop from a non-target class at *idx*."""
        image = self._load_image(idx)
        crop, _ = self._sample_single_crop(image, label=0, source_idx=idx)
        return crop, 0

    def _make_inverted_negative_crop(self, idx: int) -> tuple[Tensor, int]:
        """Make one inverted negative crop from the target class at *idx*."""
        crop, _ = self._make_positive_crop(idx)
        crop = normalize_img(crop, "minmax", "image")
        crop = 1.0 - crop
        return crop, 0

    def _make_mixed_negative_crop(
        self, source_indices: tuple[int, ...], alpha: float,
    ) -> tuple[Tensor, int]:
        """Make one mixed negative crop from the given source indices."""
        target_crop, _ = self._make_positive_crop(source_indices[0])
        target_crop = normalize_img(target_crop, "minmax", "image")

        aux_crops: list[Tensor] = []
        for aux_idx in source_indices[1:]:
            image = self._load_image(aux_idx)
            aux_crop, _ = self._sample_single_crop(image, label=0, source_idx=aux_idx)
            aux_crops.append(normalize_img(aux_crop, "minmax", "image"))

        aux_mean = torch.stack(aux_crops).mean(dim=0)
        mixed_crop = alpha * target_crop + (1.0 - alpha) * aux_mean
        return mixed_crop, 0

    def _execute_recipe(self, recipe: CropRecipe) -> tuple[Tensor, int]:
        """Execute a single :class:`CropRecipe` and return ``(crop, label)``."""
        if recipe.family == "positive":
            return self._make_positive_crop(recipe.source_indices[0])
        if recipe.family == "trivial":
            return self._make_trivial_negative_crop(recipe.source_indices[0])
        if recipe.family == "inverted":
            return self._make_inverted_negative_crop(recipe.source_indices[0])
        if recipe.family == "mixed":
            assert recipe.alpha is not None
            return self._make_mixed_negative_crop(recipe.source_indices, recipe.alpha)
        raise ValueError(f"Unknown family: {recipe.family}")

    def __getitem__(self, idx: int) -> Union[Tensor, tuple[Tensor, Tensor]]:
        recipes = self._plan[idx]
        crops: list[Tensor] = []
        labels: list[int] = []
        for recipe in recipes:
            crop, label = self._execute_recipe(recipe)
            crops.append(crop)
            labels.append(label)

        processed_crops = self._apply_per_crop_processing(torch.stack(crops))
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if self.return_label:
            return processed_crops, labels_tensor
        return processed_crops

    def _build_epoch_plan(self) -> list[list[CropRecipe]]:
        """Build a deterministic sampling plan for the current epoch.

        The epoch is sized by the minority (target) pool: slots are generated
        until the target deck is exhausted.  Neither deck is replenished, so
        every target image appears exactly once per epoch.  The non-target deck
        is partially consumed; different subsets are covered across epochs
        thanks to reshuffling.
        """
        rng = random.Random(self._epoch)

        # Prepare shuffled decks (no replenishment)
        target_deck = list(self.target_indices)
        rng.shuffle(target_deck)
        target_cursor = 0

        non_target_deck = list(self.non_target_indices)
        rng.shuffle(non_target_deck)
        non_target_cursor = 0

        def target_remaining() -> int:
            return len(target_deck) - target_cursor

        def non_target_remaining() -> int:
            return len(non_target_deck) - non_target_cursor

        def draw_target() -> int:
            nonlocal target_cursor
            idx = target_deck[target_cursor]
            target_cursor += 1
            return idx

        def draw_non_target() -> int:
            nonlocal non_target_cursor
            idx = non_target_deck[non_target_cursor]
            non_target_cursor += 1
            return idx

        families = list(self.negative_family_weights.keys())
        weights = list(self.negative_family_weights.values())

        plan: list[list[CropRecipe]] = []
        while True:
            slot_recipes: list[CropRecipe] = []
            for _ in range(self.num_crops_per_image):
                if rng.random() < self.positive_probability:
                    if target_remaining() < 1:
                        break
                    recipe = CropRecipe(
                        family="positive",
                        source_indices=(draw_target(),),
                    )
                else:
                    family = rng.choices(families, weights=weights, k=1)[0]
                    if family == "trivial":
                        if non_target_remaining() < 1:
                            break
                        recipe = CropRecipe(
                            family="trivial",
                            source_indices=(draw_non_target(),),
                        )
                    elif family == "inverted":
                        if target_remaining() < 1:
                            break
                        recipe = CropRecipe(
                            family="inverted",
                            source_indices=(draw_target(),),
                        )
                    elif family == "mixed":
                        num_aux = rng.randint(1, self.mixed_num_sources - 1)
                        if target_remaining() < 1 or non_target_remaining() < num_aux:
                            break
                        target_idx = draw_target()
                        aux_indices = tuple(
                            draw_non_target() for _ in range(num_aux)
                        )
                        alpha = rng.uniform(*self.mixed_alpha_range)
                        recipe = CropRecipe(
                            family="mixed",
                            source_indices=(target_idx, *aux_indices),
                            alpha=alpha,
                        )
                    else:
                        raise ValueError(f"Unknown negative family: {family}")
                slot_recipes.append(recipe)

            # Only keep complete slots (all num_crops_per_image recipes filled)
            if len(slot_recipes) < self.num_crops_per_image:
                break
            plan.append(slot_recipes)

        return plan

    def set_epoch(self, epoch: int) -> None:
        """Rebuild the sampling plan for a new epoch."""
        self._epoch = epoch
        self._plan = self._build_epoch_plan()

    def __len__(self) -> int:
        return len(self._plan)
