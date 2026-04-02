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
from protein_classification.data.utils import (
    crop_img,
    normalize_img,
    resize_img,
)

PathLike = Union[Path, str]


@dataclass(frozen=True, slots=True)
class ImageRecipe:
    """Pre-computed recipe for one image in the epoch plan."""

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
        valid_crop_positions: Optional[dict[int, list[tuple[int, int]]]] = None,
        crop_position_jitter: int = 8,
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
        self.valid_crop_positions = valid_crop_positions
        self.crop_position_jitter = crop_position_jitter
        self.imreader = imreader
        self.return_label = return_label
        self.augmentation_config = augmentation_config
        self.num_crops_per_image = num_crops_per_image

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
        """Extract a single random crop, preferring precomputed foreground positions.

        If *source_idx* is provided and valid crop positions are available,
        a position is sampled from them (with jitter).  Otherwise falls back
        to a uniformly random crop.
        """
        crop_size = self.augmentation_config.crop_size
        if crop_size is None:
            return image, label

        _, h, w = image.shape

        # Try to use precomputed valid positions
        if source_idx is not None and self.valid_crop_positions is not None:
            positions = self.valid_crop_positions.get(source_idx, [])
            if positions:
                y, x = random.choice(positions)
                jitter = self.crop_position_jitter
                if jitter > 0:
                    y += random.randint(-jitter, jitter)
                    x += random.randint(-jitter, jitter)
                y = max(0, min(y, h - crop_size))
                x = max(0, min(x, w - crop_size))
                crop = image[:, y : y + crop_size, x : x + crop_size]
                return crop, label

        # Fallback: uniformly random crop
        crop = crop_img(image, crop_size, random_crop=True)
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

    def _make_positive_crops(self, idx: int) -> tuple[list[Tensor], list[int]]:
        """Load one target image and extract ``num_crops_per_image`` positive crops."""
        image = self._load_image(idx)
        crops = []
        for _ in range(self.num_crops_per_image):
            crop, _ = self._sample_single_crop(image, label=1, source_idx=idx)
            crops.append(crop)
        return crops, [1] * self.num_crops_per_image

    def _make_trivial_negative_crops(self, idx: int) -> tuple[list[Tensor], list[int]]:
        """Load one non-target image and extract ``num_crops_per_image`` trivial negative crops."""
        image = self._load_image(idx)
        crops = []
        for _ in range(self.num_crops_per_image):
            crop, _ = self._sample_single_crop(image, label=0, source_idx=idx)
            crops.append(crop)
        return crops, [0] * self.num_crops_per_image

    def _make_inverted_negative_crops(self, idx: int) -> tuple[list[Tensor], list[int]]:
        """Load one target image and extract ``num_crops_per_image`` inverted negative crops."""
        image = self._load_image(idx)
        crops = []
        for _ in range(self.num_crops_per_image):
            crop, _ = self._sample_single_crop(image, label=0, source_idx=idx)
            crop = normalize_img(crop, "minmax", "image")
            crop = 1.0 - crop
            crops.append(crop)
        return crops, [0] * self.num_crops_per_image

    def _make_mixed_negative_crops(
        self, source_indices: tuple[int, ...], alpha: float,
    ) -> tuple[list[Tensor], list[int]]:
        """Load target + auxiliary images once, extract and blend ``num_crops_per_image`` crops."""
        target_idx = source_indices[0]
        target_image = self._load_image(target_idx)

        aux_images: list[tuple[Tensor, int]] = []
        for aux_idx in source_indices[1:]:
            aux_images.append((self._load_image(aux_idx), aux_idx))

        crops = []
        for _ in range(self.num_crops_per_image):
            target_crop, _ = self._sample_single_crop(target_image, label=0, source_idx=target_idx)
            target_crop = normalize_img(target_crop, "minmax", "image")

            aux_crops = []
            for aux_image, aux_idx in aux_images:
                aux_crop, _ = self._sample_single_crop(aux_image, label=0, source_idx=aux_idx)
                aux_crops.append(normalize_img(aux_crop, "minmax", "image"))

            aux_mean = torch.stack(aux_crops).mean(dim=0)
            mixed_crop = alpha * target_crop + (1.0 - alpha) * aux_mean
            crops.append(mixed_crop)
        return crops, [0] * self.num_crops_per_image

    def _execute_recipe(self, recipe: ImageRecipe) -> tuple[list[Tensor], list[int]]:
        """Execute a single :class:`ImageRecipe` and return multiple crops + labels."""
        if recipe.family == "positive":
            return self._make_positive_crops(recipe.source_indices[0])
        if recipe.family == "trivial":
            return self._make_trivial_negative_crops(recipe.source_indices[0])
        if recipe.family == "inverted":
            return self._make_inverted_negative_crops(recipe.source_indices[0])
        if recipe.family == "mixed":
            assert recipe.alpha is not None
            return self._make_mixed_negative_crops(recipe.source_indices, recipe.alpha)
        raise ValueError(f"Unknown family: {recipe.family}")

    def _build_epoch_plan(self) -> list[ImageRecipe]:
        """Build a deterministic sampling plan for the current epoch.

        Each entry is a single :class:`ImageRecipe` specifying which image(s)
        to load and the family.  ``__getitem__`` extracts
        ``num_crops_per_image`` crops from the loaded image(s).

        The epoch is sized by the minority (target) pool: entries are generated
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

        plan: list[ImageRecipe] = []
        while True:
            if rng.random() < self.positive_probability:
                if target_remaining() < 1:
                    break
                recipe = ImageRecipe(
                    family="positive",
                    source_indices=(draw_target(),),
                )
            else:
                family = rng.choices(families, weights=weights, k=1)[0]
                if family == "trivial":
                    if non_target_remaining() < 1:
                        break
                    recipe = ImageRecipe(
                        family="trivial",
                        source_indices=(draw_non_target(),),
                    )
                elif family == "inverted":
                    if target_remaining() < 1:
                        break
                    recipe = ImageRecipe(
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
                    recipe = ImageRecipe(
                        family="mixed",
                        source_indices=(target_idx, *aux_indices),
                        alpha=alpha,
                    )
                else:
                    raise ValueError(f"Unknown negative family: {family}")
            plan.append(recipe)

        return plan

    def set_epoch(self, epoch: int) -> None:
        """Rebuild the sampling plan for a new epoch."""
        self._epoch = epoch
        self._plan = self._build_epoch_plan()

    def __getitem__(self, idx: int) -> Union[Tensor, tuple[Tensor, Tensor]]:
        recipe = self._plan[idx]
        crops, labels = self._execute_recipe(recipe)

        processed_crops = self._apply_per_crop_processing(torch.stack(crops))
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if self.return_label:
            return processed_crops, labels_tensor
        return processed_crops
    
    def __len__(self) -> int:
        return len(self._plan)
