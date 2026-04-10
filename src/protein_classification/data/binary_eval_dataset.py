from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import tifffile as tiff
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data.dataset import Dataset

from protein_classification.data.utils import normalize_img, resize_img

PathLike = Union[Path, str]


class BinaryEvalDataset(Dataset):
    """Evaluation dataset for one-vs-rest binary classification.

    Extracts all crop positions for each image deterministically — no sampling,
    no augmentation, no epoch plan.

    Two extraction modes:
    - **Foreground mode** (default when ``valid_crop_positions`` is provided):
      uses cached foreground ``(y, x)`` positions from a ``BackgroundAnalyzer``.
    - **All-crops mode** (when ``valid_crop_positions=None``): generates a dense
      non-overlapping grid covering the full image, including background patches.
    """

    def __init__(
        self,
        inputs: Sequence[tuple[PathLike, int]],
        target_label: int,
        crop_size: int,
        img_size: int,
        valid_crop_positions: Optional[dict[int, list[tuple[int, int]]]] = None,
        bit_depth: Optional[int] = None,
        normalize: Optional[Literal["minmax", "std"]] = None,
        normalization_scope: Literal["dataset", "image"] = "dataset",
        dataset_stats: Optional[tuple[float, float]] = None,
        imreader: Callable[[PathLike], Union[NDArray, Tensor]] = tiff.imread,
    ) -> None:
        super().__init__()
        self.inputs = list(inputs)
        self.target_label = int(target_label)
        self.crop_size = crop_size
        self.img_size = img_size
        self.valid_crop_positions = valid_crop_positions
        self.bit_depth = bit_depth
        self.normalize = normalize
        self.normalization_scope = normalization_scope
        self.dataset_stats = dataset_stats
        self.imreader = imreader

    def _load_image(self, idx: int) -> Tensor:
        """Load one TIFF image and return it as a ``(C, H, W)`` float32 tensor."""
        fpath, _ = self.inputs[idx]
        image = self.imreader(fpath)
        if isinstance(image, torch.Tensor):
            image_t = image.to(torch.float32)
        else:
            if self.img_size is not None and image.shape[-2:] != (self.img_size, self.img_size):
                image = resize_img(image, self.img_size)
            image_t = torch.tensor(image, dtype=torch.float32)

        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        elif image_t.ndim != 3:
            raise ValueError(f"Expected 2D or 3D image, got shape {tuple(image_t.shape)}")

        if self.img_size is not None and image_t.shape[-2:] != (self.img_size, self.img_size):
            resized = resize_img(image_t.squeeze(0).cpu().numpy(), self.img_size)
            image_t = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)

        return image_t

    def _get_positions(self, idx: int, h: int, w: int) -> list[tuple[int, int]]:
        """Return crop top-left positions for a given image.

        Uses cached foreground positions if available, otherwise falls back to
        a dense non-overlapping grid covering the full image.
        """
        if self.valid_crop_positions is not None:
            positions = self.valid_crop_positions.get(idx)
            if positions:
                return positions

        # Dense non-overlapping grid (all-crops mode or missing cache entry)
        crop_size = self.crop_size
        ys = list(range(0, h - crop_size + 1, crop_size))
        if not ys or ys[-1] != h - crop_size:
            ys.append(h - crop_size)
        xs = list(range(0, w - crop_size + 1, crop_size))
        if not xs or xs[-1] != w - crop_size:
            xs.append(w - crop_size)
        return [(y, x) for y in ys for x in xs]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image = self._load_image(idx)
        _, h, w = image.shape
        binary_label = 1 if self.inputs[idx][1] == self.target_label else 0

        positions = self._get_positions(idx, h, w)
        crops = [
            image[:, y : y + self.crop_size, x : x + self.crop_size]
            for y, x in positions
        ]
        crops = torch.stack(crops)  # (N, C, crop_size, crop_size)

        if self.normalize is not None:
            crops = torch.stack([
                normalize_img(crop, self.normalize, self.normalization_scope, self.dataset_stats)
                for crop in crops
            ])

        labels = torch.full((len(positions),), binary_label, dtype=torch.long)
        return crops.to(torch.float32), labels
