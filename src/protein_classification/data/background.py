"""Background analysis for crop-based datasets.

Provides :class:`BackgroundAnalyzer` which precomputes per-label foreground
thresholds and per-image valid crop positions in a single systematic pass.
"""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

from protein_classification.data.utils import (
    compute_background_score, resize_img,
)

PathLike = Union[Path, str]


class BackgroundAnalyzer:
    """Precompute per-label background thresholds and per-image valid crop positions.

    The analysis proceeds in two phases:

    1. **Threshold estimation** -- a subset of images (up to *max_images_for_thresholds*)
       is scanned on a regular grid of overlapping crops.  The resulting scores are
       pooled by label and a per-label quantile threshold is derived.
    2. **Valid-position mapping** -- *every* image is scanned on the same grid.
       Grid positions whose score meets the label's threshold are recorded.
       Score maps computed during phase 1 are reused to avoid redundant work.

    Parameters
    ----------
    inputs : Sequence[tuple[PathLike, int]]
        ``(filepath, label)`` pairs for every image in the dataset.
    crop_size : int
        Side length of the square crop.
    stride : int
        Step between adjacent crop positions on the grid.
    img_size : int
        Images are resized to ``(img_size, img_size)`` before analysis.
    metrics : list of {"std", "entropy"}
        Metrics forwarded to :func:`compute_background_score`.
    quantile : float
        Default quantile for threshold estimation (e.g. 0.05 keeps the top
        95 % of crops).
    quantiles_by_label : dict[int, float] or None
        Per-label overrides for *quantile*.
    max_images_for_thresholds : int or None
        Cap on how many images to use for threshold estimation.  ``None``
        means use all images.
    imreader : callable
        Function that reads an image file and returns an ndarray or Tensor.
    """

    def __init__(
        self,
        inputs: Sequence[tuple[PathLike, int]],
        crop_size: int,
        stride: int,
        img_size: int,
        metrics: list[Literal["std", "entropy"]] = "entropy",
        quantile: float = 0.1,
        quantiles_by_label: Optional[dict[int, float]] = None,
        max_images_for_thresholds: Optional[int] = 50,
        imreader: Callable[[PathLike], Union[NDArray, Tensor]] = None,
        thresholds_by_label: Optional[dict[int, float]] = None,
    ) -> None:
        if imreader is None:
            import tifffile as tiff
            imreader = tiff.imread

        self._inputs = list(inputs)
        self._crop_size = crop_size
        self._stride = stride
        self._img_size = img_size
        self._metrics = metrics
        self._quantile = quantile
        self._quantiles_by_label = quantiles_by_label
        self._max_images_for_thresholds = max_images_for_thresholds
        self._imreader = imreader

        self._thresholds_by_label: dict[int, float] = {}
        self._valid_positions_by_index: dict[int, list[tuple[int, int]]] = {}

        self._analyze(precomputed_thresholds=thresholds_by_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def thresholds_by_label(self) -> dict[int, float]:
        return self._thresholds_by_label

    @property
    def valid_positions_by_index(self) -> dict[int, list[tuple[int, int]]]:
        return self._valid_positions_by_index

    def save(self, path: PathLike) -> None:
        """Save thresholds and valid positions to a JSON file.

        Positions are keyed by file path (not index) so the cache is
        independent of any particular train/val split.
        """
        data = {
            "thresholds_by_label": {
                str(k): v for k, v in self._thresholds_by_label.items()
            },
            "valid_positions_by_filepath": {
                str(self._inputs[idx][0]): positions
                for idx, positions in self._valid_positions_by_index.items()
            },
            "params": {
                "crop_size": self._crop_size,
                "stride": self._stride,
                "img_size": self._img_size,
                "metrics": self._metrics,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(path: PathLike) -> dict:
        """Load precomputed background data from a JSON file."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def to_index_keyed(
        valid_positions_by_filepath: dict[str, list[list[int]]],
        inputs: Sequence[tuple[PathLike, int]],
    ) -> dict[int, list[tuple[int, int]]]:
        """Map filepath-keyed positions to index-keyed for a given inputs list."""
        result: dict[int, list[tuple[int, int]]] = {}
        missing: list[str] = []
        for idx, (fpath, _) in enumerate(inputs):
            key = str(fpath)
            if key in valid_positions_by_filepath:
                result[idx] = [tuple(pos) for pos in valid_positions_by_filepath[key]]
            else:
                missing.append(key)
        if missing:
            warnings.warn(
                f"{len(missing)} / {len(inputs)} images not found in background "
                f"cache — they will fall back to random crops. "
                f"First missing: {missing[0]}",
                stacklevel=2,
            )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_image(self, idx: int) -> Tensor:
        """Load and normalise an image to ``(1, H, W)`` float tensor."""
        fpath, _ = self._inputs[idx]
        image = self._imreader(fpath)
        if isinstance(image, torch.Tensor):
            image_tensor = image.to(torch.float32)
        else:
            if self._img_size is not None and image.shape != (self._img_size, self._img_size):
                image = resize_img(image, self._img_size)
            image_tensor = torch.tensor(image, dtype=torch.float32)

        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D image tensor, got shape {tuple(image_tensor.shape)}"
            )

        if self._img_size is not None and image_tensor.shape[-2:] != (self._img_size, self._img_size):
            resized = resize_img(image_tensor.squeeze(0).cpu().numpy(), self._img_size)
            image_tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)

        return image_tensor

    def _compute_score_map(self, image: Tensor) -> tuple[list[tuple[int, int]], list[float]]:
        """Compute background scores on a regular grid of overlapping crops.

        Returns the grid positions and corresponding scores.  For each axis,
        if the last grid position doesn't reach the image edge, an extra
        edge-aligned position (``dim_size - crop_size``) is appended so that
        no region of the image is systematically excluded.
        """
        # TODO: make compatible with 3D
        _, h, w = image.shape

        # Build axis coordinates, adding edge positions if needed
        ys = list(range(0, h - self._crop_size + 1, self._stride))
        last_y = h - self._crop_size
        if ys[-1] != last_y:
            ys.append(last_y)

        xs = list(range(0, w - self._crop_size + 1, self._stride))
        last_x = w - self._crop_size
        if xs[-1] != last_x:
            xs.append(last_x)

        positions: list[tuple[int, int]] = []
        scores: list[float] = []
        for y in ys:
            for x in xs:
                crop = image[:, y : y + self._crop_size, x : x + self._crop_size]
                score = compute_background_score(crop, self._metrics)
                positions.append((y, x))
                scores.append(score)
        return positions, scores

    def _analyze(
        self,
        precomputed_thresholds: Optional[dict[int, float]] = None,
    ) -> None:
        """Run both phases: threshold estimation then valid-position mapping.

        If *precomputed_thresholds* is provided, phase 1 is skipped and the
        given thresholds are used directly (useful for validation sets that
        should reuse training thresholds).
        """
        n = len(self._inputs)
        cached_score_maps: dict[int, tuple[list[tuple[int, int]], list[float]]] = {}

        # -- Phase 1: estimate thresholds from a subset -----------------------
        if precomputed_thresholds is not None:
            self._thresholds_by_label = dict(precomputed_thresholds)
        else:
            if self._max_images_for_thresholds is not None and self._max_images_for_thresholds < n:
                subset_indices = set(
                    random.sample(range(n), self._max_images_for_thresholds)
                )
            else:
                subset_indices = set(range(n))

            scores_by_label: dict[int, list[float]] = {}

            for idx in tqdm(
                sorted(subset_indices),
                desc="Background analysis (thresholds)",
            ):
                image = self._load_image(idx)
                positions, scores = self._compute_score_map(image)
                cached_score_maps[idx] = (positions, scores)

                label = int(self._inputs[idx][1])
                scores_by_label.setdefault(label, []).extend(scores)

            for label, scores in scores_by_label.items():
                q = self._quantile
                if self._quantiles_by_label is not None:
                    q = self._quantiles_by_label.get(label, q)
                self._thresholds_by_label[label] = float(
                    np.quantile(np.asarray(scores, dtype=np.float32), q)
                )

        # -- Phase 2: compute valid positions for all images ------------------
        remaining_indices = [i for i in range(n) if i not in cached_score_maps]

        for idx in tqdm(
            remaining_indices,
            desc="Background analysis (valid positions)",
        ):
            image = self._load_image(idx)
            positions, scores = self._compute_score_map(image)
            cached_score_maps[idx] = (positions, scores)

        for idx in range(n):
            label = int(self._inputs[idx][1])
            threshold = self._thresholds_by_label.get(label)
            positions, scores = cached_score_maps[idx]
            if threshold is None:
                self._valid_positions_by_index[idx] = list(positions)
            else:
                self._valid_positions_by_index[idx] = [
                    pos for pos, score in zip(positions, scores)
                    if score >= threshold
                ]
