"""Quick interactive inspection of background-score thresholds on CellAtlas crops.

Use with VS Code / Jupyter-style cells (`# %%`).

The script:
- loads a subset of CellAtlas labels
- samples random crops with the same logic used by `compute_background_thresholds`
- plots score distributions per label
- visualizes crops around selected quantiles to help pick a threshold
- proposes an automatic threshold per label via Otsu on the sampled scores
"""

# %%
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
from torch import Tensor

from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.utils import (
    compute_background_score,
    crop_img,
    resize_img,
)


# %%
# Edit these values as needed.
DATA_DIR = "/group/jug/federico/data/CellAtlas"
LABELS_TO_INSPECT = ["Microtubules"]
IMG_SIZE = 2048
CROP_SIZE = 256
RANDOM_CROP = True
METRICS = ["entropy"]
SAMPLES_PER_IMAGE = 20
MAX_IMAGES_PER_LABEL = 50
QUANTILES_TO_SHOW = [0.05, 0.07, 0.1]
N_EDGE_CROPS = 8
SEED = 0

rng = np.random.default_rng(SEED)


# %%
def _load_image(path: str | Path, img_size: Optional[int]) -> Tensor:
    image = tiff.imread(path)
    if isinstance(image, torch.Tensor):
        image_tensor = image.to(torch.float32)
    else:
        if img_size is not None and image.shape != (img_size, img_size):
            image = resize_img(image, img_size)
        image_tensor = torch.tensor(image, dtype=torch.float32)

    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image tensor, got shape {tuple(image_tensor.shape)}")

    if img_size is not None and image_tensor.shape[-2:] != (img_size, img_size):
        resized = resize_img(image_tensor.squeeze(0).cpu().numpy(), img_size)
        image_tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0)

    return image_tensor


def _to_display(crop: Tensor) -> np.ndarray:
    arr = crop.squeeze(0).detach().cpu().float().numpy()
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.clip(0.0, 1.0)


def _collect_scored_crops(
    inputs: list[tuple[str, int]],
    img_size: int,
    crop_size: int,
    random_crop: bool,
    metrics: list[str],
    samples_per_image: int,
    max_images_per_label: Optional[int],
) -> dict[int, list[tuple[float, Tensor, str]]]:
    inputs_by_label: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for path, label in inputs:
        inputs_by_label[int(label)].append((path, label))

    scored_by_label: dict[int, list[tuple[float, Tensor, str]]] = {}
    for label, label_inputs in inputs_by_label.items():
        chosen_inputs = list(label_inputs)
        rng.shuffle(chosen_inputs)
        if max_images_per_label is not None:
            chosen_inputs = chosen_inputs[:max_images_per_label]

        scored_crops: list[tuple[float, Tensor, str]] = []
        for path, _ in chosen_inputs:
            image = _load_image(path, img_size)
            for _ in range(samples_per_image):
                crop = crop_img(image, crop_size, random_crop)
                score = compute_background_score(crop, metrics)
                scored_crops.append((score, crop, str(path)))
        scored_by_label[label] = scored_crops

    return scored_by_label


def _select_edge_crops(
    scored_crops: list[tuple[float, Tensor, str]],
    threshold: float,
    n_examples: int,
) -> tuple[float, list[tuple[float, Tensor, str]]]:
    ranked = sorted(scored_crops, key=lambda item: abs(item[0] - threshold))
    return threshold, ranked[:n_examples]


def _compute_otsu_threshold(scores: np.ndarray, n_bins: int = 256) -> float:
    if scores.size == 0:
        raise ValueError("Cannot compute an Otsu threshold from an empty score array.")
    if np.allclose(scores.min(), scores.max()):
        return float(scores.min())

    hist, bin_edges = np.histogram(scores, bins=n_bins)
    hist = hist.astype(np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    weight_bg = np.cumsum(hist)
    weight_fg = hist.sum() - weight_bg

    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.cumsum(hist * bin_centers) / np.maximum(weight_bg, 1e-12)
    mean_fg = (
        (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight_fg[::-1], 1e-12))[::-1]
    )

    between_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_class_var[~valid] = -1.0
    best_idx = int(np.argmax(between_class_var))
    return float(bin_centers[best_idx])


# %%
inputs, labels_dict = get_cellatlas_filepaths_and_labels(
    data_dir=DATA_DIR,
    extra_labels=["Mitochondria"],
)
label_name_by_id = {idx: name for name, idx in labels_dict.items()}
selected_label_ids = {labels_dict[name] for name in LABELS_TO_INSPECT}

scored_by_label = _collect_scored_crops(
    inputs=[item for item in inputs if item[1] in selected_label_ids],
    img_size=IMG_SIZE,
    crop_size=CROP_SIZE,
    random_crop=RANDOM_CROP,
    metrics=METRICS,
    samples_per_image=SAMPLES_PER_IMAGE,
    max_images_per_label=MAX_IMAGES_PER_LABEL,
)

print(f"Loaded {len(scored_by_label)} labels with metric(s)={METRICS}.")
for label, scored in scored_by_label.items():
    print(f"{label_name_by_id[label]}: {len(scored)} sampled crops")


# %%
for label, scored_crops in scored_by_label.items():
    scores = np.array([score for score, _, _ in scored_crops], dtype=np.float32)
    otsu_threshold = _compute_otsu_threshold(scores)
    plt.figure(figsize=(7, 4))
    plt.hist(scores, bins=40, color="#4c78a8", alpha=0.85)
    for quantile in QUANTILES_TO_SHOW:
        threshold = float(np.quantile(scores, quantile))
        plt.axvline(
            threshold,
            linestyle="--",
            linewidth=2,
            label=f"q={quantile:.2f} ({threshold:.4f})",
        )
    plt.axvline(
        otsu_threshold,
        color="#e45756",
        linestyle="-",
        linewidth=2,
        label=f"Otsu ({otsu_threshold:.4f})",
    )
    plt.title(f"{label_name_by_id[label]} | score distribution")
    plt.xlabel("Background score")
    plt.ylabel("Crop count")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
for label, scored_crops in scored_by_label.items():
    scores = np.array([score for score, _, _ in scored_crops], dtype=np.float32)
    otsu_threshold = _compute_otsu_threshold(scores)
    otsu_quantile = float(np.mean(scores <= otsu_threshold))
    print(f"{label_name_by_id[label]}: Otsu threshold={otsu_threshold:.4f}, quantile={otsu_quantile:.3f}")


    for quantile in QUANTILES_TO_SHOW:
        threshold = float(np.quantile(scores, quantile))
        _, examples = _select_edge_crops(scored_crops, threshold, N_EDGE_CROPS)
        n_cols = 4
        n_rows = int(np.ceil(len(examples) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3 * n_cols, 3 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"{label_name_by_id[label]} | q={quantile:.2f} | threshold={threshold:.4f}",
            fontsize=12,
        )

        for ax in axes.flat:
            ax.axis("off")

        for idx, (score, crop, path) in enumerate(examples):
            ax = axes[idx // n_cols][idx % n_cols]
            ax.imshow(_to_display(crop), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"score={score:.4f}", fontsize=9)
            ax.set_xlabel(Path(path).name, fontsize=7)
            ax.xaxis.set_label_position("top")

        plt.tight_layout()
        plt.show()

    _, otsu_examples = _select_edge_crops(scored_crops, otsu_threshold, N_EDGE_CROPS)
    n_cols = 4
    n_rows = int(np.ceil(len(otsu_examples) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("black")
    fig.suptitle(
        f"{label_name_by_id[label]} | Otsu threshold={otsu_threshold:.4f}",
        fontsize=12,
    )

    for ax in axes.flat:
        ax.axis("off")

    for idx, (score, crop, path) in enumerate(otsu_examples):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.imshow(_to_display(crop), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"score={score:.4f}", fontsize=9)
        ax.set_xlabel(Path(path).name, fontsize=7)
        ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.show()
# %%
