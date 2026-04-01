# %% [markdown]
# # Interactive inspector for BinaryDataset samples
#
# Shows a paginated grid of crops sampled from a BinaryDataset, with each crop
# bordered and titled by its binary sample family (positive / easy / mixed /
# inverted).

# %% Imports
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from protein_classification.config.data import DataAugmentationConfig
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data import BinaryDataset

# %% Parameters
DATA_DIR = "/group/jug/federico/data/CellAtlas"
TARGET = "Mitochondria"
IMG_SIZE = 2048
CROP_SIZE = 256
N_PER_PAGE = 25
N_COLS = 5
POS_PROB = 0.5
AUG = "geometric"  # None, "geometric", "intensity", "noise", or "all"
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)

# %% Data setup
print(f"\nLoading CellAtlas data from: {DATA_DIR}")
inputs, labels_dict = get_cellatlas_filepaths_and_labels(
    data_dir=DATA_DIR,
    extra_labels=["Mitochondria"],
)

if TARGET not in labels_dict:
    raise ValueError(f"Target '{TARGET}' not found. Available: {list(labels_dict)}")

target_label = labels_dict[TARGET]
print(f"Target: '{TARGET}'  →  label id {target_label}")
print(f"Total inputs: {len(inputs)}")

aug_config = DataAugmentationConfig(
    transform=AUG,
    crop_size=CROP_SIZE,
    random_crop=True,
)

dataset = BinaryDataset(
    inputs=inputs,
    split="train",
    img_size=IMG_SIZE,
    augmentation_config=aug_config,
    num_crops_per_image=1,
    target_label=target_label,
    positive_probability=POS_PROB,
    negative_family_weights={"trivial": 1.0, "mixed": 0.0, "inverted": 0.0},
    background_threshold_quantiles_by_label={
        labels_dict["Mitochondria"]: 0.1,
        labels_dict["Nucleus"]: 0.25,
        labels_dict["Microtubules"]: 0.1,
        labels_dict["Endoplasmic reticulum"]: 0.1,
    },
    return_label=True,
)

n_pos = len(dataset.target_indices)
n_neg = len(dataset.non_target_indices)
print(f"Target-class files : {n_pos}")
print(f"Non-target files   : {n_neg}")

# %% Sampling helpers

def _sample_with_family(ds: BinaryDataset) -> tuple[torch.Tensor, int, str]:
    """Sample one crop from a random plan slot, also returning the family name."""
    slot_idx = random.randrange(len(ds))
    recipe = ds._plan[slot_idx][0]
    crop, label = ds._execute_recipe(recipe)
    return crop, label, recipe.family


def _to_display(crop: torch.Tensor) -> np.ndarray:
    """Convert a (1, H, W) or (H, W) tensor to a display-ready (H, W) float32 in [0, 1]."""
    arr = crop.squeeze(0).float().numpy()
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.clip(0.0, 1.0)


FAMILY_COLOR = {
    "positive": "#2ca02c",   # green
    "trivial":     "#d62728",   # red
    "mixed":    "#ff7f0e",   # orange
    "inverted": "#9467bd",   # purple
}

# %% Visualize a page of samples
batch = [_sample_with_family(dataset) for _ in range(N_PER_PAGE)]

family_counts = {}
for _, _, fam in batch:
    family_counts[fam] = family_counts.get(fam, 0) + 1

n_rows = (N_PER_PAGE + N_COLS - 1) // N_COLS

fig, axes = plt.subplots(
    n_rows, N_COLS,
    figsize=(2.6 * N_COLS, 2.8 * n_rows),
    squeeze=False,
)
fig.patch.set_facecolor("black")
count_str = "  |  ".join(f"{fam}: {cnt}" for fam, cnt in sorted(family_counts.items()))
fig.suptitle(f"Target: {TARGET} - " + count_str, fontsize=18, y=1.01, color="white")

for ax in axes.flat:
    ax.axis("off")

for i, (crop, label, family) in enumerate(batch):
    row, col = divmod(i, N_COLS)
    ax = axes[row][col]
    ax.imshow(_to_display(crop), cmap="gray", vmin=0, vmax=1)
    ax.set_title(family, fontsize=12, color=FAMILY_COLOR[family])

plt.tight_layout()
plt.show()

# %%
