# %% [markdown]
# # Binary Critic Evaluation on LambdaSplit Predictions
#
# Evaluate a pretrained binary classification model on multichannel images
# (λSplit spectral unmixing predictions stored as `.npz`).
# Each channel is treated as a separate structure; one channel is the target
# (positive) and all others are trivial negatives.

# %% Parameters
from pathlib import Path

CKPT_DIR = Path("/group/jug/federico/critic_net_training/2604/CellAtlas_best/Nucleus")
# DATA_PATH = Path("/group/jug/federico/data/simulated_spectral/CellAtlas/2602/sim_data_CellAtlas_2048px_n300_bands3_pwr32_Mito/GT/test")
DATA_PATH = Path("/group/jug/federico/lambdasplit_training/2604/lambdasplit_CellAtlas_LVAE_4FP_2D/11/predictions_MMSE_5/pred_imgs.npz")
TARGET_IDX = 0
CROP_SIZE = 256         # patch size
OVERLAP = CROP_SIZE // 4
USE_CALIBRATION = True  # apply temperature scaling if calibration.json exists
N_BINS = 15             # bins for calibration plots
BG_STD_QUANTILE = 0.0   # discard patches with std below this quantile (per channel); set to 0.0 to disable
N_IMGS = 10


# %% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile as tiff
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader

from protein_classification.config import AlgorithmConfig, DataConfig
from protein_classification.data.utils import collate_multi_crop_batches, normalize_img
from protein_classification.model import BioStructClassifier
from protein_classification.utils.calibration import (
    apply_temperature,
    binary_brier,
    binary_ece,
    binary_nll,
)
from protein_classification.utils.evaluation import compute_classification_metrics
from protein_classification.utils.io import load_calibration, load_checkpoint, load_config


# %% Dataset
class LambdaSplitPatchDataset(Dataset):
    """Extract all patches from every channel of multichannel .npz images.

    For each multichannel image ``(C, H, W)`` in the archive, each channel is
    independently tiled into patches on a regular grid with the given overlap.
    Patches from the target channel are labelled 1; all others 0.
    """

    def __init__(
        self,
        data_path,
        target_idx: int,
        crop_size: int,
        overlap: int,
        num_imgs: int = None,
        bg_std_quantile: float = 0.0,
        normalize=None,
        normalization_scope="image",
        dataset_stats=None,
    ):
        super().__init__()
        self.target_idx = target_idx
        self.crop_size = crop_size
        self.overlap = overlap
        self.normalize = normalize
        self.normalization_scope = normalization_scope
        self.dataset_stats = dataset_stats
        self.num_imgs = num_imgs

        # Load all multichannel images from the .npz archive
        if data_path.suffix == ".npz":
            npz_data = np.load(data_path)
            multi_images = [img for img in npz_data.values()]  # list of (C, H, W) arrays
        else:
            multi_images = []
            for fname in os.listdir(data_path):
                if fname.endswith(".tiff") or fname.endswith(".tif"):
                    multi_images.append(tiff.imread(os.path.join(data_path, fname)))

        if self.num_imgs is not None:
            multi_images = multi_images[: self.num_imgs]

        # Extract all patches per channel, tracking std for background filtering
        per_channel_patches: dict[int, list[tuple[torch.Tensor, float]]] = {}
        per_channel_labels: dict[int, int] = {}

        for img in multi_images:
            n_channels = img.shape[0]
            for ch in range(n_channels):
                per_channel_patches.setdefault(ch, [])
                per_channel_labels[ch] = 1 if ch == self.target_idx else 0
                channel_img = torch.tensor(img[ch], dtype=torch.float32).unsqueeze(0)
                positions = self._grid_positions(channel_img.shape[-2], channel_img.shape[-1])
                for y, x in positions:
                    patch = channel_img[:, y : y + crop_size, x : x + crop_size]
                    per_channel_patches[ch].append((patch, patch.std().item()))

        # Filter out background patches per channel using std quantile threshold
        self.patches: list[torch.Tensor] = []
        self.labels: list[int] = []
        n_before = sum(len(v) for v in per_channel_patches.values())
        for ch, patch_list in sorted(per_channel_patches.items()):
            if bg_std_quantile > 0 and patch_list:
                stds = torch.tensor([s for _, s in patch_list])
                threshold = torch.quantile(stds, bg_std_quantile).item()
                kept = [(p, s) for p, s in patch_list if s >= threshold]
            else:
                kept = patch_list
            label = per_channel_labels[ch]
            for p, _ in kept:
                self.patches.append(p)
                self.labels.append(label)

        n_after = len(self.patches)
        if bg_std_quantile > 0:
            print(
                f"Background filtering (std quantile={bg_std_quantile}): "
                f"{n_before} -> {n_after} patches ({n_before - n_after} removed)"
            )

    def _grid_positions(self, h: int, w: int) -> list[tuple[int, int]]:
        stride = self.crop_size - self.overlap
        ys = list(range(0, h - self.crop_size + 1, stride))
        if not ys or ys[-1] != h - self.crop_size:
            ys.append(h - self.crop_size)
        xs = list(range(0, w - self.crop_size + 1, stride))
        if not xs or xs[-1] != w - self.crop_size:
            xs.append(w - self.crop_size)
        return [(y, x) for y in ys for x in xs]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.normalize is not None:
            patch = normalize_img(
                patch, self.normalize, self.normalization_scope, self.dataset_stats,
            )
        return patch.to(torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# %% Load configs + model
algo_config = AlgorithmConfig(**load_config(str(CKPT_DIR), "algorithm"))
data_config = load_config(str(CKPT_DIR), "data")
data_config["target_label"] = f"Nucleus"
data_config = DataConfig(**data_config)

if algo_config.architecture_config.num_classes != 1:
    raise RuntimeError("This script only supports binary checkpoints.")

temperature = 1.0
if USE_CALIBRATION:
    try:
        cal_meta = load_calibration(str(CKPT_DIR))
        temperature = float(cal_meta["temperature"])
        print(f"Loaded calibration temperature: T={temperature:.4f}")
    except FileNotFoundError:
        print("No calibration.json found — using T=1.0")

model = BioStructClassifier(config=algo_config)
ckpt = load_checkpoint(str(CKPT_DIR), best=True)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()


# %% Build dataset + dataloader
dataset = LambdaSplitPatchDataset(
    data_path=DATA_PATH,
    target_idx=TARGET_IDX,
    crop_size=CROP_SIZE,
    overlap=OVERLAP,
    num_imgs=N_IMGS,
    bg_std_quantile=BG_STD_QUANTILE,
    normalize=data_config.normalize,
    normalization_scope=data_config.normalization_scope,
    dataset_stats=data_config.dataset_stats,
)

print(f"Total patches: {len(dataset)}")
print(f"  Positive: {sum(dataset.labels)}")
print(f"  Negative: {len(dataset) - sum(dataset.labels)}")

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)


# %% Inference
trainer = Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    enable_progress_bar=True,
    logger=False,
    precision=32,
)
outputs = trainer.predict(model=model, dataloaders=dataloader)

all_preds = torch.cat([b[0] for b in outputs]).cpu()
all_logits = torch.cat([b[1] for b in outputs]).cpu()
all_labels = torch.cat([b[2] for b in outputs]).cpu()

raw_probs = torch.sigmoid(all_logits)
cal_probs = apply_temperature(all_logits, temperature)


# %% Metrics
metrics = compute_classification_metrics(
    preds=(cal_probs > 0.5).long(),
    gts=all_labels,
    probs=cal_probs,
    logits=all_logits,
    num_classes=1,
    average="macro",
    calibration_metrics=True,
)

print("\n------------- Patch-level Metrics -------------")
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"F1 (macro): {metrics['f1']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
print(f"NLL:       {metrics['nll']:.4f}")
print(f"Brier:     {metrics['brier']:.4f}")
print(f"ECE:       {metrics['ece']:.4f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
if temperature != 1.0:
    print(f"Temperature: {temperature:.4f}")


# %% Helpers for calibration plots
def _compute_bin_stats(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int,
) -> dict[str, np.ndarray]:
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().float()
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)

    centers, counts, avg_conf, avg_acc = [], [], [], []
    for idx in range(n_bins):
        lo, hi = boundaries[idx], boundaries[idx + 1]
        mask = (probs >= lo) & (probs <= hi) if idx == n_bins - 1 else (probs >= lo) & (probs < hi)
        centers.append(float((lo + hi) / 2))
        counts.append(int(mask.sum().item()))
        if mask.any():
            avg_conf.append(float(probs[mask].mean().item()))
            avg_acc.append(float(labels[mask].mean().item()))
        else:
            avg_conf.append(np.nan)
            avg_acc.append(np.nan)

    return {
        "centers": np.asarray(centers),
        "counts": np.asarray(counts),
        "avg_conf": np.asarray(avg_conf),
        "avg_acc": np.asarray(avg_acc),
    }


def _summarize(logits, probs, labels, n_bins):
    from sklearn.metrics import precision_score, recall_score
    preds = (probs >= 0.5).long()
    preds_np = preds.numpy()
    labels_np = labels.long().numpy()
    acc = float((preds == labels.long()).float().mean().item())
    return {
        "accuracy": acc,
        "precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "recall": float(recall_score(labels_np, preds_np, zero_division=0)),
        "ece": binary_ece(probs, labels, n_bins=n_bins),
    }


# %% Calibration plots
raw_metrics = _summarize(all_logits, raw_probs, all_labels, N_BINS)
cal_metrics = _summarize(all_logits / temperature, cal_probs, all_labels, N_BINS)

raw_bins = _compute_bin_stats(raw_probs, all_labels, N_BINS)
cal_bins = _compute_bin_stats(cal_probs, all_labels, N_BINS)

pos_mask = all_labels == 1
neg_mask = all_labels == 0
hist_bins = np.linspace(0.0, 1.0, 31)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Raw probability histogram ---
ax = axes[0, 0]
ax.hist(raw_probs[neg_mask].numpy(), bins=hist_bins, alpha=0.6, label="negatives", color="#d95f02")
ax.hist(raw_probs[pos_mask].numpy(), bins=hist_bins, alpha=0.6, label="positives", color="#1b9e77")
ax.set_title("Raw Probability Histogram")
ax.set_xlabel("Predicted positive probability")
ax.set_ylabel("Count")
ax.legend()
ax.grid(alpha=0.2)

# --- Calibrated probability histogram ---
ax = axes[0, 1]
ax.hist(cal_probs[neg_mask].numpy(), bins=hist_bins, alpha=0.6, label="negatives", color="#d95f02")
ax.hist(cal_probs[pos_mask].numpy(), bins=hist_bins, alpha=0.6, label="positives", color="#1b9e77")
ax.set_title("Calibrated Probability Histogram")
ax.set_xlabel("Predicted positive probability")
ax.set_ylabel("Count")
ax.legend()
ax.grid(alpha=0.2)

# --- Reliability diagram ---
ax = axes[1, 0]
valid_raw = ~np.isnan(raw_bins["avg_conf"])
valid_cal = ~np.isnan(cal_bins["avg_conf"])
ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")
ax.plot(
    raw_bins["avg_conf"][valid_raw], raw_bins["avg_acc"][valid_raw],
    marker="o", color="#7570b3",
    label=f"raw (ECE={raw_metrics['ece']:.3f})",
)
ax.plot(
    cal_bins["avg_conf"][valid_cal], cal_bins["avg_acc"][valid_cal],
    marker="o", color="#e7298a",
    label=f"calibrated (ECE={cal_metrics['ece']:.3f})",
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Reliability Diagram")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Empirical positive rate")
ax.legend()
ax.grid(alpha=0.2)

# --- Summary text ---
ax = axes[1, 1]
metric_text = (
    f"Target channel: {TARGET_IDX}\n"
    f"Crop size: {CROP_SIZE}, overlap: {OVERLAP}\n"
    f"Patches: {len(all_labels)}\n"
    f"Temperature: {temperature:.4f}\n\n"
    "Raw\n"
    f"  accuracy  = {raw_metrics['accuracy']:.4f}\n"
    f"  precision = {raw_metrics['precision']:.4f}\n"
    f"  recall    = {raw_metrics['recall']:.4f}\n"
    f"  ece       = {raw_metrics['ece']:.4f}\n\n"
    "Calibrated\n"
    f"  accuracy  = {cal_metrics['accuracy']:.4f}\n"
    f"  precision = {cal_metrics['precision']:.4f}\n"
    f"  recall    = {cal_metrics['recall']:.4f}\n"
    f"  ece       = {cal_metrics['ece']:.4f}"
)
ax.text(
    0.02, 0.98, metric_text,
    va="top", ha="left", fontsize=12, family="monospace",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)
ax.set_title("Summary")
ax.axis("off")

fig.suptitle("LambdaSplit Binary Critic — Calibration Inspection", fontsize=16)
plt.tight_layout()
plt.show()


# %% Patch gallery
def plot_patch_gallery(
    patches: list[torch.Tensor],
    labels: torch.Tensor,
    probs: torch.Tensor,
    n: int = 6,
) -> None:
    """Plot a 6-row gallery of representative patches.

    Rows (top to bottom):
    1. True positives  — highest confidence (prob > .95)
    2. False positives — highest confidence (prob > .95)
    3. True positives  — most uncertain   (prob closest to .5)
    4. True negatives  — highest confidence (prob < .05)
    5. False negatives — highest confidence (prob < .05)
    6. True negatives  — most uncertain   (prob closest to .5)
    """
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().long()
    preds = (probs > 0.5).long()

    tp_mask = (preds == 1) & (labels == 1)
    fp_mask = (preds == 1) & (labels == 0)
    tn_mask = (preds == 0) & (labels == 0)
    fn_mask = (preds == 0) & (labels == 1)

    def _pick(mask, prob_lo, prob_hi, k):
        """Randomly sample k indices from mask where probs fall in [prob_lo, prob_hi]."""
        region = mask & (probs >= prob_lo) & (probs <= prob_hi)
        idxs = region.nonzero(as_tuple=False).squeeze(-1)
        if len(idxs) == 0:
            return []
        perm = torch.randperm(len(idxs))[:k]
        return idxs[perm].tolist()

    rows = [
        ("TP — high conf (p∈[.95,1])",  _pick(tp_mask, 0.95, 1.0, n)),
        ("FP — high conf (p∈[.95,1])",  _pick(fp_mask, 0.95, 1.0, n)),
        ("TP — uncertain (p∈[.40,.60])", _pick(tp_mask, 0.4, 0.6, n)),
        ("FP — uncertain (p∈[.40,.60])", _pick(fp_mask, 0.4, 0.6, n)),
        ("TN — high conf (p∈[0,.05])",  _pick(tn_mask, 0.0, 0.05, n)),
        ("FN — high conf (p∈[0,.05])",  _pick(fn_mask, 0.0, 0.05, n)),
        ("TN — uncertain (p∈[.40,.60])", _pick(tn_mask, 0.4, 0.6, n)),
        ("FN — uncertain (p∈[.40,.60])", _pick(fn_mask, 0.4, 0.6, n)),
    ]

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n, figsize=(2.4 * n, 2.8 * n_rows))
    if n == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (row_title, idxs) in enumerate(rows):
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            if col_idx < len(idxs):
                i = idxs[col_idx]
                img = patches[i].squeeze(0).numpy()
                ax.imshow(img, cmap="gray")
                gt = "pos" if labels[i] == 1 else "neg"
                ax.set_title(f"p={probs[i]:.2f} | {gt}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_idx, 0].set_ylabel(row_title, fontsize=9, rotation=90, labelpad=40)

    fig.suptitle("Patch Gallery — Representative Examples", fontsize=14)
    plt.tight_layout()
    plt.show()


plot_patch_gallery(dataset.patches, all_labels, cal_probs, n=8)

# %%
