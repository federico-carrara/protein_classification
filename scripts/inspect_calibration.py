# %% [markdown]
# # Calibration inspection
#
# Hardcoded script to visualize calibration for a trained binary critic.

# %% Parameters
from pathlib import Path

CKPT_DIR = Path("/group/jug/federico/critic_net_training/2604/resnet18_CellAtlas_Nucleus_binary/4")
BG_CACHE = Path("/group/jug/federico/data/CellAtlas/precomputed_bg_CellAtlas_NucMitEndMic_crop256_stride64.json")
TARGET = "Nucleus"
NEGATIVE = {"trivial" : 1.0}
SPLIT_SEED = 0
N_BINS = 15
FIGSIZE = (14, 10)


# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from protein_classification.config import AlgorithmConfig, DataConfig
from protein_classification.data import BinaryDataset
from protein_classification.data.background import BackgroundAnalyzer
from protein_classification.data.biosr import get_biosr_filepaths_and_labels
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.utils import collate_multi_crop_batches, train_test_split
from protein_classification.model import BioStructClassifier
from protein_classification.utils.calibration import (
    apply_temperature,
    binary_brier,
    binary_ece,
    binary_nll,
)
from protein_classification.utils.io import load_calibration, load_checkpoint, load_config


# %% Helpers
def _infer_dataset_name(data_config: DataConfig) -> str:
    data_dir = str(data_config.data_dir)
    if "CellAtlas" in data_dir:
        return "CellAtlas"
    if "BioSR" in data_dir:
        return "BioSR"
    raise RuntimeError("Could not infer dataset name from data_config.")


def _load_input_data(dataset: str, data_config: DataConfig):
    if dataset == "CellAtlas":
        return get_cellatlas_filepaths_and_labels(
            data_dir=data_config.data_dir,
            labels=data_config.labels,
        )
    if dataset == "BioSR":
        return get_biosr_filepaths_and_labels(
            data_dir=data_config.data_dir,
            labels=data_config.labels,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _rebuild_val_dataset(
    data_config: DataConfig,
    bg_cache,
    split_seed: int,
) -> BinaryDataset:
    dataset_name = _infer_dataset_name(data_config)
    input_data, curr_labels = _load_input_data(dataset_name, data_config)

    train_data, _ = train_test_split(input_data, train_ratio=0.9, deterministic=True)
    np.random.seed(split_seed)
    _, val_data = train_test_split(train_data, train_ratio=0.9, deterministic=False)

    target_label_id = curr_labels[data_config.target_label]
    train_aug = data_config.train_augmentation_config
    val_aug = data_config.val_augmentation_config or train_aug.model_copy(update={"transform": None})

    val_bg_positions = None
    if val_aug.crop_size is not None:
        bg_stride = val_aug.crop_size // 4
        if bg_cache:
            bg_data = BackgroundAnalyzer.load(bg_cache)
            val_bg_positions = BackgroundAnalyzer.to_index_keyed(
                bg_data["valid_positions_by_filepath"],
                val_data,
            )
        else:
            train_analyzer = BackgroundAnalyzer(
                inputs=train_data,
                crop_size=train_aug.crop_size,
                stride=bg_stride,
                img_size=data_config.img_size,
                metrics=["entropy"],
            )
            val_analyzer = BackgroundAnalyzer(
                inputs=val_data,
                crop_size=val_aug.crop_size,
                stride=bg_stride,
                img_size=data_config.img_size,
                metrics=["entropy"],
                thresholds_by_label=train_analyzer.thresholds_by_label,
            )
            val_bg_positions = val_analyzer.valid_positions_by_index

    dataset = BinaryDataset(
        inputs=val_data,
        split="test",
        augmentation_config=val_aug,
        target_label=target_label_id,
        negative_family_weights=data_config.negative_family_weights,
        img_size=data_config.img_size,
        num_crops_per_image=1,
        bit_depth=data_config.bit_depth,
        normalize=data_config.normalize,
        normalization_scope=data_config.normalization_scope,
        dataset_stats=data_config.dataset_stats,
        return_label=True,
        valid_crop_positions=val_bg_positions,
        crop_position_jitter=0,
    )
    return dataset


def _collect_logits_and_labels(
    model: BioStructClassifier,
    dataset: BinaryDataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_multi_crop_batches,
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=True,
        logger=False,
        precision=32,
    )
    outputs = trainer.predict(model=model, dataloaders=dataloader)
    logits = torch.cat([batch[1] for batch in outputs]).detach().cpu()
    labels = torch.cat([batch[2] for batch in outputs]).detach().cpu()
    return logits, labels


def _compute_bin_stats(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int,
) -> dict[str, np.ndarray]:
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().float()
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)

    centers = []
    counts = []
    avg_conf = []
    avg_acc = []
    for idx in range(n_bins):
        lo = boundaries[idx]
        hi = boundaries[idx + 1]
        if idx == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
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


def _summarize(logits: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor, n_bins: int) -> dict[str, float]:
    preds = (probs >= 0.5).long()
    acc = float((preds == labels.long()).float().mean().item())
    return {
        "accuracy": acc,
        "nll": binary_nll(logits, labels),
        "brier": binary_brier(probs, labels),
        "ece": binary_ece(probs, labels, n_bins=n_bins),
    }


def _plot_calibration_figure(
    target_name: str,
    split_seed: int,
    temperature: float,
    labels: torch.Tensor,
    raw_probs: torch.Tensor,
    cal_probs: torch.Tensor,
    raw_metrics: dict[str, float],
    cal_metrics: dict[str, float],
    n_bins: int,
) -> None:
    raw_bins = _compute_bin_stats(raw_probs, labels, n_bins)
    cal_bins = _compute_bin_stats(cal_probs, labels, n_bins)

    pos_mask = labels == 1
    neg_mask = labels == 0
    hist_bins = np.linspace(0.0, 1.0, 31)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

    ax = axes[0, 0]
    ax.hist(raw_probs[neg_mask].numpy(), bins=hist_bins, alpha=0.6, label="negatives", color="#d95f02")
    ax.hist(raw_probs[pos_mask].numpy(), bins=hist_bins, alpha=0.6, label="positives", color="#1b9e77")
    ax.set_title("Raw Probability Histogram")
    ax.set_xlabel("Predicted positive probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.hist(cal_probs[neg_mask].numpy(), bins=hist_bins, alpha=0.6, label="negatives", color="#d95f02")
    ax.hist(cal_probs[pos_mask].numpy(), bins=hist_bins, alpha=0.6, label="positives", color="#1b9e77")
    ax.set_title("Calibrated Probability Histogram")
    ax.set_xlabel("Predicted positive probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    valid_raw = ~np.isnan(raw_bins["avg_conf"])
    valid_cal = ~np.isnan(cal_bins["avg_conf"])
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")
    ax.plot(
        raw_bins["avg_conf"][valid_raw],
        raw_bins["avg_acc"][valid_raw],
        marker="o",
        color="#7570b3",
        label=f"raw (ECE={raw_metrics['ece']:.3f})",
    )
    ax.plot(
        cal_bins["avg_conf"][valid_cal],
        cal_bins["avg_acc"][valid_cal],
        marker="o",
        color="#e7298a",
        label=f"calibrated (ECE={cal_metrics['ece']:.3f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    metric_text = (
        f"Target: {target_name}\n"
        f"Samples: {len(labels)}\n"
        f"Split seed: {split_seed}\n"
        f"Temperature: {temperature:.4f}\n\n"
        "Raw\n"
        f"accuracy={raw_metrics['accuracy']:.4f}\n"
        f"nll={raw_metrics['nll']:.4f}\n"
        f"brier={raw_metrics['brier']:.4f}\n"
        f"ece={raw_metrics['ece']:.4f}\n\n"
        "Calibrated\n"
        f"accuracy={cal_metrics['accuracy']:.4f}\n"
        f"nll={cal_metrics['nll']:.4f}\n"
        f"brier={cal_metrics['brier']:.4f}\n"
        f"ece={cal_metrics['ece']:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        metric_text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_title("Summary")
    ax.axis("off")

    fig.suptitle("Calibration Inspection", fontsize=16)
    plt.tight_layout()
    plt.show()


# %% Load run
data_config_dict = load_config(str(CKPT_DIR), "data")
data_config_dict["negative_family_weights"] = NEGATIVE
data_config_dict["target_label"] = TARGET
data_config = DataConfig(**data_config_dict)
algo_config = AlgorithmConfig(**load_config(str(CKPT_DIR), "algorithm"))
calibration = load_calibration(str(CKPT_DIR))
temperature = float(calibration["temperature"])

if algo_config.architecture_config.num_classes != 1:
    raise RuntimeError("This script only supports binary checkpoints.")

dataset = _rebuild_val_dataset(
    data_config=data_config,
    bg_cache=BG_CACHE,
    split_seed=SPLIT_SEED,
)

model = BioStructClassifier(config=algo_config)
ckpt = load_checkpoint(str(CKPT_DIR), best=True)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()

logits, labels = _collect_logits_and_labels(
    model=model,
    dataset=dataset,
    batch_size=max(1, algo_config.training_config.batch_size),
)

raw_probs = torch.sigmoid(logits)
cal_probs = apply_temperature(logits, temperature)

raw_metrics = _summarize(logits, raw_probs, labels, N_BINS)
cal_metrics = _summarize(logits / temperature, cal_probs, labels, N_BINS)

print(f"Target      : {data_config.target_label}")
print(f"Samples     : {len(labels)}")
print(f"Temperature : {temperature:.4f}")
print(
    f"Raw        - acc: {raw_metrics['accuracy']:.4f}, "
    f"NLL: {raw_metrics['nll']:.4f}, "
    f"Brier: {raw_metrics['brier']:.4f}, "
    f"ECE: {raw_metrics['ece']:.4f}"
)
print(
    f"Calibrated - acc: {cal_metrics['accuracy']:.4f}, "
    f"NLL: {cal_metrics['nll']:.4f}, "
    f"Brier: {cal_metrics['brier']:.4f}, "
    f"ECE: {cal_metrics['ece']:.4f}"
)


# %% Plot
_plot_calibration_figure(
    target_name=data_config.target_label,
    split_seed=SPLIT_SEED,
    temperature=temperature,
    labels=labels,
    raw_probs=raw_probs,
    cal_probs=cal_probs,
    raw_metrics=raw_metrics,
    cal_metrics=cal_metrics,
    n_bins=N_BINS,
)
