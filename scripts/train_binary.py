"""Training script for one-vs-rest binary protein classification.

Supports CellAtlas and BioSR datasets, DenseNet and ResNet architectures.

Examples
--------
# CellAtlas — Mitochondria vs rest, ResNet18
python scripts/train_binary.py --target Mitochondria --arch resnet18 \
    --crop_size 256 --num_crops 4

# CellAtlas — Mitochondria vs rest, DenseNet121, geometric augmentation
python scripts/train_binary.py --target Mitochondria --aug geometric --num_crops 4
"""
import argparse
import os
import socket

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from protein_classification.config import (
    AlgorithmConfig, DataAugmentationConfig, DataConfig, LossConfig, TrainingConfig,
)
from protein_classification.config.architectures import DenseNetConfig, ResNetConfig
from protein_classification.data import BinaryDataset
from protein_classification.data.background import BackgroundAnalyzer
from protein_classification.data.biosr import get_biosr_filepaths_and_labels
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.utils import collate_multi_crop_batches, train_test_split
from protein_classification.model import BioStructClassifier
from protein_classification.utils.callbacks import get_callbacks
from protein_classification.utils.calibration import (
    apply_temperature, binary_brier, binary_ece, binary_nll, fit_temperature,
)
from protein_classification.utils.io import (
    load_dataset_stats, get_log_dir, log_configs,
    get_checkpoint_path, load_checkpoint, save_calibration
)

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Train a one-vs-rest binary protein classifier.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# --- dataset ---
ds = parser.add_argument_group("dataset")
ds.add_argument("--dataset", type=str, required=True, choices=["CellAtlas", "BioSR"])
ds.add_argument(
    "--labels", type=str, nargs="+", default=None,
    help="Protein labels to include (determines the full label set)."
)
ds.add_argument(
    "--img-size", type=int, default=None,
    help="Image resize dimension. Defaults per dataset if omitted."
)
ds.add_argument("--norm_type", type=str, default="minmax", choices=["std", "minmax"])
ds.add_argument("--norm_scope", type=str, default="image", choices=["dataset", "image"])
ds.add_argument(
    "--stats-path", type=str, default=None,
    help="Path to the dataset statistics file."
)
ds.add_argument(
    "--aug", type=str, default=None,
    choices=["geometric", "intensity", "noise", "all"],
    help="Augmentation applied at train time."
)
ds.add_argument(
    "--bg-cache", type=str, default=None,
    help="Path to precomputed background positions JSON (from precompute_background.py)."
)
ds.add_argument(
    "--crop-size", type=int, default=256,
    help="Crop size in pixels. Defaults to img_size (no cropping)."
)
ds.add_argument(
    "--num-crops", type=int, default=8,
    help="Number of crops sampled per source image."
)

# --- binary target ---
bt = parser.add_argument_group("binary target")
bt.add_argument(
    "--target", type=str, required=True,
    help="Target label name for the positive class (e.g. 'Mitochondria')."
)
bt.add_argument(
    "--pos-prob", type=float, default=0.5,
    help="Probability of sampling a positive crop per draw."
)
bt.add_argument(
    "--neg-families", type=str, nargs="+",
    default=["trivial", "mixed", "inverted"],
    help="Negative families to enable."
)
bt.add_argument(
    "--neg-weights", type=float, nargs="+",
    default=None,
    help="Weights for each negative family (same order as --neg_families)."
)

# --- architecture ---
ar = parser.add_argument_group("architecture")
ar.add_argument(
    "--arch", type=str, default="resnet18",
    choices=[
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "resnet18",
        "resnet34"
    ],
    help="Model architecture."
)

# --- loss ---
parser.add_argument(
    "--loss", type=str, default="binary_cross_entropy",
    choices=["binary_cross_entropy", "binary_focal"],
)
parser.add_argument(
    "--calibrate",
    action="store_true",
    help="Perform post-training temperature scaling."
)

# --- training ---
tr = parser.add_argument_group("training")
tr.add_argument("--batch_size", type=int, default=32)
tr.add_argument(
    "--acc-batches", type=int, default=1,
    help="Gradient accumulation steps."
)
tr.add_argument("--lr", type=float, default=1e-3)
tr.add_argument("--epochs", type=int, default=100)
tr.add_argument("--num-workers", type=int, default=3)

# --- logging ---
lg = parser.add_argument_group("logging")
lg.add_argument("--log", action="store_true", help="Enable Weights & Biases logging.")
lg.add_argument(
    "--log-base-dir", type=str,
    default="/group/jug/federico/critic_net_training"
)
lg.add_argument(
    "--debug", action="store_true",
    help="Limit data to 200 samples for fast iteration."
)

args = parser.parse_args()

if args.neg_weights is None:
    neg_weights = [1.0] * len(args.neg_families)
else:
    if len(args.neg_families) != len(args.neg_weights):
        parser.error("--neg_families and --neg_weights must have the same length.")
    neg_weights = args.neg_weights
negative_family_weights = dict(zip(args.neg_families, neg_weights))

# ── dataset-specific defaults ────────────────────────────────────────────────

DATASET_DEFAULTS = {
    "CellAtlas": {
        "data_dir": "/group/jug/federico/data/CellAtlas",
        "stats_path": "data_stats_cellatlas.json",
        "img_size": 2048,
        "bit_depth": 8,
        "labels": ["Nucleus", "Mitochondria", "Endoplasmic reticulum", "Microtubules"],
        "background_threshold_quantiles_by_label": {0: 0.25, 1: 0.1, 2: 0.1, 3: 0.1},
    },
    "BioSR": {
        "data_dir": "/group/jug/federico/data/BioSR_v2",
        "stats_path": "data_stats_biosr.json",
        "img_size": 1004,
        "bit_depth": 16,
        "labels": ["F-actin", "CCPs", "ER", "Microtubules"]
    },
}

defaults = DATASET_DEFAULTS[args.dataset]
DATA_DIR = defaults["data_dir"]
STATS_PATH = defaults["stats_path"]
LABELS = args.labels or defaults["labels"]
IMG_SIZE = args.img_size or defaults["img_size"]
BIT_DEPTH = defaults["bit_depth"]
CROP_SIZE = args.crop_size or IMG_SIZE

torch.set_float32_matmul_precision("medium")

# ── configurations ───────────────────────────────────────────────────────────

if args.norm_scope == "dataset":
    stats_dict = load_dataset_stats(stats_path=STATS_PATH, labels=args.labels)
    if args.norm_type == "std":
        dataset_stats = (stats_dict.get("mean"), stats_dict.get("std"))
    elif args.norm_type == "minmax":
        dataset_stats = (stats_dict.get("min"), stats_dict.get("max"))
else:
    dataset_stats = None

train_aug_config = DataAugmentationConfig(
    transform=args.aug,
    crop_size=CROP_SIZE,
    random_crop=True
)
val_aug_config = train_aug_config.model_copy(update={"transform": None})

data_config = DataConfig(
    data_dir=DATA_DIR,
    labels=LABELS,
    img_size=IMG_SIZE,
    train_augmentation_config=train_aug_config,
    val_augmentation_config=val_aug_config,
    bit_depth=BIT_DEPTH,
    normalize=args.norm_type,
    normalization_scope=args.norm_scope,
    dataset_stats=dataset_stats,
)

# --- model config ---
if args.arch.startswith("resnet"):
    model_config = ResNetConfig(
        architecture=args.arch,
        num_classes=1,
        dropout_p=0.1,
    )
else:
    model_config = DenseNetConfig(
        architecture=args.arch,
        num_classes=1,
        dropout_block=args.dropout_p > 0,
        dropout_p=0.5,
    )

loss_config = LossConfig(loss_type=args.loss)

exp_name = f"{args.arch}_{args.dataset}_{args.target}_binary"
if args.log:
    log_dir = get_log_dir(
        base_dir=args.log_base_dir, exp_name=exp_name
    )
else:
    log_dir = None

# NOTE: to get an effective batch size of B with N crops 
# per image, we need to set the effective batch size to B/N.
# Indeed, in this way the `__getitem__` will be called B/N
# times, and each time it will return N crops from the same
# image, which will be collated together by the `collate_fn`
# to form an actual batch of size B.
dloader_batch_size = args.batch_size / args.num_crops
training_config = TrainingConfig(
    max_epochs=args.epochs,
    lr=args.lr,
    batch_size=dloader_batch_size,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=args.acc_batches,
)

algo_config = AlgorithmConfig(
    mode="train",
    log_dir=log_dir,
    architecture_config=model_config,
    loss_config=loss_config,
    training_config=training_config,
)

# ── data setup ───────────────────────────────────────────────────────────────

if args.dataset == "CellAtlas":
    input_data, curr_labels = get_cellatlas_filepaths_and_labels(
        data_dir=DATA_DIR, labels=LABELS,
    )
elif args.dataset == "BioSR":
    input_data, curr_labels = get_biosr_filepaths_and_labels(
        data_dir=DATA_DIR, labels=LABELS,
    )

if args.target not in curr_labels:
    parser.error(f"--target '{args.target}' not in available labels: {list(curr_labels)}")
target_label_id = curr_labels[args.target]

if args.debug:
    input_data = input_data[:20]

train_data, _ = train_test_split(input_data, train_ratio=0.9, deterministic=True)
train_data, val_data = train_test_split(train_data, train_ratio=0.9, deterministic=False)

print("-------------- Dataset Info --------------")
print(f"Dataset            : {args.dataset}")
print(f"Mode               : binary (target={args.target}, id={target_label_id})")
print(f"Architecture       : {args.arch}")
print(f"Positive prob      : {args.pos_prob}")
print(f"Negative families  : {negative_family_weights}")
print(f"Training samples   : {len(train_data)}")
print(f"Validation samples : {len(val_data)}")
print(f"Labels             : {curr_labels}")
print("------------------------------------------\n")

# ── Background analysis ─────────────────────────────────────────────────────
# Compute foreground thresholds on train data, then derive valid crop
# positions for both train and val using the same thresholds.
train_bg_positions = None
val_bg_positions = None
bg_stride = CROP_SIZE // 4

if args.bg_cache:
    # Load precomputed positions from cache
    bg_data = BackgroundAnalyzer.load(args.bg_cache)
    train_bg_positions = BackgroundAnalyzer.to_index_keyed(
        bg_data["valid_positions_by_filepath"], train_data
    )
    val_bg_positions = BackgroundAnalyzer.to_index_keyed(
        bg_data["valid_positions_by_filepath"], val_data
    )
    print(f"Loaded background cache from: {args.bg_cache}")
else:
    # Compute inline (useful for small datasets)
    train_analyzer = BackgroundAnalyzer(
        inputs=train_data,
        crop_size=train_aug_config.crop_size,
        stride=bg_stride,
        img_size=IMG_SIZE,
        metrics=data_config.background_metrics or ["std"],
        quantile=data_config.background_threshold_quantile,
        quantiles_by_label=defaults.get("background_threshold_quantiles_by_label"),
    )
    train_bg_positions = train_analyzer.valid_positions_by_index

    val_analyzer = BackgroundAnalyzer(
        inputs=val_data,
        crop_size=val_aug_config.crop_size,
        stride=bg_stride,
        img_size=IMG_SIZE,
        metrics=data_config.background_metrics or ["std"],
        thresholds_by_label=train_analyzer.thresholds_by_label,
    )
    val_bg_positions = val_analyzer.valid_positions_by_index


# ── Data modules ─────────────────────────────────────────────────────
train_dataset = BinaryDataset(
    inputs=train_data,
    split="train",
    augmentation_config=train_aug_config,
    target_label=target_label_id,
    positive_probability=args.pos_prob,
    negative_family_weights=negative_family_weights,
    img_size=IMG_SIZE,
    num_crops_per_image=args.num_crops,
    bit_depth=BIT_DEPTH,
    normalize=data_config.normalize,
    normalization_scope=data_config.normalization_scope,
    dataset_stats=data_config.dataset_stats,
    return_label=True,
    valid_crop_positions=train_bg_positions,
    crop_position_jitter=bg_stride // 2,
)
val_dataset = BinaryDataset(
    inputs=val_data,
    split="test",
    augmentation_config=val_aug_config,
    target_label=target_label_id,
    positive_probability=args.pos_prob,
    negative_family_weights=negative_family_weights,
    img_size=IMG_SIZE,
    num_crops_per_image=args.num_crops,
    bit_depth=BIT_DEPTH,
    normalize=data_config.normalize,
    normalization_scope=data_config.normalization_scope,
    dataset_stats=data_config.dataset_stats,
    return_label=True,
    valid_crop_positions=val_bg_positions,
    crop_position_jitter=0,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_multi_crop_batches,
    prefetch_factor=2 if args.num_workers > 0 else None,
    persistent_workers=args.num_workers > 0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=training_config.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_multi_crop_batches,
    prefetch_factor=2 if args.num_workers > 0 else None,
    persistent_workers=args.num_workers > 0,
)

# ── logger ───────────────────────────────────────────────────────────────────

if args.log:
    logger = WandbLogger(
        name=os.path.join(
            socket.gethostname(),
            "/".join(str(log_dir).split("/")[-3:]),
        ),
        save_dir=log_dir,
        project=algo_config.wandb_project,
    )
    log_configs(
        configs=[algo_config, data_config],
        names=["algorithm", "data"],
        log_dir=log_dir,
        logger=logger,
    )
else:
    logger = None

# ── train ────────────────────────────────────────────────────────────────────

model = BioStructClassifier(config=algo_config)
callbacks = get_callbacks(logdir=log_dir, training_config=training_config)

trainer = Trainer(
    accelerator="gpu",
    max_epochs=training_config.max_epochs,
    logger=logger,
    callbacks=callbacks,
    enable_progress_bar=True,
    enable_model_summary=True,
    precision=training_config.precision,
    gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    gradient_clip_val=training_config.gradient_clip_val,
    accumulate_grad_batches=training_config.accumulate_grad_batches,
    log_every_n_steps=10,
)
trainer.fit(model, train_loader, val_loader)

# ── post-training calibration ───────────────────────────────────────────────

if args.calibrate:
    # load best checkpoint
    best_ckpt_path = get_checkpoint_path(str(log_dir), mode="best")
    best_ckpt = load_checkpoint(str(log_dir), best=True)
    model.load_state_dict(best_ckpt["state_dict"], strict=True)
    model.eval()

    # collect logits and labels on validation set
    predict_trainer = Trainer(
        accelerator="gpu",
        enable_progress_bar=True,
        precision=training_config.precision,
    )
    outputs = predict_trainer.predict(model=model, dataloaders=val_loader)
    all_logits = torch.cat([batch[1] for batch in outputs])
    all_labels = torch.cat([batch[2] for batch in outputs])

    # pre-calibration metrics
    pre_probs = torch.sigmoid(all_logits)
    pre_nll = binary_nll(all_logits, all_labels)
    pre_brier = binary_brier(pre_probs, all_labels)
    pre_ece = binary_ece(pre_probs, all_labels)

    # fit temperature
    temperature = fit_temperature(all_logits, all_labels)

    # post-calibration metrics
    post_probs = apply_temperature(all_logits, temperature)
    post_nll = binary_nll(all_logits / temperature, all_labels)
    post_brier = binary_brier(post_probs, all_labels)
    post_ece = binary_ece(post_probs, all_labels)

    calibration_data = {
        "temperature": temperature,
        "num_samples": len(all_labels),
        "checkpoint_path": best_ckpt_path,
        "pre_calibration": {"nll": pre_nll, "brier": pre_brier, "ece": pre_ece},
        "post_calibration": {"nll": post_nll, "brier": post_brier, "ece": post_ece},
    }
    cal_path = save_calibration(log_dir, calibration_data)

    print("\n──────────── Calibration Results ────────────")
    print(f"Temperature        : {temperature:.4f}")
    print(f"Validation samples : {len(all_labels)}")
    print(f"Pre  NLL / Brier / ECE : {pre_nll:.4f} / {pre_brier:.4f} / {pre_ece:.4f}")
    print(f"Post NLL / Brier / ECE : {post_nll:.4f} / {post_brier:.4f} / {post_ece:.4f}")
    print(f"Saved to: {cal_path}")
    print("─────────────────────────────────────────────\n")

if args.log:
    wandb.finish()
