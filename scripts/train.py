"""Unified training script for protein classification.

Supports CellAtlas and BioSR datasets, multiclass and binary modes,
and DenseNet / ResNet architectures.

Examples
--------
# Multiclass DenseNet121 on CellAtlas (default)
python scripts/train.py --num_crops 4

# Binary ResNet18 on CellAtlas targeting Mitochondria
python scripts/train.py --arch resnet18 --binary --target Mitochondria \
    --loss binary_focal_loss --num_crops 4

# Multiclass DenseNet121 on BioSR
python scripts/train.py --dataset BioSR --labels F-actin Microtubules CCPs ER \
    --data_dir /group/jug/federico/data/BioSR_v2 \
    --stats_path data_stats_biosr.json \
    --img_size 1004 --crop_size 1004 --bit_depth 16 --num_crops 4
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
from protein_classification.data import BinaryDataset, MultiClassDataset
from protein_classification.data.biosr import get_biosr_filepaths_and_labels
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.utils import collate_multi_crop_batches, train_test_split
from protein_classification.model import BioStructClassifier
from protein_classification.utils.callbacks import get_callbacks
from protein_classification.utils.io import load_dataset_stats, get_log_dir, log_configs

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Train a protein classification model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# --- dataset ---
ds = parser.add_argument_group("dataset")
ds.add_argument("--dataset", type=str, default="CellAtlas", choices=["CellAtlas", "BioSR"])
ds.add_argument("--data_dir", type=str, default=None,
                help="Path to dataset root. Defaults per dataset if omitted.")
ds.add_argument("--labels", type=str, nargs="+", default=["Mitochondria"],
                help="Protein labels to include.")
ds.add_argument("--stats_path", type=str, default=None,
                help="Path to dataset stats JSON. Defaults per dataset if omitted.")
ds.add_argument("--img_size", type=int, default=None,
                help="Image resize dimension. Defaults per dataset if omitted.")
ds.add_argument("--bit_depth", type=int, default=None,
                help="Input image bit depth (8 or 16). Defaults per dataset if omitted.")

# --- binary mode ---
bm = parser.add_argument_group("binary mode")
bm.add_argument("--binary", action="store_true",
                help="Train a one-vs-rest binary classifier.")
bm.add_argument("--target", type=str, default=None,
                help="Target label name for binary mode (e.g. 'Mitochondria').")

# --- augmentation ---
ag = parser.add_argument_group("augmentation")
ag.add_argument("--aug", type=str, default=None,
                choices=["geometric", "intensity", "noise", "all"],
                help="Augmentation applied at train time.")
ag.add_argument("--crop_size", type=int, default=None,
                help="Crop size in pixels. Defaults to img_size (no cropping).")
ag.add_argument("--num_crops", type=int, required=True,
                help="Number of crops sampled per source image.")

# --- architecture ---
ar = parser.add_argument_group("architecture")
ar.add_argument("--arch", type=str, default="densenet121",
                choices=["densenet121", "densenet161", "densenet169", "densenet201",
                         "resnet18", "resnet34"],
                help="Model architecture.")
ar.add_argument("--dropout_p", type=float, default=0.1,
                help="Dropout probability (0 disables).")

# --- loss ---
parser.add_argument("--loss", type=str, default="multiclass_focal_loss",
                    choices=["multiclass_focal_loss", "binary_focal_loss", "binary_bce_loss"])

# --- training ---
tr = parser.add_argument_group("training")
tr.add_argument("--batch_size", type=int, default=32)
tr.add_argument("--acc_batches", type=int, default=1,
                help="Gradient accumulation steps.")
tr.add_argument("--lr", type=float, default=3e-4)
tr.add_argument("--max_epochs", type=int, default=100)
tr.add_argument("--grad_clip", type=float, default=1.0)
tr.add_argument("--normalize", type=str, default="std", choices=["std", "minmax"])
tr.add_argument("--num_workers", type=int, default=3)

# --- calibration ---
parser.add_argument("--calibrate", action="store_true",
                    help="Run post-training temperature scaling on the validation set "
                         "(binary mode only).")

# --- logging ---
lg = parser.add_argument_group("logging")
lg.add_argument("--log", action="store_true", help="Enable Weights & Biases logging.")
lg.add_argument("--log_base_dir", type=str,
                default="/group/jug/federico/classification_training")
lg.add_argument("--debug", action="store_true",
                help="Limit data to 200 samples for fast iteration.")

args = parser.parse_args()

if args.calibrate and not args.binary:
    parser.error("--calibrate is only supported in --binary mode.")
if args.calibrate and not args.log:
    parser.error("--calibrate requires --log (need a run directory to save calibration metadata).")

# ── dataset-specific defaults ────────────────────────────────────────────────

DATASET_DEFAULTS = {
    "CellAtlas": {
        "data_dir": "/group/jug/federico/data/CellAtlas",
        "stats_path": "data_stats_cellatlas.json",
        "img_size": 2048,
        "bit_depth": 8,
    },
    "BioSR": {
        "data_dir": "/group/jug/federico/data/BioSR_v2",
        "stats_path": "data_stats_biosr.json",
        "img_size": 1004,
        "bit_depth": 16,
    },
}

defaults = DATASET_DEFAULTS[args.dataset]
DATA_DIR = args.data_dir or defaults["data_dir"]
STATS_PATH = args.stats_path or defaults["stats_path"]
IMG_SIZE = args.img_size or defaults["img_size"]
BIT_DEPTH = args.bit_depth or defaults["bit_depth"]
CROP_SIZE = args.crop_size or IMG_SIZE

torch.set_float32_matmul_precision("medium")

# ── configurations ───────────────────────────────────────────────────────────

dataset_stats = load_dataset_stats(stats_path=STATS_PATH, labels=args.labels)

train_aug_config = DataAugmentationConfig(
    transform=args.aug,
    crop_size=CROP_SIZE,
    random_crop=True,
)
val_aug_config = train_aug_config.model_copy(update={"transform": None})

data_config = DataConfig(
    data_dir=DATA_DIR,
    labels=args.labels,
    img_size=IMG_SIZE,
    train_augmentation_config=train_aug_config,
    val_augmentation_config=val_aug_config,
    bit_depth=BIT_DEPTH,
    normalize=args.normalize,
    dataset_stats=(dataset_stats["mean"], dataset_stats["std"]),
)

# --- model config ---
num_classes = 1 if args.binary else len(args.labels) + 3  # +3 for Nucleus, Microtubules, ER
if args.arch.startswith("resnet"):
    model_config = ResNetConfig(
        architecture=args.arch,
        num_classes=num_classes,
        dropout_p=args.dropout_p,
    )
else:
    model_config = DenseNetConfig(
        architecture=args.arch,
        num_classes=num_classes,
        dropout_block=args.dropout_p > 0,
        dropout_p=args.dropout_p,
    )

loss_config = LossConfig(loss_type=args.loss)

# --- experiment name ---
mode_tag = "binary" if args.binary else f"{num_classes}Cl"
target_tag = f"_{args.target}" if args.binary else f"_{args.labels[0]}"
arch_name = args.arch.replace("dense", "Dense").replace("net", "Net")
exp_name = f"{arch_name}_{args.dataset}_{mode_tag}{target_tag}"

log_dir = get_log_dir(args.log_base_dir, exp_name) if args.log else None

training_config = TrainingConfig(
    max_epochs=args.max_epochs,
    lr=args.lr,
    batch_size=args.batch_size,
    gradient_clip_val=args.grad_clip,
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
        data_dir=DATA_DIR, extra_labels=args.labels,
    )
elif args.dataset == "BioSR":
    input_data, curr_labels = get_biosr_filepaths_and_labels(
        data_dir=DATA_DIR, protein_labels=args.labels,
    )

if args.debug:
    input_data = input_data[:200]

train_data, _ = train_test_split(input_data, train_ratio=0.9, deterministic=True)
train_data, val_data = train_test_split(train_data, train_ratio=0.9, deterministic=False)

print("-------------- Dataset Info --------------")
print(f"Dataset            : {args.dataset}")
print(f"Mode               : {'binary (target=' + args.target + ')' if args.binary else 'multiclass'}")
print(f"Architecture       : {args.arch}")
print(f"Training samples   : {len(train_data)}")
print(f"Validation samples : {len(val_data)}")
print(f"Labels             : {curr_labels}")
print("------------------------------------------\n")

# --- build datasets ---
shared_dataset_kwargs = dict(
    img_size=IMG_SIZE,
    num_crops_per_image=args.num_crops,
    bit_depth=BIT_DEPTH,
    normalize=data_config.normalize,
    normalization_scope=data_config.normalization_scope,
    dataset_stats=data_config.dataset_stats,
    return_label=True,
)

if args.binary:
    if args.target is None:
        parser.error("--target is required when --binary is set.")
    if args.target not in curr_labels:
        parser.error(f"--target '{args.target}' not found. Available: {list(curr_labels)}")
    target_label = curr_labels[args.target]

    train_dataset = BinaryDataset(
        inputs=train_data, split="train",
        augmentation_config=train_aug_config,
        target_label=target_label,
        **shared_dataset_kwargs,
    )
    val_dataset = BinaryDataset(
        inputs=val_data, split="test",
        augmentation_config=val_aug_config,
        target_label=target_label,
        **shared_dataset_kwargs,
    )
else:
    train_dataset = MultiClassDataset(
        inputs=train_data, split="train",
        augmentation_config=train_aug_config,
        **shared_dataset_kwargs,
    )
    val_dataset = MultiClassDataset(
        inputs=val_data, split="test",
        augmentation_config=val_aug_config,
        **shared_dataset_kwargs,
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_multi_crop_batches,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=training_config.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_multi_crop_batches,
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
    enable_checkpointing=True,
    precision=training_config.precision,
    gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    gradient_clip_val=training_config.gradient_clip_val,
    accumulate_grad_batches=training_config.accumulate_grad_batches,
    log_every_n_steps=10,
)
trainer.fit(model, train_loader, val_loader)

# ── post-training calibration ───────────────────────────────────────────────

if args.calibrate:
    from protein_classification.utils.calibration import (
        apply_temperature, binary_brier, binary_ece, binary_nll, fit_temperature,
    )
    from protein_classification.utils.io import get_checkpoint_path, load_checkpoint, save_calibration

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
