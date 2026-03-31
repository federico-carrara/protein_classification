import argparse
import os
import socket

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from protein_classification.config import (
    AlgorithmConfig, DataAugmentationConfig, DataConfig,
    DenseNetConfig, LossConfig, TrainingConfig
)
from protein_classification.data import MultiClassDataset
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.utils import (
    collate_multi_crop_batches,
    compute_background_thresholds,
    train_test_split,
)
from protein_classification.model import BioStructClassifier
from protein_classification.utils.callbacks import get_callbacks
from protein_classification.utils.io import load_dataset_stats, get_log_dir, log_configs

parser = argparse.ArgumentParser(description="Train a protein classification model.")
parser.add_argument("--log", action="store_true", help="Enable logging with Weights & Biases.")
parser.add_argument("--aug", type=str, default=None, choices=["geometric", "intensity", "noise", "all"])
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
parser.add_argument("--acc_batches", type=int, default=1, help="Number of batches to accumulate gradients over.")
parser.add_argument("--img_size", type=int, default=2048, help="Size of the input images.")
parser.add_argument("--crop_size", type=int, default=2048, help="Crop size for the input images.")
parser.add_argument("--num_crops_per_image", type=int, required=True, help="Number of crops to sample from each loaded image.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode for faster training with fewer samples.")
args = parser.parse_args()

LOGGING = args.log
torch.set_float32_matmul_precision('medium')


# --- Set Configurations ---
dataset_stats = load_dataset_stats(
    stats_path="data_stats_cellatlas.json", labels=["Mitochondria"]
)
train_aug_config = DataAugmentationConfig(
    transform=args.aug,
    crop_size=args.crop_size,
    random_crop=True,
    strategy=None
    # strategy="background",
    # metrics=["std"],
    # bg_threshold=3.0, # Default threshold for background crops
)
val_aug_config = train_aug_config.model_copy(update={})
data_config = DataConfig(
    data_dir="/group/jug/federico/data/CellAtlas",
    labels=["Mitochondria"],
    img_size=args.img_size,
    train_augmentation_config=train_aug_config,
    val_augmentation_config=val_aug_config,
    bit_depth=8,
    normalize="std",
    dataset_stats=(dataset_stats["mean"], dataset_stats["std"]),
)
model_config = DenseNetConfig(
    architecture="densenet121",
    num_classes=4, # 4 classes + 1 for background
    num_init_features=64,
    dropout_block=False,
    dropout_p=0.1
)
loss_config = LossConfig(loss_type="multiclass_focal_loss")
exp_name = f"DenseNet121_CellAtlas_{model_config.num_classes}Cl_{data_config.labels[0]}" # TODO: make it more general
if LOGGING:
    log_dir=get_log_dir(
        base_dir="/group/jug/federico/classification_training",
        exp_name=exp_name,
    )
else:
    log_dir = None
training_config = TrainingConfig(
    max_epochs=100,
    lr=3e-4,
    batch_size=args.batch_size,
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

# --- Data Setup ---
input_data, curr_labels = get_cellatlas_filepaths_and_labels(
    data_dir=data_config.data_dir, protein_labels=data_config.labels,
)
if args.debug:
    input_data = input_data[:200]  # Limit to 200 samples for testing
train_input_data, _ = train_test_split(
    input_data, train_ratio=0.9, deterministic=True
)
train_input_data, val_input_data = train_test_split(
    train_input_data, train_ratio=0.9, deterministic=False
)
print("--------------Dataset Info--------------")
print(f"Number training samples: {len(train_input_data)}")
print(f"Number validation samples: {len(val_input_data)}")
print(f"Labels: {curr_labels}")
print("----------------------------------------\n")
if train_aug_config.strategy == "background":
    train_aug_config.bg_thresholds = compute_background_thresholds(
        inputs=train_input_data,
        img_size=data_config.img_size,
        crop_size=train_aug_config.crop_size,
        random_crop=train_aug_config.random_crop,
        metrics=train_aug_config.metrics,
        quantile=0.1,
        samples_per_image=4,
        max_images=128 if args.debug else None,
        imreader=data_config.imreader,
    )
train_dataset = MultiClassDataset(
    inputs=train_input_data,
    split="train",
    return_label=True,
    img_size=data_config.img_size,
    augmentation_config=data_config.train_augmentation_config,
    num_crops_per_image=args.num_crops_per_image,
    bit_depth=data_config.bit_depth,
    normalize=data_config.normalize,
    normalization_scope=data_config.normalization_scope,
    dataset_stats=data_config.dataset_stats,
)
val_dataset = MultiClassDataset(
    inputs=val_input_data,
    split="test",
    return_label=True,
    img_size=data_config.img_size,
    augmentation_config=data_config.val_augmentation_config,
    num_crops_per_image=args.num_crops_per_image,
    bit_depth=data_config.bit_depth,
    normalize=data_config.normalize,
    normalization_scope=data_config.normalization_scope,
    dataset_stats=data_config.dataset_stats,
)
train_dloader = DataLoader(
    train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    num_workers=3,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_multi_crop_batches,
)
val_dloader = DataLoader(
    val_dataset,
    batch_size=training_config.batch_size,
    shuffle=False,
    num_workers=3,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_multi_crop_batches,
)

# --- Initialize Logger + Log configs ---
if LOGGING:
    logger = WandbLogger(
        name=os.path.join(socket.gethostname(), "/".join(str(log_dir).split("/")[-3:])),
        save_dir=algo_config.log_dir,
        project=algo_config.wandb_project,
    )
    log_configs(
        configs=[algo_config, data_config],
        names=["algorithm", "data"],
        log_dir=algo_config.log_dir,
        logger=logger,
    )
else:
    logger = None

# --- Setup Model & Trainer ---
model = BioStructClassifier(config=algo_config)
callbacks = get_callbacks(
    logdir=algo_config.log_dir,
    training_config=training_config,
)
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
trainer.fit(model, train_dloader, val_dloader)
wandb.finish()
