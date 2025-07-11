import argparse
import os
import shutil
import socket

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from protein_classification.config import (
    AlgorithmConfig, DataConfig, DenseNetConfig, LossConfig, TrainingConfig
)
from protein_classification.data import InMemoryDataset, ZarrDataset
from protein_classification.data.augmentations import (
    train_augmentation, geometric_augmentation, noise_augmentation
)
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.preprocessing import ZarrPreprocessor
from protein_classification.data.utils import train_test_split
from protein_classification.model import BioStructClassifier
from protein_classification.utils.callbacks import get_callbacks
from protein_classification.utils.io import load_dataset_stats, get_log_dir, log_configs

parser = argparse.ArgumentParser(description="Train a protein classification model.")
parser.add_argument("--log", action="store_true", help="Enable logging with Weights & Biases.")
parser.add_argument("--in_memory", action="store_true", help="Load the dataset in memory, else use Zarr preprocessing.")
args = parser.parse_args()

LOGGING = args.log
IN_MEMORY = args.in_memory
torch.set_float32_matmul_precision('medium')


# --- Set Configurations ---
dataset_stats = load_dataset_stats(
    stats_path="data_stats.json", labels=["Mitochondria"]
)
data_config = DataConfig(
    data_dir="/group/jug/federico/data/CellAtlas",
    labels=["Mitochondria"],
    img_size=768,
    crop_size=512,
    random_crop=True,
    transform=train_augmentation,
    bit_depth=None,
    normalize="std",
    dataset_stats=(dataset_stats["mean"], dataset_stats["std"]),
)
model_config = DenseNetConfig(
    architecture="densenet121",
    num_classes=4,
    num_init_features=64,
    dropout_block=True,
)
loss_config = LossConfig(loss_type="multiclass_focal_loss")
exp_name = f"DenseNet121_{model_config.num_classes}Cl_{data_config.labels[0]}" # TODO: make it more general
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
    batch_size=32,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=4,
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
if IN_MEMORY:
    train_dataset = InMemoryDataset(
        inputs=train_input_data,
        split="train",
        return_label=True,
        **data_config.model_dump(exclude={"data_dir", "labels"})
    )
    val_dataset = InMemoryDataset(
        inputs=val_input_data,
        split="test",
        return_label=True,
        **data_config.model_dump(exclude={"data_dir", "labels"})
    )
else:
    train_preprocessor = ZarrPreprocessor(
        inputs=train_input_data,
        output_path="./train_preprocessed_data.zarr",
        img_size=data_config.img_size,
        normalize=data_config.normalize,
        dataset_stats=data_config.dataset_stats,
        chunk_size=16,
    )
    val_preprocessor = ZarrPreprocessor(
        inputs=val_input_data,
        output_path="./val_preprocessed_data.zarr",
        img_size=data_config.img_size,
        normalize=data_config.normalize,
        dataset_stats=data_config.dataset_stats,
        chunk_size=16,
    )
    train_zarr_path = train_preprocessor.run()
    val_zarr_path = val_preprocessor.run()
    train_dataset = ZarrDataset(
        path_to_zarr=train_zarr_path,
        split="train",
        crop_size=data_config.crop_size,
        random_crop=data_config.random_crop,
        transform=data_config.transform,
    )
    val_dataset = ZarrDataset(
        path_to_zarr=val_zarr_path,
        split="test",
        crop_size=data_config.crop_size,
        random_crop=False,  # No random cropping for validation
        transform=None,  # No transformation for validation
    )
train_dloader = DataLoader(
    train_dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    num_workers=3,
    pin_memory=True,
    drop_last=True,
)
val_dloader = DataLoader(
    val_dataset,
    batch_size=training_config.batch_size,
    shuffle=False,
    num_workers=3,
    pin_memory=True,
    drop_last=False,
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

# Clean up Zarr file after training
if not IN_MEMORY:
    shutil.rmtree(train_zarr_path, ignore_errors=True)
    shutil.rmtree(val_zarr_path, ignore_errors=True)