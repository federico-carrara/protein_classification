import os
import socket

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from protein_classification.config import (
    AlgorithmConfig, DataConfig, DenseNetConfig, LossConfig, TrainingConfig
)
from protein_classification.data import PreTrainingDataset
from protein_classification.data.augmentations import (
    train_augmentation, geometric_augmentation, noise_augmentation
)
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.model import BioStructClassifier
from protein_classification.utils.callbacks import get_callbacks
from protein_classification.utils.io import load_dataset_stats, get_log_dir, log_configs


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
    bit_depth=8,
    normalize="std",
    dataset_stats=(dataset_stats["mean"], dataset_stats["std"]),
)
model_config = DenseNetConfig(
    architecture="densenet121",
    num_classes=4,
    num_init_features=64,
    dropout_block=True,
)
loss_config = LossConfig(name="multiclass_focal_loss")
exp_name = f"DenseNet121_{model_config.num_classes}Cl_{data_config.labels[0]}" # TODO: make it more general
log_dir=get_log_dir(
    base_dir="/group/jug/federico/classification_training",
    exp_name=exp_name,
)
algo_config = AlgorithmConfig(
    mode="train",
    log_dir=log_dir,
    architecture_config=model_config,
    loss_config=loss_config,
)
training_config = TrainingConfig(
    max_epochs=100,
    lr=3e-4,
    batch_size=128,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)

# --- Data Setup ---
input_data = get_cellatlas_filepaths_and_labels(
    data_dir=data_config.data_dir, protein_labels=data_config.labels,
)
train_dataset = PreTrainingDataset(
    inputs=input_data,
    split="train",
    return_label=True,
    **data_config.model_dump(exclude={"data_dir", "labels"})
)
val_dataset = PreTrainingDataset(
    inputs=input_data,
    split="test",
    return_label=True,
    **data_config.model_dump(exclude={"data_dir", "labels"})
)
train_dloader = DataLoader(
    train_dataset,
    batch_size=algo_config.batch_size,
    shuffle=True,
    num_workers=3,
    pin_memory=True,
    drop_last=True,
)
val_dloader = DataLoader(
    val_dataset,
    batch_size=algo_config.batch_size,
    shuffle=False,
    num_workers=3,
    pin_memory=True,
    drop_last=False,
)

# --- Initialize Logger + Log configs ---
logger = WandbLogger(
    name=os.path.join(socket.gethostname(), exp_name),
    save_dir=algo_config.log_dir,
    project=algo_config.wandb_project,
)
log_configs(
    configs=[algo_config, data_config],
    names=["algorithm", "data", "training", "exp_params", "dataset"],
    log_dir=algo_config.log_dir,
    logger=logger,
)

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
)
trainer.fit(model, train_dloader, val_dloader)
wandb.finish()