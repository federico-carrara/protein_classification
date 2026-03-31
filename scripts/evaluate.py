import argparse
import json

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from protein_classification.config import AlgorithmConfig, DataConfig, DataAugmentationConfig
from protein_classification.data import MultiClassDataset
from protein_classification.data.biosr import get_biosr_filepaths_and_labels
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.utils import collate_multi_crop_batches, train_test_split
from protein_classification.model import BioStructClassifier
from protein_classification.utils.evaluation import compute_classification_metrics
from protein_classification.utils.io import load_config, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--dataset", type=str, default="CellAtlas", choices=["CellAtlas", "BioSR"], help="Dataset to evaluate.")
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation (TTA) with overlapping crops.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode for faster evaluation with fewer samples.")
args = parser.parse_args()

torch.set_float32_matmul_precision('medium')


# --- Load configurations ---
algo_config = AlgorithmConfig(
    **load_config(
        config_fpath=args.ckpt_dir, config_type="algorithm",
    )
)
algo_config.training_config.batch_size = 1 # Evaluate one sample at a time
data_config = DataConfig(
    **load_config(
        config_fpath=args.ckpt_dir, config_type="data",
    )
)
data_config.test_augmentation_config = DataAugmentationConfig(
    transform=None,
    crop_size=data_config.train_augmentation_config.crop_size,
    random_crop=True,
    strategy="overlap",
    # metrics=["std"],
    # bg_threshold=3.0, # Default threshold for background crops
)

# --- Data Setup ---
if args.dataset == "BioSR":
    input_data, curr_labels = get_biosr_filepaths_and_labels(
        data_dir=data_config.data_dir, protein_labels=data_config.labels,
    )
elif args.dataset == "CellAtlas":
    input_data, curr_labels = get_cellatlas_filepaths_and_labels(
        data_dir=data_config.data_dir, protein_labels=data_config.labels,
    )
if args.debug:
    input_data = input_data[:50]  # Use only a few samples for debugging
_, test_input_data = train_test_split(
    input_data, train_ratio=0.9, deterministic=True
)
print("--------------Dataset Info--------------")
print(f"Number test samples: {len(test_input_data)}")
print(f"Labels: {curr_labels}")
print("----------------------------------------\n")
test_dataset = MultiClassDataset(
    inputs=test_input_data,
    split="test",
    return_label=True,
    img_size=data_config.img_size,
    augmentation_config=data_config.test_augmentation_config,
    num_crops_per_image=1,
    bit_depth=data_config.bit_depth,
    normalize=data_config.normalize,
    normalization_scope=data_config.normalization_scope,
    dataset_stats=data_config.dataset_stats,
)
test_dloader = DataLoader(
    test_dataset,
    batch_size=algo_config.training_config.batch_size,
    shuffle=False,
    num_workers=3,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_multi_crop_batches,
)

# --- Setup Model & load checkpoint ---
model = BioStructClassifier(config=algo_config)
ckpt = load_checkpoint(ckpt_dir=args.ckpt_dir, best=True)
model.load_state_dict(ckpt["state_dict"], strict=True)

# --- Predict ---
trainer = Trainer(
    accelerator="gpu",
    enable_progress_bar=True,
    precision=32,
)
outputs = trainer.predict(model=model, dataloaders=test_dloader)
preds, probs, labels = [], [], []
for batch in outputs:
    batch_preds, batch_probs, batch_labels = batch
    preds.append(batch_preds)
    probs.append(batch_probs)
    labels.append(batch_labels)

# aggregate results in case of test time cropping
if data_config.test_augmentation_config.strategy == "overlap":
    probs_tta = [torch.mean(p, dim=0) for p in probs]
    labels_tta = [l[0].unsqueeze(0) for l in labels]
    preds_majority = [torch.mode(p, dim=0).values.unsqueeze(0) for p in preds]
    preds_meanprobs = [torch.argmax(p, dim=0).unsqueeze(0) for p in probs_tta]

# --- Compute metrics ---
metrics = compute_classification_metrics(
    preds=torch.cat(preds),
    gts=torch.cat(labels),
    probs=torch.cat(probs),
    num_classes=len(curr_labels),
    average="macro",
)
if data_config.test_augmentation_config.strategy == "overlap":
    metrics_meanprobs = compute_classification_metrics(
        preds=torch.cat(preds_meanprobs),
        gts=torch.cat(labels_tta),
        probs=torch.cat(probs_tta),
        num_classes=len(curr_labels),
        average="macro",
    )
    metrics_majority = compute_classification_metrics(
        preds=torch.cat(preds_majority),
        gts=torch.cat(labels_tta),
        probs=torch.cat(probs_tta),
        num_classes=len(curr_labels),
        average="macro",
    )
else:
    metrics_meanprobs = metrics_majority = None

metrics = {
    "standard": metrics,
    "meanprobs": metrics_meanprobs,
    "majority": metrics_majority,
}
with open(f"{args.ckpt_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# --- Print metrics ---
print("\n------------------------------------------")
print("Accuracy:", metrics["standard"]["accuracy"])
print("F1 (macro):", metrics["standard"]["f1"])
print("Precision:", metrics["standard"]["precision"])
print("Recall:", metrics["standard"]["recall"])
print("Confusion Matrix:\n", metrics["standard"]["confusion_matrix"])

print("\n------------------------------------------")
print("Test Time Augmentation (TTA) mean-probs:")
print("Accuracy:", metrics["meanprobs"]["accuracy"])
print("F1 (macro):", metrics["meanprobs"]["f1"])
print("Precision:", metrics["meanprobs"]["precision"])
print("Recall:", metrics["meanprobs"]["recall"])
print("Confusion Matrix:\n", metrics["meanprobs"]["confusion_matrix"])

print("\n------------------------------------------")
print("Test Time Augmentation (TTA) majority voting:")
print("Accuracy:", metrics["majority"]["accuracy"])
print("F1 (macro):", metrics["majority"]["f1"])
print("Precision:", metrics["majority"]["precision"])
print("Recall:", metrics["majority"]["recall"])
print("Confusion Matrix:\n", metrics["majority"]["confusion_matrix"])
