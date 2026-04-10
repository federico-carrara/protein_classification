import argparse
import json
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from protein_classification.config import AlgorithmConfig, DataConfig
from protein_classification.data import BinaryEvalDataset
from protein_classification.data.background import BackgroundAnalyzer
from protein_classification.data.biosr import get_biosr_filepaths_and_labels
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.utils import collate_multi_crop_batches, train_test_split
from protein_classification.model import BioStructClassifier
from protein_classification.utils.evaluation import compute_classification_metrics
from protein_classification.utils.io import load_calibration, load_config, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--dataset", type=str, default="CellAtlas", choices=["CellAtlas", "BioSR"])
parser.add_argument(
    "--bg_cache", type=str, default=None,
    help=(
        "Path to precomputed background cache JSON. "
        "If provided: evaluate on foreground crops only. "
        "If omitted: evaluate on a dense grid of all patches (includes background)."
    ),
)
parser.add_argument(
    "--use_calibration", action="store_true",
    help="Apply temperature scaling from calibration.json.",
)
parser.add_argument("--debug", action="store_true", help="Limit to first 50 samples.")
args = parser.parse_args()

torch.set_float32_matmul_precision("medium")

# --- Load configs ---
algo_config = AlgorithmConfig(**load_config(args.ckpt_dir, "algorithm"))
data_config = DataConfig(**load_config(args.ckpt_dir, "data"))

if algo_config.architecture_config.num_classes != 1:
    raise SystemExit(
        f"ERROR: This script is for binary models only (num_classes=1). "
        f"Got num_classes={algo_config.architecture_config.num_classes}."
    )
algo_config.training_config.batch_size = 1

# --- Calibration ---
temperature = None
if args.use_calibration:
    cal_meta = load_calibration(args.ckpt_dir)
    temperature = cal_meta["temperature"]
    print(f"Loaded calibration temperature: T={temperature:.4f}")
T = temperature if temperature is not None else 1.0

# --- Load filepaths + labels ---
if args.dataset == "CellAtlas":
    input_data, curr_labels = get_cellatlas_filepaths_and_labels(
        data_dir=data_config.data_dir, protein_labels=data_config.labels,
    )
elif args.dataset == "BioSR":
    input_data, curr_labels = get_biosr_filepaths_and_labels(
        data_dir=data_config.data_dir, protein_labels=data_config.labels,
    )
if args.debug:
    input_data = input_data[:50]

# --- Reproduce training test split ---
_, test_data = train_test_split(input_data, train_ratio=0.9, deterministic=True)
print("--------------Dataset Info--------------")
print(f"Target label: {data_config.target_label}")
print(f"Number of test samples: {len(test_data)}")
print(f"Labels: {curr_labels}")
print("----------------------------------------\n")

# --- Load foreground positions (optional) ---
test_bg_positions = None
if args.bg_cache:
    bg_data = BackgroundAnalyzer.load(args.bg_cache)
    test_bg_positions = BackgroundAnalyzer.to_index_keyed(
        bg_data["valid_positions_by_filepath"], test_data,
    )
    print("Crop mode: foreground patches only")
else:
    print("Crop mode: dense grid (all patches — includes background)")

# --- Build dataset + dataloader ---
target_label_id = data_config.labels.index(data_config.target_label)
crop_size = data_config.train_augmentation_config.crop_size

test_dataset = BinaryEvalDataset(
    inputs=test_data,
    target_label=target_label_id,
    valid_crop_positions=test_bg_positions,
    img_size=data_config.img_size,
    crop_size=crop_size,
    bit_depth=data_config.bit_depth,
    normalize=data_config.normalize,
    normalization_scope=data_config.normalization_scope,
    dataset_stats=data_config.dataset_stats,
)
test_dloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_multi_crop_batches,
)

# --- Load model + checkpoint ---
model = BioStructClassifier(config=algo_config)
ckpt = load_checkpoint(ckpt_dir=args.ckpt_dir, best=True)
model.load_state_dict(ckpt["state_dict"], strict=True)

# --- Run inference ---
trainer = Trainer(accelerator="gpu", enable_progress_bar=True, precision=32)
outputs = trainer.predict(model=model, dataloaders=test_dloader)

# --- Collect patch-level predictions (no aggregation) ---
all_preds, all_logits, all_labels = [], [], []
for batch_preds, batch_logits, batch_labels in outputs:
    all_preds.append(batch_preds)
    all_logits.append(batch_logits)
    all_labels.append(batch_labels)

all_preds = torch.cat(all_preds)
all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)
all_probs = torch.sigmoid(all_logits / T)

print(f"\nTotal patches evaluated: {len(all_preds)}")
print(f"  Positive (label=1): {all_labels.sum().item()}")
print(f"  Negative (label=0): {(all_labels == 0).sum().item()}")

# --- Compute patch-level metrics ---
metrics = compute_classification_metrics(
    preds=all_preds,
    gts=all_labels,
    probs=all_probs,
    logits=all_logits,
    num_classes=1,
    average="macro",
    calibration_metrics=True,
)

# --- Save metrics ---
output = {"metrics": metrics}
if temperature is not None:
    output["calibration"] = {"temperature": temperature}

metrics_path = Path(args.ckpt_dir) / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(output, f, indent=4)

# --- Print results ---
print("\n------------------------------------------")
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"F1 (macro): {metrics['f1']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
print(f"NLL:       {metrics['nll']:.4f}")
print(f"Brier:     {metrics['brier']:.4f}")
print(f"ECE:       {metrics['ece']:.4f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
if temperature is not None:
    print(f"Temperature: {temperature:.4f}")
print(f"\nMetrics saved to: {metrics_path}")
