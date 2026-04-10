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
from protein_classification.utils.calibration import apply_temperature
from protein_classification.utils.evaluation import compute_classification_metrics
from protein_classification.utils.io import load_calibration, load_config, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--dataset", type=str, default="CellAtlas", choices=["CellAtlas", "BioSR"], help="Dataset to evaluate.")
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation (TTA) with overlapping crops.")
parser.add_argument("--use_calibration", action="store_true",
                    help="Apply temperature scaling from calibration.json (binary only).")
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

num_classes = algo_config.architecture_config.num_classes
is_binary = num_classes == 1

# load calibration metadata if requested
temperature = None
if args.use_calibration:
    if not is_binary:
        raise SystemExit("ERROR: --use_calibration is only supported for binary models.")
    cal_meta = load_calibration(args.ckpt_dir)
    temperature = cal_meta["temperature"]
    print(f"Loaded calibration: T={temperature:.4f}")

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

# predict_step returns (preds, logits, y)
preds, logits, labels = [], [], []
for batch in outputs:
    batch_preds, batch_logits, batch_labels = batch
    preds.append(batch_preds)
    logits.append(batch_logits)
    labels.append(batch_labels)

# convert logits to probabilities
all_logits = torch.cat(logits)
if is_binary:
    T = temperature if temperature is not None else 1.0
    all_probs = apply_temperature(all_logits, T)
else:
    all_probs = torch.softmax(all_logits, dim=1)

# aggregate results in case of test time cropping
if data_config.test_augmentation_config.strategy == "overlap":
    probs_tta = [
        apply_temperature(lg, T) if is_binary else torch.softmax(lg, dim=1).mean(dim=0)
        for lg in logits
    ]
    if is_binary:
        probs_tta = [torch.sigmoid(lg / T).mean(dim=0) for lg in logits]
    else:
        probs_tta = [torch.softmax(lg, dim=1).mean(dim=0) for lg in logits]
    labels_tta = [l[0].unsqueeze(0) for l in labels]
    preds_majority = [torch.mode(p, dim=0).values.unsqueeze(0) for p in preds]
    if is_binary:
        preds_meanprobs = [(p > 0.5).long().unsqueeze(0) for p in probs_tta]
    else:
        preds_meanprobs = [torch.argmax(p, dim=0).unsqueeze(0) for p in probs_tta]

# --- Compute metrics ---
use_cal_metrics = is_binary
metrics = compute_classification_metrics(
    preds=torch.cat(preds),
    gts=torch.cat(labels),
    probs=all_probs,
    logits=all_logits if is_binary else None,
    num_classes=num_classes,
    average="macro",
    calibration_metrics=use_cal_metrics,
)
if data_config.test_augmentation_config.strategy == "overlap":
    metrics_meanprobs = compute_classification_metrics(
        preds=torch.cat(preds_meanprobs),
        gts=torch.cat(labels_tta),
        probs=torch.cat(probs_tta) if not is_binary else torch.stack(probs_tta),
        num_classes=num_classes,
        average="macro",
    )
    metrics_majority = compute_classification_metrics(
        preds=torch.cat(preds_majority),
        gts=torch.cat(labels_tta),
        probs=torch.cat(probs_tta) if not is_binary else torch.stack(probs_tta),
        num_classes=num_classes,
        average="macro",
    )
else:
    metrics_meanprobs = metrics_majority = None

all_metrics = {
    "standard": metrics,
    "meanprobs": metrics_meanprobs,
    "majority": metrics_majority,
}
if temperature is not None:
    all_metrics["calibration"] = {"temperature": temperature}

with open(f"{args.ckpt_dir}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)

# --- Print metrics ---
print("\n------------------------------------------")
print("Accuracy:", metrics["accuracy"])
print("F1 (macro):", metrics["f1"])
print("Precision:", metrics["precision"])
print("Recall:", metrics["recall"])
print("Confusion Matrix:\n", metrics["confusion_matrix"])
if "nll" in metrics:
    print(f"NLL: {metrics['nll']:.4f}")
    print(f"Brier: {metrics['brier']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
if temperature is not None:
    print(f"Temperature: {temperature:.4f}")

if metrics_meanprobs is not None:
    print("\n------------------------------------------")
    print("Test Time Augmentation (TTA) mean-probs:")
    print("Accuracy:", metrics_meanprobs["accuracy"])
    print("F1 (macro):", metrics_meanprobs["f1"])
    print("Precision:", metrics_meanprobs["precision"])
    print("Recall:", metrics_meanprobs["recall"])
    print("Confusion Matrix:\n", metrics_meanprobs["confusion_matrix"])

if metrics_majority is not None:
    print("\n------------------------------------------")
    print("Test Time Augmentation (TTA) majority voting:")
    print("Accuracy:", metrics_majority["accuracy"])
    print("F1 (macro):", metrics_majority["f1"])
    print("Precision:", metrics_majority["precision"])
    print("Recall:", metrics_majority["recall"])
    print("Confusion Matrix:\n", metrics_majority["confusion_matrix"])
