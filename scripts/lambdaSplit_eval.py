import argparse
import json

import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from protein_classification.config import AlgorithmConfig, DataConfig
from protein_classification.data import LambdaSplitPredsDataset
from protein_classification.data.utils import train_test_split, collate_test_time_crops
from protein_classification.model import BioStructClassifier
from protein_classification.utils.evaluation import compute_classification_metrics
from protein_classification.utils.io import load_config, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--exp_id", type=int, default="λSplit experiment ID")
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation (TTA) with overlapping crops.")
args = parser.parse_args()

torch.set_float32_matmul_precision('medium')

# --- Load configurations ---
algo_config = AlgorithmConfig(
    **load_config(
        config_fpath=args.ckpt_dir, config_type="algorithm",
    )
)
algo_config.training_config.batch_size = 1 # Evaluate one sample at a time

# --- Data Setup ---
ch_to_labels_dict = {0: 1, 1: 0, 2: 3, 3: 2}
test_dataset = LambdaSplitPredsDataset(
    data_path=f"/group/jug/federico/lambdasplit_training/2507/lambdasplit_CellAtlas_4FP_2D/{args.exp_id}/predictions_MMSE_50/pred_imgs.npz",
    ch_to_labels=ch_to_labels_dict,
    split="test",
    img_size=768,
    crop_size=512,
    random_crop=False,
    test_time_crop=args.tta,
    transform=None,
    bit_depth=None,
    normalize="std",
    return_label=True,
)
test_dloader = DataLoader(
    test_dataset,
    batch_size=algo_config.training_config.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_test_time_crops if args.tta else None,
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
if args.tta:
    probs_tta = [torch.mean(p, dim=0) for p in probs]
    labels_tta = [l[0].unsqueeze(0) for l in labels]
    preds_majority = [torch.mode(p, dim=0).values.unsqueeze(0) for p in preds]
    preds_meanprobs = [torch.argmax(p, dim=0).unsqueeze(0) for p in probs_tta]

# --- Compute metrics ---
metrics = compute_classification_metrics(
    preds=torch.cat(preds),
    gts=torch.cat(labels),
    probs=torch.cat(probs),
    num_classes=len(ch_to_labels_dict),
    average="macro",
)
metrics_meanprobs = compute_classification_metrics(
    preds=torch.cat(preds_meanprobs),
    gts=torch.cat(labels_tta),
    probs=torch.cat(probs_tta),
    num_classes=len(ch_to_labels_dict),
    average="macro",
)
metrics_majority = compute_classification_metrics(
    preds=torch.cat(preds_majority),
    gts=torch.cat(labels_tta),
    probs=torch.cat(probs_tta),
    num_classes=len(ch_to_labels_dict),
    average="macro",
)
metrics = {
    "standard": metrics,
    "meanprobs": metrics_meanprobs,
    "majority": metrics_majority,
}

# --- Print metrics ---
print("\n------------------------------------------")
print("Accuracy:", metrics["standard"]["accuracy"])
print("F1 (macro):", metrics["standard"]["f1"])
print("Precision:", metrics["standard"]["precision"])
print("Recall:", metrics["standard"]["recall"])
print("Confusion Matrix:\n", metrics["standard"]["confusion_matrix"])

if args.tta:
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