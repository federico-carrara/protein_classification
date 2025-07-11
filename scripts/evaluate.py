import argparse
import json

import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from protein_classification.config import AlgorithmConfig, DataConfig
from protein_classification.data import InMemoryDataset, ZarrDataset
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.data.preprocessing import ZarrPreprocessor
from protein_classification.data.utils import train_test_split
from protein_classification.model import BioStructClassifier
from protein_classification.utils.evaluation import compute_classification_metrics
from protein_classification.utils.io import load_config, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--in_memory", action="store_true", help="Load the dataset in memory, else use Zarr preprocessing.")
args = parser.parse_args()

torch.set_float32_matmul_precision('medium')

# --- Load configurations ---
algo_config = AlgorithmConfig(
    **load_config(
        config_fpath=args.ckpt_dir, config_type="algorithm",
    )
)
algo_config.training_config.batch_size = 8 # Set batch size to 1 for evaluation
data_config = DataConfig(
    **load_config(
        config_fpath=args.ckpt_dir, config_type="data",
    )
)
data_config.random_crop = False  # No random cropping for evaluation
data_config.transform = None  # No transformation for evaluation
# TODO: do multi-crop + majority voting as TTA

# --- Data Setup ---
input_data, curr_labels = get_cellatlas_filepaths_and_labels(
    data_dir=data_config.data_dir, protein_labels=data_config.labels,
)
_, test_input_data = train_test_split(
    input_data, train_ratio=0.9, deterministic=True
)
print("--------------Dataset Info--------------")
print(f"Number test samples: {len(test_input_data)}")
print(f"Labels: {curr_labels}")
print("----------------------------------------\n")
if args.in_memory:
    test_dataset = InMemoryDataset(
        inputs=test_input_data,
        split="test",
        return_label=True,
        **data_config.model_dump(exclude={"data_dir", "labels"})
    )
else:
    test_preprocessor = ZarrPreprocessor(
        inputs=input_data,
        output_path="./test_preprocessed_data.zarr",
        img_size=data_config.img_size,
        normalize=data_config.normalize,
        dataset_stats=data_config.dataset_stats,
        chunk_size=16,
    )
    test_zarr_path = test_preprocessor.run()
    test_dataset = ZarrDataset(
        path_to_zarr=test_zarr_path,
        split="test",
        crop_size=data_config.crop_size,
        random_crop=data_config.random_crop,
        transform=data_config.transform,
    )
test_dloader = DataLoader(
    test_dataset,
    batch_size=algo_config.training_config.batch_size,
    shuffle=False,
    num_workers=3,
    pin_memory=True,
    drop_last=False,
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

# --- Compute metrics ---
metrics = compute_classification_metrics(
    preds=torch.cat(preds),
    gts=torch.cat(labels),
    probs=torch.cat(probs),
    num_classes=len(curr_labels),
    average="macro",
)
with open(f"{args.ckpt_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Accuracy:", metrics["accuracy"])
print("F1 (macro):", metrics["f1"])
print("Precision:", metrics["precision"])
print("Recall:", metrics["recall"])
print("Confusion Matrix:\n", metrics["confusion_matrix"])
report_df = pd.DataFrame(metrics["report"]).transpose()
print(report_df)