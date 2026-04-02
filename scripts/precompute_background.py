"""Precompute foreground crop positions and save to JSON.

Run this once per dataset/parameter combination. The output file can then be
passed to ``train_binary.py --bg-cache <path>`` to skip the expensive
background analysis at training time.

Examples
--------
python scripts/precompute_background.py \
    --dataset CellAtlas --crop-size 256 --stride 64 --output bg_cellatlas.json

python scripts/precompute_background.py \
    --dataset BioSR --crop-size 256 --stride 64 --output bg_biosr.json
"""
import argparse

from protein_classification.data.background import BackgroundAnalyzer
from protein_classification.data.biosr import get_biosr_filepaths_and_labels
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels

DATASET_DEFAULTS = {
    "CellAtlas": {
        "data_dir": "/group/jug/federico/data/CellAtlas",
        "img_size": 2048,
        "labels": ["Nucleus", "Mitochondria", "Endoplasmic reticulum", "Microtubules"],
        "quantiles_by_label": {0: 0.25, 1: 0.1, 2: 0.1, 3: 0.1},
    },
    "BioSR": {
        "data_dir": "/group/jug/federico/data/BioSR_v2",
        "img_size": 1004,
        "labels": ["F-actin", "CCPs", "ER", "Microtubules"],
        "quantiles_by_label": None,
    },
}

parser = argparse.ArgumentParser(
    description="Precompute foreground crop positions for a dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset", type=str, required=True, choices=list(DATASET_DEFAULTS),
)
parser.add_argument("--crop-size", type=int, default=256)
parser.add_argument(
    "--stride", type=int, default=None,
    help="Grid stride. Defaults to crop_size // 4.",
)
parser.add_argument(
    "--img-size", type=int, default=None,
    help="Override default image size for the dataset.",
)
parser.add_argument(
    "--metrics", type=str, nargs="+", default=["std"],
    choices=["std", "entropy"],
)
parser.add_argument("--quantile", type=float, default=0.1)
parser.add_argument(
    "--max-images-for-thresholds", type=int, default=50,
    help="Max images used for threshold estimation. 0 = use all.",
)
parser.add_argument("--output", type=str, required=True, help="Output JSON path.")

args = parser.parse_args()

defaults = DATASET_DEFAULTS[args.dataset]
img_size = args.img_size or defaults["img_size"]
stride = args.stride or args.crop_size // 4
max_imgs = args.max_images_for_thresholds or None

# Load ALL images (no train/val split)
if args.dataset == "CellAtlas":
    inputs, labels_dict = get_cellatlas_filepaths_and_labels(
        data_dir=defaults["data_dir"], labels=defaults["labels"],
    )
elif args.dataset == "BioSR":
    inputs, labels_dict = get_biosr_filepaths_and_labels(
        data_dir=defaults["data_dir"], labels=defaults["labels"],
    )

print(f"Dataset: {args.dataset}")
print(f"Images:  {len(inputs)}")
print(f"Labels:  {labels_dict}")
print(f"Params:  crop_size={args.crop_size}, stride={stride}, "
      f"img_size={img_size}, metrics={args.metrics}")

analyzer = BackgroundAnalyzer(
    inputs=inputs,
    crop_size=args.crop_size,
    stride=stride,
    img_size=img_size,
    metrics=args.metrics,
    quantile=args.quantile,
    quantiles_by_label=defaults["quantiles_by_label"],
    max_images_for_thresholds=max_imgs,
)

analyzer.save(args.output)
print(f"\nSaved to: {args.output}")
print(f"Thresholds: {analyzer.thresholds_by_label}")
