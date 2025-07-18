import argparse
import json
from datetime import datetime

import tifffile as tiff
from careamics.dataset.dataset_utils.running_stats import RunningMinMaxStatistics, WelfordStatistics

from protein_classification.data.biosr import get_biosr_filepaths_and_labels
from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.utils.running_stats import calculate_dataset_stats

parser = argparse.ArgumentParser(description="Compute dataset statistics for CellAtlas dataset.")
parser.add_argument("--data_dir", type=str, help="Directory containing the dataset.")
parser.add_argument("--labels", type=str, nargs="+", help="List of labels to include in the statistics.")
parser.add_argument("--dataset", type=str, default="CellAtlas", choices=["CellAtlas", "BioSR"], help="Dataset to compute statistics for.")
args = parser.parse_args()

# get input file paths
if args.dataset == "CellAtlas":
    input_data, _ = get_cellatlas_filepaths_and_labels(
        data_dir=args.data_dir, protein_labels=args.labels,
    )
elif args.dataset == "BioSR":
    input_data, _ = get_biosr_filepaths_and_labels(
        data_dir=args.data_dir, protein_labels=args.labels,
    )
input_fpaths, _ = zip(*input_data)

# compute running statistics
data_stats = calculate_dataset_stats(filepaths=input_fpaths, imreader=tiff.imread)

# create dict to store data statistics
timestamp = datetime.now().strftime("%d/%m/%y_%H:%M")
data_stats_dict = {
    "timestamp": timestamp,
    "num_samples": len(input_fpaths),
    "num_classes": 4,
}
data_stats_dict.update(data_stats)

# update existing JSON
data_stats_fname = f"data_stats_{args.dataset.lower()}.json"
with open(data_stats_fname, "r") as f:
    existing_stats: dict = json.load(f)
    existing_stats.update(
        {"+".join(args.labels): data_stats_dict}
    )

# write updated JSON
with open(data_stats_fname, "w") as f:
    json.dump(existing_stats, f, indent=4)