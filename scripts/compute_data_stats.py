import json
from datetime import datetime

import tifffile as tiff
from careamics.dataset.dataset_utils.running_stats import RunningMinMaxStatistics, WelfordStatistics

from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels
from protein_classification.utils.running_stats import calculate_dataset_stats


DATA_DIR = "/group/jug/federico/data/CellAtlas"
LABELS = ["Mitochondria"]

# get input file paths
input_data, _ = get_cellatlas_filepaths_and_labels(
    data_dir=DATA_DIR, protein_labels=LABELS,
)
input_fpaths, _ = zip(*input_data)
input_fpaths = input_fpaths

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
with open("data_stats.json", "r") as f:
    existing_stats: dict = json.load(f)
    existing_stats.update(
        {"_".join(LABELS): data_stats_dict}
    )

# write updated JSON
with open("data_stats.json", "w") as f:
    json.dump(existing_stats, f, indent=4)