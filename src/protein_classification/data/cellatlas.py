"""Functions to get file paths and labels for the Cell Atlas dataset."""
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union

import pandas as pd

PathLike = Union[str, Path]


def _load_labels_dict(
    data_dir: PathLike, rel_path: PathLike = "./labels_list.json"
) -> dict[str, int]:
    """Get the list of labels from the labels.json file and invert it, such that
    keys are names of bio structures and values are integer ids."""
    assert str(rel_path).split('.')[-1] == 'json', (
        "The labels file must be a JSON file."
    )
    labels_path = Path(data_dir) / rel_path
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    return {str(v): int(k) for k, v in labels.items()}


def _load_fname_label_pairs(
    data_dir: PathLike, rel_path: PathLike = "./train_labels.csv"
) -> pd.DataFrame:
    """Get the list of image filenames and labels from the CSV file."""
    assert str(rel_path).split('.')[-1] == 'csv', (
        "The input file must be a CSV file."
    )
    input_fnames_labels_path = Path(data_dir) / rel_path
    if not input_fnames_labels_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_fnames_labels_path}")
    
    pairs_df = pd.read_csv(
        input_fnames_labels_path, names=['filename', 'label'], header=0
    )
    pairs_df['label'] = pairs_df['label'].apply(
        lambda x: [int(val) for val in x.split(" ")] if pd.notna(x) else []
    )
    return pairs_df


def _get_filepaths_with_labels(
    pairs_df: pd.DataFrame, labels: Sequence[str], labels_dict: dict[str, int]
) -> dict[str, list[PathLike]]:
    """Get the list of image file paths with a given label."""
    labels_dict_inv: dict[int, str] = {v: k for k, v in labels_dict.items()}
    labels_num = [labels_dict[label] for label in labels if label in labels_dict]
    
    filepaths_by_label: dict[int, list[PathLike]] = defaultdict(list)
    for _, row in pairs_df.iterrows():
        fname = row['filename']
        label_ids = row['label']
        if len(label_ids) > 1:
            continue  # Skip samples with multiple labels for now
        elif len(label_ids) == 0:
            continue  # Skip samples with no labels
        elif len(label_ids) == 1:
            label_id = label_ids[0]
            if label_id in labels_num:
                filepaths_by_label[labels_dict_inv[label_id]].append(Path(fname))
    
    return filepaths_by_label


def get_cellatlas_filepaths_and_labels(
    data_dir: PathLike,
    protein_labels: Sequence[str],
    rel_data_path: PathLike = "./train_data_raw/",
    rel_labels_path: PathLike = "./labels_list.json",
    rel_fnames_labels_pairs_path: PathLike = "./train_labels.csv"
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """Get the file paths and labels for the Cell Atlas dataset."""
    labels_dict = _load_labels_dict(data_dir, rel_labels_path)
    pairs_df = _load_fname_label_pairs(data_dir, rel_fnames_labels_pairs_path)
    fpaths_by_label = _get_filepaths_with_labels(pairs_df, protein_labels, labels_dict)
    
    curr_labels_dict = {
        "Nucleus" : 0,
        "Endoplasmic reticulum": 1,
        "Microtubules": 2,
    }
    curr_labels_dict.update({label: (i + 3) for i, label in enumerate(protein_labels)})
    
    out_fpaths: list[str] = []
    out_labels: list[int] = []
    for label, fpaths in fpaths_by_label.items():
        for fpath in fpaths:
            fpath = Path(data_dir) / rel_data_path / fpath
            # append file paths
            out_fpaths.append(f"{str(fpath)}_green.tif")
            out_fpaths.append(f"{str(fpath)}_blue.tif")
            out_fpaths.append(f"{str(fpath)}_yellow.tif")
            out_fpaths.append(f"{str(fpath)}_red.tif")
            # append labels
            out_labels.append(curr_labels_dict[label])
            out_labels.append(curr_labels_dict["Nucleus"])
            out_labels.append(curr_labels_dict["Endoplasmic reticulum"])
            out_labels.append(curr_labels_dict["Microtubules"])
    
    outputs = list(zip(out_fpaths, out_labels))
    return outputs, curr_labels_dict