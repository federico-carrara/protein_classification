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
    
    pairs_df = pd.read_csv(input_fnames_labels_path, names=['filename', 'label'])
    pairs_df['label'] = pairs_df['label'].apply(
        lambda x: [int(val) for val in x.split(" ")] if pd.notna(x) else []
    )
    return pairs_df


def _get_filepaths_with_labels(
    pairs_df: pd.DataFrame, labels: Sequence[int]
) -> dict[int, list[PathLike]]:
    """Get the list of image file paths with a given label."""
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
            if label_id in labels:
                filepaths_by_label[label_ids].append(Path(fname))
    
    return filepaths_by_label


def get_cellatlas_filepaths_and_labels(
    data_dir: PathLike,
    protein_labels: Sequence[str],
    rel_labels_path: PathLike = "./labels_list.json",
    rel_input_fnames_labels_path: PathLike = "./train_labels.csv"
) -> tuple[list[tuple[Path, int]], dict[int, str]]:
    """Get the file paths and labels for the Cell Atlas dataset."""
    labels_dict = _load_labels_dict(data_dir, rel_labels_path)
    pairs_df = _load_fname_label_pairs(data_dir, rel_input_fnames_labels_path)
    protein_labels_ids = [labels_dict[label] for label in protein_labels if label in labels_dict]
    fpaths_by_label = _get_filepaths_with_labels(pairs_df, protein_labels_ids)
    
    labels = {
        0: "Nucleus",
        1: "Endoplasmic reticulum",
        2: "Microtubules",
    }
    labels.update({labels_dict[label]: label for label in protein_labels})
    outputs = list[tuple[str, int]] = []
    for label, fpaths in fpaths_by_label.items():
        for fpath in fpaths:
            fpath = Path(data_dir) / fpath
            outputs.append((f"{str(fpath)}_green.tif", label))
            outputs.append((f"{str(fpath)}_blue.tif", 0))
            outputs.append((f"{str(fpath)}_yellow.tif", 1))
            outputs.append((f"{str(fpath)}_red.tif", 2))
            
    return outputs, labels