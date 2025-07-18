"""Functions to get file paths and labels for the BioSR dataset."""
import os
from pathlib import Path
from typing import Sequence, Union

PathLike = Union[str, Path]


def _check_valid_tiff(fpath: PathLike) -> bool:
    """Check if the file is a valid TIFF file."""
    return (
        str(fpath).lower().endswith('.tiff') or
        str(fpath).lower().endswith('.tif')
    )

def get_biosr_filepaths_and_labels(
    data_dir: PathLike,
    protein_labels: Sequence[str],
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """Get the file paths and labels for the BioSR dataset."""
    curr_labels_dict = {label: i for i, label in enumerate(protein_labels)}
    
    out_fpaths: list[str] = []
    out_labels: list[int] = []
    for label in protein_labels:
        label_dir = Path(data_dir) / label
        if not label_dir.exists():
            continue
        
        for fname in os.listdir(label_dir):
            if not _check_valid_tiff(fname):
                continue
            fpath = label_dir / fname
            out_fpaths.append(fpath)
            out_labels.append(curr_labels_dict[label])
    
    outputs = list(zip(out_fpaths, out_labels))
    return outputs, curr_labels_dict