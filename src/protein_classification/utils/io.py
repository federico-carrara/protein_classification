import glob
import json
import os
from datetime import datetime
from filelock import FileLock
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import torch
from pydantic import BaseModel
from pytorch_lightning.loggers import WandbLogger

from protein_classification.config import (
    AlgorithmConfig, DataConfig, DenseNetConfig, LossConfig, TrainingConfig
)

ConfigLike = Union[BaseModel, AlgorithmConfig, DataConfig, DenseNetConfig, LossConfig, TrainingConfig]
PathLike = Union[Path, str]


def _log_config(
    config: ConfigLike, 
    name: str,
    log_dir: Union[Path, str],
    logger: Optional[WandbLogger] = None
) -> None:
    """Save the `pydantic` configuration in a JSON file.
    
    Parameters
    ----------
    config : Config
        The configuration to save.
    name : str
        The name of the configuration. File name will be "{name}_config.json"
        and as `{name}` subdictionary on Wandb config.yml file.
    log_dir : Union[Path, str]
        The directory where the configuration file is logged locally.
    logger : Optional[WandbLogger], optional
        The logger to save the configuration in WANDB, by default None.
    """
    with open(os.path.join(log_dir, f"{name}_config.json"), "w") as f:
        f.write(config.model_dump_json(indent=4))

    if logger:
        logger.experiment.config.update({f"{name}": config.model_dump()})


def log_configs(
    configs: Sequence[ConfigLike],
    names: Sequence[str],
    log_dir: Union[Path, str],
    logger: Optional[WandbLogger] = None
) -> None:
    """Save the `pydantic` configurations in JSON files.
    
    Parameters
    ----------
    configs : Sequence[BaseModel]
        The configurations to save as `pydantic.BaseModel` objects.
    names : Sequence[str]
        The names of the configurations. File names will be "{name}_config.json"
        and as `{name}` subdictionaries on Wandb config.yml file.
    log_dir : Union[Path, str]
        The directory where the configuration files are logged locally.
    logger : Optional[WandbLogger], optional
        The logger to save the configurations in Wandb, by default None.
    """   
    for config, name in zip(configs, names):
        _log_config(config, name, log_dir, logger)
        

def load_config(
    config_fpath: str, 
    config_type: Literal['algorithm', 'data']
) -> dict:
    """Load a configuration file.
    
    Parameters
    ----------
    config_fpath : str
        Configuration file path.
    config_type : Literal['algorithm', 'data']
        Configuration type.
        
    Returns
    -------
    dict
        Configuration as a dictionary.
    """
    fpath = os.path.join(config_fpath, f'{config_type}_config.json')
    if os.path.isfile(fpath):
        with open(fpath) as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found in {config_fpath}.")


def _get_new_version_dir(model_dir: Union[str, Path]) -> Path:
    """Create a unique version ID for a new model run.
    
    Parameters
    ----------
    model_dir : Union[str, Path]
        The directory where the model logs are stored.
        
    Returns
    -------
    int
        The new version ID.
    """
    lock_path = Path(model_dir) / ".lock"
    with FileLock(str(lock_path)):
        versions = [int(d.name) for d in Path(model_dir).iterdir() if d.is_dir()]
        next_version = max(versions) + 1 if versions else 1
        next_version_dir = Path(model_dir) / str(next_version)
    
    return next_version_dir


def get_workdir(root_dir: str, name: str) -> Path:
    """Get the workdir for the current model.

    Workdir path has the following structure: "root_dir/YYMM/model_name/version".
    
    Parameters
    ----------
    root_dir : str
        The root directory where all model logs are stored.
    model_name : str
        The name of the model.
            
    Returns
    -------
    cur_workdir : Path
        The current work directory.
    """
    rel_path = datetime.now().strftime("%y%m")
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True, parents=True)

    rel_path = os.path.join(rel_path, name)
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    cur_workdir = _get_new_version_dir(cur_workdir)

    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(f"Workdir {cur_workdir} already exists.")
    
    return cur_workdir


def get_log_dir(base_dir: PathLike, exp_name: str) -> str:
    """Get the log directory for the current model, used at training time."""
    ldir = get_workdir(base_dir, exp_name)
    print(f"\nLogging to directory: {ldir}\n")
    return ldir


def load_dataset_stats(
    stats_path: PathLike, labels: Sequence[str],
) -> dict[str, int]:
    """Load the dataset statistics from a JSON file."""
    assert isinstance(stats_path, (Path, str)), (
        "stats_dir must be a Path or a string."
    )
    
    with open(stats_path, "r") as f:
        data_stats: dict = json.load(f)

    # extract statistics for the specified labels
    key = "+".join(labels)
    stats: dict = data_stats.get(key, None)
    
    return {
        "mean": stats.get("mean", None),
        "std": stats.get("std", None),
        "min": stats.get("min", None),
        "max": stats.get("max", None),
    }



def get_checkpoint_path(
    ckpt_dir: str, mode: Literal['best', 'last'] = 'best'
) -> str:
    """Get the model checkpoint path.
    
    Parameters
    ----------
    ckpt_dir : str
        Checkpoint directory.
    mode : Literal['best', 'last'], optional
        Mode to get the checkpoint, by default 'best'.
    
    Returns
    -------
    str
        Checkpoint path.
    """
    output = []
    for fpath in glob.glob(ckpt_dir + "/*.ckpt"):
        fname = os.path.basename(fpath)
        if mode == 'best':
            if fname.startswith('best'):
                output.append(fpath)
        elif mode == 'last':
            if fname.startswith('last'):
                output.append(fpath)
    assert len(output) == 1, '\n'.join(output)
    return output[0]


def load_checkpoint(
    ckpt_dir: Union[str, Path], best: bool = True, verbose: bool = True
) -> dict:
    """Load the checkpoint from the given directory.
    
    Parameters
    ----------
    ckpt_dir : Union[str, Path]
        The path to the checkpoint directory.
    best : bool, optional
        Whether to load the best checkpoint, by default True.
        If False, the last checkpoint will be loaded.
    verbose : bool, optional
        Whether to print the checkpoint loading information, by default True.
    
    Returns
    -------
    dict
        The loaded checkpoint.
    """
    if os.path.isdir(ckpt_dir):
        ckpt_fpath = get_checkpoint_path(ckpt_dir, mode="best" if best else "last")
    else:
        assert os.path.isfile(ckpt_dir)
        ckpt_fpath = ckpt_dir

    ckpt = torch.load(ckpt_fpath)
    if verbose:
        print(f"\nLoading checkpoint from: '{ckpt_fpath}' - Epoch: {ckpt['epoch']}\n")
    return ckpt