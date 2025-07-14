import json
import os
from typing import Any


exp_ids = tuple(range(1, 15)) + tuple(range(17, 41))
ckpt_dir = "/group/jug/federico/classification_training/2507/DenseNet121_4Cl_Mitochondria/"
params = ["img_size", "crop_size", "transform", "batch_size", "accumulate_grad_batches"]
configs = ["data_config", "data_config", "data_config", "algorithm_config", "algorithm_config"]

def check_param_in_experiment(exp_id: int, param: str, ckpt_dir: str, config: str) -> Any:
    exp_path = os.path.join(ckpt_dir, str(exp_id))
    config_path = os.path.join(exp_path, f"{config}.json")
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} does not exist.")
        return False
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    if config == "algorithm_config":
        config_data = config_data.get("training_config", {})
    
    return config_data.get(param)


if __name__ == "__main__":
    out_dict = {}
    for exp_id in exp_ids:
        out_dict[exp_id] = []
        for param, config in zip(params, configs):
            param_value = check_param_in_experiment(exp_id, param, ckpt_dir, config)
            if param_value is not None:
                out_dict[exp_id].append(param_value)
            else:
                out_dict[exp_id].append("NA")
    
    for exp_id, param_values in out_dict.items():            
        print(f"Experiment {exp_id}: {param_values}")