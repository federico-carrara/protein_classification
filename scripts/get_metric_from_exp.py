import json
import os
from typing import Any


exp_ids = tuple(range(1, 15)) + tuple(range(17, 41))
ckpt_dir = "/group/jug/federico/classification_training/2507/DenseNet121_4Cl_Mitochondria/"
metric = "accuracy"
metric_type = "standard"

def get_metric_from_experiment(
    exp_id: int, ckpt_dir: str, metric: str = "accuracy", metric_type: str = "standard"
) -> Any:
    exp_path = os.path.join(ckpt_dir, str(exp_id))
    metrics_path = os.path.join(exp_path, f"metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} does not exist.")
        return False
    
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
        
    if metric_type in metrics_data:
        return metrics_data[metric_type].get(metric, None)
    else:
        return metrics_data.get(metric, None)


if __name__ == "__main__":
    out_dict = {}
    for exp_id in exp_ids:
        metric_value = get_metric_from_experiment(exp_id, ckpt_dir, metric, metric_type)
        if metric_value is not None:
            out_dict[exp_id] = metric_value
        else:
            out_dict[exp_id] = -1
    
    for exp_id, metric_values in out_dict.items():            
        print(f"Experiment {exp_id}: {metric_values}")