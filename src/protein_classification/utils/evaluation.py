from typing import Optional, Union

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from numpy.typing import NDArray
from torch import Tensor


def compute_classification_metrics(
    preds: Union[Tensor, NDArray],
    gts: Union[Tensor, NDArray],
    probs: Optional[Union[Tensor, NDArray]] = None,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> dict:
    """Compute common classification metrics."""
    gts = gts.cpu().numpy() if isinstance(gts, Tensor) else gts
    preds = preds.cpu().numpy() if isinstance(preds, Tensor) else preds

    metrics = {
        "accuracy": accuracy_score(gts, preds),
        "f1": f1_score(gts, preds, average=average, zero_division=0),
        "precision": precision_score(gts, preds, average=average, zero_division=0),
        "recall": recall_score(gts, preds, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(gts, preds),
        "report": classification_report(
            gts, preds, digits=3, zero_division=0, output_dict=True
        ),
    }

    # Optional: AUC if probability scores are provided
    if probs is not None and num_classes is not None:
        try:
            if num_classes == 2:
                metrics["roc_auc"] = roc_auc_score(gts, probs[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    gts, probs, multi_class="ovr", average=average
                )
        except ValueError:
            metrics["roc_auc"] = None  # e.g., single-class prediction

    return metrics