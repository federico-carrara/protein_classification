from typing import Optional, Union

import torch
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

from protein_classification.utils.calibration import binary_nll, binary_brier, binary_ece


def compute_classification_metrics(
    preds: Union[Tensor, NDArray],
    gts: Union[Tensor, NDArray],
    probs: Optional[Union[Tensor, NDArray]] = None,
    logits: Optional[Tensor] = None,
    num_classes: Optional[int] = None,
    average: str = "macro",
    calibration_metrics: bool = False,
) -> dict:
    """Compute common classification metrics.

    Parameters
    ----------
    preds : Tensor | NDArray
        Predicted class labels.
    gts : Tensor | NDArray
        Ground-truth class labels.
    probs : Tensor | NDArray, optional
        Predicted probabilities (for ROC AUC).
    logits : Tensor, optional
        Raw model logits (for binary calibration metrics).
    num_classes : int, optional
        Number of classes (used for ROC AUC computation).
    average : str
        Averaging strategy for multi-class metrics.
    calibration_metrics : bool
        If True and binary (num_classes == 1), compute NLL, Brier, ECE.
    """
    gts_np = gts.cpu().numpy() if isinstance(gts, Tensor) else gts
    preds_np = preds.cpu().numpy() if isinstance(preds, Tensor) else preds

    metrics = {
        "accuracy": accuracy_score(gts_np, preds_np),
        "f1": f1_score(gts_np, preds_np, average=average, zero_division=0),
        "precision": precision_score(gts_np, preds_np, average=average, zero_division=0),
        "recall": recall_score(gts_np, preds_np, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(gts_np, preds_np).tolist(),
    }

    # Optional: AUC if probability scores are provided
    if probs is not None and num_classes is not None:
        probs_np = probs.cpu().numpy() if isinstance(probs, Tensor) else probs
        try:
            if num_classes == 1:
                # binary with single logit -> probs is 1-D
                metrics["roc_auc"] = roc_auc_score(gts_np, probs_np)
            elif num_classes == 2:
                metrics["roc_auc"] = roc_auc_score(gts_np, probs_np[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    gts_np, probs_np, multi_class="ovr", average=average
                )
        except ValueError:
            metrics["roc_auc"] = None  # e.g., single-class prediction

    # Optional: binary calibration metrics
    if calibration_metrics and num_classes == 1 and logits is not None and probs is not None:
        logits_t = logits if isinstance(logits, Tensor) else torch.tensor(logits)
        probs_t = probs if isinstance(probs, Tensor) else torch.tensor(probs)
        gts_t = gts if isinstance(gts, Tensor) else torch.tensor(gts_np)
        metrics["nll"] = binary_nll(logits_t, gts_t)
        metrics["brier"] = binary_brier(probs_t, gts_t)
        metrics["ece"] = binary_ece(probs_t, gts_t)

    return metrics
