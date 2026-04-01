"""Post-training temperature scaling for binary critic calibration.

Temperature scaling learns a single scalar T > 0 such that
``sigmoid(logits / T)`` produces well-calibrated probabilities.
"""

import torch
import torch.nn as nn
from torch import Tensor


def fit_temperature(
    logits: Tensor,
    labels: Tensor,
    lr: float = 0.01,
    max_iter: int = 50,
) -> float:
    """Fit a scalar temperature on binary logits using LBFGS (`max_iter` steps).

    Parameters
    ----------
    logits : Tensor
        Raw model logits, shape ``[N]``.
    labels : Tensor
        Binary ground-truth labels (0/1), shape ``[N]``.
    lr : float
        LBFGS learning rate.
    max_iter : int
        Maximum LBFGS iterations.

    Returns
    -------
    float
        Optimised temperature ``T = exp(log_T)``.
    """
    logits = logits.detach().float()
    labels = labels.detach().float()

    log_T = nn.Parameter(torch.zeros(1, device=logits.device))  # T=1 init
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_T.exp().item()


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Apply temperature scaling and return calibrated probabilities.

    Parameters
    ----------
    logits : Tensor
        Raw model logits, shape ``[N]``.
    temperature : float
        Temperature scalar (must be > 0).

    Returns
    -------
    Tensor
        Calibrated probabilities, shape ``[N]``.
    """
    return torch.sigmoid(logits / temperature)


def binary_nll(logits: Tensor, labels: Tensor) -> float:
    """Binary negative log-likelihood (BCE with logits).

    Parameters
    ----------
    logits : Tensor
        Raw logits, shape ``[N]``.
    labels : Tensor
        Binary labels (0/1), shape ``[N]``.

    Returns
    -------
    float
        Mean NLL.
    """
    return nn.functional.binary_cross_entropy_with_logits(
        logits.float(), labels.float()
    ).item()


def binary_brier(probs: Tensor, labels: Tensor) -> float:
    """Brier score for binary predictions.

    Parameters
    ----------
    probs : Tensor
        Predicted probabilities for the positive class, shape ``[N]``.
    labels : Tensor
        Binary labels (0/1), shape ``[N]``.

    Returns
    -------
    float
        Mean Brier score (lower is better).
    """
    return ((probs.float() - labels.float()) ** 2).mean().item()


def binary_ece(probs: Tensor, labels: Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error for binary predictions (fixed-width bins).

    Parameters
    ----------
    probs : Tensor
        Predicted probabilities for the positive class, shape ``[N]``.
    labels : Tensor
        Binary labels (0/1), shape ``[N]``.
    n_bins : int
        Number of equally-spaced bins.

    Returns
    -------
    float
        ECE (lower is better, in [0, 1]).
    """
    probs = probs.float()
    labels = labels.float()
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)

    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n_in_bin = mask.sum().item()
        if n_in_bin == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += (n_in_bin / len(probs)) * (avg_confidence - avg_accuracy).abs()

    return ece.item()
