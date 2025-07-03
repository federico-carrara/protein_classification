import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from protein_classification.losses.loss_utils import get_hard_samples, lovasz_hinge


class BinaryFocalLoss(nn.Module):
    """Focal Loss for binary classification.
    
    This loss can also be used for multi-label classification by treating each class
    independently.
    """
    def __init__(self, gamma: float = 2) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logit: torch.Tensor, target: torch.Tensor, epoch: int = 0) -> float:
        target = target.float()
        
        # compute numerically stable BCE loss
        max_val = (-logit).clamp(min=0)
        loss = (
            logit -
            logit * target +
            max_val +
            ((-max_val).exp() + (-logit - max_val).exp()).log()
        )
        
        # compute the focal scaling factor
        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        
        # compute the focal loss
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()
    
    
class MulticlassFocalLoss(nn.Module):
    """Focal Loss for multi-class classification.
    
    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter. Default is 2.0.
    class_weights : torch.Tensor, optional
        Weights for each class. If provided, the loss will be weighted by these values.
    reduction : Literal["mean", "sum"], default="mean"
        Specifies the reduction to apply to the output: 'mean' for averaging the loss,
        'sum' for summing the loss. If 'none', no reduction will be applied
    """
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: Literal["mean", "sum"] = "mean"
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        """Forward pass for the Focal Loss.
        
        Parameters
        ----------
        logits : torch.Tensor
            The raw output logits from the model, shape [B, N], where N is the number
            of classes.
        target : torch.Tensor
            The ground truth labels, shape [B], with sach value in the range [0, N-1].
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Get the log-probability of the true class for each sample
        log_pt = log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)

        # Focal loss
        loss = -((1 - pt) ** self.gamma) * log_pt

        # Apply class weights if provided
        if self.class_weights is not None:
            class_weight = self.class_weights[target] # [B]
            loss = loss * class_weight

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'


class HardLogLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, epoch: int = 0) -> float:
        labels = labels.float()
        loss = 0
        for i in range(self.num_classes):
            logit_ac=logits[:, i]
            label_ac=labels[:, i]
            logit_ac, label_ac = get_hard_samples(logit_ac, label_ac)
            loss += self.bce_loss(logit_ac,label_ac)
        return loss / self.num_classes


# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053
class SymmetricLovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, epoch: int = 0
    ) -> float:
        labels = labels.float()
        loss=((lovasz_hinge(logits, labels)) + (lovasz_hinge(-logits, 1 - labels))) / 2
        return loss


class FocalSymmetricLovaszHardLogLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = MulticlassFocalLoss()
        self.slov_loss = SymmetricLovaszLoss()
        self.log_loss = HardLogLoss()
    
    def forward(
        self, logit: torch.Tensor, labels: torch.Tensor, epoch: int = 0
    ) -> float:
        labels = labels.float()
        focal_loss = self.focal_loss.forward(logit, labels, epoch)
        slov_loss = self.slov_loss.forward(logit, labels, epoch)
        log_loss = self.log_loss.forward(logit, labels, epoch)
        loss = focal_loss * 0.5 + slov_loss * 0.5 + log_loss * 0.5
        return loss


# https://github.com/ronghuaiyang/arcface-pytorch
class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1 + gamma * loss2) / (1 + gamma)
        return loss