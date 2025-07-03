import math

import torch
import torch.nn.functional as F
from torch import nn

from protein_classification.losses.loss_utils import get_hard_samples, lovasz_hinge


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logit: torch.Tensor, target: torch.Tensor, epoch: int = 0) -> float:
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = (
            logit -
            logit * target +
            max_val +
            ((-max_val).exp() + (-logit - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


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
        self.focal_loss = FocalLoss()
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