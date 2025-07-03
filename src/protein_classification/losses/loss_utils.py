import torch
import torch.nn.functional as F
from torch.autograd import Variable


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovasz extension w.r.t sorted errors.
    
    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary Lovasz hinge loss
    
    Parameters
    ----------
    logits : torch.Tensor
        [P] Variable, logits at each prediction (between -\infty and +\infty).
    labels : torch.Tensor
        [P] Tensor, binary ground truth labels (0 or 1).
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


# https://github.com/bermanmaxim/LovaszSoftmax/tree/master/pytorch
def lovasz_hinge(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int, per_class: bool = True
) -> float:
    """Binary Lovasz hinge loss
      
    Parameters
    ----------
    logits : torch.Tensor 
        [B, C] Variable, logits at each pixel (between -\infty and +\infty).
    labels : torch.Tensor
        [B, C] Tensor, binary ground truth masks (0 or 1).
    num_classes : int
        Number of classes.
    per_image : bool
        Compute the loss per image instead of per batch.
    """
    if per_class:
        loss = 0
        for i in range(num_classes):
            logit_ac = logits[:, i]
            label_ac = labels[:, i]
            loss += lovasz_hinge_flat(logit_ac, label_ac)
        loss = loss / num_classes
    else:
        logits = logits.view(-1)
        labels = labels.view(-1)
        loss = lovasz_hinge_flat(logits, labels)
    return loss


def hard_mining(
    neg_output: torch.Tensor, neg_labels: torch.Tensor, num_hard: int):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

def get_hard_samples(logits,labels,neg_more=2,neg_least_ratio=0.5,neg_max_ratio=0.7):
    logits = logits.view(-1)
    labels = labels.view(-1)

    pos_idcs = labels > 0
    pos_output = logits[pos_idcs]
    pos_labels = labels[pos_idcs]

    neg_idcs = labels <= 0
    neg_output = logits[neg_idcs]
    neg_labels = labels[neg_idcs]

    neg_at_least=max(neg_more,int(neg_least_ratio * neg_output.size(0)))
    hard_num = min(neg_output.size(0),pos_output.size(0) + neg_at_least, int(neg_max_ratio * neg_output.size(0)) + neg_more)
    if hard_num > 0:
        neg_output, neg_labels = hard_mining(neg_output, neg_labels, hard_num)

    logits=torch.cat([pos_output,neg_output])
    labels = torch.cat([pos_labels, neg_labels])


    return logits,labels