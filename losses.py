"""
losses.py
Loss functions designed for imbalanced multi-class image classification.
Supports:
  1. Standard Cross-Entropy Loss
  2. Weighted Cross-Entropy Loss (Inverse frequency weighting)
  3. Focal Loss (Lin et al., ICCV 2017)
  4. Class-Balanced Loss (Cui et al., CVPR 2019) with CE or Focal core
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[Union[torch.Tensor, List[float]]] = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha.float()
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: shape (N, C) - logits
        targets: shape (N,) - ground truth class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # probability of true class
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples:
    E_n = (1 - beta^n) / (1 - beta)
    Weights = (1 - beta) / (1 - beta^n_y)
    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
    """
    def __init__(self, samples_per_cls: List[int], beta: float = 0.9999, loss_type: str = "focal", gamma: float = 2.0):
        super(ClassBalancedLoss, self).__init__()
        self.loss_type = loss_type
        self.gamma = gamma

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)
        self.cb_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.cb_weights.to(inputs.device)
        if self.loss_type == "ce":
            return F.cross_entropy(inputs, targets, weight=weights)
        elif self.loss_type == "focal":
            focal = FocalLoss(gamma=self.gamma, alpha=weights)
            return focal(inputs, targets)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


def compute_class_weights(class_counts: Union[List[int], np.ndarray, torch.Tensor], mode: str = "inverse", smoothing: float = 0.1) -> torch.Tensor:
    """
    Computes class weights from frequency counts.
    mode options:
      - 'inverse': N / (C * N_c)
      - 'sqrt': sqrt(max(N_c) / N_c)
    """
    counts = np.array(class_counts, dtype=np.float32)
    total = np.sum(counts)
    num_classes = len(counts)

    if mode == "inverse":
        weights = total / (num_classes * (counts + smoothing))
    elif mode == "sqrt":
        weights = np.sqrt(np.max(counts) / (counts + smoothing))
    else:
        weights = np.ones(num_classes, dtype=np.float32)

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_loss_fn(loss_type: str, class_counts: Optional[List[int]] = None, device: str = "cpu", gamma: float = 2.0, beta: float = 0.9999) -> nn.Module:
    """
    Factory function to retrieve loss function.
    loss_type options:
      - 'ce' or 'cross_entropy': Standard Cross-Entropy
      - 'weighted_ce': Weighted Cross-Entropy with inverse frequency weights
      - 'focal': Focal Loss (gamma=2.0)
      - 'cb_loss' or 'class_balanced': Class-Balanced Loss
    """
    lt = loss_type.lower()
    if lt in ["ce", "cross_entropy"]:
        return nn.CrossEntropyLoss()

    elif lt in ["weighted_ce", "weighted_cross_entropy"]:
        if class_counts is None:
            raise ValueError("class_counts must be provided for weighted_ce")
        weights = compute_class_weights(class_counts, mode="inverse").to(device)
        return nn.CrossEntropyLoss(weight=weights)

    elif lt in ["focal", "focal_loss"]:
        alpha = None
        if class_counts is not None:
            alpha = compute_class_weights(class_counts, mode="sqrt").to(device)
        return FocalLoss(gamma=gamma, alpha=alpha)

    elif lt in ["cb_loss", "class_balanced"]:
        if class_counts is None:
            raise ValueError("class_counts must be provided for class_balanced loss")
        return ClassBalancedLoss(samples_per_cls=class_counts, beta=beta, loss_type="focal", gamma=gamma)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    counts = [6705, 1113, 1099, 514, 327, 142, 115]
    dummy_logits = torch.randn(8, 7)
    dummy_targets = torch.tensor([0, 0, 1, 3, 5, 6, 2, 0])

    ce = get_loss_fn("ce")
    wce = get_loss_fn("weighted_ce", class_counts=counts)
    fl = get_loss_fn("focal", class_counts=counts)
    cb = get_loss_fn("cb_loss", class_counts=counts)

    print("CE Loss:", ce(dummy_logits, dummy_targets).item())
    print("Weighted CE Loss:", wce(dummy_logits, dummy_targets).item())
    print("Focal Loss:", fl(dummy_logits, dummy_targets).item())
    print("Class Balanced Loss:", cb(dummy_logits, dummy_targets).item())
    print("Losses module smoke test passed!")
