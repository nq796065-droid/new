"""
model.py - Dinh nghia Kien truc Mo hinh (ResNet-50) va Ham Loss (Focal Loss).
Tong cong: ~60 dong code de hieu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SkinLesionModel(nn.Module):
    """ResNet-50 Transfer Learning cho phan loai 7 lop ton thuong da."""
    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        in_features = base.fc.in_features
        base.fc = nn.Identity()  # Loai bo layer cuoi cua ImageNet
        self.backbone = base
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def get_model(num_classes: int = 7, pretrained: bool = True) -> SkinLesionModel:
    return SkinLesionModel(num_classes=num_classes, pretrained=pretrained)


class FocalLoss(nn.Module):
    """
    Focal Loss giai quyet mat can bang lop: FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Giam trong so cua cac mau de hoc (de), tap trung vao mau kho va lop hiem.
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Xac suat du doan dung
        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            return (alpha_t * focal_weight * ce_loss).mean()
        return (focal_weight * ce_loss).mean()


def compute_class_weights(class_counts: list, device: torch.device) -> torch.Tensor:
    """Tinh trong so nghich dao tan suat lop: w_c = N / (C * N_c)"""
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = counts.sum() / (len(counts) * counts)
    return (weights / weights.mean()).to(device)
