"""
models/resnet.py
Neural network architectures for Skin-Lesion Classification.
Supports:
  - ResNet-50 (Pretrained on ImageNet, default main architecture)
  - EfficientNet-B0 (Pretrained on ImageNet, comparative architecture)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SkinLesionModel(nn.Module):
    def __init__(self, model_name: str = "resnet50", num_classes: int = 7, pretrained: bool = True, dropout_rate: float = 0.3):
        super(SkinLesionModel, self).__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes

        if self.model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        elif self.model_name in ["efficientnet_b0", "efficientnet-b0"]:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}. Use 'resnet50' or 'efficientnet_b0'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        out = self.classifier(features)
        return out

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(model_name: str = "resnet50", num_classes: int = 7, pretrained: bool = True, dropout_rate: float = 0.3) -> SkinLesionModel:
    """Helper factory function to instantiate model."""
    return SkinLesionModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    # Smoke test
    dummy = torch.randn(2, 3, 224, 224)
    m = get_model("resnet50", num_classes=7)
    out = m(dummy)
    print("ResNet-50 output shape:", out.shape)
    assert out.shape == (2, 7)
    print("Smoke test passed successfully!")
