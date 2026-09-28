"""Late-fusion ResNet18 plus biochemical MLP baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18ImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, imagenet: bool = False, freeze_backbone: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if imagenet else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(images))


class BiochemistryMLP(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LateFusionInjuryClassifier(nn.Module):
    def __init__(
        self,
        biochem_input_dim: int,
        num_classes: int = 3,
        image_embedding_dim: int = 128,
        biochem_embedding_dim: int = 32,
        imagenet: bool = False,
        freeze_image_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = ResNet18ImageEncoder(
            embedding_dim=image_embedding_dim,
            imagenet=imagenet,
            freeze_backbone=freeze_image_backbone,
        )
        self.biochemistry_encoder = BiochemistryMLP(
            input_dim=biochem_input_dim,
            embedding_dim=biochem_embedding_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_embedding_dim + biochem_embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor, biochemistry: torch.Tensor) -> torch.Tensor:
        image_embedding = self.image_encoder(images)
        biochem_embedding = self.biochemistry_encoder(biochemistry)
        fused = torch.cat([image_embedding, biochem_embedding], dim=1)
        return self.classifier(fused)


def build_late_fusion_model(biochem_input_dim: int, num_classes: int = 3, imagenet: bool = False) -> nn.Module:
    return LateFusionInjuryClassifier(
        biochem_input_dim=biochem_input_dim,
        num_classes=num_classes,
        imagenet=imagenet,
    )
