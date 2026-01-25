# src/models/model.py
import torch.nn as nn
from torchvision import models


def build_model(name: str = "resnet18", num_classes: int = 6) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return resnet18_grayscale(num_classes)
    raise ValueError(f"Unknown model: {name}")


def resnet18_grayscale(num_classes: int = 6) -> nn.Module:
    # from scratch
    model = models.resnet18(weights=None)

    # grayscale input (1 channel)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    # classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
