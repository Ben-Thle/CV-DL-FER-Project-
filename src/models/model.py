from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(
    name: str = "resnet18",
    num_classes: int = 6,
    *,
    input_channels: int = 3,  # pipeline outputs RGB JPG
    small_input: bool = True,
) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return resnet18(num_classes=num_classes, input_channels=input_channels, small_input=small_input)
    raise ValueError(f"Unknown model: {name}")


def resnet18(*, num_classes: int = 6, input_channels: int = 3, small_input: bool = True) -> nn.Module:
    model = models.resnet18(weights=None)

    if small_input:
        # best for 64x64
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        # default resnet stem, but allow grayscale if needed
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

