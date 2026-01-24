import torch
import torch.nn as nn
from torchvision import models


def resnet18_grayscale(num_classes: int = 6) -> nn.Module:
    model = models.resnet18(weights=None)

   
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

  
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
