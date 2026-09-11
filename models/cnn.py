import torch
from torch import nn
import torchvision.models as models


class SealCodeCNN(nn.Module):
    def __init__(self, code_length=7, classes=10):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")

        # Convert first layer from RGB to grayscale
        old_conv = backbone.conv1

        backbone.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Initialize grayscale conv from pretrained RGB weights
        with torch.no_grad():
            backbone.conv1.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.heads = nn.ModuleList([
            nn.Linear(512, classes)
            for _ in range(code_length)
        ])

        self.code_length = code_length
        self.classes = classes

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        logits = torch.stack(
            [head(x) for head in self.heads],
            dim=1
        )

        return logits