"""ResNet implementation for single-channel image classification.

Based on "Deep Residual Learning for Image Recognition"
(https://arxiv.org/abs/1512.03385).
"""
from collections.abc import Sequence

import torch.nn as nn


class BasicBlock(nn.Module):
    """Two-layer residual block: Conv3x3 → BN → ReLU → Conv3x3 → BN → skip → ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """ResNet for single-channel (grayscale) image classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    block_config : Sequence[int]
        Number of BasicBlocks in each of the 4 layer groups.
        ResNet18 = [2, 2, 2, 2], ResNet34 = [3, 4, 6, 3].
    channels : Sequence[int]
        Output channels for each of the 4 layer groups.
        Standard = [64, 128, 256, 512].
    num_init_features : int
        Number of filters in the stem convolution.
    dropout_p : float
        Dropout probability before the classifier head. 0.0 disables dropout.
    """

    def __init__(
        self,
        num_classes: int,
        block_config: Sequence[int] = (2, 2, 2, 2),
        channels: Sequence[int] = (64, 128, 256, 512),
        num_init_features: int = 64,
        dropout_p: float = 0.0,
        *_args, **_kwargs,
    ) -> None:
        super().__init__()

        # Stem: Conv7x7 → BN → ReLU → MaxPool (same structure as DenseNet)
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                1, num_init_features,
                kernel_size=7, stride=2, padding=3, bias=False,
            ),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 layer groups
        self.layer_groups = nn.ModuleList()
        in_ch = num_init_features
        for i, (num_blocks, out_ch) in enumerate(zip(block_config, channels)):
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock(out_ch, out_ch, stride=1))
            self.layer_groups.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.classifier = nn.Linear(in_ch, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        for group in self.layer_groups:
            x = group(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)
