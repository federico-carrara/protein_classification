from typing import Sequence

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, dropout_p: float
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_input_features, 
                out_channels=bn_size * growth_rate,
                kernel_size=1, 
                stride=1, 
                bias=False
            ),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=bn_size * growth_rate,
                out_channels=growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
        )
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layer(x)
        if self.dropout is not None:
            new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        dropout_p: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, dropout_p)
            self.layers.append(layer)
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_input_features, num_output_features,
                kernel_size=1, stride=1, bias=False
            ),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Parameters:
    ----------
    num_classes : int
        Number of classes for the classification task.
    growth_rate : int
        Number of filters to add each layer (`k` in paper).
    block_config : Sequence[int]
        Number of layers in each pooling block.
    num_init_features : int
        Number of filters in the first convolution layer.
    bottleneck_size : int
        Bottleneck size for the DenseNet blocks. This number is multiplied by the growth
        factor to get the number of features in the bottleneck of each dense block.
    dropout_block : bool
        Whether to use the so-called dropout block before the classification head.
    dropout_p : float
        Dropout probability for the model's dropout layers.
    """
    def __init__(
        self,
        num_classes: int,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bottleneck_size: int = 4,
        dropout_block: bool = True,
        dropout_p: int = 0.5,
        *_args, **kwargs
    ) -> None:
        super().__init__()

        # First convolution
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                1, num_init_features, 
                kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks + Transition layers
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bottleneck_size,
                growth_rate=growth_rate,
                dropout_p=dropout_p # TODO: check this
            )
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.dense_blocks.append(trans)
                num_features = num_features // 2
        # final batch norm
        self.dense_blocks.append(nn.BatchNorm2d(num_features))
        self.dense_blocks.append(nn.ReLU(inplace=True))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # Classifier head layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if dropout_block:
            self.droput_block = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(num_features * 2),
                nn.Dropout1d(p=0.5),
                nn.Linear(num_features * 2, num_features),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features),
                nn.Dropout1d(p=0.5),
            )
        else:
            self.droput_block = None
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes)    
        )

        # Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.dense_blocks(x)
        if self.droput_block is not None:
            x = torch.concat([self.avgpool(x), self.avgpool(x)], dim=1)
            x = self.droput_block(x)
        else:
            x = self.avgpool(x)
        return self.classifier(x)