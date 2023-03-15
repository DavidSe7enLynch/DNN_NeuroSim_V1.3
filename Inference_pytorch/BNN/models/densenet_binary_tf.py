import torch
import torch.nn as nn
import math
from .binarized_modules import BinarizeConv2d
from typing import Sequence

__all__ = ['densenet_binary_tf']


def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, dilation: int = 1):
        super(DenseBlock, self).__init__()
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.bconv = BinarizeConv2d(in_channels, out_channels, hwArgs=hwArgsGlobal, kernel_size=3, dilation=dilation, padding=1)

    def forward(self, x):
        out = self.batchnorm(x)
        out = self.bconv(out)
        return torch.concat((x, out), dim=1)


class BetweenDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(BetweenDenseBlock, self).__init__()
        self.dilation = dilation
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.batchnorm1(x)
        if self.dilation == 1:
            out = self.maxpool1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        return out


class DenseNet(nn.Module):
    name: str
    reduction: Sequence[float]
    dilation_rate: Sequence[int]
    layers: Sequence[int]

    def __init__(self):
        super(DenseNet, self).__init__()
        self.start = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            BinarizeConv2d(3, 64, hwArgs=hwArgsGlobal, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        layers = []
        cur_channels = 64
        for block, layers_per_block in enumerate(self.layers):
            for layer in range(layers_per_block):
                layers.append(DenseBlock(cur_channels, 64, self.dilation_rate[block]))
                cur_channels += 64

            if block < len(self.layers) - 1:
                out_channels = round(cur_channels // self.reduction[block] / 32) * 32
                layers.append(BetweenDenseBlock(cur_channels, out_channels, self.dilation_rate[block + 1]))
                cur_channels = out_channels

        self.dense = nn.Sequential(*layers)

        self.end1 = nn.Sequential(
            nn.BatchNorm2d(cur_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7)
        )
        self.end2 = nn.Sequential(
            nn.Linear(cur_channels, 1000),
            nn.LogSoftmax()
        )

        init_model(self)
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-3},
            30: {'lr': 5e-4},
            60: {'lr': 1e-4},
            90: {'lr': 5e-5}
        }

    def forward(self, x):
        out = self.start(x)
        out = self.dense(out)
        out = self.end1(out)
        out = out.view(-1, out.size(1))
        out = self.end2(out)

        return out


class DenseNet28(DenseNet):
    name = "binary_densenet28"
    reduction = (2.7, 2.7, 2.2)
    dilation_rate = (1, 1, 1, 1)
    layers = (6, 6, 6, 5)


hwArgsGlobal = None

def densenet_binary_tf(hwArgs, **kwargs):
    global hwArgsGlobal
    hwArgsGlobal = hwArgs
    return DenseNet28()


if __name__ == "__main__":
    DenseNet28()
