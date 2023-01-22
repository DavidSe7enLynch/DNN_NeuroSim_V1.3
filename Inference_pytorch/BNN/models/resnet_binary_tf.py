import torch.nn as nn
import math
from .binarized_modules import BinarizeConv2d
from typing import Optional, Sequence

__all__ = ['resnet_binary_tf']

def Binaryconv3x3(in_planes, out_planes, hwArgs, nameNum, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, hwArgs=hwArgs, name="Conv" + str(nameNum) + "_", kernel_size=3,
                          stride=stride, padding=1, bias=False), nameNum + 1

def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # downsample
        self.avgpool_d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batchnorm_d = nn.BatchNorm2d(out_channels)

        self.bconv = BinarizeConv2d(in_channels, out_channels, hwArgs=hwArgsGlobal, kernel_size=3, stride=stride, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        downsample = self.stride != 1 or self.in_channels != self.out_channels
        if downsample:
            residual = self.avgpool_d(x)
            residual = self.conv_d(residual)
            residual = self.batchnorm_d(residual)
        else:
            residual = x

        out = self.bconv(x)
        out = self.batchnorm(out)

        return out + residual

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.build()

    def build(self):
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv1 = BinarizeConv2d(3, 64, hwArgs=hwArgsGlobal, kernel_size=7, stride=2, padding=3)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 256, 512, 2, stride=2)

        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, self.num_classes)
        self.softmax = nn.LogSoftmax()

        init_model(self)
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-3},
            30: {'lr': 5e-4},
            60: {'lr': 1e-4},
            90: {'lr': 5e-5}
        }
        # self.regime = {
        #     0: {'optimizer': 'SGD', 'lr': 1e-2,
        #         'weight_decay': 1e-4, 'momentum': 0.9},
        #     30: {'lr': 5e-3},
        #     60: {'lr': 1e-3, 'weight_decay': 0},
        #     90: {'lr': 1e-4}
        # }

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        # downsample in first block
        blk = block(inplanes, planes, stride)
        layers.append(blk)
        for i in range(blocks * 2 - 1):
            blk = block(planes, planes, 1)
            layers.append(blk)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.batchnorm2(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu2(out)
        out = self.avgpool(out)
        out = out.view(-1, out.size(1))
        out = self.fc(out)
        out = self.softmax(out)
        return out


hwArgsGlobal = None


def resnet_binary_tf(hwArgs, **kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    global hwArgsGlobal
    hwArgsGlobal = hwArgs
    if dataset == "imagenet":
        return ResNet(1000)
    elif dataset == "cifar10":
        return ResNet(10)

