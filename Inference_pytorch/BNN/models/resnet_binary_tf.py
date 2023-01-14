import torch.nn as nn
import math
from .binarized_modules import BinarizeLinear, BinarizeConv2d

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

