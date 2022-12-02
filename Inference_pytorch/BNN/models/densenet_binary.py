import torch.nn as nn
from collections import OrderedDict
import torchvision.transforms as transforms
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from typing import Any, List, Optional, Tuple
from .binarized_modules import BinarizeLinear, BinarizeConv2d

__all__ = ['densenet_binary']


def BinaryConv(in_planes, out_planes, kernel_size, hwArgs, nameNum, stride=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, hwArgs=hwArgs, name="Conv" + str(nameNum) + "_", kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=bias), nameNum + 1


# nameNum += 2
class DenseLayer(nn.Module):
    def __init__(
            self,
            num_input_features: int,
            growth_rate: int,
            bn_size: int,
            drop_rate: float,
            hwArgs,
            nameNum,
            memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv1, nameNum = BinaryConv(num_input_features, bn_size * growth_rate, kernel_size=1, hwArgs=hwArgs, nameNum=nameNum);

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.conv2, nameNum = BinaryConv(bn_size * growth_rate, growth_rate, kernel_size=3, hwArgs=hwArgs, nameNum=nameNum);
        # self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.tanh1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output


    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.tanh2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


# nameNum += num_layers * 2
class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float,
            hwArgs,
            nameNum,
            memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                hwArgs=hwArgs,
                nameNum=nameNum,
                memory_efficient=memory_efficient,
            )
            nameNum += 2
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


# nameNum += 1
class Transition(nn.Sequential):
    def __init__(
            self,
            num_input_features: int,
            num_output_features: int,
            hwArgs,
            nameNum
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.tanh = nn.Hardtanh(inplace=True)
        self.conv, nameNum = BinaryConv(num_input_features, num_output_features, kernel_size=1, hwArgs=hwArgs, nameNum=nameNum, stride=1)
        # self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
            self,
            hwArgs,
            nameNum,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
    ) -> None:

        super().__init__()
        # _log_api_usage_once(self)

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", BinaryConv(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    # ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("tanh0", nn.Hardtanh(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                hwArgs=hwArgs,
                nameNum=nameNum,
                memory_efficient=memory_efficient,
            )
            nameNum += num_layers * 2
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features, num_output_features=num_features // 2, hwArgs=hwArgs, nameNum=nameNum)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2
                nameNum += 1

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        # need to modify to input nameNum, etc.
        self.classifier = BinarizeLinear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        # out = F.relu(features, inplace=True)
        tanh = nn.Hardtanh(inplace=True)
        out = tanh(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

