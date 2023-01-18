import torch.nn as nn
import math
from .binarized_modules import BinarizeLinear, BinarizeConv2d
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

        self.bconv = BinarizeConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
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

    # def spec(self):
    #     spec = {
    #         18: ([2, 2, 2, 2], [64, 128, 256, 512]),
    #         34: ([3, 4, 6, 3], [64, 128, 256, 512]),
    #         50: ([3, 4, 6, 3], [256, 512, 1024, 2048]),
    #         101: ([3, 4, 23, 3], [256, 512, 1024, 2048]),
    #         152: ([3, 8, 36, 3], [256, 512, 1024, 2048]),
    #     }
    #     try:
    #         return spec[self.num_layers]
    #     except Exception:
    #         raise ValueError(f"Only specs for layers {list(self.spec.keys())} defined.")

    def build(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
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
        # self.regime = {
        #     0: {'optimizer': 'Adam', 'lr': 1e-3},
        #     30: {'lr': 5e-4},
        #     60: {'lr': 1e-4},
        #     90: {'lr': 5e-5}
        # }
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 5e-3},
            60: {'lr': 1e-3, 'weight_decay': 0},
            90: {'lr': 1e-4}
        }

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

# class BinaryResNetE18Factory:
#     def spec(self):
#         spec = {
#             18: ([2, 2, 2, 2], [64, 128, 256, 512]),
#             34: ([3, 4, 6, 3], [64, 128, 256, 512]),
#             50: ([3, 4, 6, 3], [256, 512, 1024, 2048]),
#             101: ([3, 4, 23, 3], [256, 512, 1024, 2048]),
#             152: ([3, 8, 36, 3], [256, 512, 1024, 2048]),
#         }
#         try:
#             return spec[self.num_layers]
#         except Exception:
#             raise ValueError(f"Only specs for layers {list(self.spec.keys())} defined.")
#
#     def residual_block(self, x, filters, strides: int = 1):
#         downsample = x.get_shape().as_list()[-1] != filters
#         in_channels = filters[2]
#         out_channels = filters[3]
#
#         if downsample:
#             # residual = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)(x)
#             residual = nn.AvgPool2d(kernel_size=2, stride=2)(x)
#             # residual = tf.keras.layers.Conv2D(
#             #     filters,
#             #     kernel_size=1,
#             #     use_bias=False,
#             #     kernel_initializer="glorot_normal",
#             # )(residual)
#             residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)(residual)
#             # residual = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
#             #     residual
#             # )
#             residual = nn.BatchNorm2d(filters[3])(residual)
#         else:
#             residual = x
#
#         # x = lq.layers.QuantConv2D(
#         #     filters,
#         #     kernel_size=3,
#         #     strides=strides,
#         #     padding="same",
#         #     input_quantizer=self.input_quantizer,
#         #     kernel_quantizer=self.kernel_quantizer,
#         #     kernel_constraint=self.kernel_constraint,
#         #     kernel_initializer="glorot_normal",
#         #     use_bias=False,
#         # )(x)
#         x = BinarizeConv2d(in_channels, out_channels, kernel_size=3, stride=strides)(x)
#         # x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
#         x = nn.BatchNorm2d(out_channels)(x)
#
#         # return tf.keras.layers.add([x, residual])
#         return x + residual
#
#     def build(self):
#         # if self.image_input.shape[1] and self.image_input.shape[1] < 50:
#         #     x = tf.keras.layers.Conv2D(
#         #         self.initial_filters,
#         #         kernel_size=3,
#         #         padding="same",
#         #         kernel_initializer="he_normal",
#         #         use_bias=False,
#         #     )(self.image_input)
#         # else:
#         #     x = tf.keras.layers.Conv2D(
#         #         self.initial_filters,
#         #         kernel_size=7,
#         #         strides=2,
#         #         padding="same",
#         #         kernel_initializer="he_normal",
#         #         use_bias=False,
#         #     )(self.image_input)
#         x = nn.Conv2d(3, 64, kernel_size=7, stride=2)
#         # x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
#         x = nn.BatchNorm2d(64)(x)
#         # x = tf.keras.layers.Activation("relu")(x)
#         x = nn.ReLU(inplace=True)(x)
#         # x = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(x)
#         x = nn.MaxPool2d(3, stride=2, padding=1)(x)
#         # x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
#         x = nn.BatchNorm2d(64)(x)
#
#         for block, (layers, filters) in enumerate(zip(*self.spec)):
#             # This trick adds shortcut connections between original ResNet
#             # blocks. We wultiply the number of blocks by two, but add only one
#             # layer instead of two in each block
#             for layer in range(layers * 2):
#                 strides = 1 if block == 0 or layer != 0 else 2
#                 x = self.residual_block(x, filters, strides=strides)
#
#         # x = tf.keras.layers.Activation("relu")(x)
#         x = nn.ReLU(inplace=True)(x)
#
#         # if self.include_top:
#         #     x = utils.global_pool(x)
#         #     x = tf.keras.layers.Dense(
#         #         self.num_classes, kernel_initializer="glorot_normal"
#         #     )(x)
#         #     x = tf.keras.layers.Activation("softmax", dtype="float32")(x)
#
#         # model = tf.keras.Model(
#         #     inputs=self.image_input,
#         #     outputs=x,
#         #     name=f"binary_resnet_e_{self.num_layers}",
#         # )
#
#         # Load weights.
#         if self.weights == "imagenet":
#             # Download appropriate file
#             if self.include_top:
#                 weights_path = utils.download_pretrained_model(
#                     model="resnet_e",
#                     version="v0.1.0",
#                     file="resnet_e_18_weights.h5",
#                     file_hash="bde4a64d42c164a7b10a28debbe1ad5b287c499bc0247ecb00449e6e89f3bf5b",
#                 )
#             else:
#                 weights_path = utils.download_pretrained_model(
#                     model="resnet_e",
#                     version="v0.1.0",
#                     file="resnet_e_18_weights_notop.h5",
#                     file_hash="14cb037e47d223827a8d09db88ec73d60e4153a4464dca847e5ae1a155e7f525",
#                 )
#             model.load_weights(weights_path)
#         elif self.weights is not None:
#             model.load_weights(self.weights)
#         return model

def resnet_binary_tf(hwArgs, **kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == "imagenet":
        return ResNet(1000)
    elif dataset == "cifar10":
        return ResNet(10)

