import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

# import sys
# sys.path.append("../")
# import data
# sys.path.append("../../")
# # from modules.quantization_cpu_np_infer import QConv2d, QLinear
# from utee import wage_quantizer

from . import wage_quantizer

import numpy as np


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = 1.0

    def hinge_loss(self, input, target):
        # import pdb; pdb.set_trace()
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input, target)


class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction, self).__init__()
        self.margin = 1.0

    def forward(self, input, target):
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        self.save_for_backward(input, target)
        loss = output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self, grad_output):
        input, target = self.saved_tensors
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        import pdb;
        pdb.set_trace()
        grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(input.numel())
        return grad_output, grad_output


def Quantize(tensor, quant_mode='det', params=None, numBits=8):
    tensor.clamp_(-2 ** (numBits - 1), 2 ** (numBits - 1))
    if quant_mode == 'det':
        tensor = tensor.mul(2 ** (numBits - 1)).round().div(2 ** (numBits - 1))
    else:
        tensor = tensor.mul(2 ** (numBits - 1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2 ** (numBits - 1))
        quant_fixed(tensor, params)
    return tensor


# modified
# import torch.nn._functions as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        wl_input = 8
        wl_activate = 8
        wl_error = 8
        wl_weight = 8
        inference = 1
        onoffratio = 10
        cellBit = 1
        subArray = 128
        ADCprecision = 5
        vari = 0
        t = 0
        v = 0
        detect = 0
        target = 0
        is_linear = 1
        self.wl_weight = int(wl_weight)
        self.wl_activate = int(wl_activate)
        self.wl_error = int(wl_error)
        self.wl_input = int(wl_input)
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = int(cellBit)
        self.subArray = subArray
        self.ADCprecision = int(ADCprecision)
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.is_linear = is_linear

    def neurosim_linear(self):
        outputOriginal = nn.functional.linear(input, self.weight)
        # set parameters for Hardware Inference
        onoffratio = self.onoffratio
        upper = 1
        lower = 1 / onoffratio
        output = torch.zeros_like(outputOriginal)
        cellRange = 2 ** self.cellBit  # cell precision is 4
        # Now consider on/off ratio
        dummyP = torch.zeros_like(self.weight)
        dummyP[:, :] = (cellRange - 1) * (upper + lower) / 2

        # cells per weight
        numCell = int(self.wl_weight / self.cellBit)
        if self.wl_weight % self.cellBit != 0:
            numCell += 1
        # need to divide to different subArray
        numSubArray = int(self.weight.shape[1] / self.subArray)
        if self.weight.shape[1] % self.subArray != 0:
            numSubArray += 1

        inputQ = torch.round((2 ** self.wl_input - 1) / 1 * (input - 0) + 0)
        outputIN = torch.zeros_like(outputOriginal)
        for z in range(self.wl_input):
            inputB = torch.fmod(inputQ, 2)
            inputQ = torch.round((inputQ - inputB) / 2)
            outputP = torch.zeros_like(outputOriginal)
            for s in range(numSubArray):
                mask = torch.zeros_like(self.weight)

                if s == numSubArray - 1:
                    # last subArray
                    # cannot go beyond size of weight
                    mask[:, (s * self.subArray):] = 1
                else:
                    mask[:, (s * self.subArray):(s + 1) * self.subArray] = 1

                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                X_decimal = torch.round((2 ** self.wl_weight - 1) / 2 * (self.weight + 1) + 0) * mask
                outputSP = torch.zeros_like(outputOriginal)
                outputD = torch.zeros_like(outputOriginal)
                for k in range(numCell):
                    remainder = torch.fmod(X_decimal, cellRange) * mask
                    # retention
                    remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect, self.target)
                    X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                    # Now also consider weight has on/off ratio effects
                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                    remainderQ = (upper - lower) * (remainder - 0) + (
                            cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                    remainderQ = remainderQ + remainderQ * torch.normal(0.,
                                                                        torch.full(remainderQ.size(), self.vari,
                                                                                   device='cuda'))
                    outputPartial = nn.functional.linear(inputB, remainderQ * mask, self.bias)
                    outputDummyPartial = nn.functional.linear(inputB, dummyP * mask, self.bias)
                    # Add ADC quanization effects here !!!

                    # choose one from these two: linear or non-linear
                    if self.is_linear == 1:
                        # linear quantization
                        outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial,
                                                                          self.ADCprecision)
                        outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial,
                                                                               self.ADCprecision)
                    else:
                        # non-linear quantization
                        # print("calling nonlinear")
                        outputPartialQ = wage_quantizer.NonLinearQuantizeOut(outputPartial,
                                                                             self.ADCprecision)
                        outputDummyPartialQ = wage_quantizer.NonLinearQuantizeOut(outputDummyPartial,
                                                                                  self.ADCprecision)

                    scaler = cellRange ** k
                    outputSP = outputSP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                    outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                outputSP = outputSP - outputD  # minus dummy column
                outputP = outputP + outputSP
            scalerIN = 2 ** z
            outputIN = outputIN + outputP * scalerIN
            output = output + outputIN / (2 ** self.wl_input)
        output = output / (2 ** self.wl_weight)
        return output

    def forward(self, input):

        if input.size(1) != 784:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        if self.inference == 0:
            out = nn.functional.linear(input, self.weight)
        else:
            out = self.neurosim_linear()

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
