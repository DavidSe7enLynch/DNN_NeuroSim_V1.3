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

from . import wage_quantizer, wage_initializer

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

    def __init__(self, *kargs, hwArgs=None, name="BinarizeLinear", **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        # wl_input = 2
        # wl_activate = 2
        # wl_error = 2
        # wl_weight = 1
        # # inference = 1
        # onoffratio = 10
        # cellBit = 1
        # subArray = 128
        # ADCprecision = 5
        # vari = 0
        # t = 0
        # v = 0
        # detect = 0
        # target = 0
        # is_linear = 1
        # self.wl_weight = wl_weight
        # self.wl_activate = wl_activate
        # self.wl_error = wl_error
        # self.wl_input = wl_input
        # self.hw = hw
        # self.onoffratio = onoffratio
        # self.cellBit = cellBit
        # self.subArray = subArray
        # self.ADCprecision = ADCprecision
        # self.vari = vari
        # self.t = t
        # self.v = v
        # self.detect = detect
        # self.target = target
        # self.is_linear = is_linear
        # self.scale = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)
        # self.name = name

        wl_input = 2
        wl_weight = 1
        onoffratio = 10
        subArray = 128
        vari = 0
        t = 0
        v = 0
        detect = 0
        target = 0
        self.wl_weight = wl_weight

        if hwArgs:
            self.wl_input = hwArgs.m_wlInput
            self.hw = hwArgs.m_isHW
            self.ADCprecision = hwArgs.m_adcPrec
        else:
            self.wl_input = 8
            self.hw = 0
            self.ADCprecision = 7

        self.onoffratio = onoffratio
        self.subArray = subArray
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.scale = 1
        self.is_linear = 1
        self.cellBit = 1

    # hardware effect simulation
    # used only when inferencing
    # so self.weight must be already binary
    # so no need to transform weight
    # only need to transform input into bit stream
    def neurosim_linear(self, input, is_input_bin):
        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        outputOrignal = nn.functional.linear(input, weight, self.bias)
        output = torch.zeros_like(outputOrignal)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        # set parameters for Hardware Inference
        onoffratio = self.onoffratio
        upper = 1
        lower = 1 / onoffratio
        output = torch.zeros_like(outputOrignal)
        cellRange = 2 ** self.cellBit  # cell precision is 4
        # Now consider on/off ratio
        dummyP = torch.zeros_like(weight)
        dummyP[:, :] = (cellRange - 1) * (upper + lower) / 2
        # need to divide to different subArray
        numSubArray = int(weight.shape[1] / self.subArray)

        if numSubArray == 0:
            mask = torch.zeros_like(weight)
            mask[:, :] = 1
            # quantize input into binary sequence
            inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
            outputIN = torch.zeros_like(outputOrignal)
            for z in range(bitActivation):
                inputB = torch.fmod(inputQ, 2)
                inputQ = torch.round((inputQ - inputB) / 2)
                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                outputP = torch.zeros_like(outputOrignal)
                outputD = torch.zeros_like(outputOrignal)
                for k in range(int(bitWeight / self.cellBit)):
                    remainder = torch.fmod(X_decimal, cellRange) * mask
                    # retention
                    remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect, self.target)
                    X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                    # Now also consider weight has on/off ratio effects
                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                    remainderQ = (upper - lower) * (remainder - 0) + (
                            cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                    # remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari,
                    #                                                                    device='cuda'))
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
                    outputP = outputP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                    outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
                scalerIN = 2 ** z
                outputIN = outputIN + (outputP - outputD) * scalerIN
            output = output + outputIN / (2 ** bitActivation)
        else:
            inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
            outputIN = torch.zeros_like(outputOrignal)
            for z in range(bitActivation):
                inputB = torch.fmod(inputQ, 2)
                inputQ = torch.round((inputQ - inputB) / 2)
                outputP = torch.zeros_like(outputOrignal)
                for s in range(numSubArray):
                    mask = torch.zeros_like(weight)
                    mask[:, (s * self.subArray):(s + 1) * self.subArray] = 1
                    # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                    X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
                    outputSP = torch.zeros_like(outputOrignal)
                    outputD = torch.zeros_like(outputOrignal)
                    for k in range(int(bitWeight / self.cellBit)):
                        remainder = torch.fmod(X_decimal, cellRange) * mask
                        # retention
                        remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect, self.target)
                        X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
                        # Now also consider weight has on/off ratio effects
                        # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                        # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                        remainderQ = (upper - lower) * (remainder - 0) + (
                                cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
                        # remainderQ = remainderQ + remainderQ * torch.normal(0.,
                        #                                                     torch.full(remainderQ.size(), self.vari,
                        #                                                                device='cuda'))
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
            output = output + outputIN / (2 ** bitActivation)
        output = output / (2 ** bitWeight)

        output = output / self.scale
        # output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)

        return output
        # # outputOriginal = nn.functional.linear(input, self.weight)
        # # # set parameters for Hardware Inference
        # # upper = 1
        # # lower = 1 / self.onoffratio
        # # output = torch.zeros_like(outputOriginal)
        # # # cellRange = 2 ** self.cellBit  # cell precision is 4
        # # # Now consider on/off ratio
        # # dummyP = torch.zeros_like(self.weight)
        # # # dummyP[:, :] = (cellRange - 1) * (upper + lower) / 2
        # # dummyP[:, :] = 1 * (upper + lower) / 2
        # #
        # # # cells per weight
        # # # numCell = int(self.wl_weight / self.cellBit)  # in BNN, numCell shall be 1
        # # # if self.wl_weight % self.cellBit != 0:
        # # #     numCell += 1
        # #
        # # # need to divide to different subArray
        # # numSubArray = int(self.weight.shape[1] / self.subArray)
        # # if self.weight.shape[1] % self.subArray != 0:
        # #     numSubArray += 1
        # #
        # # inputQ = torch.round((2 ** self.wl_input - 1) / 1 * (input - 0) + 0)
        # # outputIN = torch.zeros_like(outputOriginal)
        # # for z in range(self.wl_input):
        # #     inputB = torch.fmod(inputQ, 2)
        # #     inputQ = torch.round((inputQ - inputB) / 2)
        # #     outputP = torch.zeros_like(outputOriginal)
        # #     for s in range(numSubArray):
        # #         mask = torch.zeros_like(self.weight)
        # #
        # #         if s == numSubArray - 1:
        # #             # last subArray
        # #             # cannot go beyond size of weight
        # #             mask[:, (s * self.subArray):] = 1
        # #         else:
        # #             mask[:, (s * self.subArray):((s + 1) * self.subArray)] = 1
        # #
        # #         # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
        # #         X_decimal = torch.round((2 ** self.wl_weight - 1) / 2 * (self.weight + 1) + 0) * mask
        # #         outputSP = torch.zeros_like(outputOriginal)
        # #         outputD = torch.zeros_like(outputOriginal)
        # #         # for k in range(numCell):
        # #         # remainder = torch.fmod(X_decimal, cellRange) * mask
        # #         # retention
        # #         remainder = wage_quantizer.Retention(X_decimal, self.t, self.v, self.detect, self.target)
        # #         # X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
        # #         # Now also consider weight has on/off ratio effects
        # #         # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
        # #         # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
        # #         # remainderQ = (upper - lower) * (remainder - 0) + (
        # #         #         cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
        # #         remainderQ = (upper - lower) * (remainder - 0) + 1 * lower  # weight cannot map to 0, but to Gmin
        # #         # remainderQ = remainderQ + remainderQ * torch.normal(0.,
        # #         #                                                     torch.full(remainderQ.size(), self.vari,
        # #         #                                                                device='cuda'))
        # #
        # #         outputPartial = nn.functional.linear(inputB, remainderQ * mask, self.bias)
        # #         outputDummyPartial = nn.functional.linear(inputB, dummyP * mask, self.bias)
        # #         # Add ADC quanization effects here !!!
        # #
        # #         # BNN doesn't need this step
        # #         # # choose one from these two: linear or non-linear
        # #         # if self.is_linear == 1:
        # #         #     # linear quantization
        # #         #     outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial,
        # #         #                                                       self.ADCprecision)
        # #         #     outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial,
        # #         #                                                            self.ADCprecision)
        # #         # else:
        # #         #     # non-linear quantization
        # #         #     # print("calling nonlinear")
        # #         #     outputPartialQ = wage_quantizer.NonLinearQuantizeOut(outputPartial,
        # #         #                                                          self.ADCprecision)
        # #         #     outputDummyPartialQ = wage_quantizer.NonLinearQuantizeOut(outputDummyPartial,
        # #         #                                                               self.ADCprecision)
        # #
        # #         # scaler = cellRange ** k
        # #         # outputSP = outputSP + outputPartialQ * scaler * 2 / (1 - 1 / self.onoffratio)
        # #         # outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / self.onoffratio)
        # #         outputSP = outputSP + outputPartial * 1 * 2 / (1 - 1 / self.onoffratio)
        # #         outputD = outputD + outputDummyPartial * 1 * 2 / (1 - 1 / self.onoffratio)
        # #
        # #         outputSP = outputSP - outputD  # minus dummy column
        # #         outputP = outputP + outputSP
        # #     scalerIN = 2 ** z
        # #     outputIN = outputIN + outputP * scalerIN
        # #     output = output + outputIN / (2 ** self.wl_input)
        # # output = output / (2 ** self.wl_weight)
        # # # print("neurosim_linear finished")
        # # # print("original output\n", outputOriginal)
        # # # print("neuro_output\n", output)
        # # return output
        #
        #
        # # copy from quant
        # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        # weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        # outputOrignal = nn.functional.linear(input, weight, self.bias)
        # output = torch.zeros_like(outputOrignal)
        #
        # bitWeight = int(self.wl_weight)
        # bitActivation = int(self.wl_input)
        # # print("bitActivation: ", bitActivation)
        # # print("numcell: ", int(bitWeight / self.cellBit))
        #
        # # if self.inference == 1 and self.model == 'VGG8':
        # # set parameters for Hardware Inference
        # onoffratio = self.onoffratio
        # upper = 1
        # lower = 1 / onoffratio
        # output = torch.zeros_like(outputOrignal)
        # cellRange = 2 ** self.cellBit  # cell precision is 4
        # # Now consider on/off ratio
        # dummyP = torch.zeros_like(weight)
        # dummyP[:, :] = (cellRange - 1) * (upper + lower) / 2
        # # need to divide to different subArray
        # numSubArray = int(weight.shape[1] / self.subArray)
        #
        # # if numSubArray == 0:
        # #     print("inside numSubArray = 0")
        # #     mask = torch.zeros_like(weight)
        # #     mask[:, :] = 1
        # #     # quantize input into binary sequence
        # #     inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
        # #     outputIN = torch.zeros_like(outputOrignal)
        # #     for z in range(bitActivation):
        # #         inputB = torch.fmod(inputQ, 2)
        # #         inputQ = torch.round((inputQ - inputB) / 2)
        # #         # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
        # #         X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
        # #         outputP = torch.zeros_like(outputOrignal)
        # #         outputD = torch.zeros_like(outputOrignal)
        # #         for k in range(int(bitWeight / self.cellBit)):
        # #             remainder = torch.fmod(X_decimal, cellRange) * mask
        # #             # retention
        # #             remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect, self.target)
        # #             X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
        # #             # Now also consider weight has on/off ratio effects
        # #             # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
        # #             # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
        # #             remainderQ = (upper - lower) * (remainder - 0) + (
        # #                     cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
        # #             # remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari,
        # #             #                                                                    device='cuda'))
        # #             outputPartial = nn.functional.linear(inputB, remainderQ * mask, self.bias)
        # #             outputDummyPartial = nn.functional.linear(inputB, dummyP * mask, self.bias)
        # #             # Add ADC quanization effects here !!!
        # #
        # #             # choose one from these two: linear or non-linear
        # #             if self.is_linear == 1:
        # #                 # linear quantization
        # #                 outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial,
        # #                                                                   self.ADCprecision)
        # #                 outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial,
        # #                                                                        self.ADCprecision)
        # #             else:
        # #                 # non-linear quantization
        # #                 # print("calling nonlinear")
        # #                 outputPartialQ = wage_quantizer.NonLinearQuantizeOut(outputPartial,
        # #                                                                      self.ADCprecision)
        # #                 outputDummyPartialQ = wage_quantizer.NonLinearQuantizeOut(outputDummyPartial,
        # #                                                                           self.ADCprecision)
        # #
        # #             scaler = cellRange ** k
        # #             outputP = outputP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
        # #             outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
        # #         scalerIN = 2 ** z
        # #         outputIN = outputIN + (outputP - outputD) * scalerIN
        # #     output = output + outputIN / (2 ** bitActivation)
        # # else:
        # #     print("inside numSubArray != 0")
        # inputQ = torch.round((2 ** bitActivation - 1) / 1 * (input - 0) + 0)
        # outputIN = torch.zeros_like(outputOrignal)
        # for z in range(bitActivation):
        #     inputB = torch.fmod(inputQ, 2)
        #     inputQ = torch.round((inputQ - inputB) / 2)
        #     outputP = torch.zeros_like(outputOrignal)
        #     for s in range(numSubArray):
        #         mask = torch.zeros_like(weight)
        #         mask[:, (s * self.subArray):(s + 1) * self.subArray] = 1
        #         # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
        #         X_decimal = torch.round((2 ** bitWeight - 1) / 2 * (weight + 1) + 0) * mask
        #         outputSP = torch.zeros_like(outputOrignal)
        #         outputD = torch.zeros_like(outputOrignal)
        #         for k in range(int(bitWeight / self.cellBit)):
        #             remainder = torch.fmod(X_decimal, cellRange) * mask
        #             # retention
        #             remainder = wage_quantizer.Retention(remainder, self.t, self.v, self.detect, self.target)
        #             X_decimal = torch.round((X_decimal - remainder) / cellRange) * mask
        #             # Now also consider weight has on/off ratio effects
        #             # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
        #             # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
        #             remainderQ = (upper - lower) * (remainder - 0) + (
        #                     cellRange - 1) * lower  # weight cannot map to 0, but to Gmin
        #             # remainderQ = remainderQ + remainderQ * torch.normal(0.,
        #             #                                                     torch.full(remainderQ.size(), self.vari,
        #             #                                                                device='cuda'))
        #             outputPartial = nn.functional.linear(inputB, remainderQ * mask, self.bias)
        #             outputDummyPartial = nn.functional.linear(inputB, dummyP * mask, self.bias)
        #             # Add ADC quanization effects here !!!
        #
        #             # choose one from these two: linear or non-linear
        #             if self.is_linear == 1:
        #                 # linear quantization
        #                 outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial,
        #                                                                   self.ADCprecision)
        #                 outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial,
        #                                                                        self.ADCprecision)
        #             else:
        #                 # non-linear quantization
        #                 # print("calling nonlinear")
        #                 outputPartialQ = wage_quantizer.NonLinearQuantizeOut(outputPartial,
        #                                                                      self.ADCprecision)
        #                 outputDummyPartialQ = wage_quantizer.NonLinearQuantizeOut(outputDummyPartial,
        #                                                                           self.ADCprecision)
        #
        #             scaler = cellRange ** k
        #             outputSP = outputSP + outputPartialQ * scaler * 2 / (1 - 1 / onoffratio)
        #             outputD = outputD + outputDummyPartialQ * scaler * 2 / (1 - 1 / onoffratio)
        #         outputSP = outputSP - outputD  # minus dummy column
        #         outputP = outputP + outputSP
        #     scalerIN = 2 ** z
        #     outputIN = outputIN + outputP * scalerIN
        # output = output + outputIN / (2 ** bitActivation)
        #
        # output = output / (2 ** bitWeight)
        #
        # # elif self.inference == 1:
        # #     weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        # #     weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        # #     weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
        # #     input = wage_quantizer.Q(input, self.wl_input)
        # #     output = nn.functional.linear(input, weight, self.bias)
        # #
        # #     # choose one from these two: linear or non-linear
        # #     if self.is_linear == 1:
        # #         # linear quantization
        # #         output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        # #     else:
        # #         # non-linear quantization
        # #         # print("calling non-linear")
        # #         output = wage_quantizer.NonLinearQuantizeOut(output, self.ADCprecision)
        # # else:
        # #     # original WAGE QCov2d
        # #     weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        # #     weight = weight1 + (wage_quantizer.Q(weight1, self.wl_weight) - weight1).detach()
        # #     weight = wage_quantizer.Retention(weight, self.t, self.v, self.detect, self.target)
        # #     output = nn.functional.linear(input, weight, self.bias)
        # # print("before scale output: ", output)
        # # print(self.scale)
        # # output = output / self.scale
        # # output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)
        # output = output * 2
        #
        # return output

    def forward(self, input):
        is_input_bin = 0
        # print("initial input: ", input)
        # print("initial weight: ", self.weight)
        # if input.size(1) != 784:
        is_input_bin = 1
        input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        # print("linear: is_input_bin = ", is_input_bin)
        # print("bin weight: ", self.weight)

        self.hw = 0
        if self.hw == 0:
            out = nn.functional.linear(input, self.weight)
            # print("linear finished")
        else:
            out = self.neurosim_linear(input, is_input_bin)
            # print("nuerosim_linear finished: ", out)
            # print("linear out:", nn.functional.linear(input, self.weight))
            # print("after weight: ", self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, hwArgs=None, name="BinarizeConv2d", **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.wl_weight = 1
        self.onoffratio = 10
        self.subArray = 128
        self.vari = 0
        self.t = 0
        self.v = 0
        self.detect = 0
        self.target = 0
        self.name = name
        self.is_linear = 1

        if hwArgs:
            self.wl_input = hwArgs.m_wlInput
            self.hw = hwArgs.m_isHW
            self.ADCprecision = hwArgs.m_adcPrec
        else:
            self.wl_input = 8
            self.hw = 0
            self.ADCprecision = 7


        # middle results
        # self.outputPartial
        # self.outputDummyPartial

    def quantizationADC(self, outputPartial, outputDummyPartial):
        # choose one from these two: linear or non-linear
        if self.is_linear:
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
        return outputPartialQ, outputDummyPartialQ

    def binaryLayer(self, input, output, upper, lower, mask, dummyP):
        # not first layer, do not quantize
        X_decimal = torch.round((2 ** self.wl_weight - 1) / 2 * (self.weight + 1) + 0) * mask
        outputP = torch.zeros_like(output)
        outputD = torch.zeros_like(output)
        remainder = wage_quantizer.Retention(X_decimal, self.t, self.v, self.detect, self.target)
        remainderQ = (upper - lower) * (remainder - 0) + 1 * lower  # weight cannot map to 0, but to Gmin
        outputPartial = nn.functional.conv2d(input, remainderQ * mask, self.bias, self.stride,
                                             self.padding,
                                             self.dilation, self.groups)
        outputDummyPartial = nn.functional.conv2d(input, dummyP * mask, self.bias, self.stride,
                                                  self.padding, self.dilation, self.groups)

        outputPartial, outputDummyPartial = self.quantizationADC(outputPartial, outputDummyPartial)

        outputP = outputP + outputPartial * 1 * 2 / (1 - 1 / self.onoffratio)
        outputD = outputD + outputDummyPartial * 1 * 2 / (1 - 1 / self.onoffratio)

        outputP = outputP - outputD
        output = output + outputP
        return output

    def firstLayer(self, input, output, upper, lower, mask, dummyP):
        inputAbsMax = max(abs(torch.min(input).item()), abs(torch.max(input).item()))
        expandRatio = (2 ** self.wl_input - 1) / inputAbsMax
        # print("inputabsmax, expandratio = ", inputAbsMax, expandRatio)
        inputQ = torch.round(expandRatio * input)
        outputIN = torch.zeros_like(output)
        for z in range(self.wl_input):
            inputB = torch.fmod(inputQ, 2)
            inputQ = torch.round((inputQ - inputB) / 2)
            outputP = torch.zeros_like(output)
            # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
            X_decimal = torch.round((2 ** self.wl_weight - 1) / 2 * (self.weight + 1) + 0) * mask
            outputD = torch.zeros_like(output)
            # for k in range(int(self.wl_weight / self.cellBit)):
            # remainder = torch.fmod(X_decimal, cellRange) * mask
            # retention
            remainder = wage_quantizer.Retention(X_decimal, self.t, self.v, self.detect,
                                                 self.target)
            remainderQ = (upper - lower) * (
                    remainder - 0) + 1 * lower  # weight cannot map to 0, but to Gmin
            outputPartial = nn.functional.conv2d(inputB, remainderQ * mask, self.bias, self.stride,
                                                 self.padding, self.dilation, self.groups)
            outputDummyPartial = nn.functional.conv2d(inputB, dummyP * mask, self.bias,
                                                      self.stride,
                                                      self.padding, self.dilation, self.groups)

            outputPartial, outputDummyPartial = self.quantizationADC(outputPartial, outputDummyPartial)

            outputP = outputP + outputPartial * 1 * 2 / (1 - 1 / self.onoffratio)
            outputD = outputD + outputDummyPartial * 1 * 2 / (1 - 1 / self.onoffratio)

            scalerIN = 2 ** z
            outputIN = outputIN + (outputP - outputD) * scalerIN
        output = output + outputIN / expandRatio
        return output

    def neurosim_conv2d(self, input):
        outputOriginal = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        # set parameters for Hardware Inference
        onoffratio = self.onoffratio
        upper = 1
        lower = 1 / onoffratio

        output = torch.zeros_like(outputOriginal)
        del outputOriginal
        # cellRange = 2 ** self.cellBit  # cell precision is 4

        # Now consider on/off ratio
        dummyP = torch.zeros_like(self.weight)
        # dummyP[:, :, :, :] = (cellRange - 1) * (upper + lower) / 2
        dummyP[:, :, :, :] = 1 * (upper + lower) / 2

        # need to divide to different subArray
        numSubArray = math.ceil(self.weight.shape[1] / self.subArray)

        # print("weight size: ", self.weight.shape)
        # print("is_linear: ", self.is_linear)
        for i in range(self.weight.shape[2]):
            for j in range(self.weight.shape[3]):
                for s in range(numSubArray):
                    mask = torch.zeros_like(self.weight)
                    mask[:, (s * self.subArray):(s + 1) * self.subArray, i, j] = 1
                    if self.weight.shape[1] == 3:
                        # if i == 0 and j == 0:
                            # print("first layer, input = ", input[0][0][0][0])
                        # first layer, convert to binary sequence
                        output = self.firstLayer(input, output, upper, lower, mask, dummyP)
                    else:
                        # if i == 0 and j == 0:
                            # print("binary layer, input = ", input[0][0][0][0])
                        # other binary layers
                        output = self.binaryLayer(input, output, upper, lower, mask, dummyP)
        return output

    def forward(self, input):
        # print("forward, input = ", input.size(), input[0][0][0])
        is_input_bin = 0
        if input.size(1) != 3:
            is_input_bin = 1
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        # print("conv: is_input_bin = ", is_input_bin)
        # self.hw = 0
        if self.hw == 0:
            out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
            # print("conv2d finished")
        else:
            out = self.neurosim_conv2d(input)
            # print("neurosim_conv2d finished: ", out.size(), out[0][0][0])
            # conv2d_out = nn.functional.conv2d(input, self.weight, None, self.stride,
            #                        self.padding, self.dilation, self.groups)
            # print("conv2d finished: ", conv2d_out.size(), conv2d_out[0][0][0])

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
