# from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import os
import torch.nn as nn
import shutil
# from modules.quantization_cpu_np_infer import QConv2d, QLinear
# from modules.floatrange_cpu_np_infer import FConv2d, FLinear
from models.binarized_modules import BinarizeConv2d, BinarizeLinear
import numpy as np
import torch
from utee import wage_quantizer
from utee import float_quantizer


def Neural_Sim(self, input, output):
    global model_n, FP

    print("quantize layer, input: ", self.name)
    input_file_name = './layer_record_' + str(model_n) + '/input' + str(self.name) + '.csv'
    weight_file_name = './layer_record_' + str(model_n) + '/weight' + str(self.name) + '.csv'
    f = open('./layer_record_' + str(model_n) + '/trace_command.sh', "a")
    f.write(weight_file_name + ' ' + input_file_name + ' ')
    # if FP:
    #     weight_q = float_quantizer.float_range_quantize(self.weight, self.wl_weight)
    # else:
    #     weight_q = wage_quantizer.Q(self.weight, self.wl_weight)
    write_matrix_weight(self.weight.cpu().data.numpy(), weight_file_name)
    # print(self.weight.shape)
    # print(len(self.weight.shape))
    # print(self.weight.shape[-1])
    if len(self.weight.shape) > 2:
        k = self.weight.shape[-1]
        padding = self.padding
        stride = self.stride
        write_matrix_activation_conv(stretch_input(input[0].cpu().data.numpy(), k, padding, stride), None,
                                     self.wl_input, input_file_name)
    else:
        write_matrix_activation_fc(input[0].cpu().data.numpy(), None, self.wl_input, input_file_name)


def write_matrix_weight(input_matrix, filename):
    cout = input_matrix.shape[0]
    weight_matrix = input_matrix.reshape(cout, -1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",", fmt='%10.5f')


def write_matrix_activation_conv(input_matrix, fill_dimension, length, filename):
    filled_matrix_b = np.zeros([input_matrix.shape[2], input_matrix.shape[1] * length], dtype=np.str)
    filled_matrix_bin, scale = dec2bin(input_matrix[0, :], length)
    for i, b in enumerate(filled_matrix_bin):
        filled_matrix_b[:, i::length] = b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",", fmt='%s')


def write_matrix_activation_fc(input_matrix, fill_dimension, length, filename):
    filled_matrix_b = np.zeros([input_matrix.shape[1], length], dtype=np.str)
    filled_matrix_bin, scale = dec2bin(input_matrix[0, :], length)
    for i, b in enumerate(filled_matrix_bin):
        filled_matrix_b[:, i] = b
    np.savetxt(filename, filled_matrix_b, delimiter=",", fmt='%s')


def stretch_input(input_matrix, window_size=5, padding=(0, 0), stride=(1, 1)):
    # input_shape = input_matrix.shape
    # item_num = ((input_shape[2] + 2 * padding[0] - window_size) / stride[0] + 1) * (
    #             (input_shape[3] + 2 * padding[1] - window_size) / stride[1] + 1)
    # output_matrix = np.zeros((input_shape[0], int(item_num), input_shape[1] * window_size * window_size))
    # iter = 0
    # for i in range(input_shape[2] - window_size + 1):
    #     for j in range(input_shape[3] - window_size + 1):
    #         for b in range(input_shape[0]):
    #             output_matrix[b, iter, :] = input_matrix[b, :, i:i + window_size, j: j + window_size].reshape(
    #                 input_shape[1] * window_size * window_size)
    #         iter += 1
    #
    # return output_matrix

    input_shape = input_matrix.shape
    i_range = int((input_shape[2] + 2 * padding[0] - window_size) / stride[0] + 1)
    j_range = int((input_shape[3] + 2 * padding[1] - window_size) / stride[1] + 1)
    item_num = i_range * j_range
    output_matrix = np.zeros((input_shape[0], int(item_num), input_shape[1] * window_size * window_size))

    # add padding to the matrix
    input_matrix_new = input_matrix.copy()
    for i in range(padding[0]):
        input_matrix_new = np.insert(input_matrix_new, 0, 0, axis=2)
        input_matrix_new = np.insert(input_matrix_new, input_matrix_new.shape[2], 0, axis=2)
    for i in range(padding[1]):
        input_matrix_new = np.insert(input_matrix_new, 0, 0, axis=3)
        input_matrix_new = np.insert(input_matrix_new, input_matrix_new.shape[3], 0, axis=3)
    iter = 0
    for i in range(i_range):
        for j in range(j_range):
            for b in range(input_shape[0]):
                temp = input_matrix_new[b, :, i * stride[0]: i * stride[0] + window_size,
                       j * stride[1]: j * stride[1] + window_size]
                # print("i, j, temp shape: ", i, j, temp.shape)
                output_matrix[b, iter, :] = temp.reshape(input_shape[1] * window_size * window_size)
            iter += 1

    return output_matrix


def dec2bin(x, n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0 / (2 ** (n - 1))
    x_int = x / delta

    base = 2 ** (n - 1)

    y[x_int >= 0] = 0
    y[x_int < 0] = 1
    rest = x_int + base * y
    out.append(y.copy())
    scale_list.append(-base * delta)
    for i in range(n - 1):
        base = base / 2
        y[rest >= base] = 1
        y[rest < base] = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out, scale_list


def bin2dec(x, n):
    bit = x.pop(0)
    base = 2 ** (n - 1)
    delta = 1.0 / (2 ** (n - 1))
    y = -bit * base
    base = base / 2
    for bit in x:
        y = y + base * bit
        base = base / 2
    out = y * delta
    return out


def remove_hook_list(hook_handle_list):
    for handle in hook_handle_list:
        handle.remove()


def hardware_evaluation(model, wl_weight, wl_activation, model_name):
    global model_n, FP
    model_n = model_name
    # FP = 1 if mode == 'FP' else 0
    FP = 0

    hook_handle_list = []
    if not os.path.exists('./layer_record_' + str(model_name)):
        os.makedirs('./layer_record_' + str(model_name))
    if os.path.exists('./layer_record_' + str(model_name) + '/trace_command.sh'):
        os.remove('./layer_record_' + str(model_name) + '/trace_command.sh')
    f = open('./layer_record_' + str(model_name) + '/trace_command.sh', "w")
    f.write('./NeuroSIM/main ./NeuroSIM/NetWork_' + str(model_name) + '.csv ' + str(wl_weight) + ' ' + str(
        wl_activation) + ' ')

    for i, layer in enumerate(model.modules()):
        if isinstance(layer, (BinarizeConv2d, nn.Conv2d)) or isinstance(layer, (BinarizeLinear, nn.Linear)):
            print("inside hardware_evaluation, layer: ", layer)
            hook_handle_list.append(layer.register_forward_hook(Neural_Sim))
    return hook_handle_list