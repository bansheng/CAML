from functools import reduce
import torch
from torch import normal
import torch.nn as nn
import torch.nn.functional as F
from models.meta_operation_darts import *
from torch.autograd import Variable
from meta_genotype import PRIMITIVES, Genotype
import numpy as np
from meta_neural_network_architectures import MetaLinearLayer

class MixedOp(nn.Module):

    def __init__(self, primitive, args, C, stride, device):
        super(MixedOp, self).__init__()
        # self.mix_op = nn.ModuleList()
        self.layer_dict = nn.ModuleDict()
        self.index = PRIMITIVES.index(primitive)
        self.layer_dict['op_{}'.format(self.index)] = OPS[primitive](args, C, stride, device)
        self.restore_backup_stats()

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name in self.layer_dict.keys():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        return self.layer_dict['op_{}'.format(self.index)](x, num_step, param_dict['op_{}'.format(self.index)], training, backup_running_statistics)

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for key in self.layer_dict.keys():
            # print(key)
            self.layer_dict[key].restore_backup_stats()

class Cell(nn.Module):

    def __init__(self, args, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, device=None):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        self.args = args
        self.device = device
        self.reduction = reduction

        self.layer_dict = nn.ModuleDict()

        if reduction_prev:
            self.layer_dict['preprocess0'] = MetaFactorizedReduce(self.args, C_prev_prev, C, device=self.device)
        else:
            self.layer_dict['preprocess0'] = MetaReLUConvBN(self.args, C_prev_prev, C, 1, 1, 0, device=self.device)
        self.layer_dict['preprocess1'] = MetaReLUConvBN(self.args, C_prev, C, 1, 1, 0, device=self.device)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        # self._ops = nn.ModuleList()
        count = 0
        stage = [0, 2, 5, 9]
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            mixop_index = index + stage[count // 2]
            self.layer_dict['mixop_{}'.format(mixop_index)] =  MixedOp(name, self.args, C, stride, self.device)
            count += 1
        self._indices = indices

    def forward(self, s0, s1, num_step, params=None, training=False, backup_running_statistics=False):
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name in self.layer_dict.keys():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        s0 = self.layer_dict['preprocess0'](s0, num_step, param_dict['preprocess0'], training, backup_running_statistics)
        s1 = self.layer_dict['preprocess1'](s1, num_step, param_dict['preprocess1'], training, backup_running_statistics)

        states = [s0, s1]
        stage = [0, 2, 5, 9]
        for i in range(self._steps):
            pre_1, pre_2 = self._indices[2*i] + stage[i], self._indices[2*i+1] + stage[i]
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            h1 = self.layer_dict['mixop_{}'.format(pre_1)](
                h1, num_step, param_dict['mixop_{}'.format(pre_1)], training, backup_running_statistics
            )
            h2 = self.layer_dict['mixop_{}'.format(pre_2)](
                h2, num_step, param_dict['mixop_{}'.format(pre_2)], training, backup_running_statistics
            )
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for key in self.layer_dict.keys():
            # print(key)
            self.layer_dict[key].restore_backup_stats()


class MetaNetwork(nn.Module):

    def __init__(self, args, genotype, in_channels, init_channels, num_classes, layers=2, device=None, multiplier=4, stem_multiplier=3):
        super(MetaNetwork, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self._C = C = init_channels
        self._num_classes = num_classes
        self._layers = layers
        # self._criterion = criterion
        self.device = device
        # self._steps = steps
        self._multiplier = multiplier

        self.layer_dict = nn.ModuleDict()

        C_curr = stem_multiplier * C
        # self.stem = MetaStem(self.args, C_curr, device=self.device)
        self.layer_dict['stem'] = MetaStem(self.args, self.in_channels, C_curr, device=self.device)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i == layers -1:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(self.args, genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, device=self.device)
            reduction_prev = reduction
            # self.cells += [cell]
            self.layer_dict['cell_{}'.format(i)] = cell
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.layer_dict['global_pooling'] = nn.AdaptiveAvgPool2d(1)
        self.layer_dict['classifier'] = MetaLinearLayer((2, C_prev), num_classes, use_bias=True)

        self.restore_backup_stats()

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        param_dict = dict()

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        s0 = s1 = self.layer_dict['stem'](x, num_step, param_dict['stem'], training, backup_running_statistics)

        for i in range(self._layers):
            cell = self.layer_dict['cell_{}'.format(i)]
            s0, s1 = s1, cell(s0, s1, num_step, param_dict['cell_{}'.format(i)], training, backup_running_statistics)

        out = self.layer_dict['global_pooling'](s1)
        logits = self.layer_dict['classifier'](
            out.view(out.size(0), -1),
            param_dict['classifier'],
        )
        return logits

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            # print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for key in self.layer_dict.keys():
            if 'pool' not in key and 'classifier' not in key:
                # print(key)
                self.layer_dict[key].restore_backup_stats()

