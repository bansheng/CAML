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
import torch.distributions.categorical as cate
import math
from meta_utils import normalize

class MixedOp(nn.Module):

    def __init__(self, args, C, stride, device):
        super(MixedOp, self).__init__()
        # self.mix_op = nn.ModuleList()
        self.layer_dict = nn.ModuleDict()
        for i, primitive in enumerate(PRIMITIVES):
            op = OPS[primitive](args, C, stride, device)
            self.layer_dict['op_{}'.format(i)] = op

        self.restore_backup_stats()


    def forward(self, x, weights, num_step, params=None, training=False, backup_running_statistics=False, selected_idx=None):

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name in self.layer_dict.keys():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None
                # print("None")

        if selected_idx is None:
            return sum(
                w * self.layer_dict['op_{}'.format(i)](
                    x,
                    num_step,
                    param_dict['op_{}'.format(i)],
                    training,
                    backup_running_statistics
                ) for i, w in enumerate(weights))
        else:  # unchosen operations are pruned
            return self.layer_dict['op_{}'.format(selected_idx)](x, num_step, param_dict['op_{}'.format(selected_idx)], training, backup_running_statistics)

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for key in self.layer_dict.keys():
            # print(key)
            self.layer_dict[key].restore_backup_stats()

class Cell(nn.Module):

    def __init__(self, args, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, device=None):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        self.args = args
        self.device = device
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier

        self.layer_dict = nn.ModuleDict()

        if reduction_prev:
            self.layer_dict['preprocess0'] = MetaFactorizedReduce(self.args, C_prev_prev, C, device=self.device)
        else:
            self.layer_dict['preprocess0'] = MetaReLUConvBN(self.args, C_prev_prev, C, 1, 1, 0, device=self.device)
        self.layer_dict['preprocess1'] = MetaReLUConvBN(self.args, C_prev, C, 1, 1, 0, device=self.device)


        # self.mixops = nn.ModuleList()
        # self._bns = nn.ModuleList()
        count = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(self.args, C, stride, device=self.device)
                # self.mixops.append(op)
                self.layer_dict['mixop_{}'.format(count)] = op
                count += 1

        self.restore_backup_stats()

    def forward(self, s0, s1, weights, num_step, params=None, training=False, backup_running_statistics=False, selected_idxs=None):

        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, para in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        s0 = self.layer_dict['preprocess0'](s0, num_step, param_dict['preprocess0'], training, backup_running_statistics)
        s1 = self.layer_dict['preprocess1'](s1, num_step, param_dict['preprocess1'], training, backup_running_statistics)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            o_list = []
            for j, h in enumerate(states):
                o = None
                if selected_idxs[offset + j] == -1: # undecided mix edges
                    # o = self.mixops[offset + j](h, weights[offset + j])
                    # print(weights[offset + j].shape)
                    o = self.layer_dict['mixop_{}'.format(offset + j)](
                        h,
                        weights[offset + j],
                        num_step,
                        param_dict['mixop_{}'.format(offset + j)],
                        training,
                        backup_running_statistics
                    )
                elif selected_idxs[offset + j] == PRIMITIVES.index('none'): # pruned edges
                    pass
                else: # decided discrete edges
                    # o = self.mixops[offset + j](h, None, selected_idxs[offset + j])
                    o = self.layer_dict['mixop_{}'.format(offset + j)](
                        h,
                        None,
                        num_step,
                        param_dict['mixop_{}'.format(offset + j)],
                        training,
                        backup_running_statistics,
                        selected_idxs[offset + j]
                    )
                if o is not None: # o is not None
                    o_list.append(o)
            s = sum(o_list)
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for key in self.layer_dict.keys():
            # print(key)
            self.layer_dict[key].restore_backup_stats()


class MetaNASNetwork(nn.Module):

    def __init__(self, args, in_channels, init_channels, num_classes, layers, device=None, steps=4, multiplier=4, stem_multiplier=3):
        super(MetaNASNetwork, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self._C = C = init_channels
        self._num_classes = num_classes
        self._layers = layers
        # self._criterion = criterion
        self.device = device
        self._steps = steps
        self._multiplier = multiplier

        self.layer_dict = nn.ModuleDict()

        C_curr = stem_multiplier * C
        # self.stem = MetaStem(self.args, C_curr, device=self.device)
        self.layer_dict['stem'] = MetaStem(self.args, self.in_channels, C_curr, device=self.device)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i == layers - 1: # first layer
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(self.args, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, device=self.device)
            reduction_prev = reduction
            # self.cells += [cell]
            self.layer_dict['cell_{}'.format(i)] = cell
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.layer_dict['global_pooling'] = nn.AdaptiveAvgPool2d(1)
        self.layer_dict['classifier'] = MetaLinearLayer((2, C_prev), num_classes, use_bias=True)

        self._initialize_alphas()
        self.restore_backup_stats()
        # self.normal_selected_idxs = None
        # self.reduce_selected_idxs = None
        # self.normal_candidate_flags = None
        # self.reduce_candidate_flags = None

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        param_dict = dict()
        alphas_normal = []
        alphas_reduce = []

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

            alphas_normal = self.alphas_normal.copy()
            alphas_reduce = self.alphas_reduce.copy()
            for name, param in param_dict.items():
                if 'normal' in name:
                    index = int(name.split('_')[2])
                    alphas_normal[index] = param
                elif 'reduce' in name:
                    index = int(name.split('_')[2])
                    alphas_reduce[index] = param
        else:
            alphas_normal = self.alphas_normal
            alphas_reduce = self.alphas_reduce

        for name in self.layer_dict.keys():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        # s0 = s1 = self.stem(input)
        s0 = s1 = self.layer_dict['stem'](x, num_step, param_dict['stem'], training, backup_running_statistics)

        # for i, cell in enumerate(self.cells):
        for i in range(self._layers):
            cell = self.layer_dict['cell_{}'.format(i)]
            if cell.reduction:
                selected_idxs = self.reduce_selected_idxs
                alphas = alphas_reduce
            else:
                selected_idxs = self.normal_selected_idxs
                alphas = alphas_normal

            weights = []
            n = 2
            start = 0
            for _ in range(self._steps):
                end = start + n
                for j in range(start, end):
                    weights.append(F.softmax(alphas[j], dim=-1))
                start = end
                n += 1

            s0, s1 = s1, cell(s0, s1, weights, num_step, param_dict['cell_{}'.format(i)], training, backup_running_statistics, selected_idxs)

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

            for param in self.arch_parameters(): # zero alpha
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            # print(param.grad)
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

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = []
        self.alphas_reduce = []
        for i in range(self._steps):
            for n in range(2 + i):
                self.alphas_normal.append(Variable(1e-3 * torch.randn(num_ops).cuda(), requires_grad=True))
                self.alphas_reduce.append(Variable(1e-3 * torch.randn(num_ops).cuda(), requires_grad=True))

        self.normal_selected_idxs = torch.tensor(len(self.alphas_normal) * [-1], requires_grad=False, dtype=torch.int).cuda()
        self.reduce_selected_idxs = torch.tensor(len(self.alphas_reduce) * [-1], requires_grad=False, dtype=torch.int).cuda()
        self.normal_candidate_flags = torch.tensor(len(self.alphas_normal) * [True], requires_grad=False, dtype=torch.bool).cuda()
        self.reduce_candidate_flags = torch.tensor(len(self.alphas_reduce) * [True], requires_grad=False, dtype=torch.bool).cuda()
        self.importance_score_front = torch.tensor(range(len(self.alphas_normal), 0, -1), requires_grad=False, dtype=torch.float).cuda()
        self.importance_score_back = torch.tensor(range(len(self.alphas_normal)), requires_grad=False, dtype=torch.float).cuda()

    def _alphas(self):
        return (self.normal_selected_idxs, self.reduce_selected_idxs,
                self.normal_candidate_flags, self.reduce_candidate_flags,
                self.alphas_normal, self.alphas_reduce
                )

    def load_alphas(self, state):
        (
            self.normal_selected_idxs, self.reduce_selected_idxs,
            self.normal_candidate_flags, self.reduce_candidate_flags,
            self.alphas_normal, self.alphas_reduce
        ) = state['alphas']

    def arch_parameters(self):
        return self.alphas_normal + self.alphas_reduce # concat lists

    def named_arch_parameters(self):
        para = []
        for i in range(len(self.alphas_normal)):
            para.append(('alphas_normal_{}'.format(i), self.alphas_normal[i]))
        for i in range(len(self.alphas_reduce)):
            para.append(('alphas_reduce_{}'.format(i), self.alphas_reduce[i]))
            # para['alpha_reduce_{}'.format(i)] = self.alphas_reduce[i]
        for p in para:
            yield p

    def check_edges(self, flags, selected_idxs, reduction=False):
        """
        remove the duplicated decided operations.
        """
        n = 2
        max_num_edges = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            if num_selected_edges >= max_num_edges:
                for j in range(start, end):
                    if flags[j]:
                        flags[j] = False
                        selected_idxs[j] = PRIMITIVES.index('none') # pruned edges
                        if reduction:
                            self.alphas_reduce[j].requires_grad = False
                        else:
                            self.alphas_normal[j].requires_grad = False
                    else:
                        pass
            start = end
            n += 1

        return flags, selected_idxs

    def check_weights(self, logging=None):

        normal_keys = []
        for i, index in enumerate(self.normal_selected_idxs):
            if index != -1: # decided
                for j in range(len(PRIMITIVES)):
                    if j != index:
                        normal_keys.append('mixop_{}op_{}'.format(i, j)) # need be pruned

        reduce_keys = []
        for i, index in enumerate(self.reduce_selected_idxs):
            if index != -1: # decided
                for j in range(len(PRIMITIVES)):
                    if j != index:
                        reduce_keys.append('mixop_{}op_{}'.format(i, j)) # need be pruned

        # print(normal_keys, reduce_keys)

        for i in range(self._layers):
            cell = self.layer_dict['cell_{}'.format(i)]
            if cell.reduction: #reduce cell
                for name, param in cell.named_parameters():
                    newname = name.replace('layer_dict.', '')
                    if ''.join(newname.split('.')[:2]) in reduce_keys:
                        if param.requires_grad:
                            param.requires_grad = False
                            # logging.info('cell_{}_{} requires_grad = False'.format(i, name))
            else:
                for name, param in cell.named_parameters():
                    newname = name.replace('layer_dict.', '')
                    if ''.join(newname.split('.')[:2]) in normal_keys:
                        if param.requires_grad:
                            param.requires_grad = False
                            # logging.info('cell_{}_{} requires_grad = False'.format(i, name))

    def edge_decision(self, epoch, logging):

        geno_normal = self.edge_decision_helper("normal", epoch, logging, make_decision=self.args.make_decision)
        geno_reduce = self.edge_decision_helper("reduce", epoch, logging, make_decision=self.args.make_decision)

        genotype = Genotype(
            normal= geno_normal,
            normal_concat=[2, 3, 4, 5],
            reduce=geno_reduce,
            reduce_concat=[2, 3, 4, 5],
        )
        logging.info("genotype = {}".format(str(genotype)))


    def edge_decision_helper(self, type, epoch, logging, make_decision):

        if type == 'normal':
            alphas, selected_idxs, candidate_flags = \
                self.alphas_normal, self.normal_selected_idxs, self.normal_candidate_flags
        else:
            alphas, selected_idxs, candidate_flags = \
                self.alphas_reduce, self.reduce_selected_idxs, self.reduce_candidate_flags

        mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
        logging.info(str(mat))

        if not make_decision: #不做实时pruning
            if epoch == self.args.warmup_dec_epoch + 7 * self.args.decision_freq: #最后一刻pruning，总共做8次
                for i in range(8): # 总共切出8条边
                    self.edge_decision_helper(type, epoch, logging, make_decision=True)

        elif torch.sum(candidate_flags.int()) > 0:
            if epoch >= self.args.warmup_dec_epoch and \
            (epoch - self.args.warmup_dec_epoch) % self.args.decision_freq == 0:

                importance = torch.sum(mat[:, 1:], dim=-1)
                probs = mat[:, 1:] / importance[:, None]

                if self.args.pruning_criterion == "criterion_front":
                    logging.info("use criterion_front for pruning")
                    # score = normalize(importance)
                    score = self.importance_score_front
                elif self.args.pruning_criterion == "criterion_back": # criterion_2
                    logging.info("use criterion_back for pruning")
                    # entropy = cate.Categorical(probs=probs).entropy() / math.log(probs.size()[1])
                    # score = normalize(importance) * normalize(1 - entropy)
                    score = self.importance_score_back
                else: # darts_like criterion
                    logging.info("use criterion_darts for pruning")
                    score = torch.max(mat[:, 1:], dim=-1)[0]

                masked_score = torch.min(score,
                                        (2 * candidate_flags.float() - 1) * np.inf)
                selected_edge_idx = torch.argmax(masked_score)
                if not self.args.rand:
                    selected_op_idx = torch.argmax(probs[selected_edge_idx]) + 1 # add 1 since none op
                else:
                    choice = np.random.choice(range(1, len(PRIMITIVES)))
                    selected_op_idx = torch.tensor(choice).to(selected_idxs.device)

                selected_idxs[selected_edge_idx] = selected_op_idx

                candidate_flags[selected_edge_idx] = False
                alphas[selected_edge_idx].requires_grad = False
                if type == 'normal':
                    reduction = False
                elif type == 'reduce':
                    reduction = True
                else:
                    raise Exception('Unknown Cell Type')
                candidate_flags, selected_idxs = self.check_edges(candidate_flags,
                                                                selected_idxs,
                                                                reduction=reduction)

                logging.info("#" * 30 + " Decision Epoch " + "#" * 30)
                logging.info("epoch {}, {}_selected_idxs {}, added edge {} with op idx {}".format(epoch,
                                                                type,
                                                                selected_idxs,
                                                                selected_edge_idx,
                                                                selected_op_idx))
                logging.info(type + "_candidate_flags {}".format(candidate_flags))
                # geno = self.genotype(selected_idxs)
                # return geno
                self.check_weights(logging)
            else:
                logging.info("#" * 30 + " Not a Decision Epoch " + "#" * 30)
                logging.info("epoch {}, {}_selected_idxs {}".format(epoch,
                                                            type,
                                                            selected_idxs))
                logging.info(type + "_candidate_flags {}".format(candidate_flags))
                # geno = self.genotype(selected_idxs)
                # return geno
        # else:
        geno = self.genotype(selected_idxs)
        return geno

    def genotype(self, selected_idxs):

        geno = []
        steps = 4
        start = 0
        n = 2
        for i in range(steps):
            end = start + n
            count = 0
            for index, j in enumerate(selected_idxs[start:end]):
                if j > 0:
                    geno.append((PRIMITIVES[j], index))
                    count += 1
            for j in range(2-count):
                geno.append(())

            n += 1
            start = end
        return geno