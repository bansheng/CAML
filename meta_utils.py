import os
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.distributions.categorical as cate
import numpy as np
import math
import shutil
from meta_genotype import PRIMITIVES

def r_makedir(paths):
    path = ""
    for p in paths:
        path = os.path.join(path, p)
        if not os.path.exists(path):
            os.makedirs(path)
    return path

def normalize(v):
    min_v = torch.min(v)
    range_v = torch.max(v) - min_v
    if range_v > 0:
        normalized_v = (v - min_v) / range_v
    else:
        normalized_v = torch.zeros(v.size()).cuda()
    return normalized_v

def count_parameters_in_KB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()
            #  if "stem" not in name and "classifier" not in name
             )/1024