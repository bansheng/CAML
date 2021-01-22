from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
# Genotype = namedtuple('Genotype_PRUNE', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

IMAGENET_V1 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5',0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('skip_connect', 3)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=[2, 3,
4, 5])

IMAGENET_V2 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('skip_connect', 3)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5])

OMNI_V1 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 4)], reduce_concat=[2, 3, 4, 5])

##
FRONT_V1 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], reduce_concat=[2, 3, 4, 5])


## imagenet_5W_5S_B4_C16_tv31_crib
BACK_V1 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('dil_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])
