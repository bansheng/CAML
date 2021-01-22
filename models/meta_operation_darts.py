import torch
import torch.nn as nn
import torch.nn.functional as F
from meta_neural_network_architectures import MetaBatchNormLayer, MetaConv2dLayer, MetaLinearLayer

OPS = {
  'none' : lambda args, C, stride, device: MetaZero(stride),
  'avg_pool_3x3' : lambda args, C, stride, device: MetaPool(args, C, stride, device),
  # count_include_pad ，这个参数表示计算均值的时候是否包含零填充
  'max_pool_3x3' : lambda args, C, stride, device: MetaPool(args, C, stride, device, type='max_pooling'),
  'skip_connect' : lambda args, C, stride, device: MetaIdentity() if stride == 1 else MetaFactorizedReduce(args, C, C, device=device),
  'sep_conv_3x3' : lambda args, C, stride, device: MetaSepConv(args, C, C, 3, stride, 1, device=device),
  'sep_conv_5x5' : lambda args, C, stride, device: MetaSepConv(args, C, C, 5, stride, 2, device=device),
  'sep_conv_7x7' : lambda args, C, stride, device: MetaSepConv(args, C, C, 7, stride, 3, device=device),
  'dil_conv_3x3' : lambda args, C, stride, device: MetaDilConv(args, C, C, 3, stride, 2, 2, device=device),
  'dil_conv_5x5' : lambda args, C, stride, device: MetaDilConv(args, C, C, 5, stride, 4, 2, device=device),
}

def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else: #top_level = conv0
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    # print("current_dict.keys()", current_dict.keys())
    # print("output_dict.keys()", output_dict.keys())

    return output_dict

class MetaReLUConvBN(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel_size=1, stride=2, padding=0, use_bias=False, device=None):
        super(MetaReLUConvBN, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.conv = MetaConv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias
        )
        self.norm_layer = MetaBatchNormLayer(
            args=args,
            num_features=out_channels,
            device=device,
            use_per_step_bn_statistics=args.per_step_bn_statistics,
        )

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        batch_norm_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if 'norm_layer' in params:
                batch_norm_params = params['norm_layer']

            conv_params = params['conv']
        else:
            conv_params = None
            #print('no inner loop params', self)

        out = self.relu(x)
        out = self.conv(out, params=conv_params)
        out = self.norm_layer.forward(
            out, num_step=num_step,
            params=batch_norm_params, training=training,
            backup_running_statistics=backup_running_statistics
        )
        return out

    def restore_backup_stats(self):
        self.norm_layer.restore_backup_stats()

class MetaZero(nn.Module):
    def __init__(self, stride):
        super(MetaZero, self).__init__()
        self.stride = stride

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)

    def restore_backup_stats(self):
        pass

class MetaPool(nn.Module):
    def __init__(self, args, out_channels, stride, device, type='avg_pooling'):
        super(MetaPool, self).__init__()
        if type == 'avg_pooling':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1, count_include_pad=False)
        elif type == 'max_pooling':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.norm_layer = MetaBatchNormLayer(
            args=args,
            num_features=out_channels,
            device=device,
            use_per_step_bn_statistics=args.per_step_bn_statistics,
        )

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        batch_norm_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if 'norm_layer' in params:
                batch_norm_params = params['norm_layer']

        out = self.pool(x)
        out = self.norm_layer.forward(
            out, num_step=num_step,
            params=batch_norm_params, training=training,
            backup_running_statistics=backup_running_statistics
        )

        return out

    def restore_backup_stats(self):
        self.norm_layer.restore_backup_stats()

class MetaIdentity(nn.Module):
    def __init__(self):
        super(MetaIdentity, self).__init__()

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        return x

    def restore_backup_stats(self):
        pass

class MetaFactorizedReduce(nn.Module):

    def __init__(self, args, in_channels, out_channels, kernel_size=1, stride=2, padding=0, use_bias=False, device=None):
        super(MetaFactorizedReduce, self).__init__()
        assert out_channels % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = MetaConv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias
        )
        self.conv2 = MetaConv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias
        )
        self.norm_layer = MetaBatchNormLayer(
            args=args,
            num_features=out_channels,
            device=device,
            use_per_step_bn_statistics=args.per_step_bn_statistics,
        )

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        batch_norm_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if 'norm_layer' in params:
                batch_norm_params = params['norm_layer']

            conv_params_1 = params['conv1']
            conv_params_2 = params['conv2']
        else:
            conv_params_1 = None
            conv_params_2 = None
            #print('no inner loop params', self)

        x = self.relu(x)
        if x.size(2)%2!=0:
            x = F.pad(x, (1,0,1,0), "constant", 0)

        out1 = self.conv1(x, params=conv_params_1)
        out2 = self.conv2(x[:,:,1:,1:], params=conv_params_2)

        out = torch.cat([out1, out2], dim=1)
        out = self.norm_layer.forward(
            out, num_step=num_step,
            params=batch_norm_params, training=training,
            backup_running_statistics=backup_running_statistics
        )
        return out

    def restore_backup_stats(self):
        self.norm_layer.restore_backup_stats()

class MetaSepConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bias=False, device=None):
        super(MetaSepConv, self).__init__()

        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = MetaConv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            groups=in_channels
        )
        self.conv2 = MetaConv2dLayer( # 1*1 conv
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=use_bias
        )
        self.norm_layer1 = MetaBatchNormLayer(
            args=args,
            num_features=in_channels,
            device=device,
            use_per_step_bn_statistics=args.per_step_bn_statistics,
        )

        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = MetaConv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            use_bias=use_bias,
            groups=in_channels
        )
        self.conv4 = MetaConv2dLayer( # 1*1 conv
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=use_bias
        )
        self.norm_layer2 = MetaBatchNormLayer(
            args=args,
            num_features=out_channels,
            device=device,
            use_per_step_bn_statistics=args.per_step_bn_statistics,
        )

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        batch_norm_params1 = None
        batch_norm_params2 = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if 'norm_layer1' in params:
                batch_norm_params1 = params['norm_layer1']
            if 'norm_layer2' in params:
                batch_norm_params2 = params['norm_layer2']

            conv_params_1 = params['conv1']
            conv_params_2 = params['conv2']
            conv_params_3 = params['conv3']
            conv_params_4 = params['conv4']

        else:
            conv_params_1 = conv_params_2 = conv_params_3 = conv_params_4 = None
            #print('no inner loop params', self)

        out = self.relu1(x)
        out = self.conv1(out, params=conv_params_1)
        out = self.conv2(out, params=conv_params_2)
        out = self.norm_layer1.forward(
            out, num_step=num_step,
            params=batch_norm_params1, training=training,
            backup_running_statistics=backup_running_statistics
        )

        out = self.relu2(out)
        out = self.conv3(out, params=conv_params_3)
        out = self.conv4(out, params=conv_params_4)
        out = self.norm_layer2.forward(
            out, num_step=num_step,
            params=batch_norm_params2, training=training,
            backup_running_statistics=backup_running_statistics
        )

        return out

    def restore_backup_stats(self):
        self.norm_layer1.restore_backup_stats()
        self.norm_layer2.restore_backup_stats()

class MetaDilConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation_rate=2, use_bias=False, device=None):
        super(MetaDilConv, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.conv1 = MetaConv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            groups=in_channels,
            dilation_rate=dilation_rate
        )
        self.conv2 = MetaConv2dLayer( # 1*1 conv
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=use_bias
        )
        self.norm_layer = MetaBatchNormLayer(
            args=args,
            num_features=out_channels,
            device=device,
            use_per_step_bn_statistics=args.per_step_bn_statistics,
        )

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        batch_norm_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if 'norm_layer' in params:
                batch_norm_params = params['norm_layer']

            conv_params_1 = params['conv1']
            conv_params_2 = params['conv2']

        else:
            conv_params_1 = conv_params_2 = None
            #print('no inner loop params', self)

        out = self.relu(x)
        out = self.conv1(out, params=conv_params_1)
        out = self.conv2(out, params=conv_params_2)
        out = self.norm_layer.forward(
            out, num_step=num_step,
            params=batch_norm_params, training=training,
            backup_running_statistics=backup_running_statistics
        )

        return out

    def restore_backup_stats(self):
        self.norm_layer.restore_backup_stats()


class MetaStem(nn.Module):

    def __init__(self, args, in_channels, out_channels, use_bias=False, device=None):
        super(MetaStem, self).__init__()

        self.conv1 = MetaConv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels//2,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=use_bias,
        )
        self.norm_layer1 = MetaBatchNormLayer(
            args=args,
            num_features=out_channels//2,
            device=device,
            use_per_step_bn_statistics=args.per_step_bn_statistics,
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = MetaConv2dLayer(
            in_channels=out_channels//2,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=use_bias,
        )
        self.norm_layer2 = MetaBatchNormLayer(
            args=args,
            num_features=out_channels,
            device=device,
            use_per_step_bn_statistics=args.per_step_bn_statistics,
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        # self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):

        batch_norm_params1 = batch_norm_params2 = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if 'norm_layer1' in params:
                batch_norm_params1 = params['norm_layer1']
            if 'norm_layer2' in params:
                batch_norm_params2 = params['norm_layer2']

            conv_params_1 = params['conv1']
            conv_params_2 = params['conv2']

        else:
            conv_params_1 = conv_params_2 = None

        out = self.conv1(x, params=conv_params_1)
        out = self.norm_layer1.forward(
            out, num_step=num_step,
            params=batch_norm_params1, training=training,
            backup_running_statistics=backup_running_statistics
        )
        out = self.pool1(out)
        out = self.relu1(out)

        out = self.conv2(out, params=conv_params_2)
        out = self.norm_layer2.forward(
            out, num_step=num_step,
            params=batch_norm_params2, training=training,
            backup_running_statistics=backup_running_statistics
        )
        out = self.pool2(out)
        # out = self.relu2(out)

        return out

    def restore_backup_stats(self):
        self.norm_layer1.restore_backup_stats()
        self.norm_layer2.restore_backup_stats()