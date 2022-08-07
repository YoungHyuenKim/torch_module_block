import torch
import torch.nn as nn
from .util import make_activate_fn


class ConvBnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, activate_fn=None):
        super(ConvBnBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activate_fn = make_activate_fn(activate_fn)

        self.deploy = False

    def forward(self, inputs):
        out = self.bn(self.conv(inputs))
        return self.activate_fn(out)


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, groups=1, use_conv11=False, act_fn="relu"):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBnBlock(in_channels, in_channels, kernel_size, stride, padding=padding, groups=groups)
        self.conv2 = ConvBnBlock(in_channels, in_channels, kernel_size, stride, padding=padding, groups=groups)
        self.use_conv11 = use_conv11
        if self.use_conv11:
            self.residual_conv = ConvBnBlock(in_channels, in_channels, 1, stride, padding=0, groups=groups)

        self.activation = make_activate_fn(act_fn)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_conv11:
            inputs = self.residual_conv(inputs)
        x = self.activation(x + inputs)
        return x


class InceptionNaiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_3, out_channels_5, use_max_pooling=True,
                 act_fn="relu"):
        super(InceptionNaiveBlock, self).__init__()
        self.conv11 = ConvBnBlock(in_channels, out_channels_1, kernel_size=1, stride=1, padding=0, activate_fn=act_fn)
        self.conv33 = ConvBnBlock(in_channels, out_channels_3, kernel_size=3, stride=1, padding=1, activate_fn=act_fn)
        self.conv55 = ConvBnBlock(in_channels, out_channels_5, kernel_size=5, stride=1, padding=2, activate_fn=act_fn)
        self.use_max_pooling = use_max_pooling
        if self.use_max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        total_outputs = out_channels_1 + out_channels_3 + out_channels_5
        total_outputs += in_channels if use_max_pooling else 0
        self.bn = nn.BatchNorm2d(total_outputs)

        self.activation = make_activate_fn(activate_fn=act_fn)

    def forward(self, inputs):
        branch_conv1 = self.conv11(inputs)
        branch_conv3 = self.conv33(inputs)
        branch_conv5 = self.conv55(inputs)
        if self.use_max_pooling:
            branch_pool = self.max_pool(inputs)
            out = torch.cat((branch_conv1, branch_conv3, branch_conv5, branch_pool), dim=1)
        else:
            out = torch.cat((branch_conv1, branch_conv3, branch_conv5), dim=1)
        out = self.activation(self.bn(out))
        return out


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1,
                 re_ch_3, out_channels_3, re_ch_5, out_channels_5,
                 out_channels_p,
                 use_max_pooling=True, act_fn="relu"):
        super(InceptionBlock, self).__init__()
        self.conv11_branch = nn.Sequential()
        self.conv11_branch.add_module("conv11",
                                      ConvBnBlock(in_channels, out_channels_1, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))

        self.conv33_branch = nn.Sequential()
        self.conv33_branch.add_module("conv33_reduction",
                                      ConvBnBlock(in_channels, re_ch_3, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))
        self.conv33_branch.add_module("conv33",
                                      ConvBnBlock(re_ch_3, out_channels_3, kernel_size=3, stride=1, padding=1,
                                                  activate_fn=act_fn))

        self.conv55_branch = nn.Sequential()
        self.conv55_branch.add_module("conv33_reduction",
                                      ConvBnBlock(in_channels, re_ch_5, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))
        self.conv55_branch.add_module("conv33",
                                      ConvBnBlock(re_ch_5, out_channels_5, kernel_size=5, stride=1, padding=2,
                                                  activate_fn=act_fn))

        self.use_max_pooling = use_max_pooling
        if self.use_max_pooling:
            self.max_pool_branch = nn.Sequential()
            self.max_pool_branch.add_module("max_pool", nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            self.max_pool_branch.add_module("reduce_pool",
                                            ConvBnBlock(in_channels, out_channels_p, kernel_size=1, stride=1,
                                                        padding=0, activate_fn=act_fn))

        total_outputs = out_channels_1 + out_channels_3 + out_channels_5
        total_outputs += out_channels_p if use_max_pooling else 0

        self.bn = nn.BatchNorm2d(total_outputs)
        self.activation = make_activate_fn(activate_fn=act_fn)

    def forward(self, inputs):
        branch_conv1 = self.conv11_branch(inputs)
        branch_conv3 = self.conv33_branch(inputs)
        branch_conv5 = self.conv55_branch(inputs)

        if self.use_max_pooling:
            branch_pool = self.max_pool_branch(inputs)
            out = torch.cat((branch_conv1, branch_conv3, branch_conv5, branch_pool), dim=1)
        else:
            out = torch.cat((branch_conv1, branch_conv3, branch_conv5), dim=1)
        out = self.activation(self.bn(out))
        return out


class InceptionV2Block(nn.Module):
    def __init__(self, in_channels, out_channels_1,
                 re_ch_3, out_channels_3, re_ch_5,re_ch2_5, out_channels_5,
                 out_channels_p,
                 use_max_pooling=True, act_fn="relu"):
        super(InceptionV2Block, self).__init__()
        self.conv11_branch = nn.Sequential()
        self.conv11_branch.add_module("conv11",
                                      ConvBnBlock(in_channels, out_channels_1, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))

        self.conv33_branch = nn.Sequential()
        self.conv33_branch.add_module("conv33_reduction",
                                      ConvBnBlock(in_channels, re_ch_3, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))
        self.conv33_branch.add_module("conv33",
                                      ConvBnBlock(re_ch_3, out_channels_3, kernel_size=3, stride=1, padding=1,
                                                  activate_fn=act_fn))

        self.conv55_branch = nn.Sequential()
        self.conv55_branch.add_module("conv33_reduction",
                                      ConvBnBlock(in_channels, re_ch_5, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))
        self.conv55_branch.add_module("conv33",
                                      ConvBnBlock(re_ch_5, re_ch2_5, kernel_size=3, stride=1, padding=1,
                                                  activate_fn=act_fn))
        self.conv55_branch.add_module("conv33_2",
                                      ConvBnBlock(re_ch2_5, out_channels_5, kernel_size=3, stride=1, padding=1,
                                                  activate_fn=act_fn))

        self.use_max_pooling = use_max_pooling
        if self.use_max_pooling:
            self.max_pool_branch = nn.Sequential()
            self.max_pool_branch.add_module("max_pool", nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            self.max_pool_branch.add_module("reduce_pool",
                                            ConvBnBlock(in_channels, out_channels_p, kernel_size=1, stride=1,
                                                        padding=0, activate_fn=act_fn))
        total_outputs = out_channels_1 + out_channels_3 + out_channels_5
        total_outputs += out_channels_p if use_max_pooling else 0

        self.bn = nn.BatchNorm2d(total_outputs)
        self.activation = make_activate_fn(activate_fn=act_fn)

    def forward(self, inputs):
        branch_conv1 = self.conv11_branch(inputs)
        branch_conv3 = self.conv33_branch(inputs)
        branch_conv5 = self.conv55_branch(inputs)

        if self.use_max_pooling:
            branch_pool = self.max_pool_branch(inputs)
            out = torch.cat((branch_conv1, branch_conv3, branch_conv5, branch_pool), dim=1)
        else:
            out = torch.cat((branch_conv1, branch_conv3, branch_conv5), dim=1)
        out = self.activation(self.bn(out))
        return out


class InceptionResBlock(nn.Module):
    """
    Base on InceptionV2Block
    """

    def __init__(self, in_channels, out_channels_1,
                 re_ch_3, out_channels_3, re_ch_5, re_ch2_5, out_channels_5,
                 out_channels_p, out_channels_result,
                 use_max_pooling=True, use_shortcut=True, act_fn="relu"):
        super(InceptionResBlock, self).__init__()
        self.use_max_pooling = use_max_pooling
        self.use_shortcut = use_shortcut

        self.conv11_branch = nn.Sequential()
        self.conv11_branch.add_module("conv11",
                                      ConvBnBlock(in_channels, out_channels_1, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))

        self.conv33_branch = nn.Sequential()
        self.conv33_branch.add_module("conv33_reduction",
                                      ConvBnBlock(in_channels, re_ch_3, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))
        self.conv33_branch.add_module("conv33",
                                      ConvBnBlock(re_ch_3, out_channels_3, kernel_size=3, stride=1, padding=1,
                                                  activate_fn=act_fn))

        self.conv55_branch = nn.Sequential()
        self.conv55_branch.add_module("conv33_reduction",
                                      ConvBnBlock(in_channels, re_ch_5, kernel_size=1, stride=1, padding=0,
                                                  activate_fn=act_fn))
        self.conv55_branch.add_module("conv33", ConvBnBlock(re_ch_5, re_ch2_5, kernel_size=3, stride=1, padding=1,
                                                            activate_fn=act_fn))
        self.conv55_branch.add_module("conv33_2",
                                      ConvBnBlock(re_ch2_5, out_channels_5, kernel_size=3, stride=1, padding=1,
                                                  activate_fn=act_fn))

        self.use_max_pooling = use_max_pooling
        if self.use_max_pooling:
            self.max_pool_branch = nn.Sequential()
            self.max_pool_branch.add_module("max_pool", nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            self.max_pool_branch.add_module("reduce_pool",
                                            ConvBnBlock(in_channels, out_channels_p, kernel_size=1, stride=1,
                                                        padding=0, activate_fn=act_fn))
        total_outputs = out_channels_1 + out_channels_3 + out_channels_5
        total_outputs += out_channels_p if use_max_pooling else 0

        if self.use_shortcut:
            self.residual_reduction = ConvBnBlock(total_outputs, out_channels_result, kernel_size=1, stride=1,
                                                  padding=0, activate_fn=act_fn)
            self.shortcut = ConvBnBlock(in_channels, out_channels_result, kernel_size=1, stride=1, padding=0,
                                        activate_fn=act_fn)
            self.bn = nn.BatchNorm2d(out_channels_result)
        else:
            self.residual_reduction = ConvBnBlock(total_outputs, in_channels, kernel_size=1, stride=1,
                                                  padding=0, activate_fn=act_fn)
            self.shortcut = nn.Identity()
            self.bn = nn.BatchNorm2d(in_channels)
        self.activation = make_activate_fn(activate_fn=act_fn)

    def forward(self, inputs):
        branch_conv1 = self.conv11_branch(inputs)
        branch_conv3 = self.conv33_branch(inputs)
        branch_conv5 = self.conv55_branch(inputs)

        if self.use_max_pooling:
            branch_pool = self.max_pool_branch(inputs)
            out = torch.cat((branch_conv1, branch_conv3, branch_conv5, branch_pool), dim=1)
        else:
            out = torch.cat((branch_conv1, branch_conv3, branch_conv5), dim=1)
        residual = self.residual_reduction(out)
        inputs = self.shortcut(inputs)
        out = self.activation(self.bn(residual + inputs))
        return out


class AsymmetricConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, stride, padding, groups=1,
                 act_fn="relu"):
        super(AsymmetricConvBlock, self).__init__()

        self.horizon_conv = nn.Conv2d(in_channels, middle_channels,
                                      kernel_size=(kernel_size, 1), stride=(stride, 1),
                                      padding=(padding, 0), groups=groups)
        self.horizon_bn = nn.BatchNorm2d(middle_channels)

        self.vertical_conv = nn.Conv2d(middle_channels, out_channels,
                                       kernel_size=(1, kernel_size), stride=(1, stride),
                                       padding=(0, padding), groups=groups)
        self.vertical_bn = nn.BatchNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = make_activate_fn(activate_fn=act_fn)

    def forward(self, inputs):
        x = self.activation(self.horizon_bn(self.horizon_conv(inputs)))
        x = self.activation(self.vertical_bn(self.vertical_conv(x)))
        x = self.activation(self.bn(x))
        return x
