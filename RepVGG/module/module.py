import numpy as np
import torch
import torch.nn as nn
from common.module import ConvBnBlock


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super(RepVGGBlock, self).__init__()
        self.deploy = False
        self.in_channels = in_channels

        self.activation = nn.ReLU()

        # self.conv33 = conv_bn(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.conv11 = conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv33 = ConvBnBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv11 = ConvBnBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.residual_addiction = nn.BatchNorm2d(num_features=in_channels) if in_channels == out_channels else None
        # residual_addiction : BatchNorm2d Because batch normalization...

    def forward(self, inputs):
        if self.deploy and hasattr(self, "reparam_conv"):
            return self.activation(self.reparam_conv(inputs))

        if self.residual_addiction is None:
            return self.activation(self.conv33(inputs) + self.conv11(inputs))
        else:
            return self.activation(self.conv33(inputs) + self.conv11(inputs) + self.residual_addiction(inputs))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv33)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv11)
        kernelid, biasid = self._fuse_bn_tensor(self.residual_addiction)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.conv33.conv.in_channels,
                                      out_channels=self.conv33.conv.out_channels,
                                      kernel_size=self.conv33.conv.kernel_size, stride=self.conv33.conv.stride,
                                      padding=self.conv33.conv.padding, dilation=self.conv33.conv.dilation,
                                      groups=self.conv33.conv.groups, bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv33')
        self.__delattr__('conv11')
        if hasattr(self, 'residual_addiction'):
            self.__delattr__('residual_addiction')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        # if isinstance(branch, nn.Sequential):
        if isinstance(branch, ConvBnBlock):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


if __name__ == '__main__':
    # Implements Test...
    from RepVGG.util import repvgg_model_convert

    in_dim, out_dim, width, height = 3, 10, 20, 20
    dummpy_input = torch.randn(1, in_dim, height, width)

    repvgg_block = RepVGGBlock(in_dim, out_dim)
    repvgg_block.eval()

    for module in repvgg_block.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)

    inference_y = repvgg_block(dummpy_input)

    deploy_model = repvgg_model_convert(repvgg_block)
    deploy_y = deploy_model(dummpy_input)
    diff = inference_y - deploy_y
    print(torch.abs(diff).sum())
