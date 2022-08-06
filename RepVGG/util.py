from RepVGG.module.module import ConvBnBlock

import torch
import torch.nn as nn
import numpy as np
import copy


def fuse_bn_tensor(module, *args, **kwargs):
    """
    :param module: inherited nn.Module
    :param args:
    :param kwargs: need key word....
    :return: re-parameterize kernel, re-parameterize bias
    """
    if module is None:
        return 0, 0

    if isinstance(module, ConvBnBlock):
        kernel = module.conv.weight
        running_mean = module.bn.running_mean
        running_var = module.bn.running_var
        gamma = module.bn.weight
        beta = module.bn.bias
        eps = module.bn.eps
    elif isinstance(module, nn.BatchNorm2d):
        # Make Kernel...
        assert "in_channels" in kwargs, "BatchNorm2d need in_channels"
        assert "kernel_size" in kwargs, "BatchNorm2d need kernel_size."
        in_channels = kwargs["in_channels"]
        in_dim = kwargs["in_dim"] if "in_dim" in kwargs else in_channels
        kernel_size = kwargs["kernel_size"]
        kernel_value = np.zeros((in_channels, in_dim, kernel_size, kernel_size))
        for i in range(in_channels):
            kernel_value[i, i % in_dim, 1, 1] = 1

        kernel = torch.from_numpy(kernel_value)
        running_mean = module.running_mean
        running_var = module.running_var
        gamma = module.weight
        beta = module.bias
        eps = module.eps
    else:
        raise NotImplemented
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
