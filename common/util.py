from typing import Optional
import torch.nn as nn


def make_activate_fn(activate_fn: Optional[str] = None):
    """
    :param activate_fn:
    :return: nn.Module, activation function..
    """
    if activate_fn is None:
        return nn.Identity()

    activate_fn = activate_fn.lower()
    if activate_fn == "relu":
        return nn.ReLU()
    if activate_fn == "sigmoid":
        return nn.Sigmoid()
    if activate_fn == "leaky_relu":
        return nn.LeakyReLU()
    if activate_fn == "tanh":
        return nn.Tanh()
    if activate_fn == "swish" or activate_fn == "silu":
        return nn.SiLU()
    if activate_fn == "elu":
        return nn.ELU()
    return nn.Identity()
