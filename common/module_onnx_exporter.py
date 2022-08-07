from common.module import *

import sys
import inspect

import torch


def onnx_export(model, input_dummy, onnx_file, input_names, output_names):
    torch.onnx.export(model, input_dummy, onnx_file, verbose=False, input_names=input_names, output_names=output_names,
                      training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False)


def example_AsymmetricConvBlock():
    name = "AsymmetricConvBlock"
    onnx_file = f"example_onnx_block/{name}.onnx"
    input_dummy = torch.randn((1, 3, 224, 224))
    model = AsymmetricConvBlock(in_channels=3, middle_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1,
                                groups=1, act_fn="relu")

    input_names = ["inputs"]
    output_names = ["outputs"]
    onnx_export(model, input_dummy, onnx_file, input_names, output_names)


def example_ConvBnBlock():
    name = "ConvBnBlock"
    onnx_file = f"example_onnx_block/{name}.onnx"
    input_dummy = torch.randn((1, 3, 224, 224))
    model = ConvBnBlock(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1, groups=1, activate_fn=None)

    input_names = ["inputs"]
    output_names = ["outputs"]
    onnx_export(model, input_dummy, onnx_file, input_names, output_names)


def example_InceptionBlock():
    name = "InceptionBlock"
    onnx_file = f"example_onnx_block/{name}.onnx"
    input_dummy = torch.randn((1, 3, 224, 224))
    model = InceptionBlock(in_channels=3, out_channels_1=5,
                           re_ch_3=4, out_channels_3=6, re_ch_5=6, out_channels_5=12,
                           out_channels_p=5,
                           use_max_pooling=True, act_fn="relu")
    input_names = ["inputs"]
    output_names = ["outputs"]
    onnx_export(model, input_dummy, onnx_file, input_names, output_names)


def example_InceptionNaiveBlock():
    name = "InceptionNaiveBlock"
    onnx_file = f"example_onnx_block/{name}.onnx"
    input_dummy = torch.randn((1, 3, 224, 224))
    model = InceptionNaiveBlock(in_channels=3, out_channels_1=4, out_channels_3=5, out_channels_5=6,
                                use_max_pooling=True,
                                act_fn="relu")

    input_names = ["inputs"]
    output_names = ["outputs"]
    onnx_export(model, input_dummy, onnx_file, input_names, output_names)


def example_InceptionResBlock():
    name = "InceptionResBlock"
    onnx_file = f"example_onnx_block/{name}.onnx"
    input_dummy = torch.randn((1, 3, 224, 224))
    model = InceptionResBlock(in_channels=3, out_channels_1=5,
                              re_ch_3=4, out_channels_3=5, re_ch_5=4, re_ch2_5=8, out_channels_5=10,
                              out_channels_p=5, out_channels_result=32,
                              use_max_pooling=True, use_shortcut=True, act_fn="relu")

    input_names = ["inputs"]
    output_names = ["outputs"]
    onnx_export(model, input_dummy, onnx_file, input_names, output_names)


def example_InceptionV2Block():
    name = "InceptionV2Block"
    onnx_file = f"example_onnx_block/{name}.onnx"
    input_dummy = torch.randn((1, 3, 224, 224))
    model = InceptionV2Block(in_channels=3, out_channels_1=5,
                             re_ch_3=4, out_channels_3=5, re_ch_5=4, re_ch2_5=6, out_channels_5=8,
                             out_channels_p=10,
                             use_max_pooling=True, act_fn="relu")

    input_names = ["inputs"]
    output_names = ["outputs"]
    onnx_export(model, input_dummy, onnx_file, input_names, output_names)


def example_ResBlock():
    name = "ResBlock"
    onnx_file = f"example_onnx_block/{name}.onnx"
    input_dummy = torch.randn((1, 3, 224, 224))
    model = ResBlock(in_channels=3, kernel_size=3, stride=1, padding=1, groups=1, use_conv11=True, act_fn="relu")

    input_names = ["inputs"]
    output_names = ["outputs"]
    onnx_export(model, input_dummy, onnx_file, input_names, output_names)


if __name__ == '__main__':

    for name, fun in inspect.getmembers(sys.modules[__name__]):
        if "example" in name and inspect.isfunction(fun):
            print(name)
            # fun()
