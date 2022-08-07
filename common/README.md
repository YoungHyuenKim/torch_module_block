# Common Block Architecture

Dummy Input : 1, 3, 224, 244

Onnx Exporter options 
- training=torch.onnx.TrainingMode.TRAINING
- do_constant_folding=False

Image : Using [Netron](https://netron.app/)

## Basic Block
example_ConvBnBlock

[ConvBnBlock Image](./example_onnx_block/ConvBnBlock.onnx.png)


example_ResBlock

[ResBlock Image](./example_onnx_block/ResBlock.onnx.png)

example_InceptionNaiveBlock

[InceptionNaiveBlock Image](./example_onnx_block/InceptionNaiveBlock.onnx.png)

example_AsymmetricConvBlock

[AsymmetricConvBlock Image](./example_onnx_block/AsymmetricConvBlock.onnx.png)

## Inception Module

example_InceptionBlock

[InceptionBlock Image](./example_onnx_block/InceptionBlock.onnx.png)

example_InceptionResBlock

[InceptionResBlock Image](./example_onnx_block/InceptionResBlock.onnx.png)

example_InceptionV2Block

[InceptionV2Block Image](./example_onnx_block/InceptionV2Block.onnx.png)
