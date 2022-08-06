#  RepVGG: Making VGG-style ConvNets Great Again

RepVGG: Making VGG-style ConvNets Great Again

https://arxiv.org/abs/2101.03697 

기존 VGG의 경우 모델 Sequential 형태의 디자인이 되어 있다.
Sqeuential 형태의 모델의 경우 모델 레이어가 깊어지면 gradient vanishing 문제가 발생한다.

Gradient Vanishing를 해결하기 위해 gradient flow를 제공하는 Multi branch(e,g residual addition in ResNet, branch-concatenation in GoogleNet's inception Module) 
를 사용하였지만 연상량 증가와 메모리 사용량이 증가하여 연산속도가 느려지게 되었다.

RepVGG에서는 이러한 문제를 학습단에서는 multi branch형태로 학습을 진행하여
실제 Inference를 위해 배포할때는 Multi branch 형태 연산을 하나의 연산으로 줄이는 형태를 제안한다.

논문에서는 Conv 3x3, Conv 1x1, residual addition 를 하나의 Block으로 사용하고
이를 Inference대는 Conv 3x3형태로 바꾼다.


구현은 RepVGG Git를 참고하였다.