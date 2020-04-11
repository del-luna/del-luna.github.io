---
layout: post
title: AlexNet
author: Jaeheon Kwon
categories: Papers
tags: [cnn]
---

# AlexNet :  ImageNet Classification with Deep Convolutional Neural Networks 



## Summery

- 기본 구조는 LeNet-5와 비슷하며 2개의 GPU 병렬연산을 수행하기 위해 병렬적 구조로 이루어져 있다.
- 기존에 사용되면 activation function인 tanh와 달리 ReLU를 사용하였다.
- Dropout 기법을 적용하였다.



## Introduction

머신러닝 작업을 수행하기 위해서 데이터의 중요성은 항상 강조되어 왔습니다.<br>

비교적 조금만 예전으로 돌아가도 큰 규모의 레이블링된 데이터 셋을 구하는건 힘든 일이었는데 LableMe, ImageNet 등의 데이터 셋이 생겨나고 이를 활용한 다양한 연구가 가능해졌습니다.<br>

하지만 많은 수의 이미지에서 수천 개의 객체에 대해 학습하려면 Large capacity model이 필요합니다.<br>

또한 객체 인식의 Complexity가 심하면 ImageNet과 같은 큰 데이터 셋으로도 이 문제를 specified 할 수 없습니다.<br>

따라서 모델에는 우리가 보유하지 못한 데이터를 보완 할 수 있는 많은 Prior knowledge가 있어야 합니다.<br>

CNN 바로 이러한 종류의 모델을 구성합니다.<br>

CNN의 Capacity는 depth와 width를 변경하여 제어할 수 있으며, 이미지의 특성에 대해 강력하고 정확한 가정을 가능하게 합니다.<br>

따라서, 비슷한 크기의 layers를 가진 standard feedforward 방식의 model과 비교할 때 CNN은 connection과 parameter가 훨씬 적기 때문에 학습하기 쉬우며 이론적으로 최상의 성능은 약간 떨어집니다.<br>

CNN의 특성과 local architecture의 상대적 효율성에도 불구하고 여전히 High resolution image에 대규모로 적용하기엔 비용이 너무 큽니다.<br>

최근 기술의 발전으로 고성능 GPU와 최적화된 Conv layer, Overfitting을 방지할만한 충분한 Label이 있는 ImageNet과 같은 데이터 셋이 존재합니다.<br>

## Dataset

ImageNet은 약 22,000개의 카테고리와 1500만개가 넘는 Label이 존재하는 고 해상도 이미지 데이터 셋입니다.

ImageNet은 가변 해상도 이미지로 구성되기 때문에 입력을 위하여 256x256으로 downsampling 하였습니다.<br>

각 pixel에 대하여 training set에 대해 mean activity를 빼는 것을 제외하곤 어떤 전처리도 하지 않았습니다.<br>

그래서 pixel(centered) raw RGB값에 대해 네트워크를 훈련했습니다.<br>

## Architecture

<img src = "https://py-tonic.github.io/images/alexnet/1.PNG">

네트워크 구조는 나온지 오래된 모델이라 비교적 단순합니다.<br>

8개의 Layer(5개의 Conv layer, 3개의 fully-connected layer)로 구성됩니다.<br>

2, 4, 5번째 합성곱층의 커널들은 같은 GPU내의 이전 층의 커널과 연결되어 있고, 3번째 층은 이전 층의 모든 커널과 연결되어 있습니다.<br>

Response-normalization 층들은 1, 2번째 합성곱층들 뒤에 있습니다.<br>

Max pooling 층들은 Response-normalization과 5번째 층 뒤에 있습니다.<br>

ReLU는 모든 층 뒤에 있습니다.<br>

1번째 합성곱층은 11x11x3인 커널 96개, 2번째 합성곱층은 5x5x48인 커널 256개, 3번째 합성곱층 3x3x256인 커널 384개, 4번째 합성곱층은 3x3x192인 커널 384개, 5번째 합성곱층은 3x3x192인 커널 256개를 사용합니다.<br>

FCN 에서는 각각 4096개의 뉴런을 갖습니다.<br>

### ReLU Nonlinearity

input x의 함수로 뉴런의 출력 f를 모델링하는 표준 방법은<br>

$f(x) = tanh(x)$<br>

$f(x) = (1+e^{-x})^{-1}$ <br>

를 사용합니다.<br>

Gradient Descent에 따른 training time의 측면에서 이러한 saturating nonlinearities는 non-saturating nonlinearity 보다 훨씬 느립니다.<br>

<img src = "https://py-tonic.github.io/images/alexnet/2.PNG">

non-saturating nonlinearity $f(x) = max(0,x)$ 를 ReLU 함수라고 부르겠습니다.<br>

ReLU를 사용하여 Deep CNN을 학습하면 tanh를 사용할 때 보다 학습시간이 훨씬 줄어듭니다.<br>

위 그림은 particular 4-layer convolutional network의 CIFAR-10 데이터 셋에서 25%의 training error에 도달하는데 필요한 반복 횟수를 보여줍니다.<br>

- solid line : ReLU
- dashed line : tanh

물론 이 논문에서 제안한 방법이 CNN에서 전통적인 모델에 대한 대안을 고려한 최초의 방법은 아닙니다.<br>

예를들어 Jarrett이 제안한 nonlinearity <br>

<br>

$f(x)=|tanh(x)|$

<br>

위 함수는 Caltech-101 데이터셋에서 local average pooling과 함께 잘 작동합니다.<br>

하지만 이 데이터 셋에서의 주요 관심사는 Overfitting을 막는것이기 때문에 training set에 대한 좀 더 빠른 훈련속도 능력과는 다릅니다.<br>

### Local Response Normalization

ReLU는 saturation을 방지하기 위해 Input Normalization이 필요하지 않습니다.<br>

적어도 일부 training example이 positive input이라면 학습이 발생합니다.<br>

그러나 여전히 아래와 같은 local normalization scheme가 일반화에 도움이 된다는 것을 알았습니다.<br>

<img src = "https://py-tonic.github.io/images/alexnet/3.PNG">

$a^{i}_{x,y}$ : 위치 (x,y)에서 kernel i를 적용하여 계산된 뉴런의 activity

그런 다음 ReLU를 적용한것이 위 식의 좌변입니다.<br>

이러한 종류의 response normalization은 실제 뉴런에서 발견되는 유형에서 영감을 얻은 측면 억제 형식을 구현하여 다른 커널을 사용하여 계산된 뉴런 출력사이에서 큰 activity에 대한 경쟁을 만듭니다.<br>

- ps :위에 실제 뉴런에서 영감을 얻은 부분은 [Lateral inhibition](https://en.wikipedia.org/wiki/Lateral_inhibition) 이라고 합니다.

즉, 어떤 뉴런이 강하게 활성화 되어있다면, 그 뉴런 주변에 대하여 normalization을 실행합니다.<br>

특정 layer에 ReLU적용 후 Max pooling 수행 전에 이 정규화를 적용합니다.<br>

이 정규화는 위에서 잠깐 언급했던 Jarrett의 contrast normalization scheme와 약간 유사하지만 mean activity를 빼지는 않습니다.<br>

ReLU 는 입력에 비례하여 출력이 그대로 증가가 되는데, 여러 feature map에서의 결과를 normalization 시켜주면 위에서 말한 Lateral inhibition(강한 자극이 주변의 약한 자극이 전달되는 것을 막는 효과)을 통해 Generalization 관점에서 훨씬 좋아지게 된다고 한다.<br>

즉 LRN은 ReLU 때문에 사용하는 것이고,  입력값을 그대로 출력하는 ReLU와 Conv, Max pooling layer의 특성상 매우 높은 한 픽셀값이 주변에 많은 영향을 미치게 될 것이고 이를 방지하기위해 다른 Activation map의 같은 위치에 있는 픽셀 끼리 정규화를 해주는 것입니다.<br>

- ps : 요즘은 BN을쓰지 LRN은 안쓴다고 합니다.

### Overlapping Pooling

<img src = "https://py-tonic.github.io/images/alexnet/6.PNG">

- 출처 : [심교훈님 블로그](https://bskyvision.com/421)

CNN의 Pooling layer는 동일한 kernel map에서 인접한 뉴런 그룹의 output을 요약합니다.<br>

Pooling layer는 Pooling unit의 위치를 중심으로하여 크기가 z * z인 근방을 요약하는 s pixel간격의 Pooling unit의 grid로 구성된것으로 생각할 수 있습니다.<br>

전통적으로는 스트라이드와 커널의 크기가 같도록 max pooling하지만(s=2, z=2) 논문에서는 스트라이드보다 커널의 크기를 늘려서 겹치도록 max pooling 했습니다.(s=2, z=3)<br>

이를 통해 에러를 top-1 : 0.4%, top-5 : 0.3% 감소시켰고, Overlapping이 더 Overfitting하기 힘들다는 것도 관찰했습니다.<br>

## Reducing Overfitting

우리의 모델은 약 6천만개의 parameter를 가집니다.<br>

많은 parameter가 있으므로 Overfitting에 약할 수 있으니 Overfitting에 대응하는 두가지 방법을 소개합니다.<br>

### Data Augmentation

Overfitting에 대응하는 가장 좋은 방법중 하나는 데이터를 많이 모으는 것입니다.<br>

우리는 Label을 보존하면서 데이터 셋을 인위적으로 늘렸습니다.<br>

우리는 아주 작은 계산으로 원본 이미지에서 변환 이미지를 생성해 냈습니다.<br>

두 가지의 뚜렷한 형태의 데이터 확대를 사용하므로 디스크에 저장할 필요가 없습니다.<br>

구현 시 변환된 이미지는 CPU에서 Python 코드로 생성되는 반면 GPU는 이전 이미지 batch를 학습합니다. 따라서 이러한 데이터 증식 체계에는 계산이 필요 없습니다.<br>

데이터를 증식하는 첫 번째 방법은 256x256 원본 이미지를 224x224로 추출하여 수평으로 뒤집어 원본데이터보다 2048배 큰 데이터셋을 얻었습니다.<br>

테스트에서 각 코너와 중앙의 5개의 패치와 그것들을 반전시킨 5개의 패치를 이용하여 softmax층에 넣은 후 평균을 낸 값을 이용했다.<br>

두 번째 방법은 전체 훈련 세트에 대하여 RGB값에 PCA를 수행하고, <br>

각 훈련용 이미지에 여러 구성요소를 추가합니다.<br>

해당 고유값에 비례하는 크기와 각 이미지의 mean = 0, var = 0.1인 정규분포에서 추출한 확률변수를 곱한값으로 위에 구성요소에 추가한다는 뜻..?<br>

따라서, 각 RGB 이미지 pixel $I_{xy} = [I^{R}_{xy}, I^{G}_{xy}, I^{B}_{xy}]$ 에 아래의 quantity를 추가합니다.<br>

<img src = "https://py-tonic.github.io/images/alexnet/4.PNG">

여기서 p와 λ 는 각각 3x3 공분산 행렬의 RGB pixel값의 고유벡터와 고유값입니다.<br>

α 는 위에서 언급한 확률변수입니다.<br>

## Dropout

많은 다른 모델의 예측을 결합하는 것(스태킹이나 앙상블)은 test errors를 줄이는 좋은 방법이지만 훈련하는데 시간이 많이 소요되는 큰 신경망 같은 경우에는 비용이 너무 비쌉니다.<br>

그러나 훈련 중에 약 2배 정도의 비용으로 매우 효율적인 모델을 조합하는 방법이 있습니다.<br>

dropout이라고 불리는 최근에 소개된 기술은 확률이 0.5인 각 숨겨진 뉴런의 출력을 0으로 설정하는 것입니다.<br>

이러한 방식으로 "제거 된" 뉴런은 순방향 패스에 기여하지 않으며 역 전파에도 참여하지 않습니다.<br>

따라서 입력이 제시될 때 마다 신경망은 다른 아키텍처를 샘플링하지만 이러한 모든 아키텍처는 가중치를 공유합니다.<br>

즉, 특정 뉴런이 다른 뉴런들에 의존할 수 없기 때문에 이 기술은 뉴런의 복잡한 공동 적응을 줄입니다.<br>

따라서 다른 뉴런의 많은 다른 임의의 하위 집합과 함께 유용하고 더 강력한 기능을 뉴런들은 배워야 합니다.<br>

테스트 타임에 모든 뉴런을 사용하지만 출력에 0.5를 곱해줍니다.<br>

이는 지수적으로 많은 드롭아웃 네트워크에 의해 생성된 예측 분포의 기하 평균을 취하는 합리적인 근사치 입니다.<br>

첫 두 개의 FCL에 dropout을 적용했고, dropout이 없을 때 네트워크가 부분적으로 overfitting되었습니다. Dropout을 사용하면 학습시키는데 2배의 시간이 걸립니다.<br>



## Details of Learning

우리는 Batchsize 128, momentum 0.9, weight decay 0.0005 의 SGD를 사용하여 모델을 훈련했습니다.<br>

우리는 이 작은 weight decay가 모델이 학습하는데 매우 중요하다는 것을 발견했습니다.<br>

다시 말해서 weight decay는 단순한 regularizer가 아닙니다. 모델의 Training error가 줄어듭니다.<br>

weight $w$의 업데이트 규칙은 아래와 같습니다.<br>

<img src = "https://py-tonic.github.io/images/alexnet/5.PNG">

우리는 표준 편차 0.01인 zero-mean 정규분포에서 각 layer의 가중치를 초기화 했습니다.<br>

또한 상수 1을 사용하여 FCN의 hidden layer 뿐만 아니라 2, 4, 5 번째 conv layer 뉴런의 bias를 초기화 했습니다.<br>

이 초기화는 ReLU에 positive inputs을 제공함으로써 학습의 초기 단계를 가속화 합니다.<br>

나머지 layer들의 뉴런 bias는 0으로 초기화 했습니다.<br>

모든 layer에 대해 동일한 Learning rate를 적용했으며, 훈련 전체에서 수동으로 조정했습니다.<br>

우리가 수행한 heuristic은 validation error가 현재 learning rate와 함께 향상을 멈췄을 때 learning rate를 10으로 나누는 것입니다.<br>

learning rate는 0.01로 초기화 되었으며 종료 전까지 3번 감소하였습니다.<br>

## Discussion

우리의 결과는 deep convolutional neural network이 순수하게 supervised-learning을 사용하여 매우 까다로운 데이터 셋에서 기록적인 결과를 달성 할 수 있음을 보여줍니다.<br>

단일 conv layer를 제거하면 성능이 저하됩니다.<br>

예를 들어, 중간 layer를 제거하면 네트워크의 최고성능에서 약 2%의 손실이 발생합니다.<br>

따라서 네트워크의 깊이는 성능에 매우 큰 영향을 미친다는 것을 알 수 있습니다.<br>

## Reference 

- [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
