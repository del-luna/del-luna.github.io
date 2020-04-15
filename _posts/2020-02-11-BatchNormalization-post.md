---
layout: post
title: Batch Normalization
author: Jaeheon Kwon
categories: Papers
tags: [tech]
---

#  Batch Normalization: Accelerating Deep Network Training b y Reducing Internal Covariate Shift 

## Abstract

DNN의 학습은 이전 레이어의 파라미터가 변경됨에 따라 학습 중에 레이어의 입력 분포가 변경되므로 학습이 어렵고 복잡합니다.<br>

DNN을 학습 시키면서 레이어의 입력분포가 계속 변경되므로 낮은 학습 속도와 신중한 파라미터 초기화를 요구하게됩니다.

 이는  학습을 늦추고 saturating non-linearities로 모델을 학습 시키는 것을 어렵게 만듭니다.<br>

우리는 이 현상을 internal covariate shift라고 부르며 레이어의 입력을 정규화하여 이 문제를 해결합니다.<br>

우리의 방법은 정규화를 모델 아키텍처의 일부로 만들고 각 학습용 mini-batch에 대해 정규화를 수행하는 것의 장점을 이끌어냅니다.<br>

BN을 통해 훨씬 높은 학습 속도를 사용하고 초기화에 대해 덜 민감할 수 있습니다.<br>

또한 일부 경우 드롭 아웃이 필요없는 regularizer의 역할을 합니다.<br>

SOTA 모델에 적용되는 BN은 14배 더 적은 step으로 동일한 정확도를 달성하고 원본 모델을 능가합니다.<br>

BN이 적용된 네트워크의 앙상블을 사용하여 ImageNet 문제에 대해 최상의 결과를 개선합니다.<br>

## Introduction

딥러닝은 다양한 분야에서 드라마틱한 성능을 내고 있습니다.<br>

SGD는 DNN을 학습하는 좋은 방법으로 검증이 되었습니다.<br>

보통 각 트레이닝 iter마다 미니 배치단위로 학습이 이뤄지며,

한 번에 하나의 예시가 아닌 미니 배치를 사용하면 여러 가지 방법으로 도움이 됩니다.<br>

첫 째 미니 배치에 대한 손실의 기울기는 훈련 세트에 대한 기울기의 추정치 이며 배치크기가 증가함에 따라 품질이 향상됩니다.<br>

둘 째 배치에 대한 계산은 병렬 처리로 인해 개별 예제의 m개 계산보다 훨씬 효율적일 수 있습니다.<br>

Stochastic gradients는 간단하고 효과적이지만 모델 하이퍼 파라미터, 특히 최적화에 사용되는 학습 속도와 모델 파라미터의 초기 값을 신중하게 조정해야 합니다.<br>

각 레이어에 대한 입력이 모든 이전 레이어의 파라미터에 의해 영향을 받기 때문에 학습은 복잡합니다.<br>

네트워크 파라미터에 대한 작은 변화는 네트워크가 깊어질수록 증폭됩니다.<br>

레이어 입력의 분포 변화는 레이어가 새로운 분포에 지속적으로 적응해야 하기 때문에 문제가 됩니다.<br>

학습 시스템에 대한 입력 분포가 변하면 covariate shift가 발생합니다.<br>

이 문제를 domain adaptation을 통해 해결하려는 시도가 있었습니다.<br>

그러나 covariate shift의 개념은 학습 시스템 전체를 넘어서 하위 네트워크 또는 레이어와 같은 부분에 적용할 수 있습니다.<br>

하위 네트워크로의 입력의 고정된 분포는 하위 네트워크의 외부 레이어에도 긍정적인 결과를 가져옵니다.<br>

> 사실 당연한 얘기죠?
>
> 이전 레이어의 출력은 다음 레이어의 입력이니까 그 입력이 일정한 분포를 따른다는 얘기는 레이어가 새로운 분포에 적응을 할 필요도 없고 그러므로 위에서 말한 Covariate shift도 발생하지 않을테니 안정적일 것이고 안정적이다 라는 뜻은 더 높은 learning rate를 사용 가능하게 합니다. 

sigmoid activation function z = g(Wu + b)를 생각해봅시다.<br>

g(x)가 sigmoid 이므로,<br>

g'(x)는 x의 값이 커지거나 작아질수록 0으로 수렴하는 경향을 보입니다.<br>

다시말해서 input의 값이 너무 크거나 작으면 이 모델은 느리게 훈련 될것입니다.(gradient vanishing)<br>

그러나 x = Wu + b에서 x는 W,b 및 아래 모든 레이어의 파라미터에 영향을 받기 때문에 훈련 중에 해당 파라미터를 변경하면 x의 많은 차원이 saturated regime of the non-linearity로 이동하고 수렴이 느려질 수 있습니다.<br>

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0052.jpg">

> Backpropagation graph를 떠올려 봅시다. 맨 뒤쪽에서부터  $∂L/∂z$ 가 넘어올텐데 이는 이전 레이어의 input인 x의 gradient에 영향을 미칩니다(chain rule에 의해 계속해서 곱해지겠죠?)
>
> x가 커진다는 것은 결론적으론 왼쪽으로 계속 전달되어야 하는 gradient flow를 감소시키는 결과를 초래하고 이는 학습의 지연으로 이어집니다.

당연하지만 위 효과는 네트워크 깊이에 따라 증폭됩니다.<br>

saturation 문제와 그에 따른 vanishing gradients는 일반적으로 ReLU function과 careful initialization 그리고 작은 학습률로 해결할 수 있습니다.<br>

그러나 non-linearity 입력의 분포가 네트워크 학습 처럼 안정적으로 유지될 수 있다면 옵티마이저는 saturated regime에 갇히지 않을 것이며 훈련은 가속화될 것입니다.<br>

ISC(internal covariate shift)를 제거하면 더 빠른 훈련을 가능하게 합니다.<br>

우리는 BN이라는 새로운 메커니즘을 제안하며 BN을 통해 ISC를 제거하고, DNN의 훈련을 극적으로 가속화합니다.<br>

BN은 레이어의 입력을 평균과 분산을 통해 정규화 함으로써 이를 수행합니다.<br>

BN은 파라미터 또는 초기 값에 대한 기울기의 의존을 줄임으로서 네트워크를 통한 gradient flow에 유리한 영향을 줍니다.<br>

이를 통해 발산 위험이 없는 훨씬 높은 학습속도를 사용할 수 있습니다.<br>

또한 BN은 모델을 정규화하고 드롭 아웃의 필요성을 줄입니다.<br>

마지막으로 BN을 통해 saturated modes에서 네트워크가 멈추지 않도록 하여 saturating non-linearity를 사용할 수 있습니다.<br>

## Towards Reduction Internal Covariate Shift

학습의 개선을 위해 ISC를 줄이려고 합니다.<br>

학습이 진행됨에 따라 레이어 입력 x의 분포를 고정함으로써 훈련 속도를 향상시킬 것으로 기대합니다.<br>

각 층이 아래 층에 의해 생성된 입력을 관찰할 때, 각 층의 입력의 동일한 whitening(maybe N(0,1)..?)을 달성하는 것이 유리할 것입니다.<br>

각 층에 대한 입력을 whitening함으로써 ISC의 악영향을 제거하고 입력의 고정된 분포를 달성하는 단계를 밟을 것입니다.<br>

네트워크를 직접 수정하거나 네트워크 활성화 값에 따라 최적화 알고리즘의 매개 변수를 변경하여 모든 학습 step 또는 일정 간격마다 whitening activation을 고려할 수 있습니다.<br>

그러나 이러한 수정들이 최적화 단계들에 산재된다면, gradient descent step은 정규화가 업데이트되어야 하는 방식으로 파라미터들을 업데이트하려고 시도할 수 있으며 이는 gradient step의 영향을 감소시킵니다.<br>

> 아마도 위에서 말하는 간단한 정규화(whitening)을 사용하게 된다면 아래와 같은 절차를 따를 것이다.
>
> 기존의 activation function $a_r = wx + b$라고 하자. 
>
> 1. 각각의 mini-batch의 mean과 var로 표준화한 activation을 다음 hidden layer의 입력으로 사용한다(이러면 입력이 고정 분포가 되니까)
>
> 2. Gradient descent를 위해 계산을 해보자 $∆b∝ -∂l/∂x$ (backpropagation 해보면 나온다.)
>
> 3. 편향을 업데이트 $b ← b+∆b$
>
> 4. 업데이트 후 activation $a_r = wx + (b+∆b)$
>
> 5. 위에서 말한 whitening(?), Centering(?) 후 activation $a^{'}_r{center} = a'r - E(a'r)$
>
>    $ = {(wx+b) +∆b } - {E[wx+b]+∆b}$
>
>    $ = (wx+b) - E[wx+b]$
>
>    위와 같이 $∆b$가 무시되는 문제가 발생한다. 
>    
> 6. 출력인 $a^{'}_r{center}$ 가 계속 그대로이니 loss도 그대로고.. b는 계속 $∆b$가 더해져서 발산해버린다.
>

<br>

단순히 특정 레이어의 output을 whitening하는 접근은 최적화 과정에서 문제가 생깁니다.

최적화에서 whitening 과정을 고려하지 않기 때문입니다.

이 문제를 해결하기 위해 모든 파라미터 값에 대해 네트워크가 항상 원하는 분포로 activation(출력)을 생성하도록 하고 싶습니다.<br>

그렇게하면 모델 파라미터에 대한 loss의 gradient가 정규화를 고려하고 모든 파라미터  θ에 대한 의존성을 고려할 수 있습니다.<br>

## Normalization via Mini-Batch Statistics

$x$ 는 특정 레이어의 입력, $\mathcal{X}$ 는 전체 데이터셋에 해당하는 집합이라고 합시다.

$\hat x = Norm (x,\mathcal{X}) $ 로 쓸 수 있습니다.

normalization은 주어진 training example $x$뿐만 아니라 전체 example $\mathcal {X}$에도 종속적인 함수입니다.

어쨋든 GD를 사용하려면 이 normalization layer를 backpropagation 과정을 통해 gradient를 계산해야 합니다.

$∂Norm(x,\mathcal{X})/∂x$ and $∂Norm(x,\mathcal{X})/∂\mathcal{X}$ 로 Jacobian행렬 형태입니다.<br>

<img src = "https://py-tonic.github.io/images/Batch_normalization/5.PNG">

각 레이어 입력의 전체 whitening은 비용이 많이 들고 어디에서나 차별화 할 수 없기 때문에 두 가지 단순화 과정을 수행합니다.<br>

첫 번째는 레이어의 입력과 출력의 피쳐를 공동으로 whitening하는 대신 mean = 0, var = 1이 되도록 하여 각 스칼라 피쳐를 독립적으로 정규화 한다는 것입니다.<br>

d 차원 입력 $x = (x^{(1)}, ..., x^{(d)})$ 가 있는 레이어의 경우 각 차원을 정규화 합니다.<br>

<img src = "https://py-tonic.github.io/images/Batch_normalization/1.PNG">

> 각 차원별로 독립적 정규화를 수행하는 의미가 뭘까..?<br>
>
> $ X^=Cov(X)^{−1/2}X,$<br>
>
> $Cov(X)=E[(X−E[X])(X−E[X])⊤]$<br>
>
> 논문을 읽다보면 처음 whitening에 대한 설명을 위처럼 하는데..<br>
>
> 위 식은 계산 비용이 매우 많이든다.(inverse square root)<br>
>
> 위 식은 모든 피쳐가 correlated 하다는 가정이지만, 만약 우리의 가정처럼 독립으로 한다면 단순 계산으로 normalize 할 수 있게 된다고 한다.<br>
>
> 또한 이렇게 독립으로 학습하는 경우 각각의 관계가 중요한 경우 제대로 학습을 하지 못할 수 있으므로 이를 방지하기위한 linear transform을 각 차원 k 마다 학습시켜준다.
>

레이어의 각 입력을 단순히 정규화하면 레이어가 나타낼 수 있는 내용이 변경될 수 있습니다.<br>

예를 들어, sigmoid의 입력을 정규화 하면 선형 영역으로 제한 됩니다.

> 아마 입력이 정규화되면 sigmoid 함수의 0에 가까운 linear한 부분에 몰린다는 얘기인 것 같다.(이러면 비 선형성이 줄어서 activation의 제대로된 기능을 수행하지 못할 것 같음)
>

이를 해결하기 위해 네트워크에 삽입된 변환이 identity transform(무조건적인 제한이 아닌 입력을 그대로 내보내는 변환)을 나타낼 수 있는지 확인합니다.<br>

이를 위해 각 activation $x^{(k)}$에 대해 정규화 된 값을 스케일링 하고 이동시키는 파라미터 쌍  $γ^{(k)} , β^{(k)}$ 을 제안합니다. <br>

<img src = "https://py-tonic.github.io/images/Batch_normalization/2.PNG">

**이 파라미터는 기존의 모델 파라미터와 함께 학습되며 네트워크의 표현력을 복원합니다.**<br>

> 이게 아까 위에서 말했던 linear form 인 듯 하다.
>
> 위 식의 장점은 감마와 베타값을 통해 표준화를 취소할 수도 있고,<br>베타가 bias처럼 행동하는데 기존의 whitening과는 달리 업데이트해도 사라지 않는다.<br>
>
> activation 값을 적당한 크기로 유지하기 때문에 vanishing 현상을 어느정도 막아준다.<br>
>
>  입력 분포가 안정되므로 더 빠른 학습을 가능하게 한다.<br>

우리는 activation을 정규화 하기 위해 전체 데이터 셋을 다 사용할 수도 있지만,

이는 stochastic optimization을 사용할 때 그다지 실용적이지 않습니다.

그래서 우리는 두 번째 단순화 가정을 세웁니다.

우리가 SGD를 할 때 미니 배치를 사용하기 때문에,

각 미니 배치마다 activation의 mean과 var을 계산해 사용하도록 합니다.

(normalization이 each activation independently에 적용되므로..)

$\gamma^{(k)} =  √var[x^{(k)}]$<br>

$\beta^{(k)}=E[x^{(k)}]$<br>

<img src = "https://py-tonic.github.io/images/Batch_normalization/3.PNG">

normalize 까지는 whitening과 동일한 것 같고

**미니 배치별로 각 차원에 대해 정규화 하는 것을 잊지말자.**

linear form을 적용시킨 것이 BN이다.(감마와 베타가 각각 scale, shift factor)<br>

$ y = \gamma\hat x +\beta$는 감마와 베타가 학습 가능한 파라미터라는 것을 보여주기위한 표기이다.

감마와 베타값에의해 변환된 y값은 다른 네트워크 레이어에 제공될 수 있습니다. 

<img src = "https://py-tonic.github.io/images/Batch_normalization/4.PNG">

Training 시에는 mini-batch의 mean과 var로 normalize하고<br>

Test 시에는 계산해놓은 파라미터들의 평균을 이용하여 normalize한다.<br>

## Batch-Normalized Convolutional Networks

z = g(Wu + b)<br>

x = Wu + b를 정규화 하여 비선형 레이어 직전에 BN변환을 추가해봅시다.<br>

우리는 레이어의 입력인 u를 정규화 할 수 있지만, u도 마찬가지로 다른 비선형 레이어의 출력일 가능성이 높기 때문에, 훈련 중에 분포의 모양이 변경될 수 있으며 첫 번째와 두 번째를 moment를 제한하면 Covariate shift가 제거되지 않습니다.<br>

반면 Wu + b는 대칭적이고 non-saprse distribution이며 더 가우시안 분포에 가깝습니다.<br>

가우시안 분포는 안정적인 분포로 활성화를 제공할 가능성이 높습니다.<br>

위 식은 z = g(BN(Wu))로 나타낼 수 있습니다.<br>

(b 가 빠지는건 어차피 mean 을 뺄 때도 빠지고 Alg.1에서 베타가 그 역할을 대신함)<br>

BN 변환은 차원당 별도로 학습된 파라미터 베타와 감마를 사용하여 x = Wu의 각 차원에 독립적으로 적용됩니다.<br>

Conv layer의 경우 정규화가 convolution의 속성을 따르기를 원합니다.<br>

서로 다른 위치에 있는 동일한 피쳐 맵의 다른 요소가 동일한 방식으로 정규화 되도록 합니다.<br>

> 하나의 feature map이 같은 weigth와 bias를 공유하고 있으니까 해당영역 전체를 하나로 normalize 하는 것 같음
>

이를 위해 모든 위치에 걸쳐 모든 배치를 미니 배치로 joint 정규화 합니다.<br>
