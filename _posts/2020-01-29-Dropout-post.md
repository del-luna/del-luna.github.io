---
layout: post
title: Dropout
author: Jaeheon Kwon
categories: Papers
tags: [tech]
---

#  Dropout: A Simple Way to Prevent Neural Networks from Overfitting 



## Abstract

많은 수의 파라미터를 갖는 DNN은 매우 강력한 성능을 낼 수 있습니다.<br>

하지만 파라미터 수의 증가는 곧 **오버피팅**이라는 문제를 야기합니다.<br>

큰 네트워크 일수록 테스트 할 때  여러개의 다른 신경망의 예측 값들을 조합하여 오버피팅에 대처하는 것은 어렵습니다.<br>

드롭 아웃은 이러한 문제를 해결할 수 있습니다.<br>

핵심 아이디어는 신경망을 학습시킬 때 유닛을 무작위로 "Drop" 하는 것입니다.<br>

이러한 방법은 유닛이 과도하게 **co-adapting(서로에게 적응)**하는 것을 막아 줍니다.<br>

신경망을 학습 시키는 동안 드롭 아웃은 서로 다른 "얕은" 신경망으로 부터 샘플링합니다.<br>

신경망을 테스트 하는 동안 더 작은 가중치를 갖는 기존 신경망을 사용 함으로써 서로 다른 "얕은" 신경망의 예측 값들의 평균화 하는 효과를 얻을 수 있습니다.<br>

드롭 아웃을 사용함으로써 오버피팅을 상당히 없앨 수 있고 다른 정규화 방법들 보다 많은 성능 개선을 이룰 수 있습니다.<br>

## Introduction

<img src = "https://py-tonic.github.io/images/dropout/1.PNG">

DNN의 가장 큰 문제중 하나는 바로 오버피팅 입니다.<br>

오버피팅문제를 해결하기 위해 다양한 연구들이 있었습니다.<br>

연산량에 제한을 두지 않는다면, 모델을 정규화하는 가장 좋은 방법은  파라미터의 모든 예측 값들을 학습 데이터에 대한 사후확률(Posterior)에 대해 가중평균 하는 것입니다.<br>

물론 위 방법은 단순하고 작은 모델의 경우 좋은 결과를 낼 수 있습니다.<br>

하지만 저희는 연산량을 줄이는 동시에 Bayesian gold standard의 성능을 향상시키려고 합니다.<br>

이를 위해 파라미터들을 공유하는 모델들의 예측에 동일한 가중치를 갖는 기하평균으로 근사함으로써 이를 수행할 수 있습니다.<br>

모델의 결합은 항상 머신러닝의 성능을 향상시킵니다.<br>

하지만 거대한 신경망에서 개별적으로 분리된 많은 수의 신경망을 평균화 하는 것은 비용이 너무 큽니다.<br>

서로 다른 구조의 모델을 학습시키는 것은 몇 가지 어려움이 있습니다.<br>

연산량의 증가(모델이 느려짐)와 많은 데이터를 필요로 하게 됩니다.

드롭 아웃은 위의 문제들을 해결하기 위해 등장했습니다.<br>

드롭 아웃은 신경망의 뉴런(유닛)을 부분적으로 "Drop" 하는 것을 의미합니다.<br>

유닛이 "Drop"될 확률은 랜덤이고, 논문에서는 유닛이 retained 될 확률을 p로 설정합니다.<br>

드롭 아웃을 적용한 신경망을 학습 시키는 것은 "얕은" 네트워크 여러개를 학습시키는 것과 같습니다.(Ensemble, Stacking..?)<br>

<img src = "https://py-tonic.github.io/images/dropout/2.PNG">

전체 신경망에 뉴런의 개수가 n 개라면 가능한 "얕은" 네트워크의 수는 $2^n$개 입니다.<br>

이 네트워크들은 모두 가중치를 공유합니다.<br>

"얕은" 네트워크는 retained된 뉴런들로 구성된 신경망을 의미합니다.<br>

Test time에서 위에 언급한 많은($2^n$) "얕은" 네트워크들의 예측을 평균화 하는것은 어렵습니다.<br>

하지만 approximate averaging method는 실제로 잘 동작합니다.<br>

한 개의 신경망을 드롭 아웃 없이 테스트에 사용하는 것입니다.<br>

이 네트워크의 가중치들은 학습된 가중치의 scale-down 버전이 됩니다.<br>

신경망을 학습 시킬 때 한 개의 유닛이 p의 확률로 retained 된다면,<br>

Test time에서 **유닛에 연결된 가중치에 p를 곱해줍니다.**(Fig.2)<br>

드롭 아웃이 있는 네트워크를 훈련시키고 Test time에 이러한 approximate averaging을 사용하면 다른 정규화 방법을 사용한 훈련에 비해 광범위한 분류 문제에서 **일반화 오류가 크게 낮아집니다.**<br>

논문에서는 드롭 아웃이 feed-forward 뿐만 아니라 Boltzmann Machine을 비롯한 일반적인 그래프 모델에서도 잘 동작한다고 말합니다.<br>

또한 RBM에서도 드롭 아웃을 적용한 모델이 실험 결과 더 낫다고 말하고 있습니다.<br>

## Motivation

드롭 아웃은 유전학이론으로부터 비롯되었습니다.<br>

유성 생식은 일반적으로 각 부모의 유전자를 절반씩 받아 돌연변이가 추가되어 자식이 생산됩니다.<br>

무성 생식은 함께 동작하는 좋은 유전자 집합이 자손에게 직접 전달될 수 있기 때문에 무성 생식이 individual fitness를 최적화 하는데 더 좋을 수도 있습니다.<br>

반면에 유성 생식은 이러한 co-adapted한 유전자 집합을 파괴할 가능성이 있습니다.<br>

그러나 저희는 이 co-adapted를 파괴하는 것에 초점을 맞춰야 합니다.<br>

유성 생식의 우월성은 장기적으로 자연 선택에 대한 기준이 individual fitness가 아니라 유전자의 혼합 가능성일 수 있다는 것입니다.<br>

유전자 집합이 다른 무작위 유전자 집합과 잘 동작할 수 있도록 하는 능력이 **유전자를 더 강력하게 만듭니다**.<br>

그래서, 유전자는 항상 존재하는 주변의 많은 파트너(위에서 말한 co-adapted한 집합)에 의존하지 않고 자체적으로 또는 소수의 다른 유전자와 협력하여 유용한 것을 수행하는 방법을 배워야 합니다.<br>

**(위의 설명이 Dropout의 핵심)**<br>

마찬가지로 드롭 아웃으로 훈련된 신경망의 각 히든 유닛은 무작위로 선택된 다른 유닛의 샘플을 다루는 법을 배워야 합니다.<br>

히든 유닛을 더욱 견고하게 만들고 실수를 해결하기 위해 다른 유닛에 의존하지 않고 자체적으로 유용한 기능을 생성하도록 유도하게 됩니다.<br>

그러나 레이어 내의 히든 유닛은 여전히 서로 다른 일을 하는 법을 배웁니다.<br>

## Related Work

드롭 아웃은 히든 유닛에 노이즈를 추가하여 신경망을 정규화 하는 방법으로 해석할 수 있습니다.<br>

위와 같이 state of unit에 노이즈를 추가하는 아이디어는 Denoising AutoEncoder(DAEs)에서 사용 되었습니다.<br>

노이즈는 오토인코더의 입력 유닛에 추가되고, 네트워크는 noise-free 입력을 재구성하도록 훈련됩니다.<br>

우리의 연구는 드롭 아웃이 히든 레이어에도 효과적으로 적용될 수 있고, 모델 평균화의 형태로 해석될 수 있음을 보여줌으로써 위 아이디어를 확장합니다.<br>

우리는 노이즈를 추가하는 것이 Unsupervised feature learning에 유용할 뿐만아니라 supervised learning problems에서 확장하는 것이 가능함을 보여줍니다.(위에 오토인코더 얘기 나와서 이런식으로 설명하는듯)<br>

## Model Description

L개의 hidden layer가 있는 신경망을 생각해봅시다.<br>

$z^{(l)}$은 layer $l$에서의 input vector를 나타냅니다.<br>

$y^{(l)}$은 layer $l$에서의 output vector를 나타냅니다.<br>

$W{(l)}$은 layer $l$에서의 가중치입니다.<br>

$b^{(l)}$은 layer $l$에서의 편향입니다.<br>

기본 신경망에서 feed-forward 연산은 아래와 같습니다.<br>

<img src = "https://py-tonic.github.io/images/dropout/3.PNG">

f는 activation function입니다.<br>

드롭 아웃을 적용하면 위의 연산은 아래의 식과 같습니다.<br>

<img src = "https://py-tonic.github.io/images/dropout/4.PNG">

<img src = "https://py-tonic.github.io/images/dropout/5.PNG">

위의 * 연산은 element-wise product 입니다.<br>

$r^{(l)}$은 각각 확률 p가 1일 때 베르누이 독립 확률 변수에서 추출한 벡터 입니다.<br>

ps: 베르누이 확률변수? <br>

동전던지기나 이항분포할 때 나오는 변수<br>

여기서는 유닛의 존재 유무값을 갖는 변수, 유닛이 존재할 확률이 p라면 평균은 p 분산은 p(1-p)값을 갖는다.<br>

수식 전개에 대한 더 자세한 논문은 [paper]( https://www.sciencedirect.com/science/article/pii/S0004370214000216 ) 참조

이 벡터는 샘플링되고 그 layer의 출력과 element-wise로 곱해집니다.<br>

그런 다음 위에 곱해진 결과물이 다음 layer의 입력으로 사용되며 이 프로세스는 각 레이어에 적용됩니다.<br>

이는 더 큰 네트워크에서 하위 네트워크를 샘플링하는 것입니다.<br>

학습을 통해 loss function의 미분값은 하위 네트워크를 통해 backprop 됩니다.<br>

테스트 타임에 가중치는 Fig.2와 같이 $W^{(l)}_{test} = pW^{(l)}$로 조정됩니다.<br>

결과적으로 마지막에는 신경망은 드롭 아웃없이 사용됩니다.<br>

## Learning Dropout Nets

### Backpropagation

드롭 아웃 신경망은 표준 신경망의 방식과 비슷하게 SGD를 사용하여 학습가능합니다.<br>

유일한 차이점은 미니 배치의 training case마다 몇 개의 유닛을 생략함으로써 "얕은" 네트워크를 샘플링 한다는 것입니다.<br>

해당 training case에 대한 forward & backpropagation은 이 "얕은" 네트워크에서만 수행됩니다.<br>

각 파라미터의 gradients는 미니 매치의 training case에 대해 평균화 됩니다.<br>

임의의 training case에 대하여 파라미터를 사용하지 않는 경우에 해당 파라미터의 gradients는 0이 된다.<br>

SGD를 개선하기위한 많은 방법들이 있는데(ex: momentum, L2 weight decay, ...)<br>

우리의 모델에서도 유용한 것을 볼 수 있었습니다.<br>

하나의 특정 형태의 정규화는 드롭 아웃에 특히 유용합니다.<br>

각 히든 유닛에서 들어오는 가중치 벡터의 표준이 고정 상수 c에 의해 상한 되도록 제한합니다.<br>

즉 W가 가중치 벡터라면 W의 L2 norm  ≤   c 입니다.<br>

이 제약은 w가 나올 때마다 반경 c의 공 표면에 W를 투사(사영)하여 최적화하는 동안 부과됩니다.<br>(L2 norm이니까 좌표평면상의 가중치 범위를 원으로 제한하겠다고 볼 수 있을 것 같다.)<br>이것은 임의의 가중치의 norm이 취할 수 있는 최대값이 c임을 암시하기 때문에 max-norm regularization이라고도 부릅니다.<br>

상수 c는 조정 가능한 hyperparameter 인데, validation set을 이용하여 조정합니다.<br>

Max-norm 정규화는 이전에 context of collaborative filtering 에서 사용되었습니다.<br>

드롭 아웃만 사용 상당한 개선 효과를 얻을 수 있지만 , Max-norm과 함께 사용하면 large decaying learning rate와 high momentum이 더 크게 향상됩니다.<br>

고정된 반경의 공 안에 놓이도록 가중치 벡터를 제한하면 가중치가 발산할 가능성 없이 높은 learning rate를 사용할 수 있습니다.<br>

 그리고 dropout으로 인해 노이즈를 주기 때문에, optimization 과정에서 여러 영역을 explore할 수 있습니다.<br>

 learning rate가 decay하면서 점차 minimum에 도달할 수 있습니다. <br>

### Unsupervised Pretraining

신경망은 사전 학습이 가능합니다.<br>

사전 학습은 레이블이 없는 데이터를 사용하는 효과적인 방법입니다.<br>

backprop을 통한 사전 학습 후 finetuning이 특정 경우에 random initialization 보다 성능을 크게 향상시키는 것으로 나타났습니다.<br> 이 기술을 사용하면 미리 훈련된 finetuning net에 드롭 아웃을 적용할 수 있습니다. 사전 학습 절차는 동일하게 유지됩니다.<br>

사전 학습으로 얻은 가중치는 1/p 을 곱해주어야 합니다.<br>

이를 통해 각 유닛에 대해 드롭 아웃 시 예상되는 출력이 사전 학습 중 출력과 동일하게 됩니다.<br>

우리는 처음에 드롭 아웃의 확률적인 특성이 사전 학습된 가중치의 정보를 지울 수 있다고 우려했습니다.<br>

finetuning중에 사용 된 learning rate가 무작위로 초기화 된 네트워크에 대한 최상의 learning rate와 비슷한 경우에 그런 결과가 발생했습니다.<br>

(즉, 더 작은 learning rate를 쓰자..!)<br>

그러나 learning rate가 더 작게 선택되면 사전 학습된 가중치의 정보가 유지되는 것 처럼 보였으며, finetuning시 드롭 아웃을 사용하지 않는 것과 비교하여 최종 일반화 오류 측면에서 개선점을 얻을 수 있었습니다.<br>

## Experimental Results

<img src = "https://py-tonic.github.io/images/dropout/6.PNG">

논문에서 사용한 데이터셋<br>

<img src = "https://py-tonic.github.io/images/dropout/7.PNG">

그냥 봐도 드롭아웃을 적용한게 훨씬 좋은 것을 볼 수 있습니다.<br>

### Comprasion with Standard Regulizers

앞서 말했듯 정규화 기법들은 신경망의 오버피팅을 막기 위해 제안되었습니다.<br>

L2 weight decay, KL-sparsity, Max-norm 정규화 등 다양한 방법이 있습니다.<br>

우리는 Mnist 데이터 셋에 대하여 몇개의 정규화 기법을 사용한 후 비교해봤습니다.<br>

<img src = "https://py-tonic.github.io/images/dropout/m.PNG">

결론적으로는 Max-norm Regularization과 함께 사용했을 때 가장 낮은 일반화 오류를 갖습니다.<br>

## Salient Features

### Effect on Features

<img src = "https://py-tonic.github.io/images/dropout/8.PNG">

일반적인 신경망은 Backpropagation을 통해 gradient를 받아 들이고 파라미터를 변경해가며 loss function을 최적화 합니다.<br>

그러므로 유닛은 다른 유닛들의 실수를 고치는 방식으로 gradient를 변화 시킬 것입니다.<br>

이게 우리가 위에서 설명했던 co-adaptive에 대한 부분입니다.<br>

이 부분이 오버 피팅을 유발하는데 이러한 co-adaptive가 새로운 테스트 데이터에 대해 일반화 능력이 떨어지기 때문입니다.<br>

우리는 각 히든 유닛에 대해 드롭 아웃이 다른 히든 유닛의 존재를 Unreliable 하게 만들어 줌으로써 co-adaptive를 막아준다는 가설을 세웠습니다.<br>

그러므로 히든 유닛은 다른 유닛에 의존하지 않게 되고 아주 다양한 context에서도 잘 동작합니다.<br>

위의 Fig.7_a 를 보면 각 유닛은 자신의 의미있는 특징을 발견하지 못한 것 처럼 보입니다.<br>

반면에 Fig.7_b를 보면 이미지의 edges, strokes, spot 등을 잘 찾은 것 같습니다.<br>

이러한 분이 드롭 아웃이 낮은 일반화 오류를 갖게 하는 가장 큰 이유라고 생각합니다.<br>

### Effect on Sparsity

<img src = "https://py-tonic.github.io/images/dropout/9.PNG">

우리는 드롭 아웃의 부수적인 부분을 발견했는데, 히든 유닛의 activation 값이 Sparse 해지는 것입니다.(심지어 Sparse해지지 않도록 정규화를 거쳐도 Sparse해짐)<br>

Fig.8을 보면 드롭 아웃을 적용한 경우 대부분의 activation 값이 Sparse해짐을 볼 수 있습니다.<br>

### Effect of Dropout Rate

<img src = "https://py-tonic.github.io/images/dropout/10.PNG">

드롭 아웃은 조정 가능한 hyperparameter p를 갖습니다.<br>

다음 두 가지 상황에서 파라미터를 조정하며 비교해보겠습니다.<br>

- 히든 유닛의 수는 상수로 고정
- 히든 유닛의 수는 변경, 드롭 아웃이 상수로 고정된 후에 히든 유닛 수의 기댓값 유지

결론 : p가 0.4~0.8일때 오류가 일정하다. p가 낮을 경우 히든 뉴런의 수를 늘려줬을 때(np가 일정) 에러가 더 낮다. p는 약 0.6일 때 최적이지만 계산의 편의성 때문에 0.5로 사용한다.<br>

### Effect of Data Set Size

<img src = "https://py-tonic.github.io/images/dropout/11.PNG">

데이터 셋의 크기가 1K 이상은 되어야 드롭 아웃의 효과를 볼 수 있다.

데이터 셋의 크기가 너무 커지면 애초에 오버 피팅에 강해지므로 드롭 아웃의 효과가 다시 줄어드는 것을 볼 수 있다.<br>

## Reference

- [Paper]( http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf )
- [luvimperfection]( https://luvimperfection.tistory.com/105 )
