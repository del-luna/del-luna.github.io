---
layout: post
title: MLE & CrossEntropy
author: Jaeheon Kwon
categories: 
tags: 
---



working....



# MLE & CrossEntropy

MLE를 공부하기위해 [ratsgo's blog]( https://ratsgo.github.io/statistics/2017/09/23/MLE/ )를 보다보니 흥미로운 내용이 있었습니다.
GAN에서 사용했던 KL-Divergence와 저희가 일반적으로 이진분류에서 사용하는 Cross entropy 그리고 MLE가 모두 관련이 있는 내용이었습니다.<br>
θ = 모수, $P_{model}$ = 모델의 예측결과, $P_{data}$= train_data 라고 할 때 <br>

$θ_{ML}$<br>
<br>
$=argmax_θP_{model}(θ|x) $ <br>
<br>
$=argmax_θ[E_{X:P_{data}}P_{model}(θ|x)]$<br>

위처럼 나타낼 수 있습니다. <br>

KL은 두 확률 분포의 차이를 계산하는데 사용한다고 [GAN]( https://jaeheondev.github.io/GAN-post/ )에서 말씀 드렸었죠?
다시한번 얘기해보자면
KL을 최소화 하는게 모델의 학습 과정으로도 볼 수 있습니다.

<img src = "https://py-tonic.github.io/images/MLECross/0.PNG">

이런식으로 나타낼 수 있습니다.
잠깐 그런데 어디서 많이보던 항이 보이지 않나요?
<img src = "https://py-tonic.github.io/images/MLECross/1.PNG">

이 식 왠지 많이 익숙한데? 맞습니다. 바로 Cross entropy 항이죠.
우리는 $P_{data}$에 대한 항은 고정값이기 때문에 $P_{model}$항만 변화시키면 됩니다.
일반적으로 우리가 사용하는 <br>

$-ylog(p)+ (1-y)-log(1-p)$와 동일한 형태입니다.
이로서 우리는 Cross entropy(KL)최소화와 MLE가 본질적으로 같음을 알 수 있었습니다.

## Reference

[ratsgo's blog]( https://ratsgo.github.io/statistics/2017/09/23/MLE/ )
