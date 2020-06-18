---
layout: post
title: Information Theory
author: Jaeheon Kwon
categories: Mathematics
tags: [information]
---

# Information Theory



자주 나오는 개념이라 정리용으로 아래의 레퍼런스를 복붙한 수준입니다.

자세한 정보는 레퍼런스를 참고해주세요.



정보라는 개념을 "학습에 있어 필요한 놀람의 정도(degree of surprise)"로 해석하자.

잘 일어날 것 같지 않은 사건을 관찰하는 경우, 빈번하게 일어나는 사건보다 더 많은 정보를 취득했다고 고려하는 것이다.

따라서 항상 발생하는 일이라면, 사건 발생 후 얻는 정보의 양은 0이다.



정보가 이런 속성을 가지기 때문에 결국 확률에 종속적인 모양이 된다.

정보의 양을 $h(x)$라고 정의하면 결국 정보는 확률 함수의 조합으로 표현이 될 것이다.

$h(x) = -log_2p(x) \tag{1}$

$p(x)$는 확률값이므로 0~1사이의 값이고 자주 일어나지 않는 사건 즉, $p(x)$가 작을 수록 정보의 양 $h(x)$가 커져야 하므로 음의 로그 값으로 볼 수 있다.

base는 어떤 값인지 상관 없지만 기본 단위 (bit)를 고려해서 2를 사용하자.

이제 랜덤 변수 하나를 송신자가 수신자에게 전달한다고 가정하자.

이때 전송되는 데이터 양의 평균은?

$H[x] = -\sum p(x)log_2p(x) \tag{2}$

이 식을 **엔트로피(entropy)**라고 정의한다.

엔트로피는 평균 정보량을 의미하며(기댓값의 정의 그 자체니까) $p(x)$인 분포에서 $h(x)$함수의 기대값을 의미하게 된다.

확률 변수 $x$가 8개의 가능한 값을 가지는 경우를 생각해보자.

각각의 경우 발생할 활률이 모두 동일하게 $\frac{1}{8}$인경우 하나의 데이터 $x$를 전송하기 위해 필요한 평균 비트 수는 3이 된다.

$H[x] = -8*\frac{1}{8}log_2\frac{1}{8} = 3bits \tag{3}$

-  현재 확률 분포가 Uniform이므로 실제 데이터를 표현할 때 동일한 정보량을 가지는 형태로 표현됨.
- 앞서 언급한데로 bit단위 정보량을 표현하므로 임의의 데이터 한개를 전송하기 위한 평균 bit량은 3이 된다.

다른 예시를 들어 보자.

각각의 경우가 발생할 확률이 ($\frac{1}{2},\frac{1}{4},\frac{1}{8},\frac{1}{16},\frac{1}{64},\frac{1}{64},\frac{1}{64},\frac{1}{64}$) 이면 필요한 비트 수는?

$H[x] = -\frac{1}{2}log_2\frac{1}{2} - -\frac{1}{4}log_2\frac{1}{4}-\frac{1}{8}log_2\frac{1}{8}-\frac{1}{16}log_2\frac{1}{16}-\frac{4}{64}log_2\frac{1}{64} = 2bits$

- 결론 : Non-Uniform이 Uniform보다 엔트로피가 낮다.

> Why?
>
> 쉽게 생각해보자. Uniform하다는 말은 모든 확률이 동일하다는 뜻이고 말 그대로 결과를 예측하기가 더 어렵다는 뜻.
>
> 결과를 예측하기 어렵다? -> 새로운 정보가 나타난다!



## KL-Divergence

<hr>

ML/DL을 공부해본 사람이라면 익숙한 단어일듯 하다.

정확한 형태를 모르는 확률 분포 $p(x)$가 있다고 해보자, 그리고 이 확률 분포를 최대한 근사한 $q(x)$가 존재한다.

그럼 당연히 데이터의 실 분포는 $p(x)$이고, 우리가 예측한 분포는 $q(x)$일 것이다.

이제 해당 데이터를 $q(x)$의 코딩 스킴으로 인코딩해서 데이터를 전송한다고 해보자.

- 이러면 이 데이터의 실 분포인 $p(x)$에 의해 얻을 수 있는 정보량은 다를 것이다.
- $p(x)$가 아닌 $q(x)$를 사용했기 때문에 추가적으로 필요한 정보량의 기댓값을 정의해보자.
- 단 이때 정보의 양은 위에서 처럼 bit가 아닌 nat을 쓴다.($log_2$대신 $ln$)

$KL(p\vert\vert q) = - \int p(x)lnq(x)dx - (-\int p(x)lnp(x)dx) = - \int p(x)ln\frac{q(x)}{p(x)}dx \tag{4}$

근사 분포인 $q(x)$를 사용했기 때문에 정보량은 $-lnq(x)$를 사용한다.

하지만 데이터의 실 분포는 $p(x)$이므로 기댓값은 실 분포를 대상으로 구하게 된다.

- $KL(p\vert\vert q) \neq KL(q\vert\vert p)$
- $KL \geq 0$
- $p(x) = q(x)$이면 $KL(p\vert\vert q)=0$

이를 증명하기 위해 **convex** 개념을 살펴보자.

어떤 함수 $f(x)$ 내에서 임의의 두 점$(a,b)$ 사이의 chord(직선)가 함수 $f(x)$와 같거나 혹은 더 위쪽으로 형성된다면 이를 convex라고 한다.

- 임의의 구간 $x=a, x=b$를 정한다 $(a<b)$
- 그 사이의 임의의 $x_λ$는 $x_λ = λa +(1-λ)b$로 정할 수 있다.

그러면 아래의 식이 성립한다.

$f(λa + (1-λ)b) \leq λf(a) + (1-λ)f(b)  \tag{5}$

위 식을 일반화하면,

$f(\sumλ_ix_i) \leq \sumλ_if(x_i)  \tag{6}$

여기서 모든 $x_i$의 집합에 대해 $λ_i \geq 0$이고 $\sum_iλ_i = 1$을 만족한다.

이를 **Jensen's inequality**라고 한다.

위 식에서 $λ_i$를 확률 분포라고 고려하면 식을 다음과 같이 정리할 수 있다.

이산 변수일 경우.

$f[E[x]] \leq E[f(x)] \tag{7}$

연속 변수 일경우.

$f(\int xp(x)dx) \leq \int f(x)p(x)dx \tag{8}$



## KLD & CrossEntropy

<hr>

이산 변수에 대한 CrossEntropy는 다음과 같이 정의됩니다.

$H(P,Q) = E[-logQ(x)] = -\sum P(x)logQ(x) \tag{9}$

풀어서 쓰면 아래와 같습니다.

$D_{KL}(P\vert\vert Q)$

$= - \sum P(x)log(\frac{Q(x)}{P(X)})$

$= -\sum P(x){logQ(x) - logP(x)}$

$= - \sum\{P(x)logQ(x) - P(x)logP(x)\}$

$= - \sum P(x)logQ(x) + \sum P(x)logP(x)$

$= H(P,Q) - H(P)$

이를 크로스 엔트로피 기준으로 다시 정리하면 아래와 같습니다.

$H(P,Q) = H(P) + D_{KL}(P\vert\vert Q)$



## Reference

[ratsgo]( https://ratsgo.github.io/statistics/2017/09/22/information/ )

[PRML]( http://norman3.github.io/prml/docs/chapter01/6 )