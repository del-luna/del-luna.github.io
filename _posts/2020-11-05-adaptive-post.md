---
layout: post
title: The Marginal Value of Adaptive Gradient Methods
in Machine Learning
author: Jaeheon Kwon
categories: Papers
tags: [tech]
---



## Abactract

overparameterized problems인 경우 adaptive(적응형) 방식은 기존의 gradient descent(GD)방식과 다른 솔루션을 가짐.

예를 들어, 선형 분리가 가능한 이진 분류 데이터를 생각해보면 GD방식의 경우 테스트 에러가 0이지만, 적응형 방식의 경우 절반에 가까문 테스트 에러를 얻게됨.(하나의 클래스로만 잘 못 분류하는 해로 수렴)

즉, 적응형 방식의 일반화 성능이 나쁘다는 점이 이 논문에서 말하고자 하는 토픽.

> 종종 기존의 GD방식 이라고 적은 것은 Non-adaptive와 동일한 의미로 사용했습니다.

## Introduction

위에서 말했던 것처럼 적응형 방법은 샘플 외에 일반화에 영향을 미치지 않는 '가짜 특징'에 과도하게 영향을 주는 경향이 있음을 보여줌.

적응형 방법의 일반화 성능이 나쁘다는 것을 보여주는 실험을 통해 세 가지 주요 결과를 찾음

1. 동일한 양의 하이퍼 파라미터 튜닝을 통해 모멘텀이 있는 SGD와 기본 SGD는 평가된 모든 모델 및 태스크의 dev/test 세트에서 적응형 방법을 능가함.(심지어 적응형 방식이 동일한 training loss 혹은 더 낮은 loss를 가진 경우에도 마찬가지.)
2. 적응형 방법은 종종 훈련 세트에서 더 빠르게 수렴하는 듯 보이지만 dev/test 세트에서 빠르게 정체.
3. 적응형 방법을 포함한 모든 방법에 대해 동일한 양의 튜닝이 필요함. 이는 적응형 방법이 튜닝을 덜 필요로 한다는 기존의 연구 결과에대한 도전.



## Background

기존의 최적화 알고리즘

- stochastic gradient methods

    - $w_{k+1}= w_k - \alpha_k \tilde\nabla f(w_k)$

- stochastic momentum methods

    - $w_{k+1} = w_k - \alpha_k \tilde\nabla f(w_k+\gamma_k(w_k - w_{k-1}))+\beta_k(w_k-w_{k-1})$
    - $\gamma_k = 0$ 일 때 Polyak's heavy-ball method(HB)
    - $\gamma_k = \beta_k$일 때 NAG

- Adaptive gradient, Adaptive momentum methods

    - $w_{k+1} = w_k - \alpha_kH^{-1}_k \tilde\nabla f(w_k+\gamma_k(w_k - w_{k-1}))+\beta_kH^{-1}_kH_{k-1}(w_k-w_{k-1})$

    - $H_k := H(w_1,..,w_l)$ is positive definite matrix

    - $H_k = diag((\sum\limits_{i=1}^k \eta_ig_i\circ g_i)^{1/2})$

    - $g_k = \tilde\nabla f(w_k + \gamma(w_k-w_{k-1}))$

        

일반화 라는 것은 솔루션 $w$의 퍼포먼스를 의미함.



## Related Work

- Ma and Belikin : gradient method들은 합리적인 시간 내에 복잡한 솔루션을 찾지 못할 수 도 있음을 보임.
- Hard et al. : SGD가 균일하게 안정적이므로 일반화 성능이 뛰어남을 보임.
- Raginsky et al. : Langevin dynamics가 non-convex 환경에서 기존의 SGD보다 더 나은 일반화성능을 보이는 솔루션을 찾을 수 있음을 보임.



## The potential perils of adaptivity

$$\min\limits_w R_s[w]:=\vert\vert Xw-y\vert\vert_2^2 \tag{1}$$

- $X = n\times d\ matrix$
- $y\ is\ n\ dimension\ vectors$, $\{-1,1\}$
- Non-adaptive methods 는 (1)에 대해 minimum $l_2\ norm$을 갖는 해로 수렴함
    - $w^{SGD} = X^T(XX^T)^{-1}y$

**lemma.1**

$Xsign(X^Ty)=cy$를 만족하는 $scalar\ c$가 존재함. $(w_0=0)$, $X^Ty$에는 0이라는 성분이 존재하지 않음.

$sign(X^Ty)$에 비례하는 $Xw=y$의 해가 존재할 때 마다 adaptive method가 수렴하는 해임.

**Proof.**

논문에서는 귀납법으로 증명.

$$w_k = \lambda_k sign(X^Ty)$$

$k=0$일 때, $\lambda_0=0$

$k\leq t$,

$$\nabla Rs(w_k+\gamma_k(w_k-w_{k-1})) = X^T(X(w_k+\gamma_k(w_k-w_{k-1}))-y) \\ \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \  = X^T\{(\lambda_k + \gamma_k(\lambda_k -\lambda_{k-1}))Xsign(X^Ty)-y\}\\ \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \  = \{(\lambda_k+\gamma_k(\lambda_k-\lambda_{k-1}))c-1\}X^Ty   = \mu_kX^Ty$$



$g_k=\nabla Rs(w_k+\gamma_k(w_k-w_{k-1}))$로 설정.

$H_k = v_kdiag(\vert X^Ty\vert)$

$w_{k+1}=\{\lambda_k - \frac{\alpha_k\mu_k}{v_k}+\frac{\beta_kv_{k-1}}{v_k}(\lambda_k - \lambda_{k-1}) \}sign(X^Ty)$

> 자세한 수식 전개는 논문을 참조하세요.
>
> $k\leq t$일 때 맨 첫번째 수식이 어디서 나온건지 잘 모르겠음..
>
> 첫 번째 $\rightarrow$ 두 번째 수식으로 갈 때는 $w_k=\lambda_k sign(X^Ty)$를 대입.
>
> 두 번째 $\rightarrow$ 세 번째 수식으로 갈 때는 $Xsign(X^Ty)=cy$를 대입.



- 결론은 lemma에 의해 $Xsign(X^Ty)=cy$를 만족하는 $c$가 존재하고, $Xw=y$의 문제에서 $sign(X^Ty)$에 비례하는 해 $w$가 존재할 경우 adaptive method는 언제나 그 해로 수렴한다. 하지만 이러한 해는 일반화 관점에서 좋지 않다.



$i=1,...,n$ 이고, $n$개에 대하여 label $y_i = \{1,-1\}$ 이며 $x$는 infinite dimension vector라고 가정하자.

$$x_{ij} = \begin{cases} y_i\quad j=1\\ 1\quad j=2,3\\ 1\quad j=4+5(i-1),...,4+5(i-1),+2(1-y_i)\\ 0\quad otherwise \end{cases}$$

첫 번째 원소만 잘 뽑아내면 분류가 쉬워지는 태스크이지만, 이런 정보가 없다면 adpative method는 잘 수행하지 못한다는 것을 확인해보자.

$b=\sum\limits_{i=1}^ny_i$,  $b>0$,  $u=X^Ty$로 정의하면 $u$는 다음과 같음.

$$u_j = \begin{cases} n\quad j=1\\ b\quad j=2,3\\ y_j\quad if\ j>3\ and\ x_j=1 \end{cases}$$

$$sign(u_j) = \begin{cases} 1\quad j=1\\ 1\quad j=2,3\\ y_j\quad if\ j>3\ and\ x_j=1 \end{cases}$$

이는 lemma의 모든 조건을 만족 시킴.

$w^{ada},x^{test}$에서 둘 다 0이 아닌 피처는 처음 3가지 dim뿐이고 나머지는 unique한 위치의 값을 가지기 때문에 곱해도 0이다. 따라서 $w^{ada}$는 다음과 같이 처음 보는 데이터에 대해 항상 positive class로 분류하게 됨.



## Conclusions

인트로때 소개한 것과 동일함.

저자는 SGD계열이 adaptive methods들 보다 낫다고 말하고 있지만, GAN이나 RL과 같은 분야의 경우 Adam이나 RMSProp과 같은 adaptive methods가 잘 동작하는 경우가 있고 이에 대해서는 정확한 이유를 알지 못하겠다고 함.

> 추가적으로 Q-learning에 대한 전원석님의 답변
>
> Deep reinforcement learning에서 RMSProp을 쓰는 주된 이유 중에 하나는 현재의 gradient update로 policy가 크게 바뀌면 앞으로 들어올 data가 망가지고 다음 policy에 악영향을 미쳐 전체 학습을 망치게 되는데 이를 방지하기 위함으로 알고 있습니다. 논문에서 최적화 문제가 아니라고 함은 아마 주어진 데이터셋이 있고 loss minimization을 하는 상황이 아니라는 걸 의미하지 않나 생각합니다.
>
> 출처: [재준님 블로그](http://jaejunyoo.blogspot.com/2017/06/marginal-value-of-adaptive-gradient-methods-in-ML2.html)

