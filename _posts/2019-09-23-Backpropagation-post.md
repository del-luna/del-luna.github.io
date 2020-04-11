---
layout: post
title: What is Backpropagation?
date: 2019-09-23 00:00:00
author: Jaeheon Kwon
categories: Ai
tags: [backpropagation]
---

# Backpropagation for vectorized data

기본적으로 Backpropagation의 원리는 아주 단순하다.  

우리가 궁극적으로 궁금해하는 Loss에 대한 어떤 특정 노드의 미분값을 알고 싶은것이다.  

그리고 이는 이전의 노드로 부터 흘러들어온 Upstream Gradients와 현재 노드를 기준으로 구한 local gradinet의 곱으로 이루어 진다.  

아래의 그림을 보자 우리는 $∂L/∂x$가 궁금하다. 우리는 chain rule을 이용해서  

$∂L/∂x = ∂L/∂z * ∂z/∂x$임을 알 수있다.  

여기서 Upstream Gradient = $∂L/∂z$ 이고,  

local gradient = $∂z/∂x$ 가 된다.

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0052.jpg">

위 그림을 자세히 보면 local gradient에 대해서 Jacobian Matrix라고 표현한다.

이는 Jacobian의 특성 때문인데 간략히 말하자면 Jacobian은 어떤 함수를 기준으로 output에 대한 input의 편미분으로 나타나는 행렬이다.

! 자세한 자료는 [Jacobian matrix](https://wikidocs.net/4053)를 참고하세요.  

또 중요한 것중에 하나는 항상 변수와 gradient는 같은 shape를 갖는다.  

아무튼 이러한 간략한 정의를 알고 직접 예제를 풀면서 익혀보자.  

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0062.jpg">

이런 단순한 그래프가있다 우리가 궁금한 부분은 $ ∂(Loss)/∂W, ∂(Loss)/∂x$이다.  

L2 노드에 대한 연산은 Squared 연산이다.  

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0065.jpg">

우리는 $W*x = q$로 표현한다.  

이제 생각을 해보자. W에서의 local gradient는 뭐가될까?  

바로 노드의 output인 q의 W에대한 변화량이 될것이고 수식으로는 $∂q/∂W$가 될것이다.  

아래의 사진에서보이는 $1_k=_ix_j$는 지시함수라는 건데 k와 i의 값이 같으면 $x_j$값을 갖는다고 보면 된다.   

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0066.jpg">  

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0070.jpg">

우리는 또한 Upstream Gradients가 $∂f/∂q = 2q$임을 알 고 있다.  

이를 이용해서 궁극적인 목표인 $∂f/∂W$를 구하면 chain rule을 이용해 위의 그림에서 식처럼 표현할 수 있다.  

여기서 중요한점은 위에서 말햇듯이 우리는 shape에 신경을 써야한다. W의 graident 또한 W와 shape와 같아야 하므로 $∂f/∂W = 2q*x^T$인 전치행렬을 곱해줘야 한다.  

이제 $∂f/∂x$를 구해보자 방법은 위와 동일하다.  

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0071.jpg">

우선 local gradient를 구하고 $∂q/∂x = W_k,_i$  

Upstream Gradients는 위와 동일하게 $∂f/∂q = 2q$ 일 것이다.  

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0072.jpg">

이를 이용해서 chain rule을 적용해보면 $2q*W$가 나온다.  

<img src = "https://py-tonic.github.io/images/backpropagation/cs231n_2017_lecture4_page-0073.jpg">

위에서 했던 것 처럼 shape를 맞춰주기 위해 W의 전치행렬을 앞에 곱해주면 된다.  

혹시나 변수에대해 전치행렬을 이용한다거나 곱하는 순서에대해 모르겠으면 직접 하나하나 component별로 위의 식을 따라가면서 적어보면 이해가 될 것이다.
