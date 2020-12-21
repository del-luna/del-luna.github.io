---
layout: post
title: Policy gradient
author: Jaeheon Kwon
categories: Reinforcement
tags: [Reinforcement Learning]
---



벌써 4 번째 강화학습 포스팅입니다.

이젠 수식이랑 친숙해지셨겠죠? 앞으로 수식위주의 포스팅이 될 것 같습니다.

제가 수식속에 들어간 직관이 이해된다면 최대한 적으려고 노력하겠습니다... :(

> 발표용 ppt
>
> [GoogleDrive](https://drive.google.com/drive/u/0/folders/1eJscbAKj5ImG7kjKOxyOlvklDDBDgzWC)
>
> 저퀄이지만 도움이 됐으면 좋겠습니다 :D



강화학습 문제를 해결하는 여러가지 방법이 존재하지만, 최종 목표는 환경으로 부터 받는 리워드를 최대화하는 optimal policy를 찾는 것입니다.

policy를 파라미터화 해서 문제를 해결하는 방법인 policy gradient를 유도해봅시다.

우선 return들의 기댓값 $J$를 최대화하는 policy를 다음과 같이 나타낼 수 있습니다.

$$\theta^* = argmax(J(\theta))$$

$$J(\theta) = E_{\tau\sim p_\theta(\tau) }[\sum\limits_{t=0}^T \gamma^tr(x_t,u_t)]$$



$r(x_t,u_t)$: 상태 $x_t$에서 액션 $u_t$를 할 때 에이전트가 받는 리워드

$p_\theta(\tau)$: 기댓값 계산시 요구되는 확률 밀도 함수

$\tau$: $(x_0,u_0,...,x_T,u_T)$인 trajectory



policy는 뉴럴넷으로 파라미터화 됩니다. 즉 $\theta = weights$로 볼 수 있습니다.

우선, 베이즈 정리로 $p_\theta(\tau)$를 전개해봅시다.

$$p_\theta(\tau)=p_\theta(x_0,u_0,...,x_T,u_T)$$

$$=p(x_0)p_\theta(u_0,...,x_T,u_T\vert x_0)$$

$$=p(x_0)p_\theta(u_0\vert x_0)p_\theta(x_1,u_1,...,x_T,u_T\vert x_0,u_0)$$

$$=p(x_0)p_\theta(u_0\vert x_0)p(x_1\vert x_0, u_0)p_\theta(u_1\vert x_2,...,x_T,u_T\vert x_0,u_0,x_1)$$

$$...$$

여기서 $p(x_1\vert x_0,u_0)$는 환경에 대한 모델로 policy와 무관하므로 아래첨자가 없습니다. 마르코프 시퀀스 가정에 의해,

$$p_\theta(u_1\vert x_0,u_0,x_1)=\pi_\theta(u_1\vert x_1)$$

$$p(x_2\vert x_0,u_0,x_1,u_1)=p(x_2\vert x_1,u_1)$$

이므로 최종적으로 다음과 같이 쓸 수 있습니다.

$$p_\theta(\tau) = p(x_0)\prod\limits_{t=0}^T\pi_{\theta}(u_t\vert x_t)p(x_{t+1}\vert x_t,u_t)$$



목적 함수 $J$를 state-value function으로 나타내봅시다.

$$J(\theta)=E_{\tau\sim p_\theta(\tau)}[\sum\limits_{t=0}^T\gamma^tr(x_t,u_t)] $$

$$\int_\tau p_\theta(\tau)(\sum\limits_{t=0}^T\gamma^tr(x_t,u_t))d\tau \tag{1}$$



여기서 궤적$(\tau)$을 두 영역으로 분할해 봅시다.

$$\tau = (x_0,u_0,x_1,u_1,...,x_T,u_T)$$

$$=(x_0)\cup(u_0,x_1,u_1,...,x_T,u_T)$$

$$=(x_0)\cup \tau_{u_0:u_T}$$

 state $x_0$에서 시작하여 policy를 따르는 형태로 나타낼 수 있습니다.

베이즈 정리에 의하여,

$$p_\theta(\tau)=p_\theta(x_0,\tau_{u_0:u_T})$$

$$=p(x_0)p(\tau_{u_0:u_T}\vert x_0)$$



이제 $p_\theta(\tau)$를 식 (1)에 대입해봅시다.

$$J(\theta)=\int_{x_0}\int_{\tau_{u_0:u_T}}p_\theta(x_0,\tau_{u_0:u_T})(\sum\limits_{t=0}^T\gamma^tr(x_t,u_t))d\tau_{u_0:u_T}dx_0$$

$$=\int_{x_0}\int_{\tau_{u_0:u_T}}p(x_0)p_\theta(\tau_{u_0:u_T}\vert x_0)(\sum\limits_{t=0}^T\gamma^tr(x_t,u_t))d\tau_{u_0:u_T}dx_0$$

$$=\int_{x_0}[\int_{\tau_{u_0:u_T}}p_\theta(\tau_{u_0:u_T}\vert x_0)(\sum\limits_{t=0}^T\gamma^tr(x_t,u_t))d\tau_{u_0:u_T}]p(x_0)dx_0$$

대괄호 안의 식은 친숙하시죠? State-value function $V^\pi(x_0)$입니다.

> 혹시 식이랑 친하지 않으실까봐 적어 보겠습니다.
>
> 일반적인 state-value funtion은 return Gt의 기댓값이죠?
>
> $V^{\pi}(x_t) = E_{\tau_{u_t:u_T}\sim p(\tau_{u_t:u_T}\vert x_t)}[\sum\limits_{k=t}^T \gamma^{k-t}r(x_k,u_k)\vert x_t]$
>
> $=\int_{\tau_{u_t:u_T}}(\sum\limits_{k=t}^T\gamma^{k-t}r(x_k,u_k))p(\tau_{u_t:u_T}\vert x_t)d\tau_{u_t:u_T}$



그래서 다음과 같이 간단히 표현 가능합니다.

$$J(\theta)=\int_{x_0}V^{\pi_0}(x_0)p(x_0)dx_0$$

$$=E_{x_0\sim p(x_0)}[V^{\pi_0}(x_0)]$$



즉, 목적 함수는 초기 상태 $x_0$에 대한 state-value의 기댓값입니다.

