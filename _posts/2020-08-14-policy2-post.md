---
layout: post
title: Policy gradient2
author: Jaeheon Kwon
categories: Ai
tags: [Reinforcement Learning]
---



이제 목적함수 $J(\theta)$를 최대로 만드는 $\theta$를 계산하기 위해 미분해봅시다.

$$\frac{\partial J(\theta)}{\partial \theta} = \nabla_\theta J(\theta) = \nabla_\theta \int_\tau p_\theta(\tau)\sum\limits_{t=0}^T\gamma^tr(x_t,u_t)d\tau$$

$$=\int_\tau \nabla_\theta p_\theta(\tau)\sum\limits_{t=0}^T\gamma^tr(x_t,u_t)d_\tau$$



여기서 한 가지 기믹을 사용합니다.

$$=\int_\tau \frac{p_\theta(\tau)}{p_\theta(\tau)}\nabla_\theta p_\theta(\tau)\sum\limits_{t=0}^T\gamma^tr(x_t,u_t)d_\tau$$

$$=\int_\tau p_\theta(\tau)\nabla_\theta logp_\theta(\tau)\sum\limits_{t=0}^T\gamma^tr(x_t,u_t)d_\tau \tag{1}$$



> 여기서 log기믹을 쓰는 이유는 정확히는 모르겠습니다. 
>
> log로 바꿔서 $p_\theta(\tau)$가 product꼴로 돼있는걸 덧셈으로 바꿔주려는 의도인지, 애초에 $p_\theta(\tau)$이걸 미분하는게 불가능해서 바꿔주는지..?
>
> 아시는분 있으면 댓글 남겨주세요!
>
> *애초에 적분이 붙었던 이유가 기댓값이기 때문인데 확률인 $p_{\theta}$가 미분형식으로 바뀌면 기댓값으로 표현되지 않기 때문에 확률의 의미를 살리기 위함 + 후반에 로그의 덧셈과 미분으로 인한 상수항($\theta$가 붙지 않는 환경에 대한 $p$값)을 지우는 편리성 때문이라고 볼 수 있을 것 같습니다.



저번 시간에 정의한 $p_\theta(\tau) = p(x_0)\prod\limits_{t=0}^T\pi_{\theta}(u_t\vert x_t)p(x_{t+1}\vert x_t,u_t)$를 여기서 쓸 수 있습니다.

$$\nabla_\theta log p_\theta(\tau)=\nabla_\theta log(p(x_0)\prod\limits_{t=0}^T\pi_{\theta}(u_t\vert x_t)p(x_{t+1}\vert x_t,u_t))$$

$$= \nabla_\theta(logp(x_0)+\sum\limits_{t=0}^Tlog\pi_\theta(u_t\vert x_t)+logp(x_{t+1}\vert x_t,u_t))$$

여기서 두 번째 항만 $\theta$에 대한 식이므로 다음과 같이 단순화 할 수 있습니다.

$$\nabla_\theta log p_\theta(\tau)=\sum\limits_{t=0}^Tlog\pi_\theta(u_t\vert x_t)$$



이제 위 식을 (1)에 대입해 봅시다.

$$\nabla_\theta J(\theta)=\int_\tau p_\theta(\tau)(\sum\limits_{t=0}^Tlog\pi_\theta(u_t\vert x_t))(\sum\limits_{t=0}^T\gamma^tr(x_t,u_t))d\tau$$

$$=E_{\tau\sim p_\theta(\tau)}[(\sum\limits_{t=0}^Tlog\pi_\theta(u_t\vert x_t))(\sum\limits_{t=0}^T\gamma^tr(x_t,u_t))]$$

여기서 중요한 점은 기존에 $p$로 정의된 확률 밀도 함수 또는 환경의 동역학 모델인 $p(x_{t+1}\vert x_t,u_t)$가 목적 함수의 미분 식에서 사라졌습니다. 이러한 방법을 Model-Free라고 합니다.



정리해봅시다 강화학습의 목적 함수가 다음과 같을 때,

$$J(\theta)=E_{\tau\sim p_\theta(\tau)}[\sum\limits_{t=0}^T\gamma^tr(x_t,u_t)]$$

목적 함수를 파라미터 $\theta$로 미분한 식, 즉, 목적 함수의 그래디언트는 다음과 같습니다.

$$\nabla_\theta J(\theta)=E_{\tau\sim p_\theta(\tau)}[(\sum\limits_{t=0}^T \nabla_\theta log\pi_\theta(u_t\vert x_t))(\sum\limits_{t=0}^T\gamma^tr(x_t,u_t))]$$

$$=E_{\tau\sim p_\theta(\tau)}[\sum\limits_{t=0}^T(\nabla_\theta log\pi_\theta(u_t\vert x_t)(\sum\limits_{k=0}^T\gamma^kr(x_k,u_k)))]$$

뒤쪽 항이 갑자기 $k$로 바뀌었죠? 왜그런지 생각해봅시다.

위 식의 오른쪽 항은 t=0에서 시작해서 에피소드가 끝날 때 까지 얻을 수 있는 전체 trajectory에 대한 return $G_t$입니다.

타임스텝 $k(t<k)$에서 실행된 policy는 이전 스텝에서의 리워드에 영향을 미치면 안됩니다. 식을 계속해서 전개해봅시다.

$$=E_{\tau\sim p_\theta(\tau)}[\sum\limits_{t=0}^T(\nabla_\theta log\pi_\theta(u_t\vert x_t)(\sum\limits_{k=t}^T\gamma^kr(x_k,u_k)))]$$

$$=E_{\tau\sim p_\theta(\tau)}[\sum\limits_{t=0}^T(\nabla_\theta log\pi_\theta(u_t\vert x_t)(\sum\limits_{k=t}^T\gamma^t\gamma^{k-t} r(x_k,u_k)))]$$

$$=E_{\tau\sim p_\theta(\tau)}[\sum\limits_{t=0}^T(\gamma^t \nabla_\theta log\pi_\theta(u_t\vert x_t)(\sum\limits_{k=t}^T\gamma^{k-t}r(x_k,u_k)))]$$

위식은 그럴 듯 하지만 $\gamma^t$가 계속 곱해져서 그래디언트가 0으로 수렴할 수 있다는 문제점이 있습니다. $\gamma^t=1$로 설정해도 분산이 너무 커지며, 무한 구간 에피소드에서 리워드의 총합이 무한대로 발산한다는 문제점이 존재합니다 그렇다면 실용적인 목적함수의 그래디언트는 무엇일까요?

$$\nabla_\theta J(\theta)=E_{\tau\sim p_\theta(\tau)}[\sum\limits_{t=0}^T(\nabla_\theta log\pi_\theta(u_t\vert x_t)(\sum\limits_{k=t}^T\gamma^{k-t}r(x_k,u_k)))]$$

위와 같이 예정 리워드에만 discout factor를 적용하는 것입니다. 위 식은 초기 목적함수의 그래디언트는 아닙니다. 따라서 편향된 그래디언트로 볼 수 있습니다. 



## REINFORCE

policy gradient를 실제 적용하는 데 있어서 $\tau$상의 기댓값 $E_{\tau\sim p_\theta(\tau)}[\cdot]$는 수학적으로 직접 계산할 수 없으므로 샘플을 이용해 추정합니다.



> 이게 저는 와닿지 않아서 고민을 정말 많이했는데... 사실 아직도 명확하게는 모르겠습니다...
>
> Monte-Carlo 방법으로 생각했습니다.
>
> 나라의 면적을 구하는 방정식은 굉장히 복잡할테고 이를 직접 계산하기 보다는 전체 면적에 점을 찍어서 나라 안에 들어간 샘플의 개수를 통해 확률을 이용한 면적을 추정하는 방식이라고 생각하면 될 것 같습니다.
>
> 점찍기 -> 에피소드 샘플링
>
> 에피소드 샘플링을 통해 나라 면적에 대한 추정 확률을 구하고
>
> 각각의 에피소드에 대한 평균을 통해 근사한다!



$$E_{\tau\sim p_\theta(\tau)}[\cdot] \approx \frac{1}{M}\sum\limits_{m=1}^M[\cdot]$$

위 방법을 통해 목적 함수의 그래디언트를 근사적으로 추정해봅시다.

$$\nabla_\theta J(\theta)\approx \frac1M\sum\limits_{m=1}^M[\sum\limits_{t=0}^T\{\nabla_\theta log\pi_\theta(u_t^{(m)}\vert x_t^{(m)})(\sum\limits_{k=t}^T\gamma^{k-t}r(x_k^{(m)},u_k^{(m)})) \}]$$

> m=1 일 때는 첫 번째 에피소드에 대한 목적 함수의 그래디언트, m=2, m=3,...이런식으로 그래디언트 들의 평균을 구해서 목적 함수의 그래디언트를 근사하겠다!



위 식의 오른쪽 항은 우리에게 익숙한 타임스텝 $k=t$에서 $T$까지의 return $G_t$이므로,

$$\nabla_\theta J(\theta)\approx \frac1M\sum\limits_{m=1}^M[\sum\limits_{t=0}^T\{\nabla_\theta log\pi_\theta(u_t^{(m)}\vert x_t^{(m)})G_t^{(m)} \}]$$

$$=\nabla_\theta \frac1M\sum\limits_{m=1}^M[\sum\limits_{t=0}^T(log\pi_\theta(u_t^{(m)}\vert x_t^{(m)})G_t^{(m)} )]$$

여기서 $G_t^{(m)}$은 $\theta$에 대한 식이 아닌 것을 이용했습니다.



매번 에피소드를 M개 만큼 생성하고 policy를 업데이트할 수도 있지만, 한 개의 에피소드마다 업데이트 할 수도 있습니다.

파라미터 $\theta$로 표현된 policy $\pi_\theta$를 신경망으로 구성할 때 에피소드의 손실 함수는 다음과 같습니다.

$$loss = -\sum\limits_{t=1}^T(log\pi_\theta(u_t^{(m)}\vert x_t^{(m)})G_t^{(m)})$$

손실함수의 구조를 살펴보면 cross entropy에 $G_t$를 곱한 형태임을 알 수 있습니다. 따라서 반환값을 크게 받은 policy의 에피소드는 그래디언트 계산 시 더 큰 영향을 끼치고, 반환값이 작은 policy의 에피소드는 작은 영향을 끼쳐서 policy는 점진적으로 개선됩니다.

REINFORCE 알고리즘의 프로세스를 글로 적으면 다음과 같습니다.

policy $\pi_{\theta_1}$을 에피소드가 종료할 때 까지 실행시켜 $T_1+1$개의 샘플 $(x_t,u_t,r(x_t,u_t),x_{t+1},...)$을 생성하고 이를 바탕으로 policy를 $\pi_{\theta_2}$로 업데이트하고 샘플을 폐기합니다. 그 후 새로운 샘플을 생성하고 학습이 일정 성과에 도달할 때 까지 반복합니다.



- Policy $\pi_{\theta}(u\vert x)$로 부터 샘플 trajectory $\tau^{(m)} = \{x_0^{(i)}, u_0^{(m)},x_1^{(m)},u_1^{(m)},...,x_t^{(m)},u_t^{(m)} \}$를 생성(즉, 에피소드 생성)한다.
- 에피소드에서 반환값 $G_t^{(m)}$를 계산한다.
- 에피소드에서 손실함수를 계산한다. $loss = -\sum\limits_{t=1}^T(log\pi_\theta(u_t^{(m)}\vert x_t^{(m)})G_t^{(m)})$
- Policy 파라미터를 업데이트한다. $\theta\leftarrow \theta+\alpha\nabla_\theta J(\theta)$



REINFORCE 알고리즘은 몇 가지 단점이 존재합니다.

- 한 에피소드가 끝나야 정책 업데이트가 가능하다.(Monte Carlo policy gradient 라고 불림.) 따라서, 에피소드가 긴 경우 학습이 굉장히 오래걸린다.
- 그래디언트 분산이 매우 심하다. 목적 함수의 그래디언트 값은 반환값에 비례하고 에피소드의 길이에 따라 변화량이 상당히 큽니다.
- On-policy 방법이다. 즉, policy를 업데이트 하기 위해서 해당 policy를 실행시켜 발생한 샘플(에피소드)이 필요하므로 효율성이 떨어집니다.



이런 이유로 현재는 잘 사용되지 않는다고 합니다.