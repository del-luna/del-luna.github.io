---
layout: post
title: Planning by Dynamic Programming
author: Jaeheon Kwon
categories: Ai
tags: [Reinforcement Learning]
---



MDP에 대한 모든 정보를 알고 있다고 해도 최적의 policy를 구하는 것은 쉽지 않은 문제입니다.

planning은 environment에 대한 모델이(Model based) 있을 때 최적의 policy를 구하는 것을 뜻합니다.

그 방법론으로 저희 타이틀 처럼 Dynamic Programming을 사용합니다.



## What is Dynamic Programming?

직관적으로 말하면, 큰 문제를 작은 문제로 나눠서 푸는 것!

작은 문제를 풀어서 솔루션들을 찾고 그 솔루션들을 모아서 큰 문제를 푸는 것입니다.

Dynamic Programming이 되기 위한 조건

- Optimal substructure
    - 작은 문제로 쪼개서 각각을 최적화 해도 기존의 문제를 최적화 할 수 있다.
- Overlapping subproblems
    - 작은 문제들을에 대한 결과값을 저장, 재사용 가능하다.

MDP가 위의 두 가지 조건을 모두 만족합니다.

> 벨만 방정식이 recursive하게 엮여 있고, 
>
> Value function이 작은 문제들의 해 -> 저장했다가 reuse 가능하다! -> Overlapping subproblems 

강화학습에서 문제를 푼다는 것은 두 가지 경우로 나눌 수 있습니다.

- For Prediction
    - Value function을 학습하는 것
    - MDP에서 Policy가 주어졌을  value function을 계산하는 것
- For Control
    - MDP가 주어졌을 때 Optimal Policy를 찾는 것

DP에 대한 좀 더 자세한 글은 [여기](https://norman3.github.io/rl/docs/chapter02)를 참고하세요

## Iterative Policy Evaluation

policy를 따라갔을 때 return을 얼마나 받을 수 있을까? 에 대한 문제입니다.

벨만 기댓값 방정식을 이용해서 Iterative하게 적용해봅시다!

- 모든 state에 대해 V를 초기화
- $v_1 \rightarrow v_2 \rightarrow ... \rightarrow v_\pi$ iterative하게 $\pi$에 수렴하도록 한다. 
- Using Synchronous backups,
    - At each iteration
    - For all states $s\in S$
    - Update $v_{k+1}(s)$ from  $v_k(s')$

매 iteration 마다 조금 더 정확한 $v_k$를 얻는 것이 목적입니다.

그럼 어떻게 iteration마다 update를 하는 걸까요?

저번 챕터에 나온 식을 활용합니다.



$$v_{k+1} = \sum\limits_{a\in A}\pi(a\vert s)(R_s^a + \gamma \sum\limits_{s' \in S}P^a_{ss'}v_k(s'))$$

$$v^{k+1} = R^\pi + \gamma P^{\pi}v^k$$



우리의 궁극적인 목적은 각 state에서 true value function인 $v_\pi(s)$입니다. 현재의 policy가 괜찮은지 아닌지를 판단하기 위해서 value function을 사용하는데, 최초의 state-value function은 true value function이 아닙니다. 현재 Policy에 대한 각 state의 경중을 에이전트는 알 수 없기 때문에 왔다 갔다 하면서 계속 업데이트를 통해 정확한 true value function을 찾는 것입니다.

> 정확한 값에 도달하면 더 이상 값이 바뀌지 않는다고 합니다.



<img src = "https://py-tonic.github.io/images/rl/3.png">



너무 추상적이니까 그림을 통해 이해해봅시다.

<img src = "https://py-tonic.github.io/images/rl/4.png">

전형적인 prediction 문제 입니다. MDP가 있고, policy가 존재할 때 value를 찾자!

random policy를 따르고 정 사면체 모양 주사위를 던져서 방향을 결정한다고 할 때, 주사위를 평균 몇 번 던져야 검은 구역에 도착할 수 있을까요?

어려운 문제입니다. 우선 reward는 매 번 진행할 때 마다 -1을 얻을테니, reward가 최대가 되는 policy를 찾는게 이 문제를 푸는 방향일 것 같다는 생각을 할 수 있습니다.(value function을 maximize!)

어떻게 풀어야 할까요?

<img src = "https://py-tonic.github.io/images/rl/5.png">

우선 $k=0$일 땐 모든 value가 0으로 초기화 됩니다.

$k=1$일 때 (1,2)의 state를 기준으로 생각해봅시다. 여기서 나온 식은 모두 위의 벨만 기댓값 방정식을 그대로 사용합니다.

벽에 부딪힐 경우 자기 자신의 위치로 돌아온다고 생각하면 됩니다.

<img src = "https://py-tonic.github.io/images/rl/p0.png">

- 위로 갈 경우 $v_1(s) = 0.25\times(-1 + 0)$
- 아래로 갈 경우 $v_1(s) = 0.25\times(-1 + 0)$
- 왼쪽으로 갈 경우 $v_1(s) = 0.25\times(-1 + 0)$
- 오른쪽으로 갈 경우 $v_1(s) = 0.25\times(-1 + 0)$
    - 모든 액션에 대한 합 = $4\times0.25\times(-1) = -1$



$k=2$일 때는 어떻게 될까요?

<img src = "https://py-tonic.github.io/images/rl/p1.png">

- 위로 갈 경우 $v_2(s) = 0.25\times(-1 + -1)$
- 아래로 갈 경우 $v_1(s) = 0.25\times(-1 + -1)$
- 왼쪽으로 갈 경우 $v_1(s) = 0.25\times(-1 + 0)$ < goal
- 오른쪽으로 갈 경우 $v_1(s) = 0.25\times(-1 + -1)$
    - 모든 액션에 대한 합 = $0.25\times(-1) + 3\times0.25\times(-2) = -1.75$

이런 식으로 목표 지점에 도착할 경우 Gt에 대한 기댓값 value가 높아지는 것을 볼 수 있습니다.

계속 update하면 다음과 같은 true value function을 얻을 수 있습니다.

<img src = "https://py-tonic.github.io/images/rl/6.png">



## How to Improve a Policy

그리드 월드에서는 한번 업데이트 후 policy를 greedy하게 두면 바로 optimal policy였습니다. 하지만리얼 월드에서 policy를 더 좋은 방향으로 업데이트 하려면 어떻게 해야 할까요?

- Given a policy $\pi$
    - Evaluate the policy $\pi$ 
        - $v_\pi(s) = E[R_{t+1} \gamma v_{\pi}(S_{t+1})\vert S_t = s]$
    - Improve the policy by acting greedily with respect to $v_\pi$
        - $\pi' = greedy(v_\pi)$

<img src = "https://py-tonic.github.io/images/rl/7.png">

초기 value function v, 초기 policy $\pi$가 존재할 때 v를 evaluate해서 update하고, 그 위에서 greedy하게 움직이는 policy를 구합니다.

이를 반복하면 optimal policy를 찾을 수 있습니다.

즉 Policy Iteration이라는 것은 두 가지 스텝으로 진행됩니다.

- Policy Evaluation을 통한 value function의 업데이트
- Policy improvement를 통해 더 나은 policy를 찾기

그럼 여기서 의문이 생깁니다.

단순히 greedy하게 policy를 update한다고 이전 policy보다 더 나은 Policy가 되는가?

deterministic policy $a = \pi(s)$를 생각해봅시다.

그리디를 통해 policy improvement를 할 수 있습니다.(q에 대해 그리디 하게 움직이는게 $\pi$보다 $\pi'$이 낫다.)

- $\pi'(s)=argmax$ $q_\pi(s,a)$
- $q_\pi(s,\pi'(s))= max$ $q_\pi(s,a)\geq q_\pi(s,\pi(s))=v_\pi(s)$

> 첫 번째 항은 첫 스텝에서만 $\pi'$을 따라 가고 그 뒤로는 $\pi$를 따르는 value 입니다.
>
> 뒤의 q에 대한 함수는 첫 스텝과 그 뒤도 $\pi$를 따라가는 value입니다.
>
> 우리는 업데이트를 통해 value가 계속 바뀔것이고, 그 상황에서 첫 번째 action이 최적인 경우는 그리디하게 선택하는 것과 동일하게 볼 수 있습니다.
>
> 그러므로 중간에 q-func를 max하는 action은 greedy하게 선택한 액션과 동일할 것이고 모든 식을 만족하게 됩니다.
>
> 계속 $\pi'$이 $\pi$보다 좋다가 어느 순간 같아지면 optimal policy!
>
> 강의 뒷편에 이게 local optimal에 빠지지 않고 global optimal에 무조건 수렴한다고 하는데 이 얘기는 왜그런지 잘 모르겠다.. :(



## Value iteration

Policy iteration과의 차이점은 Bellman Optimality Equation을 사용합니다.(policy가 없고 value만 가지고 생각)

$$v_*(s) \leftarrow max(R_s^a + \gamma\sum\limits_{s'\in S}P_{ss'}^a v_{*}(s'))$$

예시를 들어 봅시다.

<img src = "https://py-tonic.github.io/images/rl/8.png">

policy가 없고 step이 지날 때 마다 value function이 업데이트 됩니다.



<img src = "https://py-tonic.github.io/images/rl/v0.png">





## Reference

[팡요랩](https://www.youtube.com/watch?v=rrTxOkbHj-M&t=32s)

[sumniya](https://sumniya.tistory.com/10)