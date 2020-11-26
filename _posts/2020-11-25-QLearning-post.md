---
layout: post
title: Q-Learning
author: Jaeheon Kwon
categories: Ai
tags: [Reinforcement Learning]
---



Q-learning이란?

- Model-Free 알고리즘
- Action-Value function을 사용



Q-Learning의 식은 다음과 같이 나타낸다.

$$Q(s_t,a_t) \leftarrow (1-\alpha)Q(s_t,a_t) + \alpha(R_t + \gamma \max\limits_{a_{t+1}}Q(s_{t+1},a_{t+1}))$$

기존 Q값과 새로운 샘플(다음 스테이트)에 대한 weighted sum 형태임을 볼 수 있다.



전체적인 프로세스는 다음과 같다.

- intialize $Q(s,a)$ Matirx 
- Observe initial state $s$
- select and carry out an action $a$
- oberserve reward $R$ and new state $s'$
- update $Q$
- $s\leftarrow s'$
- Repeat



$\alpha = 1$인 경우에 대해 update 수식은 벨만 방정식과 동일한 것을 볼 수 있음.



이제부터 Q-Learning의 수식이 어떻게 나왔는지 자세히 알아보자.

우선 기존의 Action-Value function의 수식은 다음과 같다.

$$Q(s_t,a_t) = \int G_tp(\tau_{s_{t+1}:a_T}\vert s,a)d\tau_{s_{t+1}:a_T}$$

벨만 방정식을 통해 재귀적인 형태로 나타내보자.

$$Q(s_t,a_t) = \int(R_t + \gamma Q(s',a'))p(a'\vert s')p(s'\vert s_t,a_t)ds',a'$$



우리의 목적은 Expected Return을 최대화 하는 것이고, 이를 위해선 optimal policy가 필요하다.

그렇다면 optimal policy는 무엇이고 어떻게 구할 수 있을까?

(아래 이미지 참조)

<img src = "https://py-tonic.github.io/images/qlearning/optim.jpg">



결국 Optimal Policy라는 것은 $V^*$를 최대화 하는 정책을 뜻하며, Optimal Action Value function ($Q^*$)가 존재할 때 그 값을 최대로 하는 액션을 고르는 정책을 뜻한다.

사실상 상태 가치 함수의 정의인 Return의 기댓값을 구하는 수식에서 $p(a_t,s',a',...\vert s_t)$를 베이즈 룰을 통해 쪼개면 우리가 구하고자하는 policy $p(a_t\vert s_t),p(a'\vert s'),...$가  다 들어있다.



결국 MDP를 통한 강화학습의 목적은 Expected Return을 최대화 하는 것이고 그 방법으로는 Optimal Policy를 찾는 것이다 라고 말할 수 있다.

그런데 Optimal Policy를 위해서는 $Q^*$라는 최적 방정식을 찾아야 한다. 이건 또 어떻게 구할 수 있을까?

사실상 $Q^*$는 바로 구할 수 없고 알고리즘을 통해 모델을 수렴시키는 방법을 통해 구한다.

에피소드를 진행하면서 $Q$를 $Q^*$에 다가가도록 만드는 것이 목표(트레이닝을 통해서)이고 최종적으로 $Q^*$를 구하면 테스트 타임 때는 $greedy$하게 액션을 선택하면 됨.

$Q^*$를 구하기 위한 학습 방법엔 두 가지가 존재함

- Monte-Carlo
- Temporal difference



두 가지 방법을 여기서 다 설명하긴 어려우니 따로 포스팅 하도록 하겠다.

