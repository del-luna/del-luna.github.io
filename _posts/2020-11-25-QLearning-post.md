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

그렇다면 optimal policy는 어떻게 구할 수 있을까?

우선 최적의 행동 가치함수는 다음과 같이 나타낸다.

$$Q^*(s_t,a_t) = \max\limits_{a}Q(s_t,a_t)$$

