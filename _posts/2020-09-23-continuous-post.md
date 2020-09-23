---
layout: post
title: 함수의 극한과 연속
author: Jaeheon Kwon
categories: Mathematics
tags: [analysis]
---



기존의 $\epsilon-\delta$를 통한 연속성을 증명하는 것은 특정 point에 대해서 하기 때문에 구간에 대해 연속성을 증명하는게 상당히 귀찮음.

그런데 균등 연속이라는 것을 사용하면 구간의 연속을 상당히 쉽게 증명할 수 있음.

- $f: D\rightarrow\mathbb R$
- $\forall \epsilon>0, \exist \delta>0, s.t. \forall x,y \in D,$
- $\vert x-y\vert<\delta \Rightarrow \vert f(x)-f(y)\vert <\epsilon$
- 위가 성립하면 $f$는 $D$에서 균등 연속이다.
- $f$가 $D$에서 균등 연속이면 연속이다.



기존의 $\epsilon-\delta$과 다른점은  다른 점은 극한이 아닌 연속의 정의 이기 때문에 $\vert x-y\vert <\delta$ 여기서 $0<\vert x-y\vert$ 이라는 조건이 빠졌고, 도메인의 특정 point $a$에 대해 증명하는 것이 아닌 $x,y\in D$에 대해서 증명함.



**ex)** $f(x) = x^2$이 $[-2,2)$에서 균등 연속임을 증명.

$\forall \epsilon>0, Let \delta = ? >0$

일단 델타는 모르지만 0보단 크다고 두자.

$Then, \forall x,y \in [-2,2)\quad with \vert x-y\vert<\delta$

위에서 정의했던 식을 그대로 대입해보자.

$\vert f(x)-f(y)\vert = \vert x^2-y^2\vert = \vert x+y\vert\vert x-y\vert\leq \vert x\vert+\vert y\vert\vert x-y\vert$

여기서 $\vert x\vert \leq 2, \vert y \vert \leq 2$ 이므로 위 식은 다음과 같이 쓸 수 있다.

$\vert x\vert+\vert y\vert\vert x-y\vert\leq 4\vert x-y\vert <\epsilon$

위 식이 $\epsilon$보다 작아야 하므로 $\delta = \frac{\epsilon}4$ 이다.

$QED$

 