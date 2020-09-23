---
layout: post
title: Convex Optimization Chapter.1
author: Jaeheon Kwon
categories: Ai
tags: [islr]
---



- Linear Combination : $a_0x_0+...+a_nx_n$ $(a_i \in \mathbb R)$
- Affine Combination : $a_0x_0+...+a_nx_n$ $(\sum\limits_{i}a_i=1)$
- Convex Combination : $a_0x_0+...+a_nx_n$ ($\sum\limits_{i}a_i=1$ & $0\leq a_i\leq1$)

Convex Combination에 대해 닫힌 집합을 Convex set 이라고 부름.

$$x1,x2 \in C, 0\leq a\leq 1 \rightarrow ax_1+(1-a)x_2 \in C$$



<img src = "https://py-tonic.github.io/images/convex/0.png">



- Affine은 $x_1,x_2$를 지나는 직선이다. (외분점)
- Convex는 $x_1,x_2$를 연결하는 선분이다. (내분점)





### Convex set의 성질

- Disjoint Convex set을 분리해주는 hyperplane이 존재함.

    단, 여기서 set들은 closed set이며 둘 중 하나가 bounded set이어야 함.

- Convex set의 경계점을 지나는 접선이 항상 존재함.



<img src = "https://py-tonic.github.io/images/convex/1.png">

<img src = "https://py-tonic.github.io/images/convex/2.png">



### Convex functions

- $f:R^n\Rightarrow R$ is convex if $dom(f)$ is a convex set and $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y) \forall x,y \in dom(f), 0\leq\theta\leq 1$

직관적으로 부등식을 만족하려면 아래로 볼록해야 한다는 것을 떠올릴 수 있다.



<img src = "https://py-tonic.github.io/images/convex/3.png">

- $f$의 epigraph가 Convex set 일 때, $f$ is  Convex function (역도 성립)



<img src = "https://py-tonic.github.io/images/convex/4.png">



최적화 문제에서 $f$가 Convex임을 보이는 게 중요한 이유에 대해 알아보자.

흔히들  Convex에 대한 얘기를 하면 $f(x)=x^2$ 떠올리고 저 함수는 Convex이며 global minimum을 구할 수 있는게 보장되어있다.

그렇다면 Convex의 local minimum은 항상 global minimum일까?

Convex function은 다음과 같은 특성을 가진다.

- $f$ is Conex, $x$가 $f(x)$의 locally optimal point 일 때(즉 $f(x)$ is local minimum), $x$는 globally optimal point 이다.



**proof.**

- $x$가 locally optimal point일 때, $f(y)<f(x)$를 만족하는 $y$가 존재한다고 가정하자.
- $x$가 locally optimal point라는 것은 다음을 만족하는 $\delta>0$가 존재하는 것과 같음.
- $\vert\vert z-x\vert\vert_2 \leq \delta \Rightarrow f(z)\geq f(x),\quad z=\theta y+(1-\theta)x, (0<\theta<1)$
- $f(y)<f(x)$ 가 성립하려면 $f(z)\leq\theta f(y)+(1-\theta)f(x)<\theta f(x)+(1-\theta)f(x)=f(x)$를 만족해야함
- 그러나 $f(z)\geq f(x)$에 모순.
- $QED$





## Reference

[Ratsgo](https://ratsgo.github.io/convex%20optimization/2017/12/25/convexset/)

[모두를 위한 컨벡스 최적화](https://wikidocs.net/17206)