---
layout: post
title: Convex Optimization Chapter.2
author: Jaeheon Kwon
categories: Mathematics
tags: [optimization]
---



앞선 챕터에서 affine set은 (직선) 외분점, convex set은 (선분) 내분점이라고 설명했다.

각각의 set은 무수히 많은 line과 무수히 많은 line segment들이 모여서 만들어진 것으로 설명할 수 있다.

용어 정리부터 해보자.

- line : 일반적으로 우리가 생각하는 두 점을 지나면서 무한히 커지는 선
- line segment : 두 점 사이에서 정의되는 선 (선분)
- ray : 한 점에서 시작해서 다른 점을 지나면서 무한히 커지는 선



- line
    - $y = \theta x_1+(1-\theta)x_2, \quad \theta\in\mathbb R$
- line segment ($x_2$에서 출발해서 $(x_1-x_2)$방향으로 $\theta$배 진행하다가 $x_1$에서 멈춤)
    - $y = \theta x_1+(1-\theta)x_2,\quad\ 0\leq\theta\leq 1$
    - $y = x_2 +\theta(x_1-x_2),\quad 0\leq\theta\leq 1$
- ray ($x_2$에서 출발해서 ($x_1 - x_2$)방향으로 $\theta$배 무한히 진행)
    - $y = \theta x_1+(1-\theta)x_2,\quad \theta\geq0$
    - $y=x_2+\theta(x_1-x_2),\quad\theta\geq 0$



### Affine set

affine set은 점, 직선, 평면과 같은 선형적  특징이 있으면서 경계가 없는 집합을 뜻한다.

어떤 집합이 affine set이라고 말할 수 있으려면 집합에 속한 임의의 두 점으로 직선을 만들어서 그 직선이 집합에 포함되는지를 보면 된다.

즉, 공간에 경계가 존재한다면 affine set이 될 수 없다.



### Convex set

convex set은 직관적으로 표현하면 오목하게 들어가거나 내부에 구멍이 없는 집합을 의미한다.

어떤 집합이 convex set이라고 말할 수 있으려면 집합에서 속한 임의의 두 점으로 line segment를 만들어서 그 선분이 집합에 포함되어야 한다.



<img src = "https://del-luna.github.io/images/convex/5.png">



오직 육각형만 convex set을 만족한다.



### Convex hull

$C\sub R^n$에 포함된 모든 점들의 convex combination들의 집합을 $C$의 convex hull이라고 한다.

