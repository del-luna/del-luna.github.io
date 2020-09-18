---
layout: post
title: 해석학의 세 가지 공리
author: Jaeheon Kwon
categories: Mathematics
tags: [analysis]
---





## Field Axioms



$a,b,c \in \R$ 및 연산 $+,\cdot$ 에 대해 다음 성질들이 성립한다.(공리니까 받아들이자.)



- 덧셈에 대한 폐쇄성 : $a + b \in \mathbb R$
- 덧셈에 대한 결합법칙 : $(a+b)+c = a+(b+c)$
- 덧셈에 대한 교환법칙 : $a+b = b+a$
- 덧셈에 대한 항등원 : 모든 실수 $a$에 대해 $a+0 = 0+a = a$를 만족하는 0이 유일하게 존재함.
- 덧셈에 대한 역원 : 모든 실수 a에 대해 $a+(-a) = (-a) + a =0$을 만족하는 (-a)가 유일하게 존재함.



- 곱셈에 대한 폐쇄성 : $a\cdot b \in \mathbb R$
- 곱셈에 대한 결합법칙 : $(a\cdot b)\cdot c = a\cdot (b\cdot c)$
- 곱셈에 대한 교환법칙 : $a\cdot b$ = $b\cdot a$
- 곱셈에 대한 항등원 : 모든 실수 $a$에 대해, $a\cdot 1 = 1\cdot a = a$를 만족하는 1이 유일하게 존재함.
- 곱셈에 대한 역원 : 0을 제외한 모든 실수 $a$에 대해, $a\cdot a^{-1} = a^{-1}\cdot a=1$를 만족하는 $a^{-1}$가 유일하게 존재함.



- 분배법칙 : $a\cdot (b+c) = a\cdot b+a\cdot c$



## Order Axioms



$a,b,c \in \mathbb R$에 대해, 다음의 성질들이 성립한다고 받아들이자.



- 삼분성 : 주어진 $a,b$에 대해서, $a<b$ 혹은 $a>b$ 혹은 $a=b$이어야 한다.
- 추이성 : $a<b$ 이고 $b<c$ 이면 $a<c$
- 가산성 : $a<b$ 이고 $c\in\R$ 이면 $a+c<b+c$
- 승산성 : $a<b$ 이고 $c>0$ 이면 $ac<bc,$ 혹은 $c<0$ 이면 $ac>bc$



## Completeness Axioms



- $E\subset \mathbb R $ , $E \neq \phi$ 인 집합 $E$가 bounded above이면 supremum $sup(E)<\infty$ 이 존재한다.



<img src = "https://py-tonic.github.io/images/analysis/0.png">



> - 위로 유계(bounded above)
> - 상계(upper bound)
> - 최소 상계 or 상한(least upper bound or supremum)
>
> - $E$의 모든 원소 $a$에 대해 $a\leq M$이 성립하면 $E$를 bounded above라고 한다.
> - 이러한 조건을 만족시키는 $M$을 모두 $E$의 upper bound라고 부른다.
> - $sup(E)$는 $E$의 가장 작은 upper bound(나는 M의 가장 작은 원소?로 해석했다.)를 뜻하며, 모든 $E$의 upper bound $M$에 대해서 $sup(E)\leq M$을 만족하는 수 이다.
>
> 
>
> - 아래로 유계(bounded below)
> - 하계(lower bound)
> - 최대 하계 or 하한(greatest lower bound or infimum)
> - $E$의 모든 원소 $a$에 대해 $a \geq m$이 성립하면 $E$를 bounded below라고 한다.
> - 이러한 조건을 만족시키는 $m$을 모두 $E$의 lower bound라고 부른다. 
> - $inf(E)$는 $E$의 가장 큰 lower bound를 뜻하며, 모든 $E$의 lower bound $m $에 대해서 $inf(E)\geq m$을 만족하는 수 이다.

