---
layout: post
title: 페아노 공리계
author: Jaeheon Kwon
categories: Mathematics
tags: [analysis]
---



$\mathbb N := \{0,1,2,3,...\}$ 

위 정의는 직관적인 자연수에 대한 정의이다. (0(혹은 1)에서 시작하고 무한히 셀 수 있다.)

복잡한 과정들을 쪼개어 단순화 해서 꼭 필요한 것만 남겨봅시다.

(e.g. 지수 승은 곱셈으로 쪼갤 수 있고 곱셉은 덧셈으로 쪼갤 수 있으며 덧셈은 increment(단순히 다음 수)로 쪼갤 수 있습니.)

우리에게 익숙한 프로그래밍 언어 표기법으로 위의 직관적인 자연수 집합을 재정의 해보자.

$\mathbb N:=\{0, 0++, (0++)++, ((0++)++)++, ...\}$



이제 자연수 집합을 정의하기 위해 단순하면서 필요한 두 가지가 눈에 보입니다.



$Axiom.1:$ 0(or 1) is  a natural number

$Axiom.2 :$ If $n \in \mathbb N$,    $n$++ $\in \mathbb N$

 여기 까지만 정의해도 꽤나 그럴 듯 합니다.

위처럼 자연수 집합을 정의하고 각각 우리가 기존에 알던 자연수 집합(0, 1, 2, 3 ...)에 1대1 대응 시키면 되니까요, 그런데 몇 가지 문제점이 존재합니다.

우선 우리는 수학에 익숙하지 않으니 예시를 살펴봅시다.

우리에게 다음과 같은 수체계가 존재한다고 가정합시다.

0++ = 1, 1++ = 2,  2++ = 3, 3++ = 0

위 체계도 $Axiom$ 1, 2 를 모두 만족합니다. 따라서 0으로 되돌아 오지 않기 위한 공리가 추가적으로 필요합니다.

$Axiom.3:$ 0 is not the successor of any Natural Number



위 수체계를 그대로 확장해봅시다. 우리는 3까지 등반했습니다. 그런데 다음과 같은 문제점이 발생했습니다.

0++ = 1, 1++ = 2,  2++ = 3, 3++ = 0, 3++ = 4, 4++ = 4

이제 0으로 되돌아 가지는 않지만 위 처럼 다음 수가 같은 수여도 $Axiom$ 1,2,3을 모두 만족합니다. 따라서 successor가 달라야 한다는 공리가 추가적으로 필요합니다.

$Axiom.4:$ Different Natural Number must have different successors



이제 꽤나 만족스럽습니다. 딱 한가지만 빼면요,

자연수는 ''무한히'' 셀 수 있는데 유한한 시간을 사는 저희같은 존재는 저 수체계의 마지막을 보고 돌아올 수 없습니다.

따라서 저 수체계가 어디서나 위 공리들을 만족한다는 추가적인 공리가 필요합니다.

$Axiom.5:$ Principle of Mathematical Induction

Let $P(n)$ be any property pertaining to a Natural Number

$P(0)$ is true, $P(n)$ is true, $P(n++)$ is also true, Then $P(n)$ is true for every Natural Number $n$



위와 같은 5가지 공리와 집합론의 추가 공리에서 시작하여 다른 모든 수 체계를 구축하고, 함수를 생성하며 대수와 미적분이 가능해집니다.

수학을 다루면서 객체가 무엇인지, 무엇을 의미하는지가 아닌 객체의 property에 신경 쓰고 객체를 추상적으로 취급하는 "공리를 통한 숫자의 추상적 이해"가 중요합니다. 