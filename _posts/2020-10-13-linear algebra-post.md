---
layout: post
title: linear algebra
author: Jaeheon Kwon
categories: Mathematics
tags: [linear algebra]
---



### Vector Space

linear combination에 대해 닫혀있는 공간(더하거나 스칼라배에 대해 닫혀있어야 됨)

zero vector를 포함해야함!(이게 키 포인트)



### Sub Space

Vector space의 부분집합 이며,  linear combination에 대해 닫혀있는 공간.

Sub간의 교집합은 Sub space를 이룸(합집합은 안됨!)

### Null Space

위와 마찬가지로 '공간' 이기 때문에 linear combination에 대해 닫혀있어야 함.

Ax=b로 표현되는 선형 방정식에서 b가 zero vector 일 때, 식을 만족하는 모든 x에 대한 집합.

즉, Ax=0의 '해' 들이 이루는 공간

이러면 어쨋든 x = zero vector에 대해선 모두 만족하고 그 다음 다른 해가 있는 지가 중요한데, 그 다른 해가 존재한다면 1개 존재할 때 0벡터와 직선을 그릴 수 있고, 두 개면 평면... 이런식으로 공간을 형성할 수 있게 되고 그걸 Null Space라고 부름. 0벡터를 지나는 공간이기 때문에 여전히 linear한 성질은 만족하고, 사실상 앞서 언급한 여러 스페이스와 동일하게 생각할 수 있는 것 같음.

하지만 이게 다른 b에 대해서도 Vector space를 이루는 건 아님!

근데 여기서 '다른 해' 즉, 0벡터가 아닌 어떤 해 e.g. [-2,1,0,0]은 행렬 A와 곱해져서 [0,0,0,0]이 될 수 있다고 하자.

그렇다면 이 '다른 해' 가 존재하려면 자유 변수가 꼭 필요할까? yes!

$$x+2y+2z =0 \\ 3y+ z = 0 \\ 4z = 0$$

z=0되고 나머지도 0밖에 안됨 그런데 맨 밑에 z가 자유변수라면? → 아무거나 하나 넣을 수 있게 되고 그로 인해 zero vector가 아닌 해가 하나 존재하게 됨! 



### Ax = b의 해

생각해보자.

Ax가 나타내는 공간 (Column 기준)으로 컬럼 스페이스 를 C(A)라고 표현하면

이는 A의 각 컬럼 벡터에 대해 선형 결합으로 이루어진 공간이다.

Ax=b의 해가 존재한다는 소리는 C(A)의 공간 위에 벡터 b가 존재한다는 소리이다. (교점이 있어야 해가 존재할 것 아닌가!)

여기서 한 가지 재밌는 사실은 A의 row기준 선형결합이 zero row를 만들어 내면 해당 행에대한 b도 0이 된다는 사실이다.(0인 row에 어떤 x를 갖다 붙여도 0이 되니까!)

ps. zero row는 다른 row로 표현되기 때문에 independent 하고, 이는 row들이 만들어낸 공간 내에 존재하기 때문에 차원을 정의하는데 기여하지 못함!

위를 해가 존재하기 위해 필요한 조건인 Solvability Condition이라고 한다.

Ax=b는 두 가지 해를 가진다

- Particular

자유 변수를 0으로 설정해서 푼 해!

- Null Space

위에서 구한 Null Space 에대한 해!

이 두가지를 더해도 여전히 그대로이다 Ax_p=0 + Ax_n=b  == A(x_p+x_n)=b

여기서 재밌는 사실은 Null Space는 말그대로 '공간' 이고 x_p는 특정 벡터일 것이다. 특수해니까!

그런데 이 둘을 더하게 되면 더 이상 Subspace의 성질은 가지고 있지 않게 된다. Null space가 원점을 지나는 공간인데 이 공간을 x_p만큼 이동 시켜야 하기 때문이다.(절편 + y = ax) 라고 생각해서 y = ax + b가 된다고 생각해도 되지 않을까..?

Ax=b가 모든 b에대해 만족하려면...?

일단 A의 row가 col보다 작아야..하고 col에 자유변수가 b의 rank만큼은 있어야 하지 않을까?

b가 2차원 C(A)가 2차원이면 교점 정도야 있을 거고 C(A)에 포함되면 럭키?

C(A)가 3차원이면 b가 무조건 포함될까..? 이것도 Vector space가 4차원이면 C(A)에 포함되지 않는 축으로 b가 설정되면 의미 없지 않을까?

그렇다면 C(A)가 Vector Space (최대 차원) 만큼 되려면 자유변수가 모든 축에 다 있어야 되는데 이게 가능한 형태인가..?

### Rank

- Full Column Rank

모든 컬럼이 독립 → 종속적인 컬럼 벡터가 존재하지 않고 (선형 연산으로)0으로 만들 수 없다.

고로 자유 변수가 존재하지 않고 모든 컬럼 벡터가 pivot 이므로 Null Space는 오직 zero vector만 갖는다.

이런 행렬에 대한 Ax=b는 오직 0이나 유일해만 가진다.

- Full Row Rank

자유 변수가 존재함. 모든 row가 pivot을 가짐 근데 이렇게 되면 row> col인데 row 가 모두 pivot? 남은 col들은? → free variable

즉 자유변수가 n-r개 가 생성되므로 Null Space가 존재함(공간의 형태로)

이런 행렬에 대한 Ax=b는 모든b에 대한 솔루션을 가진다. Null Space가 공간형태니까?!