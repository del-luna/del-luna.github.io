---
layout: post
title: linear algebra
author: Jaeheon Kwon
categories: Mathematics
tags: [linear algebra]
---



선대가 ML/DL에서 필요한 부분?

- 일단 단순히 계산할 때도 필요함

- PCA를 고유값과 고유벡터를 통해 설명할 수 있음

- Spectral norm과 Singular vector

    



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



### Rank

**Full Column Rank(m>n, r=n)**

모든 컬럼이 독립 → 종속적인 컬럼 벡터가 존재하지 않고 (선형 연산으로)0으로 만들 수 없다.

고로 자유 변수가 존재하지 않고 모든 컬럼 벡터가 pivot 이므로 Null Space는 오직 zero vector만 갖는다.

이런 행렬에 대한 Ax=b는 오직 0이나 유일해만 가진다.

- No free variable
- Null space is only zero vector
- Solution is 0 or Unique solution



**Full Row Rank(m<n, r=m)**

자유 변수가 존재함. 모든 row가 pivot을 가짐 근데 이렇게 되면 row> col인데 row 가 모두 pivot? 남은 col들은? → free variable

즉 자유변수가 n-r개 가 생성되므로 Null Space가 존재함(공간의 형태로)

이런 행렬에 대한 Ax=b는 모든b에 대한 솔루션을 가진다. Null Space가 공간형태니까!

- n-r or n-m free variable
- Null space is Exist
- Solution Exist for every b



**Full Row and Column Rank(m=n=r)**

이 때 A에 대한 rref가 단위행렬인 것은 자명하다.(이러면 역행렬을 가질 수 있음)

- No free variable
- Null space is only zero vector
- Solution is Unique



### Linear Independence

0을 제외한 선형 조합으로 0을 만들 수 없는 벡터들은 독립!

$$c_1x_1+c_2x_2+\cdot\cdot\cdot+c_nx_n\neq0\ \ except\ \ \forall c_i=0$$



2차원 평면에 방향이 다른 벡터 두 개가 존재한다? e.g. (2, 1), (1, 2) 이러면 독립이다.

그런데 세 개가 존재한다면? e.g. (2, 1), (1, 2), (2.5, -1) 당연히 하나는 종속이다 어떻게 알 수 있을까?

이 경우는 Rank에서 다뤘던 m<n인 경우를 생각해볼 수 있다. 자유변수가 존재하게되고 이는 0으로만들 수 있는 선형 조합이 존재한다는 얘기이고, 그로 인해 Null Space가 공간의 형태로 존재하게되는 형태를 떠올릴 수 있다.

사실 앞선 내용을 잘 이해했다면 크게 어려운 부분이 없다. 굳이 정리해보자면 다음과 같다.

**Independence**

- Null space가 오직 zero vector
- Rank가 Column(n)과 같음
- 자유변수 존재하지 않음



**Dependence**

- Null space가 공간의 형태로 존재함
- Rank < n
- 자유변수가 존재함



### Span

span의 사전적 의미는 포괄하다 라는 개념이다.

앞선 내용을 공부했다면 감이 오겠지만 span은 우리가 가진 벡터들로 형성할 수 있는 공간을 나타내는 개념이다. 즉 벡터들의 선형 조합을 통해 만들어지는 공간을 의미한다.

- 이 때 형성되는 공간은 $R^n$일 수도 있고, Subspace일 수도 있다. 이는 직관적으로 생각해보면 자유 변수와 관련이 있을테니 앞서 언급한 종속, 독립 개념도 쓰이게 된다.



### basis

우리에게 익숙한 3차원 공간 $R^3$를 떠올려 보자.

이 공간에 대한 '기저'는 무엇일까? 기저란 $n$차원의 공간에 대해 $n$개의 벡터를 가질 때 이 벡터들이 기저 벡터이려면 $n\times n$역행렬이 존재 해야 한다.

직관적으로 생각해보면 가장 먼저 떠올릴 수 있는 것은 단위 행렬이고 각각의 단위 행렬의 요소 $[1, 0, 0]^T,\ [0,1,0]^T,\ [0,0,1]^T$는 각각 $x,y,z$축임을 알 수 있다.

우리는 앞서 굉장히 다양한 방법으로 공간을 Span할 수 있는 것을 보았다. '즉 역행렬이 존재한다' 라는 조건은 Full Col and Row 를 만족한다는 얘기이고 이는 rref를 통해 행렬을 단위행렬로 만들 수 있다는 의미이다. 이런 조건들을 만족하는 $n$차원의 $n$개의 벡터는 무수히 많다는 것을 알 수 있다.



### Dimension

차원은 그렇다면 무슨 개념일까? 앞서 우리가 언급했던 Rank 일까? 아니면 Row의 수가 차원일까? 좀 더 자세히 알아보자.

차원이란 주어진 공간들에 대한 모든 기저들은 같은 수의 벡터를 가지는데 여기서 벡터의 수가 공간의 차원을 의미한다.

즉, 어떤 공간을 표현하기 위해 필요한 기저벡터의 수가 차원이다. $n$차원의 기저가 되기 위해서는 $n$개의 벡터가 필요하다!



**Column Space**

그렇다면 Vector Space와 같은 전체 공간이 아닌 Column Space의 기저와 차원은 어떻게 될까?

일반적으로 쉽게 생각해볼 수 있는 3x4행렬을 떠올려 보자.

2개의 pivot, 2개의 free columns을 가진다고 생각해보면 pivot vector들을 통해 C(A)를 span할 수 있다. 그리고 C(A)의 기저는 pivot vector이다.(독립이고, 애초에 독립 벡터들의 선형 조합을 통해 만들어 낸 공간이 C(A)이니까! 사실 지금 당연한 얘기를 풀어서 하고 있는 것이다.)

그리고 여기서 pivot의 개수는 앞서 우리가 Rank라고 언급했는데, 여기서 C(A)의 차원과도 동일하다.



**Null Space**

Null space의 차원은 어떻게 될까? 우리의 앞선 개념을 잘 생각해보자. Null space의 '공간의 크기'는 자유변수의 개수에 의존했다. 그렇다면 자유변수를 선형 조합해서 만든 것이 Null space임을 직관적으로 떠올려 볼 수 있다.

즉 free vector의 개수가 Null space의  차원과 동일하다.

정리해보면

- C(A)의 Rank = dim C(A)
- dim N(A) = num of free = n - r = n - dim C(A)



### Four Fundamental Subspace

**Row Space**

row vector들의 선형 조합으로 만들 수 있는 공간이지만... 조금 어색하다. 어떻게 구할 수 있을까?

행렬 A를 전치시킨 후 Col Space를 구하면된다! $Row\ space = C(A^T)$



**Left Null Space**

이 또한 위의 행공간 처럼 처음 나오는 개념인데, $A^T$에 대한 Null space를 의미한다.



이제 우리가 다뤘던 모든 공간들을 총 정리해보자. 

행렬 A(mxn)의 부분 공간들

- Column Space는 $R^m$의 공간에 존재한다.
- Null Space는 $R^n$에 존재한다.
- Row Space는 $R^n$에 존재한다.
- Left Null Space는 $R^m$에 존재한다.



좀 더 자세히 알아보자. 2x3행렬인 [[1,2,3], [4,5,6]]이 존재한다고 가정하자.

이 행렬의 Row Space는 3차원 공간 안에서 2차원 평면을 이룰 것이다. 3차원 공간을 채울 수 없는 이유는 이 평면에 직교하는 벡터 하나가 부족하기 때문이다. 이 때 row vector 두 개에 대해 직교하여 3차원 공간을 형성할 수 있도록 해주는 것이 Null Space이다.

즉, row space와 null space는 직교한다.. 도대체 무슨 소리일까?



<img src = "https://del-luna.github.io/images/linear/1.png">

Null Space를 구해보면 [-1, 2, -1]과 같은데, 이 공간이 두 row vector와 직교하는지 내적을 통해 확인할 수 있다. 내적해서 0이되면 두 벡터 사이의 각도는 0이된다.

재밌는 사실은 Ax=0을 통해 Null Space를 구하는 과정 자체가 row1, row2에 대해 $[x_1,x_2,x_3]^T$를 각각 내적해서 0을 만드는 벡터를 뽑은 것이나 마찬가지이다.(이게 중요함!)

또한 여기서 Rank(r)이 row space의 차원이다. rank가 2이기 때문에 row space가 2차원 평면을 이루는 것이다. (null space가 1차원 인 것은 자명 n-r을 떠올려 보자.)



Col Space는 그럼 어떨까? $R^m$공간에 존재한 다는 것은 자명하게 알 수 있다.

여기서도 Rank(r)이 col space의 차원인데 row와 마찬가지로 2이다. 이게 꽤 중요한 부분이다. row와 col space의 차원은 항상 같다. 즉, rank가 같다.



마지막으로 Left Null Space의 차원은 어떻게 될까?

$A^Tx=0$에 대한 해를 구한다고 생각하면 된다. $[[1,2,3]^T, [4,5,6]^T][x_1,x_2]^T = [0,0,0]^T$

여기서 $[1,2,3]^T, [4,5,6]^T$가 서로 독립이므로 만족하는 해는 $[0,0]$밖에 없다는 것을 알 수 있다. 즉 $N(A^T)= zero\ vector$임을 알 수 있다.



<img src = "https://del-luna.github.io/images/linear/2.png">



Col space는 $R^m$에 존재하므로 2차원 공간에 존재하고, rank=2이기 때문에 2차원 공간 전체를 표현할 수 있다. 또한 Left Null Space는 zero vector이므로 점으로 표현된다.

정리해보자.

- $C(A)$
    - basis : pivot columns of A
    - dim : r
- $N(A)$
    - basis : special solution of Ax=0
    - dim : n-r
- $C(A^T)$
    - basis : pivot columns of $A^T$
    - dim : r
- $N(A^T)$
    - basis : special solution of $A^Tx=0$
    - dim : m-r



### Eigenvalue, Eigenvector

$Ax=\lambda x$

직관적인 의미는 선형 변환을 해도 방향이 보존되는 벡터가 존재한다. 이 벡터가 고유벡터

어차피 방향이 같으면 크기는 스칼라를 통해 맞춰주면 되는데, 이 때 값이 고유값.

일반적으로 $n\times n$ 행렬은 $n$개의 고유벡터를 가진다.

또한 이런식의 변형도 가능하다.

$AP = P\lambda I$

여기서 P는 고유벡터들을 열벡터로 하는 행렬이고, $\lambda I$는 고유값들을 대각 원소로 하는 대각 행렬이다.

위와 같은 분해를 eigen decomposition이라고 한다.

- 단 고유값 분해가 가능하려면 $A$가 $n$개의 선형 독립인 고유 벡터를 가져야함. ($n$개의 기저!)



대칭행렬은 고유값 분해와 관련하여 좋은 성질 2가지를 가짐.

- (실수)대칭 행렬은 항상 고유값 대각화가 가능하다
- 직교 행렬로 대각화가 가능하다(직교 행렬은 역행렬이 전치행렬임!)





### Singular Value Decomposition

모든 $m\times n$행렬에 대해 적용 가능함!

$$A=U\Sigma V^T$$

- $U=m\times m$ 직교행렬 
    - $AA^T$를 고유값 분해해서 얻어진 직교 행렬, $U$의 열벡터들을 $A$의 left singular vector라고 부른다.
- $V = n\times n$ 직교행렬
    - $A^TA$를 고유값 분해해서 얻어진 직교 행렬, $V$의 열벡터들을 $A$의 right singular vector라고 부른다.
- $\Sigma = m\times n$ 직사각 대각행렬
    - $AA^T,A^TA$를 고유값 분해해서 나오는 고유값들의 square root를 대각원소로 하는 직사각 대각 행렬, 대각 원소를 $A$의 Singular value라고 부른다.(특이값은 항상 0 이상)



직교행렬의 기하학적 의미는 회전 변환, 혹은 반전 변환이고, 대각 행렬의 기하학적 의미는 스케일 변환 이다. 즉 행렬의 특이값은 이 행렬로 표현되는 선형 변환의 스케일 변환을 나타내는 값을 나타냄!





