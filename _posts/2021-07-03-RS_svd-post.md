---
layout: post
title: Application of Dimensionality Reduction in Recommender System -- A Case Study
author: Jaeheon Kwon
categories: Papers
tags: [Recommend]
---



Sarwar의 00년도 논문

추천 시스템에서 SVD를 사용하여 차원 축소를 적용하고 두 가지 실험을 제시함.

- 유저 선호도 예측
- Top-N list 생성

실험은 두 가지 데이터 셋으로 진행되는데 MovieLens, E-Commerce 데이터 셋에 대해 실험한다.

본 논문에서 언급하는 추천 시스템은 correlation-based collaborative filtering이다.

위 시스템의 세 가지 한계점에 대해 설명하는데 다음과 같다.

- Sparsity : CF에서 사용되는 rating matrix는 굉장히 sparse하다. 본 논문이 쓰여진 당시(?)를 기준으로 약 1%이하의 아이템에 유저가 레이팅을 매겼다고 한다. 무튼 이러한 희소성 때문에 상관관계 기반의 알고리즘은 정확도가 떨어질 수 밖에 없다. 이러한 희소성으로 인해 생겨나는 문제점들은 유저 A,B의 상관관계가 높고, B,C의 상관관계가 높다고 해서 A,C의 상관관계가 높다고 볼 수 없다는 점과 공통된 레이팅이 적다고 해서 음의 상관관계를 가진다고 잘 못 판단할 수 있다는 점이다.
- Scalability : 유저 아이템이 늘어날 수록 계산량이 많아지는데, 이는 (웹 기반의 서빙에서) 어플리케이션을 확장하기 어려운 구조라고 한다.
- Synonymy : 상관관계 기반의 CF는 이름이 다르지만 비슷한 아이템에 대해 별도의 제품으로 판단한다. 즉, item 뒤에 있는 latent를 캡처하지 못한다.



본 논문에선 위와 같은 한계를 극복하기 위해 LSI(Latent Sementic Indexing)을 사용한다.

이 기법을 사용해서 synonymy를 극복할 수 있다. TF-IDF는 단어 자체의 의미까지는 고려하지 못하는데(위에서 언급한 Synonymy 문제가 발생할 수 있음) LSI를 통해 이를 극복할 수 있다.

> 이 부분에 대해선 할 말이 좀 있는데, 단순히 SVD(LSA)를 썻다고 해서 단어 간의 의미를 포착 한다는게 잘 와닿지 않았는데, 차원을 축소 함으로 인해서 새로운 기저가 생성되고, 이로 인해 기저가된 영화들의 선형 결합으로 다른 영화들을 ''표현'' 할 수 있게 되면 기존에 의미가 없던 부분들이 의미가 생기는 것은 아닐까? 라고 생각했다.

rating matrix 혹은 TF-IDF로 가중치가 매겨진 행렬이 주어지면, LSI를 통해 차원이 축소된 두 행렬로 분해할 수 있다. 이 두 매트릭스는 본질적으로 Latent 속성을 표현한다(represent).

item space의 차원을 축소하면 우리는 density를 증가시킬 수 있고 레이팅을 더 찾을 수 있다.(희소성으로 인해 발생하는 문제를 해결할 수 있을 것으로 기대함.)

선행 연구에서 이미 SVD를 사용하여 차원을 축소하고 Latent를 캡처하는 것이 효과적임을 볼 수 있다.

(이후로 SVD에 대한 설명이 나오는데 이 부분은 별 내용이 없다. 그냥 SVD모르는 사람들을 위해 넣은 듯..?)

SVD를 수행하면 $R = USV^T$가 나오는데

여기서 $US^{1/2}$를 앞선 [논문](http://pages.stern.nyu.edu/~atuzhili/pdf/TKDE-Paper-as-Printed.pdf)에서 처럼 ContentBasedprofile로 보고, (user-feature)

$S^{1/2}V$를 Content (feature-item)으로 해석했다. 그래서 이 둘의 유사도를 통해 predict를 수행한다.

또한 SVD를 수행하는 측면에서 computational cost를 고려하지 않을 수가 없는데,

추천 시스템을 Offline, Online 두 가지 프로세스로 나누면 SVD를 어차피 offline에서 수행하니 더 효율적이라고 언급한다.



연산 측면에서 SVD와 correlation은

- SVD : $O((n+m)^3)$
- correlation : $O(m^2n)$

이라서 SVD가 더 높지만 어차피 오프라인이라 의미가 없고 저장하는데 들어가는 비용은

- SVD : $O(m+n)$
- correlation : $ O(m2)$

라서 SVD가 더 효율적이라고 한다. (또한 차원 축소한 결과가 온라인 퍼포먼스에서 더 좋았다고 한다.)

Evaluation Metric은 다음과 같이 사용했다.

- Prediction : MAE
- Top-N : F1 score

결론적으로 가지고 있는 데이터에서, train set 비율이 작을수록 SVD의 성능이 좋아지고, train set 비율이 높아질 수록 기존 성능이 조금 더 좋아진다고 한다.

또한 앞서 언급했듯 두 가지 데이터 셋에 대해 실험하는데 이 두 데이터에 대한 결과가 다르다.

MovieLens의 경우 e-commerce보다 sparsity가 낮은데,

movielens는 저차원 데이터일 때 성능이 더 좋지만 e-commerce는 고차원에서 성능이 항상 더 좋았다.

아마 논문에선 언급하지 않지만 sparsity 자체가 가지고 있는 정보가 얼마 없는데 이 상태에서 노이즈를 줄이겠다고 SVD를 썻다가 가뜩이나 없는 정보를 더 줄이는게 오히려 역효과가 나지 않았나 싶다.

논문에서는 최적의 하이퍼파라미터를 찾기 위한 실험과정을 써놨는데,, 이거 자체가 크게 의미 있어 보이진 않지만 (어차피 도메인에 따라 다를테니..) 그래도 적어보자면..

결과적으론 실험을 통해 train-test의 비율과 dimension을 얼마나 줄일지에 대한 두 하이퍼 파라미터를 각각 구하고 있다.

dimension 파라미터(k)는 14정도가 최적이었고, train-test 비율은 movielens에서 0.8, e-commenre에서 0.6으로 설정했을 때 최적이었다.



## Reference

- [paper](http://robotics.stanford.edu/~ronnyk/WEBKDD2000/papers/sarwar.pdf)