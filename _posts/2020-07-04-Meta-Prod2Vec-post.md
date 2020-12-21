---
layout: post
title: Meta-Prod2Vec
author: Jaeheon Kwon
categories: Papers
tags: [recommendation]
---

# Meta-Prod2Vec - Product Embeddings Using Side-Information for Recommendation



## Abstract

---

기존의 아이템 메타데이터를 활용하는 추천 아이템의 유사도를 계산하는 새로운 방법인 Meta-Prod2vec를 제안합니다.

이러한 시나리오는 컨텐츠 추천, 광고 타겟팅 및 웹 검색과 같은 응용 프로그램에서 자주 발생합니다.

논문의 방법은 아이템 및 해당 속성과의 과거 유저 상호작용을 활용하여 아이템의 저차원 임베딩을 계산합니다.

구체적으로, 아이템 메타 데이터는 아이템 임베딩을 정규화하기 위해 추가 정보로서 모델에 주입됩니다.



## Introduction

---

실제 추천 시스템에는 추가적인 제약 조건이 존재하는데 그 중 일부는 다음과 같습니다.

추천 시스템이 대규모 유저 상호 작용 정보를 처리할 수 있도록 추천 시스템을 스케일링하여 추천의 실시간 변경 지원 및 cold-start 처리 문제 등이 존재합니다.

지난 몇 년 동안 유저 및 제품 임베딩을 생성할 수 있는 유망한 새로운 종류의 신경망 확률 모델이 등장하여 유망한 결과를 보여주었습니다.

새로운 방법은 수백만 개의 아이템으로 스케일 될 수 있으며 cold-start문제에서 좋은 모습을 보였습니다.

본 논문에서는 처음에 제안된 Prod2Vec 알고리즘의 확장을 제시합니다.

Prod2Vec 알고리즘은 제품 시퀀스에 의해 설정된 제품 동시 발생 정보만을 이용하여 제품의 분포 표현을 생성하지만 메타 데이터는 활용하지 않습니다.

저자들은 순차 구조와 함께 텍스트 컨텐츠 정보를 고려한 알고리즘의 확장을 제시했지만, 이 접근 방식은 텍스트 메타 데이터에만 적용되며, 결과 아키텍쳐는 계층적이므로 우리의 방법과 비교하여 일부 추가 정보가 누락되었습니다.

본 연구에서는 추가 정보를 활용한 추천 작업과 연결하여 Meta-Prod2Vec을 제안하는데, 이는 간단하고 효율적인 방식으로 Prod2Vec모델에 범주형 부가 정보를 추가하는 일반적인 방법입니다.

## Related Work

---

추천 시스템을 위한 기존의 방법은 CF기반, CB기반, 및 하이브리드 방식으로 분류될 수 있습니다.

CF기반 방법은 클릭과 같은 아이템과 유저 상호작용에 기반하고 도메인 지식을 요구하지 않습니다.

컨텐츠 기반 방법은 사용자 또는 제품 컨텐츠 프로필을 사용합니다.

실제로 CF 기반 방법은 컨텐츠 기반 방법에 필요한 많은 지식을 수집하지 않고도 제품간의 흥미로운 연관성을 발견할 수 있기 때문에 더 널리 사용됩니다.

그러나 CF 기반 방식은 새 아이템과 상호작용이 거의 없거나 전혀 없을 때 발생하는 cold-start문제로 고통받습니다.



### Latent factor models

---

MF 방법은 넷플릭스 대회로 인해 유명해졌습니다.

이 방법은 재구성 오류에 대한 square loss를 최소화 하여 희소 유저-아이템 상호작용 매트릭스의 저차원 분해를 학습합니다.

사용자와 아이템 잠재 벡터 사이의 내적결과는 추천을 수행하는데 사용됩니다.

MF 방법을 추천 목표에 더 잘 맞추기 위해 여러가지 수정이 제안되었습니다.(e.g. Bayesian Personalized Ranking 및 Logistic MF)

전자는 pairwise ranking loss를 통해 사용자 및 아이템 잠재 벡터를 학습하여 관련성 기반 아이템 순위를 강조합니다.

후자는 MF 방법의 square loss를 logistic loss로 대체하여 사용자가 아이템과 상호작용할 확률을 모델링합니다.

신경망을 통해 사용자 및 아이템 잠재 표현을 학습하는 첫 번째 방법은 중 하나는 Restricted Boltzmann Machines를 사용하여 사용자 아이템 상호작용을 설명하고 추천을 수행합니다.

최근 다양한 NLP 작업에서 단어 임베딩의 성공으로 인해 얕은 신경망이 주목을 받고 잇으며, Word2Vec모델에 중점을 두고 있습니다.

추천 작업에 Word2Vec 모델을 적용하는 것은 Prod2Vec 모델에서 제안되었습니다.

구매 시퀀스로 부터 제품 임베딩을 생성하고 임베딩 공간에서 가장 유사한 제품을 기반으로 추천을 수행합니다.



### Latent factor models with content information

---

잠재적인 요소와 컨텐츠 정보로부터 통일된 표현을 만들기 위해 최근 많은 기술이 적용되었습니다.

사용자 및 아이템 컨텐츠 정보를 통합하는 한 가지 방법은 이 정보를 사용하여 회귀를 통해 사용자 및 아이템 잠재 요소를 추정하는 것입니다.

또 다른 접근법은 CF 및 컨텐츠 기능 모두에 대한 잠재 요인을 배우는 것입니다.

추가 정보를 고려하기 위해 MF의 일반화로 텐서 분해가 제안되었습니다.

이 접근법에서, 사용자 아이템 컨텐츠 행렬은 공통 잠재 공간에서 인수분해 됩니다.



그래프 기반 모델 또한 통합  표현을 만들 수 있습니다.

특히, 사용자-아이템 상호작용 및 추가 정보는 사용자 및 아이템 잠재 요소를 통해 공동으로 모델링 됩니다.

사용자-아이템 상호작용 요소 및 추가 정보 요소는 사용자 요인을 공유합니다.



## Proposed Approach

---



### Prod2Vec

---

Prod2Vec 논문에서는 전자 메일에서 오는 제품 시퀀스 영수증에 Word2Vec 알고리즘 사용을 제안했습니다.

제품 시퀀스의 집합 $S$가 주어질 때, $s = (p_1,...,P_M), s \in S$ 목적은 결과 벡터 공간에서 유사한 제품들의 D-차원 실제 값 표현 을 찾는 것입니다.

Word2Vec은 원래 텍스트에서 단어 임베딩을 학습하기 위한 확장성이 뛰어난 예측 모델이며 더 큰 신경망 언어 모델 클래스에 속합니다.

이 분야에서 수행되는 대부분의 작업은 분포 가설을 기반으로 하며, 동일한 맥락에서 나타나는 단어는 동일한 의미가 아닌 경우에 가깝습니다.

온라인 쇼핑, 음악 및 미디어 소비와 같은 더 큰 맥락에서 유사한 가설이 적용될 수 있으며 CF 방법의 기초가 되었습니다.

CF 설정에서 서비스의 사용자는 제품이 함께 발생하는 분산 컨텍스트로 사용되어 고전적인 아이템 동시 발생 방식으로 이어집니다.

co-count 기반 추천 방법과 Word2Vec 사이의 추가 유사성에 대한 연구가 있습니다.

저자는 임베딩 방법의 목적이 국소적으로 동시 발생하는 아이템(단어)의 Shifted Positive PMI를 엔트리로 포함하는 행렬의 분해와 밀접한 관련이 있음을 보여줍니다.

여기서 PMI는 Point-Wise Mutual Information입니다.

> 뭔소린지 모르겠다.
>
> 논문 봐야 알듯.. [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)

$PMI_{i,j} = log\frac{X_{ij}\cdot \vert D\vert}{X_iX_j} \tag{1}$

$SPMI_{ij} = PMI_{ij} - logk \tag{2}$

$X_i, X_j$는 아이템 빈도 수 이고 $X_{ij}$는 $i$와$j$의 동시 발생 수, $D$는 전체 데이터 셋, $k$는 negative, positive의 비율입니다.



**Prod2Vec Objective.**

GloVe 모델의 저자는 대상 제품에 대해 주어진 문맥적 제품의 경험적 분포와 모딜렝 된 조건부 분포 사이의 가중 cross entropy를 최소화 하는 최적의 문제로 Word2Vec 목표(및 유사하게 Prod2Vec)를 재 작성할 수 있음을 보여줍니다.(보다 정확하게 이는 Word2Vec-Skip-Gram 모델을 나타내며 일반적으로 큰 데이터 세트 일수록 더 좋습니다.)

또한 조건부 분포의 예측은 대상 제품과 컨텍스트 제품 벡터 사이의 내부 제품에 대한 소프트 맥스로 모델링됩니다.

$L_{P2V} = L_{J\vert I}(\theta)$

$\:\:\:\:\:\:\:\quad = \sum\limits_{ij}(-X_{ij}^{POS}log\:q_{j\vert i}(\theta) - (X_i - X_{ij}^{POS})log(1-q_{j\vert i}(\theta))$

$\:\:\:\:\:\:\:\quad =\sum\limits_{ij}X_i(-p_{j\vert i}log\:q_{j\vert i}(\theta) - p_{\neg i\vert j}log\:q_{\neg j\vert i}(\theta))$

$\:\:\:\:\:\:\:\quad = \sum\limits_i X_iH(p_{.\vert i},q_{.\vert i}(\theta))$

여기서 $H(p_{.\vert i},q_{.\vert i}(\theta))$는 입력 제품 $i\in I$ 에 의해 조절된 출력 공간 $J$에서 임의의 제품을 볼 수 있는 경험적 확률 $p_{.\vert i}$ cross-entropy 입니다.

$p_{.\vert i}$ : empirical conditional probability of any product given product $i$

$q_{.\vert i}(\theta)$: predicted conditional probability, modeled as softmax of the dot product between embedding vectors



예측된 조건부 확률 $q_{.\vert i}$는 다음과 같습니다.

$q_{j\vert i}(\theta) = \frac{exp(w_i^Tw_j)}{exp(w_i^Tw_j)+\sum_{j'\in(V_{J-j})}exp(w_i^Tw'_j)}$

여기서 $X_i$는 제품 빈도수 이고, $X_{ij}^{POS}$는 학습 데이터에서 제품 쌍$(i,j)$가 관찰된 횟수 입니다.



<img src = "https://del-luna.github.io/images/meta-prod2vec/1.png">



Prod2Vec의 아키텍처는 Fig.1에 나와 있습니다. 가운데 창에 위치한 모든 제품의 입력 공간은 단일 히든 레이어와 softmax 레이어가 있는 신경말을 사용하여 주변 제품의 값을 예측하도록 훈련되었습니다.

그러나 Prod2Vec에 의해 생성된 제품 임베딩은 사용자 구매 순서 정보, 즉 로컬 발생 정보만 고려합니다.

Collaborative Filtering에 사용된 글로벌 동시 발생 빈도보다 풍부하지만 다른 유형의 아이템 정보(아이템 메타 데이터)는 고려하지 않습니다.

예를 들어 입력이 분류된 제품 시퀀스라고 가정하면 표준 Prod2Vec 임베딩은 다음과 같은 상호작용을 모델링하지 않습니다.

- 카테고리 $c$를 가진 제품 $p$ 의 현재 방문을 고려할 때, 다음에 방문한 제품 $p'$이 동일한 카테고리 $c$에 속할 가능성이 더 높습니다.
- 현재 카테고리 $c$를 고려할 때 다음 카테고리가 $c$ 또는 관련 카테고리 $c'$중 하나일 가능성이 높습니다. (e.g. 수영복 구매 후 완전히 다른 완전히 다른 제품 카테고리에 속하는 선크림을 볼 가능성이 존재함)
- 현재 제품 $p$가 주어지면 다음 카테고리는 $c$ 또는 관련 범주 $c'$이 될 가능성이 높습니다.
- 현재 카테고리 $c$가 주어지면 방문할 가능성이 높은 현재 제품은 $p$ 혹은 $p'$ 입니다.

서론에서 언급한 바와 같이 Prod2Vec저자는 알고리즘을 확장하여 제품 시퀀스와 제품 텍스트를 동시에 고려했습니다.

비-텍스트 메타 데이터에 확장 방법을 적용하는 경우 알고리즘은 제품 시퀀스 정보, 제품 메타 데이터와 제품 ID간의 종속성을 추가로 모델링하지만 메타 데이터 시퀀스와 제품 ID 시퀀스를 서로 연결하지는 않습니다.



### Meta-Prod2Vec

---

Related-Work 섹션에서 본 것 처럼, 특히 CF방법과 CB방법을 결합한 하이브리드 방법의 추천 시스템에 대한 추가 정보 사용의 광범위한 작업이 존재했습니다.

임베딩의 경우 가장 비슷한 작업은 단어와 단락이 함께 훈련된 Doc2Vec 모델 이지만 단락 임베딩만 최종 작업에 사용됩니다.

우리는 Fig.2와 같이 신경망의 입력 및 출력 공간에 추가 정보를 통합하고 내장될 아이템 메타 데이터 사이의 각각의 상호 작용을 개별적으로 파라미터화 하는 유사한 아키텍처를 제안합니다.

> 페이퍼를 읽다보면 constraints라는 용어가 자주 나온다.
>
> "external info를 사용해서 product co-occurrences자체에 대한 constraints를 걸어서 더 연관성이 높아야만 임베딩이 되도록 만든다." 라고 해석했다.
>
> 아래의 4 가지 term도 결국 모두 추가적인 제약조건으로 해석 가능하다.
>
> We place additional constraints on prodcut co-occurrences using external info.
>
> We can create more noise-robust embeddings for product suffering from cold-start

**Meta-Prod2Vec Objective.**

Meta-Prod2Vec loss는 아이템의 메타 데이터와 관련된 4가지 추가 상호작용 용어를 고려하려 Prod2Vec loss를 확장합니다.

$L_{MP2V} = L_{J\vert I} + \lambda \times (L_{M\vert I}+L_{J\vert M}+L_{M\vert M}+ L_{I\vert M}) \tag{3}$





<img src = "https://del-luna.github.io/images/meta-prod2vec/2.png">



M은 메타 데이터 공간(e.g. artist ID), $\lambda$는 정규화 파라미터입니다.

$L_{I\vert M}$: 메타데이터가 주어질 때 입력 제품 ID의 관측된 조건부 확률과 예측된 조건부 확률 사이의 가중 cross-entropy. 이 추가 정보는 아이템을 자체 메타 데이터(시퀀스에서 동일한 인덱스)의 함수로 모델링 하기 때문에 다음 세 가지 유형과 조금 다릅니다. 대부분의 경우 아이템 메타 데이터가 ID보다 일반적이며 특정 ID의 관찰을 부분적으로 설명할 수 있기 때문입니다.

$L_{J\vert M}$ : 입력 제품의 메타 데이터가 주어질 때 주변 제품 ID의 조건부 확률과 예측된 조건부 확률 사이의 가중 cross-entropy. 이 상호작용 용어 만으로 정상적인 Word2Vec loss가 증대되는 아키텍처는 Doc2Vec모델과 상당히 유사합니다. 여기서 문서 ID정보를 보다 일반적인 유형의 아이템 메타 데이터로 대체합니다.

$L_{M\vert I}$: 입력 제품에 대해 주변 제품의 메타 데이터 값에 대한 관측된 조건부 확률과 예측된 조건부 확률 사이의 가중 cross-entropy.

$L_{M\vert M}$: 입력 제품 메타 데이터가 주어진 주변 제품의 메타 데이터 값에 대한 관측된 조건부 확률과 예측된 조건부 확률 사이의 가중 cross-entropy. 이는 관찰된 메타 데이터의 시퀀스를 모델링하고 그 자체로 메타 데이터의 Word2Vec과 같은 임베딩을 나타냅니다. 



> 솔직히 논문읽으면서 제일 와닿지 않았던 부분인데,
>
> 나는 추천시스템을 노래에 적용하려하고, Word2Vec모델의 심플하지만 강력한 성능에 meta-data를 쓰고자 하는 목적으로 이 논문을 읽게 되었는데 Recsys에서 만든 slide share에 아주 좋은 그림이 있어서 가져왔다.



<img src = "https://del-luna.github.io/images/meta-prod2vec/3.png">



요약하면, $L_{J\vert I}, L_{M\vert M}$은 아이템 및 메타 데이터의 시퀀스의 likelihood를 개별적으로 모델링하여 발생하는 loss term을 개별적으로 인코딩합니다.

$L_{I\vert M}$은 메타 데이터가 주어진 아이템 ID의 조건부 가능성을 나타내고, $L_{J\vert M}$과 $L_{M\vert I}$는 아이템 ID와 메타 데이터 사이의 교차 아이템 상호작용 term을 나타냅니다.

Fig.3에서 Prod2Vec에 의해 인수 분해된 아이템 매트릭스와 Meta-Prod2Vec에 의해 인수 분해된 아이템 매트릭스 간의 관계를 보여줍니다.

Meta-Prod2Vec에 대한 일반적인 방정식은 $\lambda_{mi}, \lambda_{jm},\lambda_{mm},\lambda_{im}$네 가지 유형의 추가 정보 각각에 대해 별도의 $\lambda$를 도입합니다.

**Experiments** 섹션에서는 각 유형의 추가 정보의 상대적 중요성을 분석합니다.

또한 여러 메타 데이터 소스를 사용하는 경우 각 소스는 global loss와 자체 정규화 파라미터에서 개별 term을 갖습니다.

소프트맥스 정규화 요소와 관련하여 아이템의 출력 공간과 메타 데이터의 분리 여부를 선택할 수 있습니다.

Word2Vec에 사용된 단순화 가정과 마찬가지로, 각 동시 발생 제품 쌍을 독립적으로 예측할 수 있습니다.

제품과 해당 메타 데이터를 동일한 공간에 임베딩하므로 정규화 제약 조건을 공유할 수 있습니다.

Word2Vec의 주요 장점 중 하나는 확장성입니다. Negative sampling을 통해 가능한 모든 단어 공간에 대한 기존의 softmax loss를 근사화합니다.

수정된 likelihood 함수 $L_{SG-NS}(\theta)$를 최대화 하기 위해 네거티브 예시의 작은 샘플과 함께 포지티브 동시 발생에만 모형을 적합시킵니다.

$L_{J\vert I}(\theta) = \sum\limits_{ij}(-X_{ij}^{POS}log\:q_{j\vert i}(\theta)-(X_{ij}^{NEG}log(1-q_{j\vert i}(\theta))\approx L_{SG-NS}(\theta)\\ and: \\ L_{SG-NS}(\theta) = \sum\limits_{ij}-X_{ij}^{POS}(log\:\sigma (w_i^Tw_j)-kE_{N\sim P_D}\:log\:\sigma(-w_i^Tw_N)) $

$P_D$ 확률 분포는 negative context 예시를 샘플링하는데 사용되며 $k$는 긍정적인 예시당 음성 예시의 수를 지정하는 하이퍼 파라미터입니다.

Meta-Prod2Vec의 경우 $L_{SG-NS}(\theta)$손실에 대한 공동 임베딩 제품 및 해당 메타데이터에 대한 결정의 영향은 아이템과 메타데이터 값입니다.



## Conclusions

---

본 논문에서는 훈련 시 아이템 메타 데이터로 기존 Prod2Vec 방법을 향상시키는 새로운 아이템 임베딩 방법인 Meta-Prod2Vec을 소개했습니다.

이 작품은 임베딩의 맥락에서 부가 정보를 이용한 학습을 도입함으로써 최근의 임베딩 기반 방법과 consecrated MF방법 사이에 새로운 연결을 만듭니다.

우리는 각 유형의 추가 정보의 상대적인 가치를 별도로 분석하고 네 가지 유형 중 하나가 유익하다는 것을 증명했습니다.

마지막으로, Meta-Prod2Vec은 globally 및 cold-start체제 모두 추천 작업에서 Pord2Vec 보다 지속적으로 우수한 성능을 보이며 표준 Collaborative Filtering 방식과 결합하면 테스트 된 모든 다른 방법보다 성능이 우수함을 보여주었습니다.

이러한 결과는 구현 비용 절감 및 우리의 방법이 온라인 추천 시스템 아키텍처에 영향을 미치지 않는다는 사실과 함께 아이템 임베딩이 이미 사용중인 경우 이 솔루션을 매력적으로 만듭니다.

앞으로의 작업은 아이템 메타 데이터를 추가 정보로 사용하는 방법과 이미지 및 연속 변수와 같은 비 범주적 정보의 지원으로 확장 될 것입니다.

