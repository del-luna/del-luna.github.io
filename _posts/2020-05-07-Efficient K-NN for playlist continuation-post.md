---
layout: post
title: Efficient K-NN for playlist continuation
author: Jaeheon Kwon
categories: Papers
tags: [recommendation,knn]
---

#  Efficient K-NN for Playlist Continuation 

## Abstract

------

우리는 RecSys Challenge 2018의 main track 리더보드에서 9위를 차지한 솔루션을 제시합니다.

트랙 및 플레이리스트 메타 데이터와 함께 (플레이리스트-트랙) matrix를 사용하여 플레이리스트를 완성하는 K-NN을 개발했습니다.

 우리 솔루션은 추천에대한 품질 향상을 위해 여러 도메인 별 Heuristics를 사용합니다.

이러한 방법의 주요 장점중 하나는 계산에 대한 리소스 사용량이 적다는 것입니다.

최송 솔루션은 1시간 안에 기존의 컴퓨터에서 계산할 수 있습니다.

## Introduction

<hr>

위 Challenge의 task는 automatic playlist continuation입니다.

일부 시드 트랙이 포함된 사용자 플레이리스트가 있을 때, 플레이 리스트를 이어가려면 트랙 리스트를 추천해야 합니다.

task는 Spotify에서 발표한 Million Playlist Dataset을 기반으로 합니다.

접근 방식의 핵심은 유사한 플레이리스트로부터 item을 추천하는 플레이리스트 기반의 nearest neighbor 방법입니다.

우리의 솔루션은 트랙의 인기를 고려해 코사인 유사도 metric을 수정해서 사용합니다.

플레이 리스트에서 트랙의 위치를 고려하여 유사도 계산을 향상시킬 수 있었고, amplification과 같은 transforming function을 적용하여 알고리즘의 점수를 더욱 향상시킵니다.

서로 다른 기술과 하이퍼파라미터 값을 적용하여 각 하위 작업에 대해 모델을 분리하여 계산했습니다.

task의 세 가지 평가 기준중 NDCG에 맞게 솔루션을 최적화 했습니다.

더 나은 솔루션들과 비교해서 우리 방식은 비교적 가볍고 간단합니다. GPU에 액세스할 필요가 없으며 몇 줄의 파이썬 코드로 구현할 수 있습니다.

## Challenge Task

<hr>

챌린지 참가자는 

1. 첫 번째 n개의 트랙
2. 임의의 n개의 트랙
3. 플레이 리스트의 이름

위의 3가지에 대한 조합이 주어질 때, 10,000개의 불완전한 플레이리스트 각각에 대해 길이가 500인 상위 추천 리스트를 제공해야 합니다.

각 플레이리스트에 대해 제출을 평가할 목적으로 일부 holdout 트랙이 존재합니다.

챌린지 작업은 10개의 하위 작업으로 나누어 지며, 각 카테고리의 플레이리스트에서 다른 정보를 사용할 수 있습니다.

1. the name of playlist
2. the name and the first track
3. the name and the first 5 tracks
4. the first 5 tracks
5. the name and the first 10 tracks
6. the first 10 tracks
7. the name and the first 25 tracks
8. the name and 25 random tracks
9. the name and the first 100 tracks
10. the name and 100 random tracks

각 하위 작업에는 1,000개의 플레이 리스트가 포함되어 있고, 1, 2의 경우 트랙 정보가 누락되었거나 제한적으로 사용 가능합니다.

 가장 높은 성능 점수를 산출하는 하위 작업은 임의의 위치의 트랙이 제공되는 8, 10이었습니다.

## Data

<hr>

각 플레이리스트에 대해 트랙과 플레이리스트의 이름이 모두 지정되었습니다.

또한 각 노래의 아티스트 및 앨범 이름도 데이터 셋에서 사용할 수 있습니다.

데이터 셋의 플레이리스트 길이는 5~250까지 이며, 전체 287,740명의 아티스트와 571,628개의 앨범, 2,262,292개의 트랙을 포함합니다.

플레이리스트의 이름을 사전 처리하기 위해 모든 문자를 소문자로 변환하고 특수문자를 제거했습니다.

일부 플레이리스트 이름은 이모티콘만으로 구성돼서 이 경우 중복 무자를 제거하고 나머지 문자를 정렬하여 이름을 정규화 했습니다.

이러한 정규화 단계를 통해 원래 92,941개의 플레이리스트 이름 수가 16,752개로 줄어들어 챌린지 셋에 6개의 일치하지 않는 이름이 남았습니다.

## Prediction Methods

<hr>

재생 목록 추천에 대한 몇 가지 nearest neighbor 기반 접근 방식은  [Geoffray Bonnin and Dietmar Jannach]( https://dl.acm.org/doi/epdf/10.1145/2652481 )에 설명되어 있습니다.

우리의 주요 솔루션은 CF기반 입니다. 플레이리스트의 유사성을 계산하고 유사한 플레이리스트의 트랙을 추천하기 위해 user K-NN을 사용합니다.

유사성을 계산하기 위해 일반적으로 플레이리스트 트랙 co-occurrence matrix를 사용합니다.

또한 트랙, 아티스트 및 앨범 메타 데이터 및 플레이리스트 이름을 기준으로 유사도 점수를 정의합니다.

각 하위 작업에 대해 개별적으로 학습 및 파라미터 최적화를 수행했습니다.

K-NN기반 방법은 먼저 해당 플레이리스트와 유사한 플레이리스트를 찾은 다음 이 목록을 기반으로 트랙을 추천합니다.

pairwise similarities $s_{uv}$ of all playlist pairs $u$, $v$ 일 때,

플레이리스트 $u$에 속하는 트랙 $i$의 점수를 아래와 같이 표기합니다.

$ \hat{r}_{ui} = \frac{\sum_{v\in N_{k}(u)}s_{uv}r_{vi}}{\sum_{v\in N_{k}(u)}s_{uv}} \tag{1} $

여기서 $\hat{r}_{ui}$는 플레이리스트 $u$에 대한 트랙 $i$의 predict relevance이고, 

$r_{ui}$는 플레이리스트 $u$에 대한 트랙 $i$의 known relevance입니다.

$N_k(u)$는 u와 가장 유사한 k개의 플레이리스트 집합입니다.

n개의 가장 높은 점수를 가진 리스트를 가져와서 길이 n의 상위 리스트를 계산합니다.

유사도 계산을 위해 코사인 유사도 측정값을 사용합니다.

$s_{uv} = \sum\limits_{i \in I} \frac{r_{ui}r_{vi}}{\vert \vert R_u \vert \vert_2 \vert \vert R_v \vert \vert_2 } \tag{2}$

$R_u$는 relevance value가 $r_{ui}$인 벡터입니다.

relevance value는 트랙이 플레이리스트에 존재하는지 여부에 따라 1 or 0입니다.

 우리는 식(1)의 수정된 버전(다른 논문에서 제안한)을 실험적으로 사용합니다.

지수 $\alpha$를 정의하고 $s_{uv}$대신 $s_{uv}^{\alpha}$를 사용합니다. $\alpha > 1$의 값이 덜 유사한 플레이리스트에 비해 더 유사한 플레이리스트의 중요성을 증폭시키는 효과를 가지므로 이 방법을 amplification이라고 부릅니다.

이렇게 하면 추천 정확도가 향상됩니다.

또한 우리는 증폭을 적용하기 전에 스코어를 구간[0,1]로 정규화 하는 것이 유용하다는 것을 알았습니다.

$S_u =$ {$s_{uv}\vert v \in N_k(u)$}

<br>
$\tilde{s}_{uv} = \frac{s_{uv}-minS_u}{maxS_u - minSu} \tag{3}$
<br>

위와 같이 정의하고 식(1)에서 $s_{uv}$대신 $\tilde{s}_{uv}^{\alpha}$를 사용합니다.

## Weighting by Inverse Item Frequency

<hr>

information retrieval에 IDF를 사용하는 것은 희귀 품목이 흔한 품목보다 유사성을 더 잘 정의한다는 사실에 의해 정당화됩니다.

이 아이디어를 활용하여 Inverse Item Frequency를 정의합니다.

두 개의 플레이리스트는 다른 많은 목록이 포함된 트랙을 공유하는 경우 보다 인기도가 낮은 트랙을 포함하는 경우 유사할 가능성이 더 높습니다.

> TF는 문서내에 단어가 얼마나 자주 등장하느냐를 나타내는 값,
>
> 이 값이 높을수록 문서에서 중요함을 나타내는 지표로 사용될 수 있음
>
> 하지만 다른 문서에서 자주 등장하면 반대로 중요도가 낮아짐.
>
> IDF는 TF의 역수(정확히는 아니지만)이므로 unique한 단어가 등장할수록 높은 수치를 가짐.

플레이리스트 간의 유사성 정의를 수정하기 위해 Inverse Item Frequency를 적용하여 플레이리스트 간의 공유 트랙을 빈도에 반비례하여 계산합니다.

실험을 통해 $((f_i - 1)^{\rho}+1)^{-1}$이 성능이 좋은 가중치 계수인 것을 발견했습니다.

여기서 $f_i$는 트랙 $i$를 포함하는 플레이리스트의 수를 나타냅니다.

아래의 식을 이용하여 식(2)를 대체합니다.

$S_{uv} = \sum\limits_{i \in I}((f_i - 1)^{\rho}+1)^{-1}\frac{r_{ui}r_{vi}}{\vert \vert R_u \vert \vert_2 \vert \vert R_v \vert \vert_2 } \tag{4}$

<img src = "https://py-tonic.github.io/images/eknn/1.PNG">

<img src = "https://py-tonic.github.io/images/eknn/2.PNG">

최적의 $\rho$값은 약 0.4입니다.

## Weighting by Position

<hr>

플레이리스트의 첫 번째 k개의 트랙이 플레이리스트에 주어지고 작업이 연속을 예측하는 경우, k보다 위의 위치를 가진 트랙은 처음 몇 개의 트랙보다 유사성을 정의하는 데 더 연관성이 있을 수 있습니다.

쿼리 플레이리스트 u와 인접 플레이리스트 v가 주어지면 식(2)의 기존 유사성 공식에서 합의 0이아닌 요소 (플레이리스트 간의 공유된 트랙)는 동일한 가중치로 계산합니다.

플레이리스트에서의 위치에 따라 플레이리스트의 항목 $i$의 관련성을 수정합니다.

위치 $p$에 대한 가중치를 정의하고 재생목록 $u$에서 항목 $i$의 관련성을 다음과 같이 수정합니다.
<br>

$\tilde{r}_{ui} = r_{ui}(1+\frac{max(l,p_{u}(i))}{d}) \tag{5}$
<br>

여기서 $p_u(i)$는 $u$에서 항목 $i$의 위치를 나타내며 변수 $l,d$는 공식에서 하이퍼파라미터로 처리됩니다.

공식의 이론적 근거는 쿼리 플레이리스트 $u$에 가까운 트랙에 더 높은 가중치를 할당하고, 첫 번째 $l$트랙에 동일한 값 $1+l/d$를 사용하는 것입니다.

이것에 의해 식(2)는 아래와 같이 변경됩니다.<br>

$s_{uv} = \sum\limits_{i \in I} \frac{\tilde{r}_{ui}r_{vi}}{\vert \vert R_u \vert \vert_2 \vert \vert R_v \vert \vert_2 } \tag{6}$

## Metadata based similarity

<hr>

플레이리스트 또는 트랙에 대한 메타데이터는 플레이리스트간의 유사성을 계산하는데에도 사용될 수 있습니다.

플레이리스트의 유사성을 계산하기위해 트랙 대신 아티스트나 앨범을 사용할 수 있습니다. 또한 특정 텍스트 정규화 단계 후에 이름이 동일한 경우 두 개의 플레이리스트가 유사할 수 있습니다.

트랙, 아티스트, 앨범 또는 이름 기반의 유사점은 각 유형에 대해 가장 유사한 k개의 플레이리스트를 개별적으로 사용하거나 유사성값에대한 가중평균을 사용할 수 있습니다.

전반적으로 메타 데이터 기반 유사성은 CF(ex: track-based similarity)보다 신뢰성이 훨씬 낮으며 트랙 정보가 없거나 매우 제한적인 경우에만 유용했습니다.(subtask 1 & 2)

## Results

<hr>

실험을 위해 플레이리스트 데이터셋을 무작위로 train-test split을 사용했으며, 테스트 셋에선 하위 작업에 적합한 길이의 재생목록만 포함되어 있습니다.

모델 또는 하이퍼파라미터를 비교할 때 우리는 NDCG점수를 비교하는데 의존했습니다. 이 점수는 세 가지 기준중 전체 성과를 가장 잘 나타냅니다.

<img src = "https://py-tonic.github.io/images/eknn/3.PNG">

<img src = "https://py-tonic.github.io/images/eknn/4.PNG">
