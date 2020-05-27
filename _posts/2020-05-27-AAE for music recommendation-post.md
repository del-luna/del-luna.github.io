---
layout: post
title: Using Adversarial Autoencoders for Multi-Modal Automatic Playlist Continuation
author: Jaeheon Kwon
categories: Papers
tags: [recommendation,Autoencoder]
---

# Using Adversarial Autoencoders for Multi-Modal Automatic Playlist Continuation



## Abstract

<hr>

Automatic playlist continuation 작업은 기존 재생 목록에 추가 할 수 있는 추천 트랙 목록을 생성하는 것입니다.

 적절한 트랙을 제안하는(i.e. 재생 목록에 노래를 추가하는 것) 추천 시스템은 현재 재생 목록의 끝을 넘어 목록을 확장할 뿐만 아니라, 재생 목록을 보다 쉽게 만들 수 있습니다.

Spotify는 다수의 재생 목록 및 관련 트랙 목록을 포함하는 재생 목록 데이터 셋을 출시했습니다.

다수의 트랙이 보류된 재생 목록 세트가 제공되면 목표는 플레이 리스트에서 누락된 트랙을 예측하는 것입니다.

우리는 adversarial autoencoders를 automatic playlist continuation 문제로 확장합니다.

재생 목록 타이틀, 아티스트 및 앨범과 같은 여러 입력을 작업에 통합하는 방법을 보여줍니다.

## Introduction

<hr>

Automatic playlist continuation은 재생 목록 feature와 초기 트랙이 제공되면 해당 재생 목록을 "계속"할 수 있는 추천 트랙 목록을 생성합니다.

트랙의 sequential order가 더 나은 모델을 만드는 데 도움이 되고, 재생 목록 품질을 평가하기가 어렵기 때문에 이는 어려운 작업입니다.

최근 adversarial autoencoders를 활용한 이미지 분야에서 발전이 있었고, 우리는 선행 연구를 통해 추천 시스템에서도 적용할 수 있다는 것을 봤습니다.

AAE는 입력을 재구성 할 뿐만 아니라 선택된 사전 분포와 일치시키도록 훈련됩니다.

> AAE란?
>
> 기존 VAE에서  GAN의 discriminator를 합친 것,
>
> VAE가 GAN에서 generator 역할을 한다. 
>
> encoder를 통해 latent variable z를 sampling하고, discriminator가 encoder가 sampling한 가짜 z와 P(z)로 부터 직접 sampling한 real z를 구분하는 역할을 한다.
>
> reference: [ratsgo]( [https://ratsgo.github.io/generative%20model/2018/01/28/VAEs/](https://ratsgo.github.io/generative model/2018/01/28/VAEs/) )

이 논문에서는 ACM Recommender System Challenge 2018의 맥락에서 AAE가 automatic playlist continuation에 적용될 수 있는지를 분석합니다.

이전 연구에서는 item 속성(track title, album title, artist name)을 활용하지 않았습니다.

우리는 item 속성을 playlist-level로 집계할 수 있는지 가능성에 대해 연구합니다.

실제로 이는 재생목록 제목이 쓸모없는 제목일 수도 있으므로 유용한 속성들입니다.

> Prior work:[Multi-Modal Adversarial Autoencoders for Recommendations of Citations and Subject Labels]( https://dl.acm.org/doi/epdf/10.1145/3209219.3209236 )

<img src = "https://py-tonic.github.io/images/AAE/1.PNG">

기존의 추천 문제는  set of users $U$와 set of items $I$  의 $U$x$I$ 행렬의 missing ratings를 예측하는 모델링됩니다.

우리의 경우에는 재생 목록에서 트랙의 양상 matrix $P$x$T$를 고려합니다.

여기서 $P$는 재생 목록 집합이고, $T$는 트랙 집합입니다. 

우리의 목적은 missing occurences를 예측하는 것입니다.

결과적으로 AAE가 다른 모델보다 지속적으로 성능이 우수하며 여러 입력 방ㅎ식을 통합하는 기능이 모델의 성능을 향상시킨다는 것을 보여줍니다.

## Problem Statement

<hr>

재생 목록은 기존 추천 시나리오에서 user로 간주될 수 있지만 item은 트랙입니다.

이것은 우리는 트랙을 bipartite 그래프로 고려해야 한다는 뜻입니다.

<img src = "https://py-tonic.github.io/images/AAE/2.PNG">

m개의 재생 목록 $P$와 n개의 트랙 $T$가 주어질 때, 전형적인 추천 시스템은 spanned space $P$x$T$를 모델화 합니다.

우리는 sparse matrix $X \in {0,1}^{m\times n}$ 를 고려합니다. $X_{jk}$는 트랙 $k$가 재생 목록 $j$에 포함되는 것을 뜻합니다.

우리는 binary occurrence만 생각합니다. 트랙이 재생 목록에 존재하면 1이고 L1 norm을 통해 플레이리스트 내의 트랙의 수를 정규화합니다.

> 재생 목록 내의 트랙 수를 정규화 한다는게 무슨 의미가 있지..?

학습하는 동안, 모델에는 Side information(title, album, artist)과 함께재생 목록에 트랙이 완전히 나타나는 것이 제공됩니다.

>  For training, the models are supplied with the complete occurrences of the tracks in the playlists, $X_{train} = P_{train} ▹◃ X$, along with side information, $S_{train} = P_{train} ▹◃ S$.

테스트 시에는, representation $X_{test}$와 $S_{test}$가 유사하게 얻어집니다.

예비 실험을 하기 위해 우리는 선행 연구의 설정과 비슷하게 인위적인 test set을 생성합니다.

우리는 각 행에서 0이 아닌 entry 하나를 0으로 설정하여$X_{test}$에서 무작위로 선택된 item을  제거합니다.

> ...? 뭔 의미지..?
>
> 각 행에서 0이아닌 entry(항목) 하나를 0으로 설정한다...
>
> 그리곤 X_test에서 임의로 선택된 item을 제거한다..?
>
>  we remove randomly selected items in $X_{test}$ by setting one non-zero entry in each row to zero. 

그리고 이 테스트 셋을 $\hat X_{test}$로 표기합니다.

real test set은 대회에서 제공되지만 우리가 인위적으로 구성한 development set은 challenge test set에 대한 represent 입니다.(자세한건 논문 뒤편에서 서술함.)

추가적인 정보$S_{test}$와 함께 $\hat X_{test}$가 주어질 때 모델의 예측 값은 $X_{pred} \in [0,1]^{m_{test} \times n}$ 입니다.

목표는 올바른 트랙(continuation)이 $X_{pred}$에서 높은 순위를 얻는 것입니다.



## Our Approach

<hr>

예비 실험에서 item 동시 발생 기준으로 두 가지 기준을 활용했습니다.

> item co-occurrence의 의미..?

먼저 이러한 기준을 제시한 다음 두 개의 AE의 변형에 대한 building block으로 multi-layer perceptron을 소개합니다.

마지막으로, 불완전한 AE를 간략하게 설명하고 AAE를 얻기 위해 확장할 수 있는 방법과 재생 목록의 타이틀, 트랙 타이틀, 아티스트 이름 및 앨범 제목과 같은 부가 정보를 두 모델에 통합하는 방법을 보여줍니다.

**$Item Co-Occurrense.$** 첫 번째 기준은, 트랙 동시 발생에 기초한 동시 발생 점수를 고려하는 것입니다.

이론적 근거는 과거에 같은 재생 목록에서 함께 발생했던 두 개의 트랙이 앞으로도 함께 발생할 가능성이 높다는 것입니다.

training data $X_{train}$이 주어지면, 우리는 전체 item co-occurrence matrix $C = X_{train}^{T} \cdot X_{train} \in \R^{n \times n}$를 계산합니다.

예측 시간에 행렬 곱셈 $X_{test} \cdot C$를 통해 동시 발생 값을 집계하여 점수를 얻습니다.

diagnoal of C에서 각 item의 (제곱) 발생 횟수가 유지되어 사전 확률을 모델링 합니다.

Singular Value Decomposition(SVD)는 item $X^T \cdot X$의 동시 발생 행렬을 인수 분해 하는 접근 방식입니다.

선행 연구에서 SVD가 추천 시스템에 유용하다는 것을 보여주었습니다.

우리는 side information을 통합하여 SVD를 확장합니다.

textual features를 TF-IDF weighted bag-of-words로 item과 연결하고 결과 matrix에 대해 SVD를 수행합니다.

예측 값을 얻기 위해 item과 관련된 재구성 matrix의 index만 사용합니다.

<img src = "https://py-tonic.github.io/images/AAE/3.PNG">

**$Multi-Layer Perceptron$.** MLP는 하나 혹은 여러개의 히든 레이어를 가지는 완전 연결 feed-forward 방식의 신경망입니다. 

출력은 $h^{(i)}=f(h^{(i-1)} \cdot W^{(i)}+b^{(i)})$ 를 연속으로 적용하여 계산합니다.

여기서 f는 activation function입니다.

우리는 MLP-2에 의해 두 개의 히든 레이어 퍼셉트론 모듈을 축약합니다.

이 MLP-2 모듈은 후속 아키텍처 빌딩 블록 뿐만 아니라, 재생 목록 타이틀에서만 작동하는 전체 모델로도 사용됩니다.

이 경우 $BCE(x,MLP-2(s))$를 최적화합니다. 여기서 재생 목록 타이틀 s는 입력으로 사용되며 x는 target output입니다.

우리는 TF-IDF weighted embedded bag-of-words 표현을 사용하여 아래에 설명된 AE의 변형과 공정한 비교를 수행합니다.

**$Undercomplete Autoencoders$.** 일반적인 AE에는 두 가지 구성 요소가 포함됩니다. (encoder를 $enc$, decoder를 $dec$로 표기합니다.)

인코더는 입력을 hidden representation으로 변환합니다.($z = enc(x)$)

그런 다음 디코더가 입력을 재구성합니다. ($r=dec(z)$) 

두 개의 구성 요소는 loss function $BCE(x,r)$ 를 최소화 하기 위해 함께 훈련됩니다.

입력 $x$를 출력 $r$에 단순 복사하는법을 배우지 않기 위해 차원을 축소합니다(undercomplete).

AE는 재구성을 위한 가장 중요한 변형 요소를 포착하도록 훈련됩니다.

인코더와 디코더 모두 MLP-2 모듈을 선택했습니다.

$r = MLP-2_{dec}(MLP-2_{enc}(x))$

side information을 이용가능한 경우 디코더에 추가 입력으로 제공합니다.

$r = MLP-2_{dec}([MLP-2_{enc}(x);s])$

사전 훈련된 단어 임베딩을 사용하여 textual features를 저차원 공간에 임베딩합니다.

여기서의 이론적 근거는 low code dimension이 많은 양의 어휘 용어에 의해 압도되지 않는다는 것입니다.

> low dimension이 많은 양의 어휘에 압도되지 않는다..?

모델을 공정하게 비교하기 위해 위에서 설명한 MLP에도 입력과 동일한 텍스트 표현이 제공됩니다.

보다 정확하게는 정보 검색에 유용한 것으로 입증된 TF-IDF weighted bag of embedded words 표현을 사용합니다.



**$Adversarial\ Autoencoders$.** 우리는 기존의 AAE를 GAN과 결합하여 확장합니다.

AE의 구성 요소는 sparse item vector를 재구성 하는 반면 discriminator는 생성된 code와 선택된 사전 분포로부터 샘플된 code를 구분합니다.

따라서 latent code는 사전 분포와 일치하도록 형성됩니다.

우리는 smooth prior로 부터 code를 구별하여 undercomplete autoencoders보다 sparse input vector에 더 강력한 모델로 latent representation이 학습된다고 가정합니다.

이론적 근거는 smoothness가 explanatory factors of variation을 구분하는 좋은 표현의 주요 기준이라는 점입니다.

>  **Factors of variation** are some factors which determine varieties in observed data. If that factors change, the behavior of the data will change. 

우리는 먼저 $h = MLP-2_{enc}(x)$와 $r = MLP-2_{dec}(h)$를 계산하고 $BCE(x,r)$를 통해 인코더와 디코더의 파라미터를 업데이트합니다. 

따라서 정규화 단계에서 $h$의 크기와 일치하지 않는 독립 가우스 분포에서 표본 $z \sim N(0,I)$을 추출합니다.

그런 다음 discriminator $MLP-2_{disc}$의 파라미터를 업데이트하여$logMLP-2_{disc}(z)+log(1-MLP-2_{disc}(h))$를 최소화합니다.

마지막으로 인코더의 파라미터는 $logMLP-2_{disc}(h)$를 최대화 하도록 업데이트되어 인코더가 discriminator를 속이도록 훈련됩니다.

결과적으로 인코더는 사전 분포와 일치하고, 입력의 재구성을 위해 공동으로 최적화 됩니다.

Prediction time에 우리는 discriminator를 버리고, 모든 고려된 항목에 대한 예측을 얻기 위해 한 번의 인코딩 및 디코딩 패스를 수행합니다.



**$Application\ to\ playlist\ continuation$.** playlist continuation의 경우 고려할 item은 트랙입니다.

입력측에서 트랙은 L1 정규화된 bag-of-tracks vector에 의해 표현됩니다.

원하는 출력은 각 트랙에 대한 추정 확률 $p(track\vert playlist)$입니다.

별개의 트랙 수가 많기 때문에 고려된 트랙의 수를 가장 빈번한 $n_{track}$으로 제한하고, 이 트랙은 하이퍼 파라미터로 제어됩니다.

확률이 추정 된 후, 트랙이 내림차순으로 순위를 매기기 전에 원래 재생 목록에 이미 존재하는 트랙을 제거합니다.

우리는 모델에 side information, 즉 재생 목록 타이틀, 트랙 타이틀, 아티스트 및 앨범 등을 통합합니다.

이를 위해 먼저 재생 목록의 각 트랙에 대해 이러한 모든 메타 데이터가 포함된 문자열을 생성하여 이 정보를 집계합니다.

side information은 디코더가 예측을 하기 위해 그것을 사용하기 전에 AE의 $Code$ 블록과 연결됩니다.

이런 식으로 훈련하는 동안 모델의 파라미터는 이미 널리 사용된 트랙 세트 또는 제공된 side information에서 예측에 필요한 정보를 얻도록 최적화됩니다.

예측 단계의 예시로, "Workout"재생 목록에서  "Walk of like" 트랙이 존재할 때 "We are the champion"이 연속되길 원하면(Fig.1) 이 경우 입력 트랙 세트는 단일 트랙 "Walk of like"로만 구성 되며, 해당 트랙의 벡터(one-hot)는 $Code$에 매핑됩니다.

그런 다음 $Code$는 재생 목록 타이틀 "Workout"과 이미 널리 사용되는 트랙의 모든 단어(이 경우 "Work","of","life")로 구성된 bag-of-words 표현과 연결됩니다.

디코더는 $Code$와 side information 모두를 입력으로 받아 "We are the champion"트랙에 대한 높은 확률을 추정합니다.



## Experiments

<hr>

챌린지의 목적은 automatic playlist continuation입니다.

playlist features와 some initial tracks 집합이 주어질 때, 시스템은 플레이리스트에 추가 가능한 추천 트랙 리스트를 생성합니다.

입력은 사용자가 만든 재생 목록으로, 일부 재생 목록 메타 데이터와 재생 목록의 K 트랙 목록으로 표시됩니다.(K = 0, 1, 5, 10, 15 ,25 or 100)

출력은 500개의 추천 후보 트랙 목록이며, 내림차순으로 정렬됩니다.

초기 시드 트랙이 제공되지 않은 재생 목록에 대처해야 합니다.

제출물의 성능을 평가하기 위해 출력 트랙 예측이 원래 재생목록의 GT 트랙과 비교됩니다.



### Procedure

<hr>

**$Preliminary\ experiments$.** 

$P$x$T$ matrix를 자르는 대신 선행 연구에서 처럼 테스트 세트를 위해 일부 행(재생목록)을 무작위로 선택합니다.

하이퍼 파라미터는 초기에 다른 데이터 세트에 대한 이전 실험을 기반으로 선택됩니다.

hidden layer의 size는 100이고 ReLU를 사용합니다.

dropout == 0.2이고, 최적화는 초기 학습 속도 0.001로 Adam에 의해 수행됩니다.

두 개의 오토 인코더 variants는 50 크기의 $Code$를 사용합니다.

또한 AAE에 대한 가우시안 사전 분포를 선택합니다.

SVD의 경우 특이 값의 수를 최대 1,000개 까지 연속적으로 늘렸습니다.

특이값이 많을 수록 성능이 저하되었습니다.

전처리 단계로서 훈련 세트에서 50,000개의 가장 빈번한 트랙을 추출했습니다.

해당 item만 유지하기 위해 training 및 test set을 모두 필터링했습니다.

또한 하나 이상의 트랙이 남아있는 재생 목록을 제거하여 테스트를 위해 삭제 해야할 트랙이 하나 이상 있습니다.

극단적인 경우 주어진 재생 목록에 대한 입력으로 트랙이 없습니다.

**$Optimization\ on\ the\ development\ set$.** 우리는 예비 실험에서 가장 유망한 접근법 즉, AAE를 선택하고 $dev set$을 활용하는 도전 과제에 최적화했습니다.

학습 세트에서는 $dev set$에 속하지 않은 모든 재생 목록이 포함됩니다.

데이터 세트를 사전 처리하기 위해 먼저 고려된 메타 데이터에서 가장 자주 사용되는 50,000개의 단어만 포함하여 학습 세트에 어휘(플레이리스트 타이틀, 트랙 타이틀, 아티스트 네임, 앨범 타이틀)를 작성했습니다.

이후에 모델을 $n_{tracks}$의 가장 빈번한 개별 트랙으로 제한합니다.

그 이유는 덜 빈번한 item에 대해 사용 가능한 트레이닝 데이터가 너무 적고, 이를 고려하면 전체 성능에 해를 끼칠 수 있기 때문입니다.

고려 된 트랙($n_{tracks}$), 히든 유닛, epoch 수, 및 $Code$의 크기 등 다양한 구성을 테스트했습니다. 각 파라미터에 대해 테스트된 값은 아래와 같습니다.

<img src = "https://py-tonic.github.io/images/AAE/4.PNG">

Google news와 $n_{tracks}$가 50,000인 사전 정의된 어휘에서 $Code$크기 뿐만 아니라 히든 유닛 및 epochs의 다양한 값을 시도했습니다.

그런 다음 $n_{tracks}$를 변경하면서 최고 성능 값(200, 20 및 100)을 선택합니다.

재생 목록 타이틀을 사용한는 것이 효과적이므로 더 많은 메타데이터를 활용하면 결과가 더 향상 될 것으로 예상했습니다.

이러한 이유로, 트랙 타이틀, 아티스트 및 앨범과 함께 재생 목록 타이틀에 의존하고 재생 목록 타이틀에 의존하는 AAE모델의 모든 구성을 실행합니다.

더 많은 메타 데이터를 고려할 때 코드에 연결된 모든 메타 데이터가 포함된 문자열을 생성하고 이 정보를 집계했습니다.

전체적으로 20개의 서로 다른 구성을 실행합니다.

**$Final\ experiments$.** 마지막으로 우리는 챌린지 세트에 대한 몇가지 실험을 수행합니다. 우리는 모델에 트랙 메타 데이터와 함께 플레이리스트 타이틀을 제공하여 $devset$에서 최적화 하는 동안 효과적임을 입증했습니다.

학습은 이제 도전 과제의 모든 재생 목록을 고려합니다.

$devset$에서 최적화를 위해 수행된 것과 동일한 전처리 단계를 적용하고 히든 유닛, epochs 및 $Code$ 크기를 각각 200, 20 및 100으로 설정하여 이전과 같이 $n_{tracks}$을 변경하는 몇 가지 구성을 테스트했습니다.



### Results

<hr>

예비 실험의 결과를 아래의 표에 나타냈습니다.

각 모델을 재생 목록 타이틀을 사용하여 한 번, 제목없이 한 번 실행합니다.

Item co-occurrence(IC)는 타이틀을 이용할 수 없으며 MLP는 타이틀에만 의존할 수 있습니다.

테스트 된 모든 사례에서 타이틀을 사용하면 성능이 향상되었습니다.

AAE는 타이틀 유무에 관계없이 가장 높은 MRR을 얻었습니다.

그러므로 $devset$에서 이 모델을 추가로 최적화하기로 결정했습니다.



<img src = "https://py-tonic.github.io/images/AAE/5.PNG">

## Reference

<hr>

[Paper]( https://dl.acm.org/doi/pdf/10.1145/3267471.3267476?download=true )

