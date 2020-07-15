---
layout: post
title: Cosine similarity vs Euclidean distance
author: Jaeheon Kwon
categories: Ai
tags: [distance]
---



# Cosine similarity vs Euclidean distance



이번 카카오 아레나 대회를 계기로 추천시스템을 공부하다보니 SparseMatrix를 다룰일이 많았다.

추천시스템의 모델 인풋으로 많이 사용되는 User-Item Matrix도 BoW로 표현된 Sparse Matrix이다.

추천을 하려면 보통 유사도를 기반으로한 Collaborative Filtering 방식을 많이 사용하는데, 이 **유사도**를 정의하는 방법이 참 다양하다.

그 중 우리는 우리가 학교에서 배우고 익숙한 Cosine Similarity와 Euclidean distance를 비교해보자.

Matrix는 벡터들의 결합으로 볼 수 있고, 각 벡터들의 유사도를 비교하는 측면에서 우리는 **내적**을 가장 쉽게 떠올릴 수 있다.

우리가 일반적으로 떠올릴 수 있는 유사도를 정의할 수 있는 **거리 함수**들은 대부분 내적의 변형된 버전으로 볼 수 있다.

우리가 어릴 때 부터 배운 좌표평면 상의 두 점의 거리를 구할 때 사용하는 유클리드 거리도 풀어서 쓰면 내적이 들어간다.



$d_{vec}(v1,v2) = \sqrt{v_{1}^{2}+v_2^{2}-2v_1v_2}\tag{1}$



|      | Word1 | Word2 | Word3 | Word4 | Word5 |
| :--: | :---: | :---: | :---: | :---: | :---: |
| Doc1 |   1   |   1   |   1   |       |       |
| Doc2 |       |       |   2   |   1   |   1   |
| Doc3 |   2   |   2   |   2   |       |       |



위 그림에서 Doc1과 Doc3은 같은 단어분포를 가지는 문서이지만, 문서3은 단어를 두 배 많이 사용했습니다.

Doc1과 Doc3의 L2norm은 $\sqrt3$과 $\sqrt{12}$이며 유클리드 거리는 $\sqrt3$입니다.

그런데 Doc1과 Doc2의 유클리드 거리 또한 $\sqrt3$입니다.

Doc1과 Doc3은 벡터 크기에 의해 거리가 생겼고, Doc1과 Doc2는 서로 다른 단어 분포를 사용해서 거리가 생겼습니다.

> 고등학교 때 벡터는 크기와 방향을 갖는다고 배웠죠? 전자는 크기만 다를 뿐 같은 방향의 벡터이고, 후자는 방향이 다른 벡터입니다.



우리는 단어 분포가 비슷할 때 두 문서가 비슷하다고 생각합니다.

코사인 유사도를 사용하면 Doc1과 Doc3의 거리는 0입니다.(같은 벡터니까요!)

코사인 유사도는 아래와 같이 두 벡터의 내적을 두 벡터의 L2norm으로 나눈 값이고, 코사인 거리는 1 - Cosine Similarity입니다.

$d_{cos}(v_1,v_2) = 1 - \frac{v_1v_2}{\vert v_1\vert\vert v_2\vert}\tag{2}$

이는 두 벡터를 unit vector로 바꾼 뒤 내적하는 것과 같습니다.(모든 벡터의 크기가 무시됩니다.)

그리고 내적을 하기 때문에 두 벡터에 공통으로 들어있는 단어들이 무엇인지, 그 비율이 어떤지 측정합니다.

고차원의 Sparse Vector사이 거리는 두 벡터의 포함된 공통 성분(여기서는 단어겠죠?)을 잘 측정하는 것이 중요하기 때문에 Jaccard distrance, Pearson correlation, Cosine distance와 같은 척도를 쓰는것이 좋다고 합니다



## Reference

<hr>

[https://lovit.github.io/nlp/machine%20learning/2018/10/16/spherical_kmeans/](https://lovit.github.io/nlp/machine learning/2018/10/16/spherical_kmeans/)







