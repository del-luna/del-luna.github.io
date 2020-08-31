---
layout: post
title: Statistics modeling two cultures
author: Jaeheon Kwon
categories: Papers
tags: [statistics]
---





데이터 분석을 통한 두 가지 목표

- Prediction: 입력 x를 통한 종속 변수 y의 예측
- Information: 입력 변수와 종속 변수간의 관계를 나타내는 방법에 대한 정보 추출



위 목표를 위한 두 가지 접근 방법



### 1. The data modeling culture

이 분야 사람들은 대부분의 전통 통계학자들(98%)이고, 블랙 박스 안에 확률적 데이터 모델이 존재한다고 가정함.

예를 들어 일반적인 데이터 모델은 다음과 같이 데이터를 독립적으로 도출함

response variables = f(predictor variables, random noise, parameters)



파라미터는 데이터로 부터 추정되고, 모델은 정보 및 예측에 사용된다. 위 블랙박스에 들어가는 모델로는 linear regression, logistic regression 등이 있다.

- 우선 데이터 생성 방법에 대한 그럴듯한 메커니즘을 구축한다.
- 연구자는 직관, 경험을 통해 독립변수와 종속 변수를 연결하는 선형 방정식을 생각한다.
- 모델 계수(가중치)는 데이터 세트에 피팅하여 찾는다.



결과 선형 방정식은 실제 데이터 생성 메커니즘 즉, 자연(nature 인데 뭐라 번역해야될 지 모르겠다..)이 종속변수 및 독립 변수 값을 생성하는 블랙 박스를 나타낸다.

검증이 발생하는 경우 데이터 모델링에서 $R^2$ 또는 residual analysis와 같은 적합도 측정으로 수행된다. 둘 다 훈련 데이터 세트에서 측정됨.

예측 정확도에 대해서는 거의 생각하지 않는 대신 모델이 연구중인 현상을 얼마나 잘 설명하는지에 중점을 둔다.



요약 : high interpretability low performance



### 2. The alorithmic modeling culture

논문쓸 당시에는 비인기 분야(약2%)

알고리즘 모델을 사용하고 데이터 메커니즘을 알 수 없는 것으로 취급한다.

모델을 선택할 때 모델이 데이터를 생성하는 기본 메커니즘을 나타내는 지 여부는 고려하지 않고 새로운(또는 홀드아웃)관찰에 대해 신뢰할 수 있는 추정을 할 수 있는지 여부만 고려한다.

논문에 나오는 대표적인 예시 모델로는 Random forest, SVM, Neural Network 등이 있다.

predict validation accuracy가 높은 모델을 선택함.(걍 딥러닝이네 ㅋㅋ..)

이쪽 분야의 중심 아이디어는 자연이 블랙 박스이고, 모델도 블랙 박스 라는 것이다. 하지만 새로운 관찰에 대한 예측은 제공 가능하다.

기본 목표가 모델링을 통해 자연에 대한 정보를 추출하는 것이더라도 가장 우선순위가 정확도여야 한다. 복잡하고 정확한 모델은 특징과 대상간의 관계 일부를 포착해야하기 때문에 문제 영역에 대해 알려줄 수 있다고 한다.

논문의 저자는 위에 두 가지 목표에 복잡하더라도 성능이 높은 algorithmic modeling이 더 적합하다고 주장함. 



요약 : low interpretability high performance