---
layout: post
title: Collaborative Filtering for Implicit Feedback Datasets
author: Jaeheon Kwon
categories: Ai
tags: [Recommend]
---



Implicit Feedback은 유저가 싫어하는 아이템에 대한 실질적인 증거가 부족함.(explicit에 비해)

따라서 본 논문에서는 implicit feedback의 고유한 속성을 식별한다.

데이터를 다양한 컨피던스와 관련된 긍정적이고, 부정적인 선호도의 지표로 취급하는 것을 제안한다.

우선 implicit feedback의 고유한 특성에 대해 알아보자.

- No negative feedback
- Noisy
- numerical value of implicit feeback indicates confidence
- Evaluation of implicit-feedback recommender requires appropriate measures

논문의 핵심은 기존의 explicit feedback과 달리 implicit이 고유하게 가지는 특성들이 존재하므로(특히 선호도를 정량화해서 나타내기 힘들다거나, negative feedback이 없다는점) 이를 고려해서 confidence로 접근하는 것이다.

$r_{ui}$

- Explicit : 레이팅
- Implicit : 선호도

이진 변수 $p_{ui}$ 는 선호도를 나타내는 indicate function

$$p_{ui} = \begin{cases} 0, \ \ \ r_{ui}>0  \\ 1, \ \ \ r_{ui}=0\end{cases}$$

그런데 $r_{ui} = 0$ 일 지라도 그 아이템을 좋아하지 않는다고 말할 수 있을까? implicit은 앞서 말했던 것 처럼 noisy 하다.

따라서 이를 보완해줄 다른 방법을 제안한다.

$$c_{ui} = 1 + \alpha r_{ui}$$

보는 것 처럼 $r_{ui}$ 가 증가할 수록 유저가 아이템을 좋아한다는 ''신뢰도''가 강해지는 것을 정량화 했다.

따라서 이를 포함하는 로스 함수는 다음과 같다.

$$\min \sum\limits_{u,i} c_{ui}(p_{ui} - x_u^Ty_i)^2 + \lambda(\sum\limits_u \vert\vert x_u \vert\vert^2 + \sum\limits_i \vert\vert y_i \vert\vert^2)$$

위 식에서  선호도 $p_{ui}$ 는 기존의 user-item factor의 내적으로 사용한다.(위의 indicate랑 헷갈릴까봐 다시 적어준듯)



## Referecne

- [paper](http://yifanhu.net/PUB/cf.pdf)



