---
layout: post
title: CapsuleNet
author: Jaeheon Kwon
categories: Papers
tags: [vision]
---



데이콘 MNIST대회중 SOTA논문에서 캡슐넷에 영감을 받아 작성한 것을 보고 관련 자료를 찾다가, PR12에서 재준님이 발표하신 영상을 참고해서 정리했습니다.



기존의 CNN이 가지는 문제점

- feature detect는 잘하지만 feature간의 spatial realation을 잘 학습하지 못한다.
- slow increase in receptive field를 개선하기 위해 data augmentation & maxpool을 사용했지만 여기서도 문제가 발생한다.

augmentation을 예시로 들면 조금씩 회전된 이미지의 경우 각각의 경우에 대한 피처를 모두 학습해서 비교합니다.

> 이건 inefficient한 방법이죠? 조금 기울어진 같은 object를 비교하기 위해 low-level feature부터 다시 다 학습해서 비교해야하니까요.

Maxpool이 resolution을 줄이면서 Global한 feature를 보게한 점은 좋았습니다. 그러나 pooling을 통해 Position에 영향을 받지 않는 피처를 학습하고자 했지만, 반대로 position이 달라도 component만 잘 가지고 있으면 높은 확률 값을 출력하게 되었습니다.

<img src = "https://del-luna.github.io/images/capsule/1.jpg">

논문에서 주장하는 바는 캡슐이 이것을 해결할 수 있다!

CNN의 기본 단위가 뉴런 하나라면, 캡슐은 뉴런들의 벡터를 한 단위로 합니다. 기존의 뉴런 하나가 component 하나를 캡처 했다면, 캡슐은 여러개의 뉴런들이 하나의 component를 보면서 추가적으로 다른 정보를 캡처합니다.(e.g. 위의 예시에서 회전 )

아웃풋은 activation vector로 다음과 같이 표현됩니다.(고차원이면 아래의 Orientation도 고차원에서 표현된다고 보면 됩니다.)

- Length: Object가 존재할 확률
- Orientation: Object angle, position과 같은 parameter

<img src = "https://del-luna.github.io/images/capsule/2.jpg">



특이한 점은 Squash라는 activation function을 사용합니다.

$$Squash(u)=\frac{\vert\vert u\vert\vert^2}{1+\vert\vert u\vert\vert^2}\frac{u}{\vert\vert u\vert\vert}$$

유닛벡터에다가 scaling factor를 곱한 형태입니다.(아웃풋이 벡터라 그런지 특이한 형태입니다.)

기존 CNN필터에선 아웃풋의 채널을 그대로 넣지만, (e.g. out_channel=80이면 다음 input은 80 채널을 뉴런으로 생각하면 될 것 같습니다.) 여기선 8(캡슐)x10개 이런 식으로 캡슐의 형태로 Reshape을 해줘야 합니다. 



## Routing Algorithm

캡슐넷의 핵심인 알고리즘입니다. 천천히 알아봅시다.

layer$ l$의 $i$번 째 캡슐의 아웃풋이 $u_i$일 때(e.g. $u_i$는 8차원 벡터), 

FCN에서 처럼 $W_{ij}$를 곱해줘서 회전 변환을 시켜줍니다. 여기서 회전 변환된 벡터 $W_{ij}u_i = \hat u_{j\vert i}$는 prediction vector 라고 부릅니다. 

그리고 어떤 계수 $c_{ij}$를 곱한 뒤 weighted sum을 통해 나온 $\sum\limits_i c_{ij}\hat u_{j\vert i} = s_j$를 squash에 넣은 뒤 나온 아웃풋 $v_j$(아래 그림에선 $u_j$)을 다음 레이어의 인풋으로 넣어줍니다.

<img src = "https://del-luna.github.io/images/capsule/3.jpg">

Predict vector와 아웃풋 벡터간 코사인 유사도를 구해서 비슷한 방향인지를 확인하고, 방향이 비슷하면 coefficient를 강하게 업데이트하는 식인 것 같습니다.



<img src = "https://del-luna.github.io/images/capsule/4.jpg">

첫 번째 캡슐의 첫 번 째 뉴런?에 대한 prediction vector를 계산했을 때 위와 같이 나왔습니다.

그 다음 뉴런에 대해서 예측하면 아래와 같다고 합시다.

<img src = "https://del-luna.github.io/images/capsule/5.jpg">

각각의 prediction vector의 왼쪽은 집, 오른쪽은 보트라고 얘기하겠습니다.

보트들은 첫 번째나 그 다음 뉴런에 대해서도 비슷한 예측을 뱉어냈지만 집은 회전이 섞인 다른 형태의 모양을 예측했습니다.

<img src = "https://del-luna.github.io/images/capsule/6.jpg">

처음에는 모두 같은 가중치로 시작합니다.

하지만 weighted sum을 해주게 되면 보트는 비슷한 형태로 유지되지만, 집은 아래와 같이 이상한 방향을 보이게 됩니다.

<img src = "https://del-luna.github.io/images/capsule/7.jpg">

이제 아웃풋으로 activation vector가 나왔으니 기존의 predict vector들과 코사인 유사도를 바탕으로 가중치를 업데이트 할 수 있습니다.

<img src = "https://del-luna.github.io/images/capsule/8.jpg">

<img src = "https://del-luna.github.io/images/capsule/9.jpg">

보시는 것 처럼 비슷한 방향을 가지는 녀석들 끼리는 유사도가 높으므로 높은 가중치로 업데이트 될 것입니다.



<img src = "https://del-luna.github.io/images/capsule/10.jpg">

<img src = "https://del-luna.github.io/images/capsule/11.jpg">

결국 최종적으로 나온 벡터의 Length를 가지고 분류할 수 있습니다.

하지만 Length는 어떻게보면 클래스의 존재 유무에 대한 확률 값으로 볼 수 있으므로 분류를 위해 각 클래스에 대해 margin loss를 적용합니다.

> low layer에서 length는 삼각형, 사각형의 유무 겠지만, high layer에서는 집, 보트의 유무로 생각할 수 있습니다.

<img src = "https://del-luna.github.io/images/capsule/12.jpg">



그대로 사용해도 되지만 뒤에 FCN레이어를 더 연결하 Reconstruction을 생성할 수 있습니다.

오토 인코더의 구조를 따르는 것 같고, Mnist기 때문에 28*28=784의 아웃풋이며, 마지막 16차원의 캡슐 10개 에서 사용하고자 하는 클래스에 대한 캡슐만 남기고 나머지는 0으로 마스킹해버립니다.

<img src = "https://del-luna.github.io/images/capsule/13.jpg">

<img src = "https://del-luna.github.io/images/capsule/14.jpg">

16차원의 벡터에 대한 각각의 파라미터가 의미하는 건 뭘까요? 아래의 사진처럼 thickness, scale등의 변화를 볼 수 있습니다.

<img src = "https://del-luna.github.io/images/capsule/15.jpg">

<img src = "https://del-luna.github.io/images/capsule/16.jpg">

하위레벨의 캡슐과 상위레벨의 캡슐이 보는 부분이 다르기 때문에 위와 같은 overlap에 대해서 각각을 분리해서 볼 수 있다고 합니다.

> 데이콘 대회랑 태스크가 똑같네요...?

우선 제가 이 논문을 읽은 이유는 Mnist데이터에 대해 적용하기 위해서 읽었지만

cnn이 정말 spatial 정보를 잘 포착하지 못하는가? 못한다면 어떻게 증명할 수 있을까?라는 본질적인 고민을 하게 되는 논문인 것 같습니다.



## Reference

[PR-12 재준님 발표 영상](https://www.youtube.com/watch?v=_YT_8CT2w_Q)

[CapsuleNet Paper](https://arxiv.org/pdf/1710.09829.pdf)