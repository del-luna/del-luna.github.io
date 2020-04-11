---
layout: post
title: DCGAN
author: Jaeheon Kwon
categories: Paper
tags: [cnn,gan]
---

#  Deep Convolutional Generative Adversarial Networks 

GAN에 이은 DCGAN입니다.
DCGAN은 GAN의 단점인 학습이 불안정하다는 것을 극복한 모델입니다.
이름에서 알 수 있듯 Convolution layer를 이용하여 이미지의 feature map을 추출하고 그것을 GAN모델에 이용하는 방식입니다.

DCGAN의 Generator 구조는 아래와 같습니다.
<img src = "https://py-tonic.github.io/images/DCGAN/0.PNG">

CNN을 공부하셨다면 의문을 가질만한 그림입니다.

Generator에서 Conv layer를 적용하는데 왜 image size가 커지지?
그 해답은 Transpose Convolution입니다.

<img src = "https://py-tonic.github.io/images/DCGAN/transe.gif">

Transpose Convolution은 예전 [ESPCN]( https://jaeheondev.github.io/ESPCN(review)-post/ )을 다룰 때에도 논문에서 나왔던 내용인데요.
기존의 CNN의 Conv연산을 Stretch하면 아래의 그림처럼 되겠죠?
<img src = "https://py-tonic.github.io/images/DCGAN/T0.PNG">

input과 filter간의 element-wise 곱을 행렬의 곱으로 표현하기위해서 0이 추가된 것일 뿐 
기존의 연산과 동일한 결과물입니다. (Y0를 만들기 위해 어떤 값들이 사용되었는지 확인해보세요!)
이제 여기서 Sparse Matrix C를 Transpose하여 Y vector에 곱합니다.
<img src = "https://py-tonic.github.io/images/DCGAN/T2.PNG">

짜란~ 그러면 Conv연산을 수행했는데 오히려 size가 커지게 되었죠?
이것을 Transpose Conv연산이라고 부릅니다!
역연산처럼 보여서 deconv연산으로 부르는 곳도 있는데 잘못된 표현이라고 합니다!

DCGAN의 장점은
GAN에서의 성능 개선 뿐만이 아닙니다.

NLP에서 쓰이는 Word2Vec이라는 개념이있는데 이게 image에서도 가능합니다.

Word2Vec은 쉽게말해서 단어를 벡터화 한다는 의미입니다.
유명한 예시로는 'King' - 'Man' + 'Woman' = 'Queen' 처럼 단어의 의미를 통해 연산이 가능하게 됩니다.

image에서도 이게 된다는 말은 아래의 사진과 같습니다.

<img src = "https://py-tonic.github.io/images/DCGAN/2.PNG">

놀랍지않나요? 조금 기괴하긴 하지만 의미를 이해하고있다는 점이 중요합니다.
회전 또한 가능합니다.
<img src = "https://py-tonic.github.io/images/DCGAN/3.PNG">



아래의 사진은 논문의 예제에서 DCGAN을 epoch별로 나타낸 그림입니다.
<img src = "https://py-tonic.github.io/images/DCGAN/1.PNG">

이렇게 epoch마다 변해가는 것을 보면서 논문에서는 "Walking in the latent space"라고 표현합니다.

이제 이 DCGAN을 이용하여 mnist 데이터셋을 생성해보겠습니다.


DCGAN으로 생성한 mnist 데이터셋
  <img src = "https://py-tonic.github.io/images/DCGAN/DCGAN.gif">

GAN으로 생성한 mnist 데이터셋
<img src = "https://py-tonic.github.io/images/DCGAN/gan.gif">

어떠신가요? 기존의 GAN과 차이점이 느껴지시나요?
중간중간에 학습이 잘되지않고 뭉친듯한 이미지가 있는데

<img src = "https://py-tonic.github.io/images/DCGAN/011.png">

<img src = "https://py-tonic.github.io/images/DCGAN/020.png">

11.epoch과 20.epoch에서 위와 같은 이미지가 출력되었습니다.

이유를 찾아보니  [mode collapse]( https://www.quora.com/What-causes-mode-collapse-in-GANs )라는 글이 있었습니다.

mode collapse가 생기는 이유는 GAN이 minmax problem을 풀어야 하기 때문이고, 실제 학습을 할 때 가정한 공간과 NN를 사용하면서 생기는 이론적 가정과의 괴리 때문이라고 합니다.

mode collapse를 해결하기위해 Unrolled GAN이라는 게 있던데 다음에 포스팅 해보도록 하겠습니다.

## Implement

[FullCode]( https://github.com/jaeheondev/Implement_GANs) 는 여기서 보실 수 있습니다.

## Referecne

[어쩐지 오늘은]( https://zzsza.github.io/data/2018/02/23/introduction-convolution/ )

[학부생의 딥러닝 DCGAN : Deep Convolutional GAN](https://haawron.tistory.com/9)

[Jaejun Yoo's Playground](http://jaejunyoo.blogspot.com/)
