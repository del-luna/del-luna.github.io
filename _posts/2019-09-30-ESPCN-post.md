---
layout: post
title: ESPCN
author: Jaeheon Kwon
categories: Paper
tags: [sr]
---

# Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network(ESPCN) - Review

[ESPCN](https://arxiv.org/pdf/1609.05158.pdf) 논문 리뷰입니다.  

처음으로 읽는 논문 및 리뷰이기때문에 부족한점이 많습니다. 양해 부탁드립니다.  

이전까지의 SISR에서의 DNN모델들은 HR space에서 Convolution 연산을 하기 때문에 계산 복잡도가 굉장히 높았습니다.  

그런데 해당 논문은 새로운 방식을 제안하는데 바로 논문 제목인 ESPCN(Efficient Sub-Pixel Convolutional Neural Network)입니다.  

이 subpixel 방식은 아래의 논문에 있는 그림을 통해 직관적으로 이해할 수 있습니다.  

<img src = "https://py-tonic.github.io/images/SR/1.png">

기존의 DNN방식들이 LR -> HR 로 변환 후 Conv연산을 수행해서 SR을 진행했다면 이 논문에서는 Sub-pixel Conv 연산을 이용하여 SR space에서 Conv 연산으로 feature map들을 추출한 후에 pixel들의 재조합으로 HR 이미지를 생성합니다.  

그렇기 때문에 이전의 모델과는 달리 계산에 필요한 비용이 굉장히 낮고 이로 인해 Real-Time으로 가능해서 논문 제목에 Real-Time이 들어갑니다.  

Conv-net부분은 다른 DNN과 동일하므로 sub-pixel 부분만 자세히 다뤄 보겠습니다.  

LR 이미지의 feature map들을 input으로 받아서 HR 이미지를 만드는 Layer입니다.  

논문에선 수식으로 다음과 같이 표현합니다.  

<img src = "https://py-tonic.github.io/images/SR/2.png">

<img src = "https://py-tonic.github.io/images/SR/3.png">

위의 그림과 수식을 비교해서 보시면 아시겠지만 feature map들의 $r^2$개의 채널들에서 pixel 들을 하나 하나 가져와서 순차적으로 붙이는 것입니다.

즉 PS(Periodic Shuffling)함수를 거치기 전의 이미지의 Shape이 (H * W * C$r^2$ ) 이었다면 PS 연산 후 rH * rW * C 로 변하게 됩니다.  

해당 수식을 Numpy로 구현하면 아래와 같습니다.  

```python
def PS(I, r):
  assert len(I.shape) == 3
  assert r>0
  r = int(r)
  O = np.zeros((I.shape[0]*r, I.shape[1]*r, I.shape[2]/(r*2)))
  for x in range(O.shape[0]):
    for y in range(O.shape[1]):
      for c in range(O.shape[2]):
        c += 1
        a = np.floor(x/r).astype("int")
        b = np.floor(y/r).astype("int")
        d = c*r*(y%r) + c*(x%r)
        print a, b, d
        O[x, y, c-1] = I[a, b, d]
  return O
```

TensorFlow로 구현하면 아래와 같습니다.  

```python
def _phase_shift(I, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  #
    bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
  # Main OP that you can arbitrarily use in you tensorflow code
  if color:
    Xc = tf.split(3, 3, X)
    X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
  else:
    X = _phase_shift(X, r)
  return X
```

[source](https://github.com/atriumlts/subpixel)



