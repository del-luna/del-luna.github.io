---
layout: post
title: Keras Neural Style Transfer
author: Jaeheon Kwon
categories: Ai
tags: [keras]
---
# Keras Neural Style Transfer를 구현해보자.

들어가기에 앞서 잠깐 기본적인 개념을 설명하자면  
Style transfer 구현 이면에 있는 핵심 개념은 모든 딥러닝 알고리즘과 동일합니다.  
목표를 표현한 Loss function을 정의하고 이 Loss를 최소화 합니다.  
여기서 원하는 것은 reference image의 style을 적용하면서 original image의 content를 보존하는 것입니다.  
콘텐츠와 스타일을 수학적으로 정의할 수 있다면 Loss function은 다음과 같을 것입니다.  
<br>  

Loss = distance(style(reference_image) - style(generated_image)) +  
       distance(content(original_image) - content(generated_image))  

여기서 distance는 L2 Norm 같은 Norm function입니다.  
Content function은 이미지의 content 표현을 계산합니다.  
Style function은 이미지의 style 표현을 계산합니다.  
<br>
이 Loss를 최소화하면 style(generated_image)는 style(reference_image)와 가까워 지고, content(generated_image)는 content(original_image)와 가까워 집니다. 
<br>

## Content Loss

우리는 CNN을 통해 Network의 하위 층의 activation은 이미지에 대한 Local imformation을 담고 있고, 상위 층의 activation은 이미지에 대한 Global & Abstract information을 담고 있는 것을 알 수 있습니다.  

다른방식으로 생각하면 Conv층의 activation은 이미지를 다른 크기의 content로 분해한다고 볼 수 있습니다.  Convnet의 상위 층의 표현을 사용하면 Global & Abstract Content를 찾을 것입니다.  

target_image와 generated_image를 pre-trained Convnet에 주입하여 상위 층의 활성화를 계산합니다. 이 두 값 사이의 L2 Norm이 Content Loss로 사용하기 좋습니다.  
상위 층에서 보았을 때 generated_image와 original_image를 비슷하게 만들 것입니다. 
Convnet의 상위 층에서 보는 것이 Input(original_image)의 Content라고 가정하면 이미지의 Content를 보존하는 방법으로 사용할 수 있습니다.  

## Style Loss

Content Loss는 상위 층만 사용합니다.  
게티스 등이 정의한 Style Loss는 Convnet의 여러 층을 사용합니다.  
하나의 Style이 아니라 reference_image에서 Convnet이 추출한 모든 크기의 Style을 잡아야 합니다.  
게티스 등은 층의 activation output의 Gram matrix를 Style Loss로 사용했습니다.
[Gram matrix란?](https://ko.wikipedia.org/wiki/%EA%B7%B8%EB%9E%8C_%ED%96%89%EB%A0%AC)  
Gram matrix는 층의 feature map들의 내적입니다.  
내적은 층의 특성 사이사이에 있는 상관관계를 표현한다고 이해할 수 있습니다.  
이런 특성의 상관관계는 특정 크기의 공각적인 패턴 통계를 잡아냅니다.  
경험에 비추어 보았을 때 이 층에서 찾은 텍스처에 대응됩니다.  

Style reference_image와 generated_image로 층의 activation을 계산합니다.  
Style Loss는 그 안에 내재된 상관관계를 비슷하게 보존하는 것이 목적입니다.  
결국 Style referenc_image와 generated_image에서 여러 크기의 텍스처가 비슷하게 보이도록 만듭니다.  
<br>
요약하면 사전 훈련된 Convnet을 사용하여 다음 Loss들을 정의할 수 있습니다.

- 콘텐츠를 보존하기 위해 타깃 콘텐츠 이미지와 생성된 이미지(generated_image) 사이에서 상위층의 활성화를 비슷하게 유지합니다. 이 Convnet은 타깃 이미지와 생성된 이미지 사이에서 동일한 것을 보아야 합니다. 
- 스타일을 보존하기 위해 저수준 층과 고수준 층에서 활성화 안에 상관관계를 비슷하게 유지합니다. 특성의 상관관계는 텍스처를 잡아냅니다. 생성된 이미지와 스타일 참조 이미지는 여러 크기의 텍스처를 공유할것 입니다.
  <br>

일반적인 과정은 다음과 같습니다.  

1. 스타일 참조 이미지, 타깃 이미지, 생성된 이미지를 위해 VGG19의 층 활성화를 동시에 계산하는 네트워크를 설정합니다.
2. 세 이미지에서 계산한 층 활성화를 사용하여 앞서 설명한 손실 함수를 정의합니다. 이 손실을 최소화하여 스타일 트랜스퍼를 구현할 것입니다.
3. 손실 함수를 최소화할 경사 하강법 과정을 설정합니다.
   <br>

이제 어떻게 하는 것인지 알아봅시다!  



```python
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, save_img

# 변환하려는 이미지 경로
target_image_path = './datasets/han.jpg'
# 스타일 이미지 경로
style_reference_image_path = './datasets/gogh.jpg'

# 생성된 사진의 차원
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)
```

```
Using TensorFlow backend.
```

한옥을 고흐의 별이빛나는 밤에 풍으로 Style transfer 해보자!

```python
import numpy as np
from keras.applications import vgg19

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # ImageNet의 평균 픽셀 값을 더합니다. vgg19.preprocess_input 함수에서 일어나는 변환을 복원합니다.
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB' vgg19.preprocess_input 함수에서 일어나는 변환을 복원하기 위해서
    x = x[:, :, ::-1] # ::-1 은 해당 인덱스를 뒤집음 즉 위의 주석처럼 channel = BGR -> RGB가 됨
    x = np.clip(x, 0, 255).astype('uint8') # np.clip 으로 값 범위 제한
    return x
```

VGG19 네트워크를 설정해 보죠.  
스타일 참조 이미지, 타깃 이미지 그리고 생성된 이미지가 담긴 플레이스홀더로 이루어진 배치를 입력으로 받습니다.  
플레이스홀더는 심볼릭 텐서로 넘파이 배열로 밖에서 값을 제공해야 합니다.  
스타일 참조 이미지와 타깃 이미지는 이미 준비된 데이터이므로 K.constant를 사용해 정의합니다.  
반면 플레이스홀더에 담길 생성된 이미지는 계속 바뀝니다.



```python
from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# 생성된 이미지를 담을 플레이스홀더
combination_image = K.placeholder((1, img_height, img_width, 3))

# 세 개의 이미지를 하나의 배치로 합칩니다
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# 세 이미지의 배치를 입력으로 받는 VGG 네트워크를 만듭니다.
# 이 모델은 사전 훈련된 ImageNet 가중치를 로드합니다
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('모델 로드 완료.')
```


```
모델 로드 완료.
```

input_tensor 는 타깃이미지, 스타일 이미지, 생성된 이미지를  
(axis=0)행으로 쌓게 됩니다.  
결국 input_tensor의 차원은 (3,400,width,3) 이 됩니다.

콘텐츠 손실을 정의해 보죠.  
VGG19 컨브넷의 상위 층은 타깃 이미지와 생성된 이미지를 동일하게 바라봐야 합니다.

```python
# 위에서 설명했듯 Content Loss는 생성된 이미지와 Input간의 Norm과 같다.
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
```

다음은 스타일 손실입니다.  
유틸리티 함수를 사용해 입력 행렬의 그람 행렬을 계산합니다.  
이 행렬은 원본 특성 행렬의 상관관계를 기록한 행렬입니다.

```python
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    #내적은 Layer의 feature 사이사이에 있는 상관관계를 표현한다고 이해할 수 있습니다.
    #그리고 이러한 상관관계가 위에서 말했듯 공간적인 패턴이 된다.
    gram = K.dot(features, K.transpose(features)) 
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
```

두 손실에 하나를 더 추가합니다.  
생성된 이미지의 픽셀을 사용해 계산하는 총 변위 손실입니다.  
이는 생성된 이미지가 공간적인 연속성을 가지도록 도와주며 픽셀의 격자 무늬가 과도하게 나타나는 것을 막아줍니다.  
이를 일종의 규제 항으로 해석할 수 있습니다.

```python
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

```

최소화할 손실은 이 세 손실의 가중치 평균입니다.  
콘텐츠 손실은 block5_conv2 층 하나만 사용해서 계산합니다.  
스타일 손실을 계산하기 위해서는 하위 층과 상위 층에 걸쳐 여러 층을 사용합니다.  
그리고 마지막에 총 변위 손실을 추가합니다.

사용하는 스타일 참조 이미지와 콘텐츠 이미지에 따라 content_weight 계수(전체 손실에 기여하는 콘텐츠 손실의 정도)를 조정하는 것이 좋습니다.  
content_weight가 높으면 생성된 이미지에 타깃 콘텐츠가 더 많이 나타나게 됩니다.

```python
# 층 이름과 활성화 텐서를 매핑한 딕셔너리
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# 콘텐츠 손실에 사용할 층
content_layer = 'block5_conv2'
# 스타일 손실에 사용할 층
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# 손실 항목의 가중치 평균에 사용할 가중치
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# 모든 손실 요소를 더해 하나의 스칼라 변수로 손실을 정의합니다
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)
for layer_name in style_layers: # 각 타깃 층에 대한 스타일 손실을 더함.
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image) #combination_image는 총 변위 손실을 더함

```


마지막으로 경사 하강법 단계를 설정합니다.   
게티스의 원래 논문에서 L-BFGS 알고리즘을 사용하여 최적화를 수행했으므로 여기에서도 이를 사용하겠습니다.  
L-BFGS 알고리즘은 싸이파이에 구현되어 있는데 두 가지 제약 사항이 있습니다.  

- 손실 함수의 값과 그래디언트 값을 별개의 함수로 전달해야 합니다.  
- 이 함수는 3D 이미지 배열이 아니라 1차원 벡터만 처리할 수 있습니다.  
  <br>

손실 함수의 값과 그래디언트 값을 따로 계산하는 것은 비효율적입니다.  
두 계산 사이에 중복되는 계산이 많기 때문입니다. 한꺼번에 계산하는 것보다 거의 두 배 가량 느립니다.   
이를 피하기 위해 손실과 그래디언트 값을 동시에 계산하는 Evaluator란 이름의 파이썬 클래스를 만들겠습니다.  
처음 호출할 때 손실 값을 반환하면서 다음 호출을 위해 그래디언트를 캐싱합니다.

```python
# 손실에 대한 생성된 이미지의 그래디언트를 구합니다
grads = K.gradients(loss, combination_image)[0]

# 현재 손실과 그래디언트의 값을 추출하는 케라스 Function 객체입니다
fetch_loss_and_grads = K.function([combination_image], [loss, grads])


class Evaluator(object): # 이 클래스는 fetch_loss_and_grads 호출을 감쌉니다. 싸이파이 옵티마이저에서
                         # 호출할 수 있도록 손실과 그래디언트를 각각 반환하는 2개의 메서드를 만듭니다.
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

```

마지막으로 싸이파이 L-BFGS 알고리즘을 사용하여 경사 하강법 단계를 수행합니다.  
알고리즘 반복마다 생성된 이미지를 저장합니다(여기에서는 한 번 반복이 경사 하강법 단계 20번입니다)

```python
from scipy.optimize import fmin_l_bfgs_b
import time

result_prefix = 'style_transfer_result'
iterations = 20
# 뉴럴 스타일 트랜스퍼의 손실을 최소화하기 위해 생성된 이미지에 대해 L-BFGS 최적화를 수행합니다
# 초기 값은 타깃 이미지입니다
# scipy.optimize.fmin_l_bfgs_b 함수가 벡터만 처리할 수 있기 때문에 이미지를 펼칩니다.
x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('반복 횟수:', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20) # 뉴럴 스타일 트랜스퍼의 손실을 최소화 하기위해
    print('현재 손실 값:', min_val)                                       # L-BFGS최적화를 수행합니다.
    # 생성된 현재 이미지를 저장합니다                                           
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('저장 이미지: ', fname)
    print('%d 번째 반복 완료: %ds' % (i, end_time - start_time))

```

```
반복 횟수: 0
현재 손실 값: 3238513400.0
저장 이미지:  style_transfer_result_at_iteration_0.png
0 번째 반복 완료: 85s
반복 횟수: 1
현재 손실 값: 868721660.0
저장 이미지:  style_transfer_result_at_iteration_1.png
1 번째 반복 완료: 84s
반복 횟수: 2
현재 손실 값: 482545860.0
저장 이미지:  style_transfer_result_at_iteration_2.png
2 번째 반복 완료: 84s
반복 횟수: 3
현재 손실 값: 338746020.0
저장 이미지:  style_transfer_result_at_iteration_3.png
3 번째 반복 완료: 84s
반복 횟수: 4
현재 손실 값: 268660450.0
저장 이미지:  style_transfer_result_at_iteration_4.png
4 번째 반복 완료: 84s
반복 횟수: 5
현재 손실 값: 227697400.0
저장 이미지:  style_transfer_result_at_iteration_5.png
5 번째 반복 완료: 84s
반복 횟수: 6
현재 손실 값: 188268720.0
저장 이미지:  style_transfer_result_at_iteration_6.png
6 번째 반복 완료: 84s
반복 횟수: 7
현재 손실 값: 165110020.0
저장 이미지:  style_transfer_result_at_iteration_7.png
7 번째 반복 완료: 84s
반복 횟수: 8
현재 손실 값: 150870600.0
저장 이미지:  style_transfer_result_at_iteration_8.png
8 번째 반복 완료: 84s
반복 횟수: 9
현재 손실 값: 141272270.0
저장 이미지:  style_transfer_result_at_iteration_9.png
9 번째 반복 완료: 84s
반복 횟수: 10
현재 손실 값: 127483400.0
저장 이미지:  style_transfer_result_at_iteration_10.png
10 번째 반복 완료: 84s
반복 횟수: 11
현재 손실 값: 116587570.0
저장 이미지:  style_transfer_result_at_iteration_11.png
11 번째 반복 완료: 88s
반복 횟수: 12
현재 손실 값: 108689020.0
저장 이미지:  style_transfer_result_at_iteration_12.png
12 번째 반복 완료: 84s
반복 횟수: 13
현재 손실 값: 102448680.0
저장 이미지:  style_transfer_result_at_iteration_13.png
13 번째 반복 완료: 84s
반복 횟수: 14
현재 손실 값: 98351760.0
저장 이미지:  style_transfer_result_at_iteration_14.png
14 번째 반복 완료: 84s
반복 횟수: 15
현재 손실 값: 94984270.0
저장 이미지:  style_transfer_result_at_iteration_15.png
15 번째 반복 완료: 84s
반복 횟수: 16
현재 손실 값: 92599310.0
저장 이미지:  style_transfer_result_at_iteration_16.png
16 번째 반복 완료: 84s
반복 횟수: 17
현재 손실 값: 89533540.0
저장 이미지:  style_transfer_result_at_iteration_17.png
17 번째 반복 완료: 84s
반복 횟수: 18
현재 손실 값: 87306380.0
저장 이미지:  style_transfer_result_at_iteration_18.png
18 번째 반복 완료: 84s
반복 횟수: 19
현재 손실 값: 85712490.0
저장 이미지:  style_transfer_result_at_iteration_19.png
19 번째 반복 완료: 84s

```



```python
from matplotlib import pyplot as plt
# 콘텐츠 이미지
plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.figure()

# 스타일 이미지
plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.figure()

# 생성된 이미지
plt.imshow(img)
plt.show()

```

<img src="https://py-tonic.github.io/images/Style_transfer_files/Style_transfer_19_0.png"> 



<img src="https://py-tonic.github.io/images/Style_transfer_files/Style_transfer_19_1.png"> 

<img src="https://py-tonic.github.io/images/Style_transfer_files/Style_transfer_19_2.png"> 


