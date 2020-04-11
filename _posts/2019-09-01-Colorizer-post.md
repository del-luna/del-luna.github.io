---
layout: post
title: Colorizer(with openCV)
date: 2019-09-01 00:00:00
author: Jaeheon Kwon
categories: Ai
tags: [colorizer,opencv]
---

# 흑백 사진 복원하기(with openCV DNN module)

<br>
웹 서핑을하다가 재밌는 사이트를 찾았는데 이 분이 만든 것들이 다 재미있는 주제가 많아서 한번씩 따라 만들어 볼 생각입니다. 
<br>
[파이썬으로 재밌는거 만들기](https://opentutorials.org/module/3811/22946)<br>  
[Github링크](https://github.com/kairess/colorizer)<br>
<br>
<img src = "https://py-tonic.github.io/images/Colorizer_files/result.jpg">
<br>
<br>
들어가기 전에 이번 프로젝트는 openCV의 DNN moudule을 사용합니다.<br>
openCV를 설치해주세요.<br>
또한 사용할 Model도 다운 받아야 하는데,<br>
위의 깃허브 링크에서 get_models.sh를 사용하시면 됩니다.<br>
그럼 시작해보겠습니다!<br>


```python
import cv2 # opencv 3.4.2+ required
import os
import numpy as np
import matplotlib.pyplot as plt
```


```python
proto = './models/colorization_deploy_v2.prototxt' #모델 아키텍쳐 파일
weights = './models/colorization_release_v2.caffemodel' # 모델 Weights 파일 (일반 색체화 모델)
# weights = './models/colorization_release_v2_norebal.caffemodel' (class rebalancing을 안한 모델)

# load cluster centers
pts_in_hull = np.load('./models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)

# load model
net = cv2.dnn.readNetFromCaffe(proto, weights) #저자가 Caffe를 사용함
# net.getLayerNames()

# populate cluster centers as 1x1 convolution kernel
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
# scale layer doesn't look work in OpenCV dnn module, we need to fill 2.606 to conv8_313_rh layer manually
# openCV DNN 모듈에 한 숫자로 채우는 기능을 직접 구현(특정 Layer에 있어야 하는 부분인데 없어서 해주는 것)
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]
```

weights 모델이 두 종류인데<br>
모델을 만든사람에 의하면 norebal모델을 사용하면 약간은 둔하지만 안전한 색체화를 한다고 되어 있습니다.

# Preprocessing


```python
img_path = 'img/sample_23.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_input = img.copy()

# convert BGR to RGB
# 이미지를 Gray_scale로 불러 온다음 RGB로 바꿔줌
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

img_rgb = img.copy()

# normalize input
img_rgb = (img_rgb / 255.).astype(np.float32)

# convert RGB to LAB
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab) #RGB 3채널 체계에서 LAB라는 채널로 변경
# only L channel to be used (0번채널 = L채널)
img_l = img_lab[:, :, 0]

input_img = cv2.resize(img_l, (224, 224))
input_img -= 50 # subtract 50 for mean-centering 논문 쓰신분이 이렇게 mean centering 하셨습니다.

# plot images
# fig = plt.figure(figsize=(10, 5))
# fig.add_subplot(1, 2, 1)
# plt.imshow(img_rgb)
# fig.add_subplot(1, 2, 2)
plt.axis('off')
plt.imshow(input_img, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f3ff04109b0>



<img src = "https://py-tonic.github.io/images/Colorizer_files/Colorizer_5_1.png">


# Prediction


```python
net.setInput(cv2.dnn.blobFromImage(input_img))# image를 blob 데이터로 변환한 뒤 모델의 Input으로 설정
pred = net.forward()[0,:,:,:].transpose((1, 2, 0))# 실제로 예측을 하는 부분

# resize to original image shape
# 원본크기 -> (224, 224) -> Prediction -> 원본크기
pred_resize = cv2.resize(pred, (img.shape[1], img.shape[0]))


# concatenate with original image L
#L 채널을 Input으로 넣으면 AB채널을 예측해주는 모델이므로 concatenate로 다시 합쳐줌 axis=2 즉, 채널방향으로 합쳐서 LAB로 만들어줌)
pred_lab = np.concatenate([img_l[:, :, np.newaxis], pred_resize], axis=2) 

# convert LAB to RGB 위에서 만든 LAB채널을 다시 컴퓨터가 잘 이해하는 RGB로 바꿔줌
pred_rgb = cv2.cvtColor(pred_lab, cv2.COLOR_Lab2RGB)
pred_rgb = np.clip(pred_rgb, 0, 1) * 255
pred_rgb = pred_rgb.astype(np.uint8) #unsigned int로 바꿔줌 잘은 모르겠는데 OpenCV에서 사용하는 것 같습니다.

# plot prediction result
fig = plt.figure(figsize=(20, 10))
fig.add_subplot(1, 2, 1).axis('off')
plt.imshow(img_l, cmap='gray')
fig.add_subplot(1, 2, 2).axis('off')
plt.imshow(pred_rgb)
# plt.savefig(output_filename)

# save result image file
filename, ext = os.path.splitext(img_path)
input_filename = '%s_input%s' % (filename, ext)
output_filename = '%s_output%s' % (filename, ext)

pred_rgb_output = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite(input_filename, img_input)
cv2.imwrite(output_filename, np.concatenate([img, pred_rgb_output], axis=1))
```




    True



<img src = "https://py-tonic.github.io/images/Colorizer_files/Colorizer_7_1.png">

