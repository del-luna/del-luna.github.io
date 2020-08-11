---
layout: post
title: Pytorch tutorial
author: Jaeheon Kwon
categories: Python
tags: [pytorch,tips]
---



오랜만에 코딩하려니까 프레임워크에 적응이 되지 않아서, 토치 공식 튜토리얼과 구글링을 통해 얻은 정보를 기록하려고 합니다. 토치에 언제쯤 적응할 수 있을까요 ... :(

코드펜스에서 >>>로 표기된건 출력입니다.



## in-place

```python
x = torch.rand(5,3) #0~1균등분포에서 난수 생성 
y = torch.rand(5,3)
torch.add(x,y) # origin
y.add_(x) # in-place
```



## view

토치는 np나 tf에서 사용되는 reshape기능을 view로 할 수 있습니다.

```python
x = torch.randn(4, 4) # 표준 정규 분포를 따르는 난수 생성
y = x.view(16)
z = x.view(-1,8)
print(x.size(), y.size(), z.size())
>>>torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```



## change between numpy and torch

```python
a = np.ones(5)
b = torch.from_numpy(a) # change np to torch

a = torch.ones(5)
b = a.numpy() # change torch to np
```



## CUDA Tensors

```python
if torch.cuda.is_available():
	device = torch.device('cuda') # cuda device object 
    y = torch.ones_like(x, device=device) # GPU에 직접 생성
    y = x.to(device) # .to('cuda')를 통해 변환
    z = x + y
    z.to('cpu', torch.double) # ''.to''는 dtype도 변경함.
```



## Autograd

autograd 패키지는 텐서의 모든 연산에 대해 자동 미분을 제공합니다.

Define-by-run 프레임워크로, 코드를 어떻게 작성하여 실행하느냐에 따라 backprop이 정의되며, backprop은 학습 과정의 매 스텝마다 달라집니다.

**torch.Tensor** 클래스는 **.requires_grad**가 **True**인 경우 그 텐서에서 이뤄진 모든 연산들을 track하기 시작합니다. 계산이 완료된 후 **.bakcward()**를 호출하여 모든 그래디언트를 자동으로 계산할 수 있고, 이 텐서의 그래디언트는 **.grad**에 누적됩니다.

텐서가 기록을 track하는 것을 멈추려면, **.detach()**를 호출하여 연산 기록으로부터 분리(detach)하여 연산들이 추적되는 것을 방지할 수 있습니다.

기록을 track하는 것(과 메모리를 사용하는 것)을 방지하기 위해, 코드 블럭을 **with torch.no_grad():** 로 감쌀 수 있습니다. 이는 그래디언트는 필요 없지만 requires_grad=True 로 설정되어 학습 가능한 매개변수를 갖는 모델을 평가할 때 유용합니다.

**Fucntion** 클래스 또한 중요합니다. **Tensor**와  **Function**은 연결되어 있습니다.

각 텐서는 **grad_fn**속성을 가지고 있는데, 이는 텐서를 생성한 함수를 참조하고 있습니다.(단 사용자가 만든 텐서는 예외로 **grad_fn = None** 입니다.)

도함수를 계산하기 위해서는 **Tensor**의 **.backward()**를 호출하면 됩니다. 만약 텐서가 스칼라인 경우( 하나의 값만 갖는 등) backward에 인자를 지정해줄 필요가 없습니다. 하지만 여러 개의 요소를 갖고 있을 때는 텐서의 모양을 **gradient**의 인자로 지정할 필요가 있습니다.



```python
import torch
x = torch.ones(2,2, requires_grad=True)
y = x+2
z = y*y*3
out = z.mean()
```

```python
print(y,z,out)
>>>tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
>>>tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)
>>>tensor(27., grad_fn=<MeanBackward0>)
```

보는 것 처럼 각각의 텐서가 연산되는 Function과 매핑되서 grad_fn에 기록되는 것을 볼 수 있습니다.



```python
print(x.grad)
>>>None
out.backward()
print(x.grad)
>>>tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

out.backward()를 통해 도함수를 계산하면 .grad에 그래디언트가 누적됩니다.



```python
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)
    
>>>True
   True
   False
```

with torch.no_grad()로 감싸서 트래킹을 방지할 수 있습니다.

