---
layout: post
title: Pytorch Tips
author: Jaeheon Kwon
categories: Python
tags: [pytorch,tips]
---



# Pytorch 삽질 후기

매번 논문만 읽다가 pytorch로 구현중인데 생각보다 많은 삽질을 하게 되어서 여기에 기록하려 합니다.<br>

**계속 업데이트 될 예정입니다!**<br>

## 공부하기 좋은 사이트

- [tutorial_pytorch]( https://tutorials.pytorch.kr/beginner/saving_loading_models.html )
- [How to Use PyTorch]( https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/ )

<br>

## tqdm

[tqdm_github]( https://github.com/tqdm/tqdm )<Br>

정말 편리한 progress bar 라이브러리 입니다.<br>

사용도 엄청 쉽고 간편합니다.<br>

<img src = "https://jaeheondev.github.io/assets/img/pytorch/tqdm.gif">

```python
for epoch in tqdm(range(training_epoch), position=0):
```

위 처럼 for문 을 감싸주는 형태로 쓰면 되고,<br>

trange를 이용할 수도 있습니다.(자세한 사용법은 위의 github 링크를 참조하세요!)<br>

```python
pbar = trange(training_epoch, desc='Loss : 0', leave=True, position=0)

for epoch in pbar:
```

## Parameter

model을 정의하고 <br>

```python
model.state_dict()
```

위 방법으로 key,value로 레이어명과 텐서를 볼 수 있습니다.(딕셔너리)<br>

특정 레이어의 접근하고 싶은 경우 <br>

```python
model.state_dict()['layer_name']
```

깔끔하게 공식 튜토리얼 처럼 state_dict를 모두 출력할 수도 있습니다.<br>

```python
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
>>>Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])
```



## Scehduler

Scheduler는 종류가 많은데 우선 제가 사용한 것은 loss에따라 learning rate를 바꿔주는 것입니다.

optimizer와 scheduler를 정의한 뒤<br>

```python
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min') 
```

epoch별로 도는 loop 안에다가<br>

```python
for param_group in optimizer.param_groups:
        lr = param_group['lr']
        if epoch%10 == 0:
            print('epoch: {:3d}, lr={:.6f}'.format(epoch, lr))
```



위와 같이 사용하면 loss의 변화량을 볼 수 있습니다.<br>

## Max & ArgMax

torch.max(dim)는 return값이 두 개 입니다.<br>

원소의 최대값과 최대값을 가진 인덱스를 return합니다.<br>

dim 기준으로 가장 큰 값과 인덱스를 보여줍니다.<br>

 ```python
t = torch.FloatTensor([[[1, 11, 9], 
                        [3, 4, 2], 
                        [7, 1, 3]],
                       [[9, 5, 7], 
                        [3, 4, 4], 
                        [1, 6, 3]]])

x = torch.FloatTensor([[1, 2, 6], 
                       [3, 4, 2], 
                       [7, 1, 3]])
 ```

```python
t.shape
>>>torch.Size([2, 3, 3])
```

```python
t.max(0)
>>>torch.return_types.max(
values=tensor([[9., 11., 9.],
        [3., 4., 4.],
        [7., 6., 3.]]),
indices=tensor([[1, 0, 0],
        [1, 1, 1],
        [0, 1, 1]]))
```

<br>

t에 대해 살펴보면 (2,3,3) 의 shpae 을 가지니까 우리는 가장 앞쪽차원에 대해서 max 연산을 수행 할 것입니다.<br>

첫 번째로 [1, 11, 6] & [9, 5, 7]에서 각 index별로 큰 값을 뽑으면 [9, 11, 9],<br>

두 번째로 [3, 4, 2] & [3, 4, 4]에서 뽑으면 [3, 4, 4],<br>

마지막으로 [7, 1, 3] & [1, 6, 3]에서 뽑으면 [7, 6, 3]이 되겠죠?<br>

위의 values 값과 일치하는 것을 볼 수 있습니다.<br>

마찬가지로 indices도 두 가지 (3,3)행렬에 대해 각각 0과 1이 할당된다고 생각하면<br>

위의 value에 대한 index로 비교하면 정확히 똑같은 값이 나옵니다!<br>

헷갈리면 손으로 직접 해보세요!<br>

<br>

```python
x.shape
>>>torch.Size([3, 3])
```

```python
x.max(0)
>>>torch.return_types.max(
values=tensor([7., 4., 6.]),
indices=tensor([2, 1, 0]))
```



## Save

torch.save로 모델을 저장할 수 있습니다.<br>

토치에서는 모델을 저장할 때 .pt 또는 .pth확장자를 사용하는 것이 국룰입니다.<br>

```python
torch.save({
            'epoch': training_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'train_accuracy': train_acc_list,
            'test_accuracy' : test_acc_list
            }, '../save/VGG11_checkpoint.pt')
```



save한 모델을 불러와서 내가 정의한 특정 모델의 레이어에 파라미터 값을 대입하려면<Br>

```python
cp = torch.load('../save/VGG11_checkpoint.pt')
```

```python
model.state_dict()['layer.weight'] = cp['model.stat_dict']['layer.weight']
```

이렇게 하면 값 대입이 되지 않습니다.(아마 데이터 타입은 같은데 저 데이터 타입에서 저렇게 대입이 안되는 것 같다.)<br>

```python
model.state_dict()['layer0.weight'].copy_(cp['model_state_dict']['layer0.weight'])#weight
model.state_dict()['layer0.bias'].copy_(cp['model_state_dict']['layer0.bias'])#bias
```

위 처럼 copy_를 사용하면 됩니다.<br>



## cfg & make_layers

```python
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```

```python
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10) # 32x32 img -> MaxPool:5 = 1x1xC(output)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                
        return nn.Sequential(*layers)
```

위처럼 cfg(config)딕셔너리를 생성하고 (아키텍처의 구조를 대략적으로 기입한 뒤)<br>

make_layers로 layer들을 쌓으니 아주 깔끔하고 보기 좋았습니다!<br>

(처음에 깃허브에서 저걸 보고 신세계였는데 생각보다 자주 쓰이는 테크닉인 것 같습니다.)

cfg가 커지면 따로 .cfg 파일로 빼서 사용한다고 합니다.(물론 이 경우 따로 파싱하는 코드가 필요합니다.)<br>

## Initializer

Torch는 기본적으로 제공되는 초기화가 있습니다.<br>

<img src = "https://jaeheondev.github.io/assets/img/pytorch/0.png">

Linear layer의 경우 위 처럼, Conv layer의경우 아래처럼 초기화 된다고 합니다.<br>

기본 제공되는 초기화 함수로도 어느정도의 성능은 얻을 수 있지만,<br>

xavier나 He같은 유명한 알고리즘을 쓰면 더 좋은 성능을 얻을 수 있습니다.<br>

보통의 경우 initializer 함수를 만들어서 쓰는데 저의 경우엔 아래와 같이 함수를 만들어 사용했습니다.<br>

```python
def init_weights(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1: #Linear일 때
    nn.init.xavier_uniform(m.weight)
    if type(m.bias) != type(None):
      nn.init.constant_(m.bias.data, 0)
   
  elif classname.find('Conv') != -1: #Conv일 때
    nn.init.xavier_uniform(m.weight)
    if type(m.bias) != type(None):
      nn.init.constant_(m.bias.data, 0)

  elif classname.find('BatchNorm') != -1: #Bn일 때
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)
```

