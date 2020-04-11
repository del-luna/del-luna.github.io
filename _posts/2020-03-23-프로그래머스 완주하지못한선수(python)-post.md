---
layout: post
title: 프로그래머스 완주하지 못한 선수
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 완주하지못한선수(python)

출처 : [완주하지 못한 선수](https://programmers.co.kr/learn/courses/30/lessons/42576) 

###### 문제 설명

수많은 마라톤 선수들이 마라톤에 참여하였습니다. 단 한 명의 선수를 제외하고는 모든 선수가 마라톤을 완주하였습니다.

마라톤에 참여한 선수들의 이름이 담긴 배열 participant와 완주한 선수들의 이름이 담긴 배열 completion이 주어질 때, 완주하지 못한 선수의 이름을 return 하도록 solution 함수를 작성해주세요.

##### 제한사항

- 마라톤 경기에 참여한 선수의 수는 1명 이상 100,000명 이하입니다.
- completion의 길이는 participant의 길이보다 1 작습니다.
- 참가자의 이름은 1개 이상 20개 이하의 알파벳 소문자로 이루어져 있습니다.
- 참가자 중에는 동명이인이 있을 수 있습니다.

##### 입출력 예

| participant                             | completion                       | return |
| --------------------------------------- | -------------------------------- | ------ |
| [leo, kiki, eden]                       | [eden, kiki]                     | leo    |
| [marina, josipa, nikola, vinko, filipa] | [josipa, filipa, marina, nikola] | vinko  |
| [mislav, stanko, mislav, ana]           | [stanko, ana, mislav]            | mislav |

<br>

**solution.py**

```python
def solution(participant, completion):
    answer = []
    counter = {}
    for a in participant:
        if a not in counter:
            counter[a] = 0
        counter[a] +=1
    for b in completion:
        if b not in counter:
            counter[b] = 0
        counter[b] -=1
    for x,y in list(counter.items()):
        if y != 0:
            answer.append(x)
    return answer[0]
```

딕셔너리 써서 for문 돌면서 딕셔너리에 key : name, value : count 형식으로 등록해서<br>

달린선수는 +count , 완주한 선수는 -count 로 계산했습니다.<br>

그냥 별 생각 없이 '어 카운팅하는 문제네? 딕셔너리 써야겠다!' 라는 생각이 들어서 이렇게 풀었습니다.<br>

### **다른  사람 풀이**

```python
def solution(participant, completion):
    completion.append("z" * 20)
    
    for p_name, c_name in zip(sorted(participant), sorted(completion)):
        if p_name != c_name:
    return(p_name)
```

문제를 읽어보면 <br>

```python
len(participant) = len(completion) + 1
```

임을 알 수 있으므로 임의로 completion에 값을 하나 넣어놓고

zip으로 list를 묶어서 for문을 돌면서 이름이 다르면 return합니다.(다른사람 한명이 바로 정답일테니)



### 다른 사람 풀이 2

```python
import collections


def solution(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer.keys())[0]
```

collections 모듈에 Counter 함수를 이용했습니다.<br>

dictionary는 (-)연산이 안되는데, Counter는 가능해서 저렇게 풀 수 있구나 싶었습니다.

파이썬은 라이브러리 많이 아는게 중요한 것 같습니다.<br>