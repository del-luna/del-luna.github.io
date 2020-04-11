---
layout: post
title: 프로그래머스 모의고사
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---

# 프로그래머스 모의고사(python)

출처 : [모의고사]( https://programmers.co.kr/learn/courses/30/lessons/42840 )

###### 문제 설명

수포자는 수학을 포기한 사람의 준말입니다. 수포자 삼인방은 모의고사에 수학 문제를 전부 찍으려 합니다. 수포자는 1번 문제부터 마지막 문제까지 다음과 같이 찍습니다.

1번 수포자가 찍는 방식: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...
2번 수포자가 찍는 방식: 2, 1, 2, 3, 2, 4, 2, 5, 2, 1, 2, 3, 2, 4, 2, 5, ...
3번 수포자가 찍는 방식: 3, 3, 1, 1, 2, 2, 4, 4, 5, 5, 3, 3, 1, 1, 2, 2, 4, 4, 5, 5, ...

1번 문제부터 마지막 문제까지의 정답이 순서대로 들은 배열 answers가 주어졌을 때, 가장 많은 문제를 맞힌 사람이 누구인지 배열에 담아 return 하도록 solution 함수를 작성해주세요.

##### 제한 조건

- 시험은 최대 10,000 문제로 구성되어있습니다.
- 문제의 정답은 1, 2, 3, 4, 5중 하나입니다.
- 가장 높은 점수를 받은 사람이 여럿일 경우, return하는 값을 오름차순 정렬해주세요.

##### 입출력 예

| answers     | return  |
| ----------- | ------- |
| [1,2,3,4,5] | [1]     |
| [1,3,2,4,2] | [1,2,3] |

##### 입출력 예 설명

입출력 예 #1

- 수포자 1은 모든 문제를 맞혔습니다.
- 수포자 2는 모든 문제를 틀렸습니다.
- 수포자 3은 모든 문제를 틀렸습니다.

따라서 가장 문제를 많이 맞힌 사람은 수포자 1입니다.

입출력 예 #2

- 모든 사람이 2문제씩을 맞췄습니다.

<br>

**solution.py**

```python
import numpy as np

def solution(answers):
    answer = []
    temp = []
    counter = []
    supo = [[1,2,3,4,5],[2,1,2,3,2,4,2,5],[3,3,1,1,2,2,4,4,5,5]]

    for a in supo:
        if len(answers)//len(a) > 0: temp.append(a*(len(answers)//len(a)) + a[:len(answers)%len(a)])
        else: temp.append(a[:len(answers)])
    
    counter.append((np.array(answers)-np.array(temp[0])).tolist().count(0))
    counter.append((np.array(answers)-np.array(temp[1])).tolist().count(0))
    counter.append((np.array(answers)-np.array(temp[2])).tolist().count(0))
    
    for i, count in enumerate(counter):
        if count == max(counter):
            answer.append(i+1)
    
    return answer
```

우선 answers 와 학생들이 찍는 패턴 (supo)을 비교하려면 for문을 한번 돌아야 된다고 생각했고,<br>

비교를 위해서 list들의 길이를 맞춰 주려고 했습니다.<br>

길이를 맞춰주는 방법은 다양하겠지만, 나는 우선 answers의 길이를 모르니까 찍는 패턴으로 나눠서 answers > supo 인 경우 몫이 나올텐데 그 몫만큼 길이를 곱해주고 나머지 길이를 더해줬고

반대인 경우에는 supo가 더 길테니까 supo를 answers 길이만큼 slicing해줬습니다.<br>

그 뒤에는 단순합니다.

 answers와 temp(slicing 한 supo)값 을 빼줬을 때(여기서 list끼리 (-)연산이 되지 않아서 numpy를 사용했다.) index의 값이 '0'인 경우가 정답이 맞은 케이스 일 테니, 각 temp별로 0의 개수를 구해줬습니다.<br>

그리곤 enumerate를 활용해 index와 count를 가지고와서 answer에 입력시켰습니다.

<br>

