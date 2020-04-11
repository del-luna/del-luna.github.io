---
layout: post
title: 프로그래머스 문자열 내 마음대로 정렬하기
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 문자열 내 마음대로 정렬하기(python)

출처 : [문자열 내 마음대로 정렬하기]( https://programmers.co.kr/learn/courses/30/lessons/12915 )

###### 문제 설명

문자열로 구성된 리스트 strings와, 정수 n이 주어졌을 때, 각 문자열의 인덱스 n번째 글자를 기준으로 오름차순 정렬하려 합니다. 예를 들어 strings가 [sun, bed, car]이고 n이 1이면 각 단어의 인덱스 1의 문자 u, e, a로 strings를 정렬합니다.

##### 제한 조건

- strings는 길이 1 이상, 50이하인 배열입니다.
- strings의 원소는 소문자 알파벳으로 이루어져 있습니다.
- strings의 원소는 길이 1 이상, 100이하인 문자열입니다.
- 모든 strings의 원소의 길이는 n보다 큽니다.
- 인덱스 1의 문자가 같은 문자열이 여럿 일 경우, 사전순으로 앞선 문자열이 앞쪽에 위치합니다.

##### 입출력 예

| strings           | n    | return            |
| ----------------- | ---- | ----------------- |
| [sun, bed, car]   | 1    | [car, bed, sun]   |
| [abce, abcd, cdx] | 2    | [abcd, abce, cdx] |

##### 입출력 예 설명

**입출력 예 1**
sun, bed, car의 1번째 인덱스 값은 각각 u, e, a 입니다. 이를 기준으로 strings를 정렬하면 [car, bed, sun] 입니다.

**입출력 예 2**
abce와 abcd, cdx의 2번째 인덱스 값은 c, c, x입니다. 따라서 정렬 후에는 cdx가 가장 뒤에 위치합니다. abce와 abcd는 사전순으로 정렬하면 abcd가 우선하므로, 답은 [abcd, abce, cdx] 입니다.

**solution.py**

```python
def solution(strings, n):
    answer = []
    strings.sort()
    c = []
    for i in strings:      
        c.append(i[n])
    c = dict(zip(range(len(c)),c))
    c = list(dict(sorted(c.items(), key=lambda x:x[1])).keys())
    for i in c:
        answer.append(strings[i])
    return answer
```

문제에서 같은 문자열이 여럿 일 경우 사전순으로 앞선 문자열이 앞쪽에 온다고 돼있습니다.

나중에 처리하기 귀찮을 것 같아서 문자열 자체를 먼저 정렬하고 시작했습니다.

c에 strings의 n번째 문자들을 집어 넣은 뒤 딕셔너리를 만들어 주고

value를 기준으로 정렬했습니다.