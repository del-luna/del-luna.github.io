---
layout: post
title: 프로그래머스 문자열 다루기 기본
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 문자열 다루기 기본(python)

출처 : [문자열 다루기 기본]( https://programmers.co.kr/learn/courses/30/lessons/12918 )

###### 문제 설명

문자열 s의 길이가 4 혹은 6이고, 숫자로만 구성돼있는지 확인해주는 함수, solution을 완성하세요. 예를 들어 s가 a234이면 False를 리턴하고 1234라면 True를 리턴하면 됩니다.

##### 제한 사항

- `s`는 길이 1 이상, 길이 8 이하인 문자열입니다.

##### 입출력 예

| s    | return |
| ---- | ------ |
| a234 | false  |
| 1234 | true   |

**solution.py**

```python
def solution(s):
    answer=True
    try:int(s)
    except ValueError:answer=False
    if len(s)!=4 and 6:answer=False
    return answer
```

try-except문을 처음 써본 문제입니다.

쓰지 않고 풀려면 아무래도 isdigit()를 사용해서 풀었을 것 같습니다.