---
layout: post
title: 프로그래머스 짝수와 홀수
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 짝수와 홀수(python)

출처 : [짝수와 홀수]( https://programmers.co.kr/learn/courses/30/lessons/12937 )

###### 문제 설명

정수 num이 짝수일 경우 Even을 반환하고 홀수인 경우 Odd를 반환하는 함수, solution을 완성해주세요.

##### 제한 조건

- num은 int 범위의 정수입니다.
- 0은 짝수입니다.

##### 입출력 예

| num  | return |
| ---- | :----: |
| 3    |  Odd   |
| 4    |  Even  |

**solution.py**

```python
def solution(num):
    if num%2==0:answer='Even'
    else:answer='Odd'
    return answer
```

