---
layout: post
title: 프로그래머스 자연수 뒤집어 배열로 만들기
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 자연수 뒤집어 배열로 만들기(python)

출처 : [자연수 뒤집어 배열로 만들기]( https://programmers.co.kr/learn/courses/30/lessons/12932 )

###### 문제 설명

자연수 n을 뒤집어 각 자리 숫자를 원소로 가지는 배열 형태로 리턴해주세요. 예를들어 n이 12345이면 [5,4,3,2,1]을 리턴합니다.

##### 제한 조건

- n은 10,000,000,000이하인 자연수입니다.

##### 입출력 예

| n     | return      |
| ----- | ----------- |
| 12345 | [5,4,3,2,1] |

**solution.py**

```python
def solution(n):
     return [int(i) for i in str(n)][::-1]
```

[::-1] 로 리스트를 뒤집을 수 있다! 

Pythonic한 코드 너무 좋다.