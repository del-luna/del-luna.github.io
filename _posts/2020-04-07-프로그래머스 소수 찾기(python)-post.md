---
layout: post
title: 프로그래머스 소수 찾기
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 소수 찾기(python)

출처 : [소수 찾기]( https://programmers.co.kr/learn/courses/30/lessons/12921 )

###### 문제 설명

1부터 입력받은 숫자 n 사이에 있는 소수의 개수를 반환하는 함수, solution을 만들어 보세요.

소수는 1과 자기 자신으로만 나누어지는 수를 의미합니다.
(1은 소수가 아닙니다.)

##### 제한 조건

- n은 2이상 1000000이하의 자연수입니다.

##### 입출력 예

| n    | result |
| ---- | ------ |
| 10   | 4      |
| 5    | 3      |

##### 입출력 예 설명

입출력 예 #1
1부터 10 사이의 소수는 [2,3,5,7] 4개가 존재하므로 4를 반환

입출력 예 #2
1부터 5 사이의 소수는 [2,3,5] 3개가 존재하므로 3를 반환

**solution.py**

```python
temp = []
def solution(n):
    prime = [1]*(n+1)
    m = int(n**0.5)
    for i in range(2,m+1):
        if prime[i] == 1:
            for j in range(i+i, n+1, i): prime[j] = 0
    for i in range(2,n+1):
        if prime[i] == 1: temp.append(i)
    answer = len(temp)
    return answer
```

에라스토테네스의 체를 이용해서 풀었습니다.

sqrt(n)까지만 검사하여 효율성을 높였습니다.

