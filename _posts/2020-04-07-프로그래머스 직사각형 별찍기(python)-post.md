---
layout: post
title: 프로그래머스 직사각형 별찍기
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 직사각형 별찍기(python)

출처 : [직사각형 별찍기]( https://programmers.co.kr/learn/courses/30/lessons/12969 )

###### 문제 설명

이 문제에는 표준 입력으로 두 개의 정수 n과 m이 주어집니다.
별(*) 문자를 이용해 가로의 길이가 n, 세로의 길이가 m인 직사각형 형태를 출력해보세요.

------

##### 제한 조건

- n과 m은 각각 1000 이하인 자연수입니다.

------

##### 예시

입력

```
5 3
```

출력

```
*****
*****
*****
```

**solution.py**

```python
a, b = map(int, input().strip().split(' '))
temp = '*'
for i in range(b):
    print(''.join(list(temp)*a))
```

별찍기라길래 바로 for문부터 떠올렸는데..

```python
a, b = map(int, input().strip().split(' '))
answer = ('*'*a +'\n')*b
print(answer)
```

이렇게도 풀 수 있었습니다.