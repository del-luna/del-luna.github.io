---
layout: post
title: 프로그래머스 수박수박수박..
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 수박수박수박수박수박수?(python)

출처 : [수박수박수박수박수박수?]( https://programmers.co.kr/learn/courses/30/lessons/12922 )

- 수박수박수박수박수박수?

- darklight

  sublimevimemacs

  Python3 

###### 문제 설명

길이가 n이고, 수박수박수박수....와 같은 패턴을 유지하는 문자열을 리턴하는 함수, solution을 완성하세요. 예를들어 n이 4이면 수박수박을 리턴하고 3이라면 수박수를 리턴하면 됩니다.

##### 제한 조건

- n은 길이 10,000이하인 자연수입니다.

##### 입출력 예

| n    | return   |
| ---- | -------- |
| 3    | 수박수   |
| 4    | 수박수박 |

**solution.py**

```python
def solution(n):
    answer = ''
    for i in range(1,n+1):
        if i % 2==1:answer+='수'
        elif i%2==0:answer+='박'
    return answer
```

