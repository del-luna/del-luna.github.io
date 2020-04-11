---
layout: post
title: 프로그래머스 가운데 글자 가져오기
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---

# 프로그래머스 가운데 글자 가져오기(python)

출처 : [가운데 글자 가져오기](https://programmers.co.kr/learn/courses/30/lessons/12903)

###### 문제 설명

단어 s의 가운데 글자를 반환하는 함수, solution을 만들어 보세요. 단어의 길이가 짝수라면 가운데 두글자를 반환하면 됩니다.

###### 재한사항

- s는 길이가 1 이상, 100이하인 스트링입니다.

##### 입출력 예

| s     | return |
| ----- | ------ |
| abcde | c      |
| qwer  | we     |

**solution.py**

```python
def solution(s):
    answer = ''
    if len(s)%2 != 0:
        answer = s[int(len(s)/2):int(len(s)/2)+1]
    elif len(s)%2 == 0:
        answer = s[int(len(s)/2-1):int(len(s)/2)+1]
    return answer
```

별로 어려운 문제는 아닙니다.

길이가 짝수일 때 홀수일 때를 나눠서 따로 처리했습니다.