---
layout: post
title: 프로그래머스 문자열 내림차순으로 배치하기
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 문자열 내림차순으로 배치하기(python)

출처 : [문자열 내림차순으로 배치하기]( https://programmers.co.kr/learn/courses/30/lessons/12917 )

###### 문제 설명

문자열 s에 나타나는 문자를 큰것부터 작은 순으로 정렬해 새로운 문자열을 리턴하는 함수, solution을 완성해주세요.
s는 영문 대소문자로만 구성되어 있으며, 대문자는 소문자보다 작은 것으로 간주합니다.

##### 제한 사항

- str은 길이 1 이상인 문자열입니다.

##### 입출력 예

| s       | return  |
| ------- | ------- |
| Zbcdefg | gfedcbZ |

**solution.py**

```python
def solution(s):
    answer = ''
    temp = []
    for i in s:
        temp.append(ord(i))
    temp.sort(reverse=True)
    c = list(filter(lambda x:x>96, temp)) + list(filter(lambda x:x<91,temp))
    for i in c:
        answer+=chr(i)
    return answer
```

ord를 이용한 아스키코드로 접근해서 문제를 해결했습니다.

소문자,대문자인 경우를 나눠서 리스트로 만들어 합쳤습니다.

