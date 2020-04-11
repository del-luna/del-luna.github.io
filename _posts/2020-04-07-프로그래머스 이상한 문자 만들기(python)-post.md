---
layout: post
title: 프로그래머스 이상한 문자 만들기
author: Jaeheon Kwon
categories: Python
tags: [ps, level_1]
---



# 프로그래머스 이상한 문자 만들기(python)

출처 : [이상한 문자 만들기]( https://programmers.co.kr/learn/courses/30/lessons/12930 )

###### 문제 설명

문자열 s는 한 개 이상의 단어로 구성되어 있습니다. 각 단어는 하나 이상의 공백문자로 구분되어 있습니다. 각 단어의 짝수번째 알파벳은 대문자로, 홀수번째 알파벳은 소문자로 바꾼 문자열을 리턴하는 함수, solution을 완성하세요.

##### 제한 사항

- 문자열 전체의 짝/홀수 인덱스가 아니라, 단어(공백을 기준)별로 짝/홀수 인덱스를 판단해야합니다.
- 첫 번째 글자는 0번째 인덱스로 보아 짝수번째 알파벳으로 처리해야 합니다.

##### 입출력 예

| s               | return          |
| --------------- | --------------- |
| try hello world | TrY HeLlO WoRlD |

##### 입출력 예 설명

try hello world는 세 단어 try, hello, world로 구성되어 있습니다. 각 단어의 짝수번째 문자를 대문자로, 홀수번째 문자를 소문자로 바꾸면 TrY, HeLlO, WoRlD입니다. 따라서 TrY HeLlO WoRlD 를 리턴합니다.

**solution.py**

```python
def solution(s):
    temp = s.split(' ')
    answer = []
    for i in temp:
        s=''
        for j in range(len(i)):
            s+=i[j].upper() if j%2==0 else i[j].lower()
        answer.append(s)
    return ' '.join(answer)
```

처음에 단어 기준으로 홀짝을 판단하는게 아니라 문자열로 해서 계속 틀렸던 문제입니다.

문제를 잘 읽읍시다 ㅜㅜ..