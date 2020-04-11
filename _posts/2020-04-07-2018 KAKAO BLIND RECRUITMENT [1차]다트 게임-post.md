---
layout: post
title: 2018 KAKAO BLIND RECRUITMENT [1차]다트 게임
author: Jaeheon Kwon
categories: Python
tags: [ps, kakao]
---

# 2018 KAKAO BLIND RECRUITMENT [1차]다트 게임

출처 : [다트 게임]( https://programmers.co.kr/learn/courses/30/lessons/17682 )

###### 문제 설명

## 다트 게임

카카오톡에 뜬 네 번째 별! 심심할 땐? 카카오톡 게임별~

![Game Star](http://t1.kakaocdn.net/welcome2018/gamestar.png)

카카오톡 게임별의 하반기 신규 서비스로 다트 게임을 출시하기로 했다. 다트 게임은 다트판에 다트를 세 차례 던져 그 점수의 합계로 실력을 겨루는 게임으로, 모두가 간단히 즐길 수 있다.
갓 입사한 무지는 코딩 실력을 인정받아 게임의 핵심 부분인 점수 계산 로직을 맡게 되었다. 다트 게임의 점수 계산 로직은 아래와 같다.

1. 다트 게임은 총 3번의 기회로 구성된다.
2. 각 기회마다 얻을 수 있는 점수는 0점에서 10점까지이다.
3. 점수와 함께 Single(`S`), Double(`D`), Triple(`T`) 영역이 존재하고 각 영역 당첨 시 점수에서 1제곱, 2제곱, 3제곱 (점수1 , 점수2 , 점수3 )으로 계산된다.
4. 옵션으로 스타상(`*`) , 아차상(`#`)이 존재하며 스타상(`*`) 당첨 시 해당 점수와 바로 전에 얻은 점수를 각 2배로 만든다. 아차상(`#`) 당첨 시 해당 점수는 마이너스된다.
5. 스타상(`*`)은 첫 번째 기회에서도 나올 수 있다. 이 경우 첫 번째 스타상(`*`)의 점수만 2배가 된다. (예제 4번 참고)
6. 스타상(`*`)의 효과는 다른 스타상(`*`)의 효과와 중첩될 수 있다. 이 경우 중첩된 스타상(`*`) 점수는 4배가 된다. (예제 4번 참고)
7. 스타상(`*`)의 효과는 아차상(`#`)의 효과와 중첩될 수 있다. 이 경우 중첩된 아차상(`#`)의 점수는 -2배가 된다. (예제 5번 참고)
8. Single(`S`), Double(`D`), Triple(`T`)은 점수마다 하나씩 존재한다.
9. 스타상(`*`), 아차상(`#`)은 점수마다 둘 중 하나만 존재할 수 있으며, 존재하지 않을 수도 있다.

0~10의 정수와 문자 S, D, T, *, #로 구성된 문자열이 입력될 시 총점수를 반환하는 함수를 작성하라.

### 입력 형식

점수|보너스|[옵션]으로 이루어진 문자열 3세트.
예) `1S2D*3T`

- 점수는 0에서 10 사이의 정수이다.
- 보너스는 S, D, T 중 하나이다.
- 옵선은 *이나 # 중 하나이며, 없을 수도 있다.

### 출력 형식

3번의 기회에서 얻은 점수 합계에 해당하는 정수값을 출력한다.
예) 37

### 입출력 예제

| 예제 | dartResult | answer | 설명                        |
| ---- | ---------- | ------ | --------------------------- |
| 1    | `1S2D*3T`  | 37     | 11 * 2 + 22 * 2 + 33        |
| 2    | `1D2S#10S` | 9      | 12 + 21 * (-1) + 101        |
| 3    | `1D2S0T`   | 3      | 12 + 21 + 03                |
| 4    | `1S*2T*3S` | 23     | 11 * 2 * 2 + 23 * 2 + 31    |
| 5    | `1D#2S*3S` | 5      | 12 * (-1) * 2 + 21 * 2 + 31 |
| 6    | `1T2D3D#`  | -4     | 13 + 22 + 32 * (-1)         |
| 7    | `1D2S3T*`  | 59     | 12 + 21 * 2 + 33 * 2        |

**solution.py**

```python
def solution(dartResult):
    d = dartResult
    z = list(dartResult)
    score =[]
    for i in range(len(d)):
        if d[i].isalpha()==True:z[i]='_'+z[i]+'_'
        elif d[i]=='#'or d[i]=='*':z[i]='_'+z[i]+'_' 
    z = list(filter(None,''.join(z).split('_')))
    for i in z:
        if i.isdigit()==1:score.append(int(i))
        elif i.isalpha()==1:
            if i=='D':score[-1]=score[-1]**2
            elif i=='T':score[-1]=score[-1]**3
        elif i=='*':
            if len(score)==1:score[-1]=score[-1]*2
            else:
                score[-1]=score[-1]*2
                score[-2]=score[-2]*2
        elif i=='#':score[-1]=score[-1]*-1
    return sum(score)
```

우선 보자마자 느낀점은 dartResult를 문자를 기준으로 슬라이싱 해야 겠다고 생각했다.

그런데 마땅한 방법이 떠오르지않아 그냥 문자 양옆에 기호 를 추가해버리고 기호를 기준으로 슬라이싱 한 뒤 합쳐버렸다.

그 뒤엔 숫자 계산을 해야되니까 숫자일 땐 score에 넣고

문자 일땐 score의 맨 마지막 값을 연산한 뒤 다시 넣어줬다.

다 풀고나서 생각한건데 굳이 문자열을 슬라이싱 했으면

D,T를 제곱기호, 세제곱 기호로 바꾼뒤 eval()함수를 썻으면 어땠을까 라는 생각도 하게 됐다.

```python
import re

def solution(dartResult):
    bonus = {'S' : 1, 'D' : 2, 'T' : 3}
    option = {'' : 1, '*' : 2, '#' : -1}
    p = re.compile('(\d+)([SDT])([*#]?)')
    dart = p.findall(dartResult)
    for i in range(len(dart)):
        if dart[i][2] == '*' and i > 0:
            dart[i-1] *= 2
        dart[i] = int(dart[i][0]) ** bonus[dart[i][1]] * option[dart[i][2]]

    answer = sum(dart)
    return answer
```

위 방법은 정규식을 이용한 코드인데..

처음엔 나도 정규식을 쓸 생각은했는데 쓰고나서 어떻게 활용하는지에 대한 생각이 부족했던 것 같다.