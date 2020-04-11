---
layout: post
title: 2018 KAKAO BLIND RECRUITMENT [1차]비밀 지도
author: Jaeheon Kwon
categories: Python
tags: [ps, kakao]
---



# [1차] 비밀지도

# 2018 KAKAO BLIND RECRUITMENT [1차] 비밀지도

출처 : [비밀지도]( https://programmers.co.kr/learn/courses/30/lessons/17681 )

###### 문제 설명

## 비밀지도

네오는 평소 프로도가 비상금을 숨겨놓는 장소를 알려줄 비밀지도를 손에 넣었다. 그런데 이 비밀지도는 숫자로 암호화되어 있어 위치를 확인하기 위해서는 암호를 해독해야 한다. 다행히 지도 암호를 해독할 방법을 적어놓은 메모도 함께 발견했다.

1. 지도는 한 변의 길이가 `n`인 정사각형 배열 형태로, 각 칸은 공백(" ) 또는벽(#") 두 종류로 이루어져 있다.
2. 전체 지도는 두 장의 지도를 겹쳐서 얻을 수 있다. 각각 지도 1과 지도 2라고 하자. 지도 1 또는 지도 2 중 어느 하나라도 벽인 부분은 전체 지도에서도 벽이다. 지도 1과 지도 2에서 모두 공백인 부분은 전체 지도에서도 공백이다.
3. 지도 1과 지도 2는 각각 정수 배열로 암호화되어 있다.
4. 암호화된 배열은 지도의 각 가로줄에서 벽 부분을 `1`, 공백 부분을 `0`으로 부호화했을 때 얻어지는 이진수에 해당하는 값의 배열이다.

![secret map](http://t1.kakaocdn.net/welcome2018/secret8.png)

네오가 프로도의 비상금을 손에 넣을 수 있도록, 비밀지도의 암호를 해독하는 작업을 도와줄 프로그램을 작성하라.

### 입력 형식

입력으로 지도의 한 변 크기 `n` 과 2개의 정수 배열 `arr1`, `arr2`가 들어온다.

- 1 ≦ `n` ≦ 16
- `arr1`, `arr2`는 길이 `n`인 정수 배열로 주어진다.
- 정수 배열의 각 원소 `x`를 이진수로 변환했을 때의 길이는 `n` 이하이다. 즉, 0 ≦ `x` ≦ 2n - 1을 만족한다.

### 출력 형식

원래의 비밀지도를 해독하여 `'#'`, `공백`으로 구성된 문자열 배열로 출력하라.

### 입출력 예제

| 매개변수 | 값                                            |
| -------- | --------------------------------------------- |
| n        | 5                                             |
| arr1     | [9, 20, 28, 18, 11]                           |
| arr2     | [30, 1, 21, 17, 28]                           |
| 출력     | `["#####","# # #", "### #", "# ##", "#####"]` |

| 매개변수 | 값                                                           |
| -------- | ------------------------------------------------------------ |
| n        | 6                                                            |
| arr1     | [46, 33, 33 ,22, 31, 50]                                     |
| arr2     | [27 ,56, 19, 14, 14, 10]                                     |
| 출력     | `["######", "### #", "## ##", " #### ", " #####", "### # "]` |

**solution.py**

```python
def solution(n, arr1, arr2):
    answer = []
    temp1 = []
    temp2 = []
    final=[]
    s=[]
    zero=['0']
    
    for i in arr1:
        x = format(i,'b')
        if len(x)<n:
            temp1.append(''.join(list(zero*(n-len(x)))+list(x)))
        else:temp1.append(x)
        
    for i in arr2:
        x = format(i,'b')
        if len(x)<n:
            temp2.append(''.join(list(zero*(n-len(x)))+list(x)))
        else:temp2.append(x)
    
    for a,b in zip(temp1,temp2):
        for c,d in zip(a,b):
            if int(c)+int(d) == 1: final.append(1)
            elif int(c)+int(d) == 0:final.append(0)
            else:final.append(1)
    
    for i in final:
        if i == 1: s.append('#')
        else:s.append(' ')
    s=''.join(s)
    
    for i in range(n):
        answer.append(s[n*i:n*(i+1)])
    return answer
```

보자마자 이진법으로 바꿔서 풀어야겠다는 생각을 했습니다.

그런데 16보다 작은 수일 경우 자릿수가 4자리밖에 나오지 않아서 경우를 추가해줬고,

각각의 자릿수별로 더해줘서 1인경우 :1 0인경우:0 으로 구분해서 리스트를 만들어 줬습니다.

최종적으로는 1인경우 #, 0인경우 ' '으로 지도와 똑같이 만들었고, n개만큼 분할해서 가져왔습니다.

문제를 풀생각만했지 어떻게 풀지에 대한 고민을 적게해서 코드가 길어진 것 같습니다.

```python
def solution(n, arr1, arr2):
    answer = []
    for i,j in zip(arr1,arr2):
        a12 = str(bin(i|j)[2:])
        a12=a12.rjust(n,'0')
        a12=a12.replace('1','#')
        a12=a12.replace('0',' ')
        answer.append(a12)
    return answer
```

위에서 자릿수별로 더해줘서 1이면 1, 0이면 0을 해주는 부분을

비트연산자로 or연산을 하면되는데 왜 이 생각을 못했을까 부끄러웠습니다.

또한 리스트 자릿수에 맞게 0을 채워주는부분도 rjust를 사용하면 한줄로 간단히 할 수 있었습니다..

replace도 그렇고 다 한번 씩 본 함수인데 사용할 생각을 못했다는게 너무 안타까웠습니다.