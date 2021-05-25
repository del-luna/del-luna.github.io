---
layout: post
title: Transformer: Attention Is All You Need
author: Jaeheon Kwon
categories: Papers
tags: [transformer]
---





기존 Seq2Seq 모델들의 한계점

- Context vector $$v$$에 소스 문장의 정보를 압축해야함.
- 병목현상이 발생하여 성능 하락의 원인이됨. 

![1](C:\Users\user\git\blog\images\transformer\1.png)

하나의 Context Vector가 아닌 입력 문장에서의 출력을 전부 입력으로 받자!

즉, Context Vector는 마지막 hidden state인데, 이 것만 사용하는 것이 아니라 모든 hidden state를 사용하는 것이 핵심이다.

