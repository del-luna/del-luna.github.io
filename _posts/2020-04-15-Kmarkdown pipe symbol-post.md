---
layout: post
title: Kmarkdown pipe symbol
author: Jaeheon Kwon
categories: Tips
tags: [markdown]
---

# Kmarkdown pipe symbol problem

예전 블로그도 지금 블로그도 깃 블로그라서 지킬 테마를 사용하고

Kmarkdown을 따르는 것으로 알고 있다.

그런데 컴퓨터 상에서 엔터 키 위에 있는 "|" 기호를 MathJax 태그로 감싸면 이상하게 알 수 없는 공백이 생긴다.

<img src = "https://py-tonic.github.io/images/kmark/1.PNG">

<img src = "https://py-tonic.github.io/images/kmark/2.PNG">

마크다운 툴에서는 제대로 적용이 됬지만 페이지에선 적용이 되지 않은 모습이다.

원인을 찾지 못하고 그냥 다른 유사한 기호로 대체해서 쓰다가 짜증나서 찾아봤는데.

Kmarkdown이 원인이었다.

[Issues](https://github.com/atom-community/markdown-preview-plus/issues/185)

해결 됬다고 나와있는데 내 경우에는 블로그 테마가 예전거라 그런지 적용이 되지 않고 있었다.

그러다가 우연히 LaTex 문법을 보던 도중 \vert가 "|" 기호와 똑같은 것을 보고 이걸 사용하고 있다.

