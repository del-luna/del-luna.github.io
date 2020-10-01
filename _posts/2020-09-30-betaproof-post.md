---
layout: post
title: 베타의 통계량 증명
author: Jaeheon Kwon
categories: Mathematics
tags: [statistics]
---



$y_i = \beta_0+\beta_1x_i + \epsilon_i$

- $\hat\beta_0 = \bar y - \hat\beta_1\bar x$
- $\hat\beta_1 = \frac{\sum_{i=1}^n(x_i - \bar x)(y_i-\bar y)}{\sum_{i=1}^n (x_i-\bar x)}$



$Assumption.$

- $E[\epsilon_i] = 0$
- $V[\epsilon] = \sigma^2$

$Remark$

- $E[y_i] = \beta_0 + \beta_1x_i$
- $V[y_i] = \sigma^2$



$E[\hat\beta_0]=\beta_0 ,\quad E[\hat\beta_1] =\beta_1$



$Proof.$

$E[\hat\beta_0] = E[\bar y - \hat\beta_1 \bar x]$ 

$= \frac1n \sum\limits_{i=1}^nE[y_i]-E[\hat\beta_1]\bar x $

$ = \frac1n \sum\limits_{i=1}^n(\beta_0+\beta_1x_i) - \beta_1\frac1n\sum\limits_{i=1}^nx_i $

$ = \beta_0 \quad Q.E.D$

<br>



$\hat\beta_1 = \frac{\sum_{i=1}^n(x_i - \bar x)(y_i-\bar y)}{\sum_{i=1}^n (x_i-\bar x)}$

$\frac{\sum_{i=1}^n(x_i - \bar x)y_i - \bar y(x_i-\bar x)}{\sum_{i=1}^n (x_i-\bar x)} $

$ =\frac{\sum_{i=1}^n(x_i - \bar x)y_i}{\sum_{i=1}^n (x_i-\bar x)}$

$Thus$

$E[\hat\beta_1] = \frac{\sum_{i=1}^n(x_i - \bar x)}{\sum_{i=1}^n (x_i-\bar x)}E[y_i]$

$=\frac{\sum_{i=1}^n(x_i - \bar x)}{\sum_{i=1}^n (x_i-\bar x)}(\beta_0+\beta_1x_i) $

$ = \beta_1 \quad Q.E.D $



$V[\hat\beta_0] = \frac{\sigma^2 x_i^2}{\sum_{i=1}^n(x_i-\bar x)^2}, \quad V[\hat\beta_1] = \frac{\sigma^2}{\sum_{i=1}^n(x_i-\bar x)^2}$

$Remark.$

- $\sum_{i=1}^n (x_i-\bar x)^2 = S_{xx}$



$Proof.$

$V[\hat\beta_0] = V[\bar y - \hat\beta_1 \bar x] $

$ = V[\bar y] + V[-\hat\beta_1\bar x] + 2Cov[\bar y , -\hat\beta_1 \bar x] $

$ = V[\bar y]+\bar x^2V[\hat\beta_1] - 2\bar xCov[\bar y, \hat\beta_1]$

$ = \frac{\sigma^2}{n} + \bar x^2\frac{\sigma^2}{S_{xx}} \quad Q.E.D$



$V[\hat\beta_1] = V[\frac{\sum_{i=1}^n(x_i-\bar x)y_i}{S_{xx}}]$

$ = (\frac1{S_{xx}})^2(\sum\limits_{i=1}^n(x_i-\bar x)^2)V[y_i]$

$= (\frac1{S_{xx}})^2(S_{xx}) \sigma^2 $

$ = (\frac1{S_{xx}})\sigma^2 \quad Q.E.D$

