---
layout: post
title: EM algorithm
author: Jaeheon Kwon
categories: Mathematics
tags: [statistics]
---

# Expectation Maximization Algorithm

실제 문제에서는 prior와 likelihood의 parameter를 알고있지 못한 경우가 많습니다.
Variational Inference는 posterior $p(z|x)$에 근사한 $q(z)$를 찾는 것이 목적이므로 p(z)의 parameter는 임시적으로 고정시켜도 상관 없다고 합니다.

우리는 VI문제를 해결하기위해 posterior에 근사한 $q(z)$의 parameter $θ_q$를 찾으면서
동시에 likelihood $p(x|z)$의 parameter $θ_l$를 추정해야 합니다.

EM algorithm은 2-step으로 나눠서 볼 수 있습니다.
<br>
Expectation(E-step) :  $D_{KL}(q(z)||p(z|x))$ 를 줄이는 $θ_q$를 찾는다.<br>
Maximization(M-step) : E-step에서 찾은 $θ_q$를 고정하고 $logp(x)$의 하한을 최대화하는 $p(x|z)$의 parameter $θ_l$를 찾는다.<br>

$θ_q$와 $θ_l$를 jointly optimize하는 문제가 어려운 문제라면 
이 문제를 해결하는 가장 간단한  방법은 하나의 variable을 고정하고 다른 하나를 update한 다음 또 다른 variable을 update하는 방식으로 해결할 수 있습니다.

우리는 각각의 alternating update method를 E-step과 M-step으로 부르겠습니다. 

E-step은 사실 Vi의 문제해결과정입니다.

식을 그대로 가져오면

<img src = "https://py-tonic.github.io/images/EM/0.PNG">

위와 같음을 알 수 있습니다.(이젠 KL-divergence는 너무 익숙하시죠?)

그런데 생각을 해봅시다.
우리는 E-step에서 $q(z)$에 대한 parameter인 $θ_q$를 찾으며 업데이트 할 것이고,
이는 위의 맨 마지막 식에서 $logp(x)$인 항과 무관합니다.

하지만 궁극적인 목표는 KL-divergence를 최소화 하는 것이므로
결국 $logp(x)$도 최소화 해야 합니다.

$log p(x)$를 중심으로 식을 전개하면 다음과 같습니다.
<img src = "https://py-tonic.github.io/images/EM/1.PNG">

여기서 우리는  $D_{KL}(q(z)||p(z|x))≥0$임을 알고 있습니다. 
따라서 $logp(x)$의 Lower Bound는 다음과 같습니다.
<img src = "https://py-tonic.github.io/images/EM/2.PNG">

$p(x)$는 Bayes theorem에서 evidence라고 불려서 위 식의 우변을 'Evidence Lower Bound(ELBO)'라고 부릅니다.

마지막으로 위 식의 ELBO를 줄이면 $logp(x)$가 줄어들 것이고 결과적으로 KLD를 최소화할 수 있게 됩니다.


## Reference

[Ratsgo's blog](https://ratsgo.github.io/generative%20model/2017/12/19/vi/)<br>

