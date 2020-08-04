---

---

# Attention Is All You Need

그 유명한 Transfomer입니다.

저는 NLP쪽은 관심이 전혀 없었는데 최근 Gpt-3나 추천시스템에 대한 공부를 하다보니 자연스럽게 조금씩 관심이 생겨서 리뷰하게 됐습니다.

트랜스포머 모델은 어텐션 메커니즘에만 기반을 두고 기존의 RNN, CNN 구조를 완전히 배제한 모델입니다. 



## Attention

---

어텐션에 대해 잠깐 살펴보고 넘어갑시다.

기존의 seq2seq모델은 encoder-decoder형태로 구성되며 인코더가 입력을 통해 fixed size context vector를 만들어 내고 decoder가 context vector를 통해 출력 시퀀스를 만들어 냅니다.

하지만 fixed size이기 때문에 모든 정보를 다 포함할 수 없고, RNN의 고질적인 gradient vanishing 문제가 존재합니다. 어텐션은 이러한 문제를 해결하기 위한 방법입니다.

기본 아이디어는 디코더에서 출력 단어를 예측하는 매 시점(time step)마다 인코더에서 인풋을 참고한다는 점입니다. 여기서 참고 할 때 모든 문장의 모든 단어를 동일한 비율로 참고하는 것이 아닌 해당 시점에서 예측해야 하는 아웃풋과 연관이 있는 부분을 좀 더 Attention해서 보게 됩니다.

