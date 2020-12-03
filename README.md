# BERT
BERT 논문 리뷰와 IMDb 문장 분류 학습

## 1.BERT란?  
*Google!*  

<image src="" width="50%>

2019년 자연어 처리에서 혜성처럼 등장한 SOTA가 있었으니, Bidirectional Encoder Representations from Transformers 이름하야 BERT이다.  

<p align="center"><i>지금은 GPT3에게 위협받고 있을지도?</i></p>

BERT는 사전 훈련된 모델을 fine-tuning 한다는 특징이 있다. Books Corpus와 Wikipedia를 사용하여 pre-training 하였다. Feature-based 모델과 *(ex ELMo...)* 달리 자원소모가 적다는 장점이 있다. GPT도 같은 방식이지만 가장 큰 차이점은 **Bidirectional** 하다는 점이다. 말 그대로 문장의  앞/뒤 문맥을 모두 살펴본다는 뜻이다.  

BERT는 Transformer(tensor2tensor)를 사용한다. Transformer는 이전에 시계열 예측에 사용한 RNN이나 LSTM 없이 시퀀스를 계산한다. 또한 attention 메커니즘을 사용해 Word2Vec에서 발생하는 손실과 RNN에서 발생하는 기울기 소실 문제를 해결한다. 그럼 Attention이 뭔가? Attention is all you need(Vaswani et al)에 등장한 attention은 Query-Key-Value를 계산해 예측 시점에 전체 문장에 **중요한 부분을 집중해** 다시 참고한다. *(현재 SOTA 자연어 처리는 다 Transformer에 기반한다)* BERT는 이 Transformer에서 Encoder만 사용한다.  Transformer 작동에 대해 [코드]()를 직접 보면 더 이해가 잘 갈수 있을 것이다.  
