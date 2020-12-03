# BERT
BERT 논문 리뷰와 IMSDb 문장 분류 학습

## 1.BERT란?  
*Google!*  


<p align="center"><image src="https://raw.githubusercontent.com/JWHer/BERT/main/paper/intro.png" width="80%"></p>

2019년 자연어 처리에서 혜성처럼 등장한 SOTA가 있었으니, Bidirectional Encoder Representations from Transformers 이름하야 BERT이다.  

<br/>  
<p align="center"><i>지금은 GPT3에게 위협받고 있을지도?</i></p>
<br/>  

BERT는 사전 훈련된 모델을 fine-tuning 한다는 특징이 있다. Books Corpus와 Wikipedia를 사용하여 pre-training 하였다. Feature-based 모델과 *(ex ELMo...)* 달리 자원소모가 적다는 장점이 있다. GPT도 같은 방식이지만 가장 큰 차이점은 **Bidirectional** 하다는 점이다. 말 그대로 문장의  앞/뒤 문맥을 모두 살펴본다는 뜻이다.  

BERT는 Transformer(tensor2tensor)를 사용한다. Transformer는 이전에 시계열 예측에 사용한 RNN이나 LSTM 없이 시퀀스를 계산한다. 또한 attention 메커니즘을 사용해 Word2Vec에서 발생하는 손실과 RNN에서 발생하는 기울기 소실 문제를 해결한다. 그럼 Attention이 뭔가? Attention is all you need(Vaswani et al)에 등장한 attention은 Query-Key-Value를 계산해 예측 시점에 전체 문장에 **중요한 부분을 집중해** 다시 참고한다. *(현재 SOTA 자연어 처리는 다 Transformer에 기반한다)* BERT는 이 Transformer에서 Encoder만 사용한다.  Transformer 작동에 대해 [코드](https://github.com/JWHer/BERT/blob/main/paper/Transformer.ipynb)를 직접 보면 더 이해가 잘 갈수 있을 것이다.  

BERT Pretraining은 2단계로 이루어진다. Masked LM은 입력 문장에서 임의의 단어를 버리고 맞추는 학습이다. Next Sentence Prediction은 50%의 확률로 연관된 두 문장/아닌 문장을 제공해 맞춘다. 이 학습은 BERT가 문맥을 파악할 수 있도록 한다.  

## 2. 직접 사용해보자

[IMSDb](https://www.imsdb.com/)에는 다양한 영화 스크립트가 준비되어 있다.  BERT를 사용해 대본 구성 요소의 분류를 해 보자.  

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/BERT/main/imsdb/joker.png" width="80%"></p>

<p align="center"><i>Joker</i></p>  

영화 대본의 구성 요소를 다음과 같이 나누었다.  

| <!-- -->    | <!-- -->    |  
|---|---------|  
| 1 | 상황설명 |
| 2 | 등장인물 |
| 3 | 대사    |
| 4 | 지시/감정|
| 5 | 감정    |
| 6 | 시간    |
| 7 | 시대    |
| 0 | 그외    |
  
하지만 IMSDB의 가장 큰 문제점은 사용된 여백이 기준없이 천차만별이라는 것이다.  
이를 고려해 [preprocess 코드](https://github.com/JWHer/BERT/blob/main/imsdb/preprocess.py)를 짜 csv로 추출하였다.  


5개의 영화 데이터를 태깅하였다.  
| IMSDb 번호 | 이름 |  
|---|---------|  
| 51  | Conan the barbarian |
| 186 | Pirates of Caribbean |
| 229 | Surrogates |
| 230 | Swordfish  |
| 231 | Terminator |

<br/>  
<p align="center"><i>단순노동이다... 중국이 AI 1위인 이유가 있다</i></p>
<br/>  

**소스코드**
> [pytorch를 사용해 학습한 버전](https://github.com/JWHer/BERT/blob/main/imsdb/BERT_torch.py)  

> [Simpletransformers를 사용한 버전](https://github.com/JWHer/BERT/blob/main/imsdb/BERT_simple.ipynb)

## 3. 결과  
*효과는 굉장했다!*  

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/BERT/main/imsdb/result.png" width="50%"></p>

Simpletransformers 버전의 Confusion Matrix  
3Epoch만에 96%의 정확도를 얻었다!  

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/BERT/main/imsdb/test.png" width="50%"></p>  
  
테스트 결과 "JOHN"을 인물(2) "No thanks, I\'m not staying long."을 대사(3)으로 잘 분류해 냄을 볼 수 있었다!  

## 참고 사이트  
[1] https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9C%BC%EB%A1%9C-%EB%8F%99%EB%84%A4%EC%83%9D%ED%99%9C-%EA%B2%8C%EC%8B%9C%EA%B8%80-%ED%95%84%ED%84%B0%EB%A7%81%ED%95%98%EA%B8%B0-263cfe4bc58d
