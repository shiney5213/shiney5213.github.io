---
layout: page
title: [GRU] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
description: >
  GUR paper review
hide_description: true
sitemap: false
---


- Title: Sequence to Sequence Learning with Neural Networks
- paper: [download]
- journal:  
- year : 2014
- Subjects:	Computation and Language (cs.CL) 
-Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- summary
- seq2seq를 end2end로 해결
- 입력, 출력 차원이 고정되지 않은 경우 
[download]: https://arxiv.org/abs/1409.3215

1. this unordered seed list will be replaced by toc as unordered list
{:toc}
<br>

## Abstract
### DNN
- 지도학습에서는 잘 학습
- sequence를 sequence로 매핑할 때는 사용 불가

### 본 연구
- end to 2nd Sequence 학습 방법 제시
- 입력 seq -> encoder(multiple LSTM) -> 고정 차원의 벡터 표현 -> decoder( 입력 seq정보가 고려된 multiple LSTM) -> 출력 seq(= input seq)
- LSTM 장점
  > - 긴 문장에서도 잘 학습
  > - 단어 순서에 민감하고, 능동태,수동태에 합리적인 구문 및 문장 학습
  > - 입력 문장의 단어 순서를 반전시키면, 성능 향상 


## (1) Introduction
### DNN
- 장점
 > - 음성 인식 및 시각 객체 인식과 같은 어려운 문제에서 우수한 성능
 > - 적은 수의 단계로 임의 병렬 계산을 수행할 수 있기 때문에 강력
 > - 복잡한 계산을 학습
 > - 지도학습에서 네트워크의 매개변수를 지정하는 데 충분한 정보를 가지고 있을 때 감독된 역전파를 사용하여 훈련 가능

- 단점
  > - 입력과 타겟을 고정 차원의 벡터로 합리적으로 인코딩할 수 있는 문제에만 적용
  > - 사전에 알려지지 않은 길이의 시퀀스로 표현되는 경우에는 제한 있음.
  > - 예: 음성 인식 및 기계 번역 등 순차적 문제, 질문 응답 


### seq2seq
-  입력과 출력의 차원이 알려져 있지 않은 문제 해결
-  Long Short-Term Memory (LSTM) 아키텍처의 간단한 응용으로 해결
- idea
  > - 1st lstm: 입력 시퀀스를 하나의 LSTM을 사용하여 시간 단계별로 하나씩 읽어 전체 시퀀스의 고정 차원의 벡터 표현 생성
  > - 2nd lstm: 이전 단계에서 얻은 벡터 표현의 출력 시퀀스를 디코딩 -> 입력 시퀀스에 조건이 부여된 반복적인 신경망 언어 모델
  > - 모델이 입력 시퀀스의 요소간 종속성을 파악하고 해당하는 출력 시퀀스를 생성
  > - 출력 시퀀스를 생성할 때 입력 시퀀스의 정보를 고려
- 언어 모델에 기반 
  > - 단어가 나올 확률을 예측 
- lstm 사용 이유
  > - seq 내 요소 간의 장기적인 종속성을 캡처하여 시퀀스를 처리하도록 설계된 순환 신경망 (RNN) 의 한 유형 -> 장거리 종속성 캡쳐
  > - 많은 시퀀스-시퀀스 작업에서 입력 요소와 출력 요소 간에 상당한 시간 지연 발생-> LSTM은 이러한 시간 지연을 효과적으로 처리 -? 시간 지연 처리 가능


> ![figure1](/assets/img/nlp/9_Seq2Seq_Figure1.png){: width="70%" height="70%"}

## (5) Conclusion

## (2) The model
## (3)  Experiment
###  Dataset details
###  Decoding and Rescoring
###  Reversing the Source Sentences
### Training details
### Parallelization
### Experimental Results
### Performance on long sentences
### Model Analysis
## (4) Related work


