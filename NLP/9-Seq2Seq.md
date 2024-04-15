---
layout: page
title: [Seq2Seq] Sequence to Sequence Learning with Neural Networks
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
### large deep LSTM
- 어휘에 대한 제한이 있고, 문제 구조에 대해 가정하지 않은 lstm 모델은 대규모 기계 번역 task에서 어휘의 가정이 없는 SMT-based system을 능가
- 충분한 data가 있다면 많은 sequnece 학습 문제에서도 효율적

### 단어 순서 역방향학습
- 소스 문장의 단어순서를 역방향으로 바꾸는 것이 성능 개선
- 주변 단어의 의존성이 중요(단기 의존성: 서로 가까운 단어들과의 관계)
- 단기 의존성을 극대화하는 인코딩 전략을 찾는 것이 중요

### 긴 문장에서 정확도 높은 번역
- 역방향 데이터로 학습했을 때 긴 문제 번역 능력 향상
- 

## (2) The model
### RNN
- input seq와 output seq의 정렬이 미리 알려진 경우에는 쉽게 매핑
- but input과 output의 관계가 복잡하거나 길이가 다를 때는 비효율적 
- input seq : (x1, . . . , xT)
- output seq :  (y1, . . . , yT)
- ![formula1](/assets/img/nlp/9_Seq2Seq_formula1.png){: width="70%" height="70%"}

### sequence learning stratege: encoder and decoder
- 입력 시퀀스를 고정 크기의 벡터로 매핑하는 한 개의 RNN을 사용한 후 다른 RNN을 사용하여 이 벡터를 대상 시퀀스로 매핑
- 장기 의존성 문제로 RNN에서는 학습이 어려움
- 해결: LSTM으로 장기 의존성 문제 해결

### LSTM
- 목적: 주어진 입력 시퀀스 (x1,..., xT) 가 주어지면 출력 시퀀스 (y1,..., yT') 를 생성할 조건부 확률을 추정
- 여기서 T, T'는 길이가 다를 수 있음.
- 1.  입력 seq(x1, . . . , xT)를 넣어 고정 차원 벡터 v(context vector)를 생성
  >  LSTM은 입력 시퀀스를 순차적으로 처리하고, 마지막 숨겨진 상태를 사용하여 입력 시퀀스를 요약하는 벡터 v를 생성
- 2.  v를 초기 은닉 상태로 사용하여 출력 시퀀스인 y1, . . . , yT′의 확률을 계산
  > - 표준 LSTM 언어 모델(LSTM-LM)을 사용
  > -  초기 은닉 상태로부터 시작하여 출력 시퀀스를 생성하는데, 이때 LSTM이 현재 상태를 고려하여 다음 단어의 확률을 계산
- 각 문장 끝에 "<EOS>"기호를 두어 다양한 길이의 시퀀스를 효과적으로 생성
![formula2](/assets/img/nlp/9_Seq2Seq_formula2.png){: width="70%" height="70%"}

### 논문의 LSTM
- 두 개의 서로 다른 LSTM 사용:  모델 파라미터 수가 거의 증가하지 않으면서 여러 언어 쌍에 대해 LSTM을 동시에 훈련하기가 자연스러워지기 때문
- 깊은 LSTM이 성능 우수: 4개의 layer로 구성된 LSTM 사용
-  입력 문장의 단어 순서를 반전하는 것이 매우 유용
-  

## (3)  Experiment
-  WMT’14 English to French 번역 Task에 두 가지 방식 적용
  > 참조 SMT 시스템을 사용하지 않고 입력 문장을 직접 번역
  > SMT 기준의 n-최상의 목록을 재점수화하는 데 사용

###  Dataset details
- WMT'14 영어에서 프랑스어로의 데이터셋
- 12백만 문장의 하위 집합에서 모델을 훈련
- OOV : "UNK" 토큰으로 대체
  
###  Decoding and Rescoring
#### 목적:  소스 문장 S가 주어졌을 때 올바른 번역 T의 로그 확률을 최대화하여 훈련
- ![formula3](/assets/img/nlp/9_Seq2Seq_formula3.png){: width="70%" height="70%"}
- |s|: 훈련 셋의 크기
- ![formula4](/assets/img/nlp/9_Seq2Seq_formula4.png){: width="70%" height="70%"}

####  beam search decoder
- 가장 가능성이 높은 번역을 효율적으로 검색하기 위해 왼쪽에서 오른쪽 빔 검색 디코더 사용
-  각 타임스텝에서 가능한 모든 단어를 사용하여 빔의 각 편가설을 확장하고 모델의 로그 확률을 기반으로 가장 가능성이 높은 가설 B개를 선택

###  Reversing the Source Sentences
- 소스 문장을 뒤집을 때 LSTM이 훨씬 더 잘 학습한다는 것을 발견
- 데이터셋에 많은 짧은 기간 의존성이 도입되었기 때문
- 일반적으로 소스 문장을 대상 문장에 연결할 때 각 단어는 해당되는 단어와 멀리 떨어져 있음. -> 최소 시간 지연
- 소스 문장의 단어를 반전시키면, 소스 언어의 처음 몇 단어는 이제 대상 언어의 처음 몇 단어와 매우 가까워
- 소스와 대상 언어의 대응되는 단어 사이의 평균 거리는 변경되지 않음 -> 최소 시간 지연이 크게 줄어듦
  
### Training details
- 모델
  > - 각각 4개의 층, 각 층당 1000개의 셀, 그리고 1000차원의 단어 임베딩을 가진 깊은 LSTM을 사용
  > - input vocab: 160.000
  > - output vocat : 80,000
  > - 매개변수: 384M개, 이 중 64M개는 순환 연결 (32M개는 "인코더" LSTM에 해당하고 32M개는 "디코더" LSTM에 해당)

- hyperparams
  > - LSTM 매개변수:  -0.08과 0.08 사이의 균일한 분포로 초기화
> - 학습률:  모멘텀 없이 고정된 학습률인 0.7을 사용하는 확률적 경사 하강법 , 5번의 epoch 후에는 매 반 에폭마다 학습률을 절반으로 줄이기
> - epochs: 총 7.5 
> - 그래디언트에 대한 128개의 시퀀스 배치를 사용,  이를 배치의 크기(즉, 128)로 나누기
> - 그래디언트의 노름에 대한 강제 제한을 설정, 각 훈련 배치에 대해, 우리는 g를 128로 나눈 그래디언트를 계산 -> 그래디언트 폭발 문제 해결
> - 만약 s > 5라면, 우리는 g = 5g/s로 설정
> - 미니배치의 모든 문장이 대략적으로 동일한 길이를 가지도록 보장 ->  2배의 속도 향상

### Parallelization
- 8-GPU 시스템을 사용하여 모델을 병렬화
-  LSTM의 각 층은 다른 GPU에서 실행되고, 계산이 완료되는 즉시 다음 GPU/층으로 활성화를 전달
-  GPU에 상주하는 4개의 LSTM layer + 나머지 4개의 GPU는 소프트맥스를 병렬화하는 데 사용
-  train 시간: 10일
  
### Experimental Results
- BLEU 점수를 사용
- 무작위 초기화와 미니배치 순서에 차이가 있는 LSTM 앙상블을 사용
- 
### Performance on long sentences
- 긴 문장에서도 잘 수행
-  ![figure3](/assets/img/nlp/9_Seq2Seq_Figure3.png){: width="70%" height="70%"}

### Model Analysis

## (4) Related work



