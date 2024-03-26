---
layout: page
title: [RNN] Recurrent neural network based language model
description: >
  RNN paper review
hide_description: true
sitemap: false
---


- Title: Recurrent neural network based language model
- paper: [download]
- journal:  Conference of the International Speech Communication Association
- year : 2015
- Subjects:	Computation and Language (cs.CL) 
- Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký, Sanjeev Khudanpur
- summary
  > 시계열 데이터의 패턴을 활용할 수 있는 순환 신경망(RNN)
- 사용 기법
  > - Dynamic model: 보통 train시 학습되고, test에서는 예측값만 계산하지만, RNN은 test에서도 학습이 진행됨

[download]: https://arxiv.org/abs/1506.00019

1. this unordered seed list will be replaced by toc as unordered list
{:toc}
<br>

## (0) Abstract
- 모델
  > - recurrent neural network based language model (RNN LM)
  > - 여러 RNN LM 모델을 함께 사용

- 성능
  > - 여러 RNN 모델을 홉합하여 사용하면 최첨단 백오프 언어 모델과 비교하여 perplexcity 약 50% 감소
  > - 음성 인식 실험에서는 같은 양의 데이터로 훈련된 모델을 비교할 때 오류율 약 18% 감소
  > - 표준 n-gram 모델보다 우수
  > - 한계 : 높은 계산 복잡도

## (5) Conclusion and future work
### conclusion
- 재귀 신경망(RNN)은 상태 최첨단 백오프 모델보다 유의미하게 우월한 성과
- 새로운 데이터를 추가하지 않아도 성능이 높아질 수 있음
-  어휘력이 큰 언어나 어휘가 큰 언어를 처리할 수 있기 때문에 기계 번역이나 OCR 같은 작업에 특히 유용### future work

### future work
- 온라인 학습은 캐시와 유사하고 트리거와 유사한 정보를 자연스럽게 얻을 수 있는 방법을 제공하므로 더 자세한 조사 필요

## (1) Introduction
### 통계적 언어 모델의 목표
  > - 문맥이 주어졌을 때, 다음 단어를 예측하는 것 -> 언어 모델을 만들 때 연속적인 예측 문제를 다룹

### 기존 연구  
- 특정 언어 도메인에 대한 접근 방식을 취함
  > - 자연어 문장을 트리로 분석하거나, 단어의 형태학, 구문적, 의미를 고려함
  > - n gram 모델 : 문장을 만드는 단어의 순서가 매우 중요하다고 가정

-  캐시 모델 및 클래스 기반 모델
  > - 캐시 모델 : 긴 문맥 정보를 설명하는 모델 
  > - 클래스 기반 모델 : 유사한 단어 간의 매개변수를 공유함으로써 짧은 문맥에 대한 매개변수 추정을 개선하는 모델
  > - 음성 인식이나 기계 번역 시스템과 같은 고급 언어 모델링 기법을 실제로 적용하는 데는 효과가 미미

- 고급 언어 모델링 기법들
  > - 베이스라인에 비해 미미하게 개선
  > - 연구 환경에서 개발되고, 데이터가 정해져있을 때 효과가 있음.
  > - 실제 시스템에 적용할 경우 효과가 떨어짐

## (2) Model description
### RNN 연구 
-  Bengio(2013) : 고정 길이 문맥을 갖는 피드포워드 신경망을 사용->  다음 단어를 예측할 때 5~10개의 단어만 고려 -> 더 많이 활용해야 함.
-  Goodman(2001) : 클래스 기반 모델과 다양한 기술 사용 -> 이 단일 모델이 다른 기술을 기반으로 한 여러 모델의 혼합보다 우수한 성능을 보임
-  Schwenk(2005) : 신경망 기반 모델이 여러 작업에 대해 우수한 베이스라인 시스템에 비해 음성 인식에서 상당한 개선을 제공
- cache 모델: 신경망 모델에 보완정인 정보 제공-> 임의의 길이를 갖는 문맥에 시간 정보 인코딩 가능

### 본 연구
- 재귀 신경망이 제한된 크기의 문맥을 사용하지 않음.
- 재귀적 연결을 사용하여 임의적으로 긴 context 단어를 사용할 수 있음.
- 한계: 확률적 경사 하강법을 사용하여 장기 의존성을 가질 수 있음.

- 모델 : simple recurrent neural network
  > - input data(x) : 벡터 w + 시간 t-1에서 컨텍스트 층(hidden layer) s의 뉴런 출력
  > - input layer(x)
> ![formula1](/assets/img/nlp/7_RNN_formula1.png){: width="70%" height="70%"}
  > - hidden layer(s)
  ![formula2](/assets/img/nlp/7_RNN_formula2.png){: width="70%" height="70%"}
  > - output layer(y)
  ![formula3](/assets/img/nlp/7_RNN_formula3.png){: width="70%" height="70%"}
> - activation function (f(z))
![formula4](/assets/img/nlp/7_RNN_formula4.png){: width="70%" height="70%"}
> - s softmax function (g(z))
![formula5](/assets/img/nlp/7_RNN_formula5.png){: width="70%" height="70%"}

- 크기
 > - 벡터 x의 크기 : 어휘 V의 크기 +  context layer 크기
 > - context layer : 30 - 500 unit -> training data 크기에 따라 다름
 > - 은닉층 크기가 커도 과적합 되지 않음.

- hyper params
> - epochs : 10 - 20
> - 가중치 초기화: 평균이 0이고 분산이 0.1인 랜덤 가우시안 노이즈
> - 확률적 경사 하강법을 사용한 표준 역전파 알고리즘을 사용
> - init learning rate : α = 0.1-> log- likelihood가 감소할 때마다 반으로 줄임
> - 정규화는 유의미하지 않음.

- loss
> - cross entropy
> - ![formula6](/assets/img/nlp/7_RNN_formula6.png){: width="70%" height="70%"}
> - desired : 정답 단어를 나타내는 벡터
> - y(t) : 네트워크의 실제 output

- `Dynamic model`
  > - 일반적인 통계 언어모델에서는 test과정에서 모델 업데이트 안됨 
  > - 그러나 RNN 모델은 test 과정에서 네트워크가 시냅스를 업데이트하고 장기 기억을 유지할 수 있게 함 
  > - 테스트 데이터를 처리하는 동안 1번만 update -> 정적 모델에 비해 perplexity 감소할 수 있음.
  > - 학습률 = 0.1
  > - 동적으로 update 되는 모델은 세로운 도메인에 적용 가능

- 시간을 통한 역전파(backpropagation through time, BPTT) 알고리즘
  > - 네트워크의 가중치는 현재 시간 단계에 대해서만 계산된 오류 벡터를 기반으로 업데이트
  > - 상기 단순화를 보완하기 위한 알고리즘 사용

- 다른 연구와 다른점
> - RNN LM: 은닉층 크기만 선택
> - Bengio(2013), Schwenk(2005) : 훈련 전 정해야 하는 매개변수의 양이 많음.
  
### Optimization
- Optimization
> - ![formula7](/assets/img/nlp/7_RNN_formula7.png){: width="70%" height="70%"}
> - 성능을 향상시키기 위해서 threshold 보다 적게 나타나는 단어를 rate token으로 병합
> - C_rare : vocabulary에서 threshol보다 적게 나타나는 단어들의 수

- 성능 향상을 위한 방법
> - BLAS 라이브러리 사용 -> 계산 속도 향상

## (3) WSJ experiments

## (4) NIST RT05 experiments

