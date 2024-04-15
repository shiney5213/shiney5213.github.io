---
layout: page
title: [GRU] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
description: >
  GUR paper review
hide_description: true
sitemap: false
---


- Title: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
- paper: [download]
- journal:  
- year : 2014
- Subjects:	Computation and Language (cs.CL) 
- Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
- summary
- 
[download]: https://arxiv.org/abs/1406.1078

1. this unordered seed list will be replaced by toc as unordered list
{:toc}
<br>

## Abstract
- RNN Encoder-Decoder
  > - Encoder : 심볼 seq를 고정 길이의 벡터로 표현
  > - Decoder :  다른 심복 seq
  > - 둘다 source seq가 주어졌을 때 target seq의 조건부 확률을 최대화하도록 훈련
  > - RNN Encoder-Decoder가 계산한 구문 쌍의 조건부 확률을 기존 로그-선형 모델의 추가 기능으로 사용하여 통계 기계 번역 시스템의 성능이 경험적으로 향상
  > - 의미론적, 구문론적으로 의미있는 언어 문구 표현을 학습 

## (1) Introduction
### DNN의 성과
- 객체 인식(Krizhevsky et al., 2012)
- 음성 인식(Dahl et al., 2012)
- 자연어 처리
 > - 언어 모델링 (Bengio et al., 2003)
 > - 패러프레이즈 감지 (Socher et al., 2011)
 > - 단어 임베딩 추출 (Mikolov et al., 2013) 
 > - 통계 기반 기계 번역(SMT) 
 > - 피드포워드 신경망이 구문 기반 SMT 시스템의 프레임워크에서 성공적으로 사용(Schwenk, 2012)

###  Statistical Machine Translation (SMT): 통계 기반 기계 번역
- RNN 인코더-디코더
  > - 인코더 : 가변 길이의 source seq -> 고정 길이 벡터로 매핑
  > - 디코더 : 고정 길이 벡터 -> 가변 길이 target seq로 매칭
  > - 두 RNN 모두 source seq가 주어졌을 때 garget seq의 조건부 확률을 최대화 하기 위해 공동 훈련
  > - 복잡한 hidden unit 사용 -> 메모리 용량, 훈련의 용이

- english -> franch : SMT에 적용
  > - 구문 쌍을 점수화한 결과 성능 향상

### 성과
- 구문 테이블의 언어 규칙 잘 포착 -> 전반적인 번역 성능 개선
- 모델의 추가 분석 결과,구문의 의미론적 및 구문론적 구조를 보존하는 구문의 연속 공간 표현을 학습
  
## (5) Conclusion
### RNN 인코더-디코더
 - 인코더 : 가변 길이의 source seq -> 고정 길이 벡터로 매핑
 - 디코더 : 고정 길이 벡터 -> 가변 길이 target seq로 매칭
 - 활용
  > - 조건부 확률을 이용해서 sequence pair를 점수화 -> 평가에 활용
  > - source seq 가 주어졌을 떼 target seq 생성 -> SMT에 활용
  
### a novel hidden unit 제안
- reset gate :   이전 시간 스텝에서 어떤 정보를 삭제할지 결정
- update gate : 현재 숨겨진 상태에 포함해야 하는 새 정보의 양을 결정
- 동적 메모리 관리를 통해 장거리 종속성을 순차적으로 캡쳐하는 모델의 기능 향상
  
### evaluation
- 구문 쌍에 나타나는 언어적 패턴과 규칙성을 포착하는 데 탁월
- 체계적인 대상 문구를 제안 -> 의미있는 번역 가능
- 번역 성능 향상 -> BLEU 점수 증가
- 
## (2) RNN ENcoder_Decoder
### 2.1. Preliminary: Recurrent Neural Networks
#### RNN
- 순환 신경망(RNN):  숨겨진 상태 h와 선택적 출력 y로 구성된 신경망

- 숨겨진 상태(h)의 업데이트
> - ![formula1](/assets/img/nlp/10_GRU_formula1.png){: width="100%" height="100%"}
> - f : 비선형 활성화 함수 (eg: 로지스틱 시그모이드)
> - unit : LSTM 등

- 학습
  > - seq에 대한 확률 분포 학습 -> seq 의 다음 simbol
  > - 각 시간 t의 출력 : 조건부확률  p(xt | xt−1, . . . , x1)
  > softmax 활성화 함수 사용한 분포
  > - ![formula2](/assets/img/nlp/10_GRU_formula2.png){: width="100%" height="100%"}
  > - seq x의 확률
  > - ![formula3](/assets/img/nlp/10_GRU_formula3.png){: width="100%" height="100%"}


### 2.2 RNN Encoder–Decoder
- 다른 길이의 seq를 받아, 가변 길이의 seq에 대한 조건부 분포를 학습
- 입력 seq, 출력 seq의 길이가 다를 수 있음.
-  p(y1, . . . , yT' | x1, . . . , xT) : T', T가 다를 수 있음.
  
#### Incoder
- 입력 시퀀스(x)의 각 기호를 순차적으로 읽는 RNN
- 각 기호를 읽을 때마다 RNN의 숨겨진 상태가 변경되며, 시퀀스의 끝까지 읽은 후에는 전체 입력 시퀀스를 요약하는 숨겨진 상태(c)가 생성

#### decoder
- 출력 시퀀스를 생성하기 위해 훈련
- 숨겨진 상태 h(t) 가 주어졌을 때 다음 기호 y(t)를 예측하여 출력 seq 생성
- 시간 t에서 디코더의 숨겨진 상태 :
- ![formula4_1](/assets/img/nlp/10_GRU_formula4_1.png){: width="100%" height="100%"}
- 다음 simbol의 조건부 확률 : 
- ![formula4_2](/assets/img/nlp/10_GRU_formula4_2.png){: width="100%" height="100%"}
- g: softmax 등 activation 홤수
  
#### 조건부 로그 우도 최대화
- ![formula4](/assets/img/nlp/10_GRU_formula4.png){: width="100%" height="100%"}
- θ : 모델 파라미터 집합
- (xn, yn): 훈련 세트에서 가져온 (입력 시퀀스, 출력 시퀀스) 쌍
  
# 활용
- 입력 시퀀스가 주어졌을 때 모델을 사용하여 대상 시퀀스를 생성
- 모델:  주어진 입력 및 출력 시퀀스 쌍을 점수화하는 데 사용
  
### 2.3 Hidden Unit that Adaptively Remembers and Forgets
#### hidden activation function
- ![figure2](/assets/img/nlp/10_GRU_figure2.png){: width="100%" height="100%"}
- z(update gate) :   숨겨진 상태가 새로운 숨겨진 상태(h)로 업데이트 될지 여부 선택
 로 업데이트될지 여부를 선택
- r(reset gate) : 이전 숨겨진 상태가 무시될지 여부 결정
  
#### j번째 은닉 유닛의 활성화 계산
- r_j(reset gate)
> -  ![formula5](/assets/img/nlp/10_GRU_formula5.png){: width="100%" height="100%"}
> -  σ:  로지스틱 시그모이드 함수
> - [·]_j:  벡터의 j번째 요소
> - x: input
> - h_t-1 :  이전 hidden state
> - W_r ,  U_r :  weight matrix

- z_j(update gate)
  > - ![formula6](/assets/img/nlp/10_GRU_formula6.png){: width="100%" height="100%"}

- h_j의 activation 
  > - ![formula7](/assets/img/nlp/10_GRU_formula7.png){: width="100%" height="100%"}
  > - ![formula8](/assets/img/nlp/10_GRU_formula8.png){: width="100%" height="100%"}
  > - ⊙: 요소별 곱셈
  > - ϕ :  활성화 함수
  > - [Wx]j : 입력에 대한 가중치
 > - U_r : reset gate에 대한 가중치
 > - h^(t-1) : 이전 시간 단계의 hidden state
 > - ~h^(t)_j : j 번째 숨겨진 상태의 업데이트된 값

#### gate
- reset gate
  > -  0에 가까우면 이전 hidden state 상태를 무시하고 현재 입력으로 재설정 -> 간결한 표현
- update gate :  이전 숨겨진 상태에서 현재 숨겨진 상태로 전달되는 정보의 양을 제어
  > - LSTM 네트워크의 메모리 셀과 유사하게 작용하여 RNN이 장기적인 정보를 기억하는 데 도움
- 각 은닛 유닛은 각자 reset, update gate를 가짐
  > - 서로 다른 시간 스케일에서 종속성 포착
  > - 짧은 시간의 종속성을 포착한 유닛: 자주 활성화되는 reset gate가 더 영향
  > - 장기적인 종속성을 포착한 유닛: update gate가 더 활성화

## (3) Statistical Machine Translatio
### 3.1 Scoring Phrase Pairs with RNN Encoder–Decoder
### 3.2 Related Approaches: Neural Networks in Machine Translation

## (4) Experiments
### 4.1 Data and Baseline System
#### 4.1.1 RNN Encoder–Decoder
#### 4.1.2 Neural Language Model
### 4.2 Quantitative Analysis
### 4.3 Qualitative Analysis
### 4.4 Word and Phrase Representations
