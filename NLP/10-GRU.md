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

## (3) Statistical Machine Translation
- SMT의 목표 : source 문장(e)가 주어졌을 때 번역 문장(f) 찾기
- p(f | e) ∝ p(e | f)p(f)
  > -  p(f | e) : 소스 문장이 주어졌을 때 번역 문장이 나타날 조건부 확률 -> maximize해야 함
  > -  p(e | f) : 번역 모델
  > -  p(f) : 언어 모델

-  log p(f | e)  
  > - SMT 시스템에서 log p(f | e)의 추가적인 특성과 해당 가중치를 가진 로그 선형 모델로 모델링
  > - 로그 선형 모델: 추가적인 특징과 그에 상응하는 가중치(weights)를 사용하여 번역의 확률 결정

- ![formula9](/assets/img/nlp/10_GRU_formula9.png){: width="100%" height="100%"}
> - f_n: n번째 특징(feature)
> - w_n은 해당 특징에 대한 가중치(weight)
> -Z(e): 가중치에 의존하지 않는 정규화 상수(normalization constant)
>

- BLEU 점수
  > - 번역의 품질을 측정하는 데 널리 사용되는 메트릭 중 하나
  > - 실제 번역과 인간의 참조 번역 간의 일치 정도를 고려한 자동 번역 품질 평가 도구

- log p(e | f) : 번역 모델
  > -  소스와 타겟 문장에서 일치하는 구문들의 번역 확률로 분해
  > -  이러한 확률들은 다시 한 번 로그 선형 모델에서 추가적인 특성으로 간주
  > -  BLEU 점수를 최대화하기 위해 이에 상응하는 가중치가 부여

### 3.1 Scoring Phrase Pairs with RNN Encoder–Decoder
#### 학습 
- Phrase Pairs 데이터를 이용하여 RNN Encoder–Decoder 학습
  > - 신경망 모델은 서로 다른 언어의 문구 쌍에 노출되어 번역 목적에 맞게 효과적으로 인코딩 및 디코딩하는 방법 학습

- 학습한 모델에서 생성한 phrase pairs 에 대한 점수를 로그 선형 모델에 feature로 사용
  > - SMT 디코더 튜닝할 때 RNN 인코더-디코더에서 얻은 점수를 로그 선형 모델에 통합
  > -  학습한 프레이즈 쌍의 표현을 활용하여 번역 프로세스를 개선
  > - 기존의 phrase table에 phrase pairs의 점수 추가
  > - 새로운 점수가 최소한의 추가 계산 비용으로 기존의 튜닝 알고리즘에 들어갈 수 있음

- corpora 안에 있는 각 phrase pairs의 정규화된 빈도 무시
  > - 학습 과정 간소화
  > - 발생 빈도에 따른 단순 암기 방지

- 모델이 언어의 규칙성을 학습하는데 초점
  > - 목표 : 그럴듯한 번역과 그럴듯하지 않은 번역을 구별하는 것
  > - 문구와 번역의 기본 구조를 학습하여 언어의 뉘앙스를 이해하고 번역 품질을 향상시키는 데 도움
### 3.2 Related Approaches: Neural Networks in Machine Translation
#### SMT(SMT)에서 신경망을 활용하는 연구
- Schwenk(Schwenk, 2012):구문 쌍의 점수화에 대한 유사한 접근법을 제안
- 데블린 등(Devlin et al., 2014) : 번역 모델을 모델링하기 위해 피드포워드 신경망을 사용하는 것을 제안
- Zou 등(Zou et al., 2013):  단어/구문의 이중 어휘 임베딩을 학습 제악
- Chandar(Chandar et al., 2014): 피드포워드 신경망을 훈련하여 입력 구문의 단어 가방 표현(bag-of-words representation)에서 출력 구문으로의 매핑을 학습
- Kalchbrenner 등(Kalchbrenner and Blunsom, 2013) : Recurrent Continuous Translation Model(Model 2)

## (4) Experiments
- English/French translation task
### 4.1 Data and Baseline System
#### data
- Moore and Lewis (2010) 와 Axelrod et al. (2011) 이 제안한  데이터 선택 방법 사용
- 언어 모델링: 20억 단어 중 41.8억 단어의 하위 집합 선택
-  RNN 인코더-디코더 훈련:  8.5억 단어 중 3.48억 단어의 하위 집합을 선택
-  데이터 선택 및 가중치 튜닝:  MERT 사용
-  테스트 세트:  newstest2012와 2013를 사용

#### 신경망 훈련
- 데이터세트의 93% 를 차지하는 영어 및 프랑스어 상위 15,000개 단어로 어휘를 제한
-  어휘를 벗어난 단어: 특수 토큰([UNK])에 매핑

### 실험 결과
- Moses를 사용하여 구축한 베이스라인 프레이즈 기반 SMT 시스템
- train set :  BLEU 점수 30.64점
- test set:  BLEU 점수 33.3점

#### 4.1.1 RNN Encoder–Decoder
##### Architecture
- input unit: 100개
- hidden unit : 1000개
- output unit: 500개
-  활성화 함수: 쌍곡선 탄젠트 함수

##### hyperparams
- 가중치 매개변수: 평균이 0이고 표준 편차가 0.01로 고정된 등방성 제로 평균(백색) 가우시안 분포에서 샘플링을 통해 초기화
- 순환 가중치 행렬 :  백색 가우시안 분포에서 샘플링하고 그 왼쪽 특이벡터 행렬 사용
- Adadelta와 확률적 경사 하강법 사용 (하이퍼파라미터: ε = 10^-6 및 ρ = 0.95 )

##### data
- 무작위로 선택된 64개의 구문 쌍을 사용
  
#### 4.1.2 Neural Language Model
##### Continuous Space Language Model(CSLM) 훈련
- 목적: 디코딩 과정 중에 부분 번역에 점수를 매기기 위해 사용
- 모델
> - 입력 단어: 임베딩 공간 R^512로 투영 후 3072차원 벡터를 형성
> - 모델 : 512 -> 3072 -> ReLU 레이어(1536 ) -> ReLU 레이어(1024) ->  소프트맥스 레이어(출력 레이어)
> - 모든 가중치 매개변수 :  균등하게 -0.01과 0.01 사이에서 초기화
> - 검증 퍼플렉서티가 10 에포크 동안 개선되지 않을 때까지 훈련
- 결과 : 훈련 후, 언어 모델의 퍼플렉서티 :  45.80
- 계산 복잡성 해결하기 위해 CSLM에 의한 n-gram 스코어링이 구현
  > - 디코더에는 스택 서치를 수행하는 동안 n-그램을 집계하기 위한 버퍼 사용
  > - 버퍼가 가득 차거나 스택이 제거될 때에만 n-그램이 CSLM에 의해 스코어링
> - Theano를 사용하여 GPU에서 빠른 행렬-행렬 곱셈 수행

### 4.2 Quantitative Analysis
- RNN Encoder- Decoder 모델에서 계산된 점수를 추가했을 때 성능 향상
-  CSLM과 RNN 인코더-디코더로부터의 구문 점수를 함께 사용했을 때 최고  성능
-  
### 4.3 Qualitative Analysis
- RNN 인코더-디코더: 대부분 실제 번역이나 문자 그대로 번역에 더 가까움
  
### 4.4 Word and Phrase Representations
![figure4](/assets/img/nlp/10_GRU_figure4.png){: width="100%" height="100%"}
-  RNN 인코더-디코더에 의해 학습된 단어 임베딩 행렬을 사용하여 단어의 2차원 임베딩
> - 의미적으로 유사한 단어들이 함께 클러스터링되는 것 확인


![figure5](/assets/img/nlp/10_GRU_figure5.png){: width="100%" height="100%"}
-  Barnes-Hut-SNE를 사용하여 시각화
  > - 구문과 의미론적 구조를 모두 포착
> - 왼쪽 아래 플롯:  대부분의 구문은 시간의 기간에 관한 것이며, 구문적으로 유사한 것들은 함께 클러스터링
> - 오른쪽 아래 플롯:  국가 또는 지역 등 의미적으로 유사한 구문들의 클러스터 시각화
> - 위쪽 오른쪽 플롯: 구문적으로 유사한 구문