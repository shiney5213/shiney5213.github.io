---
layout: page
title: [LSTM] Long Short Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
description: >
  RNN paper review
hide_description: true
sitemap: false
---


- Title: Long Short Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
- paper: [download]
- journal:  
- year : 2014
- Subjects:	Computation and Language (cs.CL) 
- Haşim Sak , Andrew Senior , Françoise Beaufays
- summary
- RNN의 한계
  > - 학습 데이터가 커질 수록 앞서 학습한 정보가 충분히 전달 안됨 -> 장기 의존성 문제(Long-term dependencies) 발생
  > - 하이퍼볼릭 탄젠트함수나 ReLU 함수의 특성으로 인해 역전파 과정에서  기울기 소실 및 폭주 문제
  
- LSTM(Long Short Term memory)  
  > - 메모리 셀과 다양한 게이트를 사용하여 네트워크 내의 정보 흐름을 제어함으로써  LSTM 모델이 해결 


- 본 연구
  > - 기존의 LSTM 모델의 계산 복잡도를 줄이기 위해 projection layer를 추가한 두 가지 architecture 제안
  > - standard LSTM 모델에 recurrent projection layer, non-recurrent projection layer 추가하여 계산 복잡도 낮추고, 성능 향상
  > - 비동기적 확률적 경사 하강(ASGD)
  > - truncated BPTT( backpropagation through time) : 오차 역전파의 기울기 소실 문제를 해결하기 위해 현재 단계에서 몇 단계까지만(보동 5단계) 전파시키는 방법

[download]: https://arxiv.org/abs/1402.1128

1. this unordered seed list will be replaced by toc as unordered list
{:toc}
<br>

## (0) ABSTRACT
- RNN 모델
  > -  순환 연결을 가지고 있어서 순차적인 데이터를 모델링하는데 효과적
  > - 손글씨 인식, 언어 모델링, 음성 프레임의 음성 라벨링과 같은 순차 라벨링 및 순차 예측 작업에 성공적으로 사용
  > - 한계:  그래디언트 소실과 폭주 문제

- LSTM
  > - 일종의 순환 신경망 (RNN) 아키텍처
  > - RNN 모델의 기울기 소실, 폭발 문제 해결
  > - vanishing gradient : 네트워크에서 가중치를 업데이트하는 데 사용되는 그래디언트가 매우 작아져 학습 속도가 느려지거나 학습이 전혀 되지 않는 문제
  > - exploding gradient : 네트워크에서 가중치를 업데이트하는 데 사용되는 그래디언트가 너무 커지면 기울기가 폭발적으로 증가하여 학습 과정 불안정
 > -  메모리 셀과 다양한 게이트를 사용하여 네트워크 내의 정보 흐름을 제어함으로써 이러한 문제를 극복

- 본 논문
  > - RNN: 음성 인식 분야에서 전화 인식에만 제한적으로 사용
  > - 대량 어휘 음성 인식을 위한 음향 모델을 훈련시키기 위해 모델 매개변수를 더 효과적으로 활용하는 새로운 LSTM 기반 RNN 아키텍처를 제안
  > - LSTM 모델이 빠르게 수렴하고 상대적으로 작은 크기의 모델에 대해 최신의 음성 인식 성능을 제공

## (4) CONCLUSION
- 새로운 LSTM architecture 제안 (출력  unit이 많은 LSTM의 확장성 문제를 해결하기 위해 )
  > - 재귀가 없는 LSTM 계층과 출력 계층 사이에 `recurrent projection layer`를 추가한 모델 
  > - 순환 연결을 더 추가하지 않고도 프로젝션 계층의 크기를 늘리는 `non-recurrent projection layer` 추가한 모델  -> 유연성 증가

- 새로운 LSTM 모델의 성능
  > - 표준 LSTM 아키텍처에 비해 LSTM 네트워크의 성능을 크게 향상
  > - 출력 상태가 많은 대규모 어휘 음성 인식 작업에서 심층 신경망 (DNN) 보다 성능 향상 

## (1) INTRODUCTION
### RNN
- 특징
  > - 이전 시간 스텝의 활성화 값을 네트워크에 입력으로 제공하여 현재 입력에 대한 결정을 내림
  > - 이전 타임스텝의 활성화를 내부 상태에 저장하여 무한한 시간적 컨텍스트 정보를 제공
  > - 고정된 크기의 창이 아닌 전체 시퀀스 기록을 포함하는 동적으로 변화하는 컨텍스트 창을 사용하여 시퀀스 모델링에 적합
- 한계
  > - BPTT(gradient-based back-propagation through time) : gradient vanishing and explding problem
  > - 멀리 있는 context의 의존성을 모델링 하는 기능이 제한되며, 일반적으로 5-10개의 time step만 모델링 가능

### LSTM
- 순환 은닉 layer에 있는 memory block이라는 특수 유닛으로 구성
  > - memory cell :  네트워크의 시간 상태 저장 및 기억
  > - 게이트: 네트워크의 정보 흐름을 제어

- 게이트 type
  > - input gate: 메모리 셀로 향하는 입력 확성화 흐름 조절
  > - output gate: : 메모리 셀에서 다음 셀로 가는 활성화 흐름 제어
  > - forget gate: 자체 순환 연결을 통해 셀의 입력이 들어 오기 전 셀의 내부 상태 scaling -> 셀의 메모리 삭제 or reset하여 입력 stream을  효과적으로 할 수 있게 함
  > - peephole connections : LSTM 내부 셀에서 동일한 셀 내의 게이트로 연결 -> LSTM이 정확한 타이밍으로 출력하도록 학습 

### 기존 연구
- LSTM
  > - 문맥에 자유롭거나, 문맥에 의존적인 언어에서 순차 예측 및 라벨링 task에서 RNN 보다 우수

- 양방향 LSTM 네트워크
  > - 입력 시퀀스를 양방향으로 처리하여 현재 입력에 대한 결정을 내림
  > - TIMIT 음성 데이터베이스의 음향 프레임에 대한 음성 라벨링을 위한 task에 사용
  > - 온라인 및 오프라인 필기 인식에서는 연결주의적 시간 분류(CTC) 출력 레이어를 사용

- deep LSTM RNN
  > -  CTC 출력 레이어 및 전화 번호 시퀀스를 예측하는 RNN 변환기와 결합되어 TIMIT 데이터베이스에서 전화 인식에서 최첨단 결과
  > - 언어 모델링에서는 표준 n-gram 모델에 비해 엄청난 퍼플렉서티 감소 달성

### 본 연구
- LSTM 기반 RNN 아키텍처
  > - 수천 개의 문맥 의존(CD) 상태가 있는 대규모 어휘 음성 인식 시스템에서 최첨단 성능을 달성
  > - 큰 모델에서의 계산 효율성 문제를 해결하기 위해 LSTM 네트워크의 표준 아키텍처를 수정하여 사용

## (2) LSTM ARCHITEXTURES
### LSTM networks
#### standard LSTM networks
- input layer, recurrent LSTM layer, output layer로 구성
- recurrent connection : output unit-> input unit-> input gate -> output gate, forget gate
  
- 각 메모리 븍록에 하나의 셀이 있는 표준 LSTM 네트워크의 가중치
- ![formula1](/assets/img/nlp/8_LSTM_formula1.png){: width="70%" height="70%"}
> - n_c: 메모리 셀의 수(이 경우 메모리 블록의 수)
> - n_i:  입력 단위의 수
> - n_o:  출력 단위의 수

- 계산 복잡도
> -  O(1): 확률적 경사 하강법(SGD) 최적화 기술을 사용하여 LSTM 모델을 학습하는 계산 복잡성
> -  O(W): 시간 단계당 학습의 계산 복잡성
> - 적은 수의 입력을 가진 네트워크의 학습 시간:  nc × (nc + no) 요소에 의해 지배
> - 시간적 문맥 정보를 저장하기 위해 대규모의 출력 단위 및 메모리 셀이 필요한 작업의 경우, LSTM 모델을 학습하는 것은 계산적으로 비용이 많이 필요

####  1. LSTMP(Long Short-Term Memory Projected) - LSTM with Recurrent Projection Layer
- 셀 출력 단위를 셀 입력 장치 및 재귀 게이트에 연결하는 순환 투영 계층 필요
- 네트워크 출력 유닛에 연결하여 출력값 예측
- 파라미터 수 :  n_c × n_r × 4 + n_i × n_c × 4 + n_r × n_o + n_c × n_r × 3 
- n_c:  셀의 단위 수
- n_r: 순환 투영 계층의 단위 수
- n_i: 입력 계층의 단위 수
- n_o: 출력 계층의 단위 수

#### 2. Deep LSTMP
- 순환 투영 계층 외에 출력 계층에 직접 연결된 또 다른 비순환 투영 계층 추가
- 추가 계층을 사용하면 순환 연결의 파라미터 수를 늘리지 않고도 투영 계층의 단위 수를 늘릴 수 있음
- 파라미터 수: n_c × n_r × 4 + n_i × n_c × 4 + (n_r + n_p) × n_o + n_c × 3 
- n_p: 비순환 투영 레이어의 단위 수
- nr + np 단위의 단일 투영 레이어를 추가할 수 있음. 
- LSTMP와 같은 기능이나, 투영 계층의 단위 수를 늘릴 수 있는 유연성 제공

#### LSTM network 계산
- input seq :  (x1, ..., xT)
- output seq : y = (y1, ..., yT)
- ![model1](/assets/img/nlp/8_LSTM_model1.png){: width="70%" height="70%"}

- LSTM architecture with recurrent projection layer
> - ![formula2](/assets/img/nlp/8_LSTM_formula2.png){: width="70%" height="70%"}
> - i_t: input gate
> - f_t: forget gate
> - c_t : memory cell
> - o_t : output gate
> - m_t : memory block
> - y_t : output

- LSTM architecture with both recurrent and non-recurrent projection layer
> - ![formula3](/assets/img/nlp/8_LSTM_formula3.png){: width="70%" height="70%"}

### IMPLEMENTATION
- 단일 시스템의 멀티 코어 CPU 사용( not GPU)
  > - CPU의 상대적으로 간단한 구현 복잡성과 디버깅의 용이성 Eoans
  > -  대규모 네트워크의 학습 시간이 단일 머신에서 주요 병목 현상이 되는 경우 대규모 클러스터의 머신에 대한 분산 구현을 용이

- 행렬 연산: Eigen 행렬 라이브러리를 사용
- > - Eigen : 벡터화된 명령어를 사용하여 CPU에서 행렬 연산을 효율적으로 구현할 수 있는 템플릿 C++ 라이브러리
  > - SIMD (single instruction multiple data) 명령어 사용 -> 행렬 연산을 병렬화 -> 계산 효율성 향상
  > - 행렬의 활성화 함수 및 gradient 계산할 때 병렬화 사용 ->  LSTM 아키텍처의 학습 프로세스를 가속화

- `비동기적 확률적 경사 하강(ASGD)` 최적화 기술을 사용
  > - 가중치 업데이트 할 때 멀티 core의 여러 thread에서 비동기적으로 수행
  > - 각 thread는 계산 효율성을 위해 병렬로 일괄 시퀀스에 수행 -> 계산 효율성 향상
  > - 모델 매개변수가 여러 입력 시퀀스로부터 동시에 업데이트될 수 있기 때문에 보다 스토캐스틱한 특성을 얻을 수 있음
  > - 여러 스레드로 학습하는 것은 사실상 훨씬 큰 시퀀스 일괄 처리 (스레드 수 곱하기 일괄 크기)가 병렬로 처리하는 것

- `truncated BPTT( backpropagation through time)`
  > - 모델 파라미터 업데이트 할 때 사용
  > - T_bptt : 고정된 time step : 활성화 값을 순방향으로 전파하고 기울기를 역전파할 때 사용
  > - 입력 seq를 크기가 T_bpttdls 하위 시퀀스로 분할
  > - 1.  네트워크 입력값과 Tbptt 시간 스텝에 대해 첫 번째 프레임부터 이전 시간 스텝의 활성화 값을 사용하여 활성화 수를 반복해서 계산하고 순방향 전파
  > - 2. 각 타임스텝에서 네트워크 비용 함수를 사용하여 네트워크 오류를 계산
  > - 3. 각 시간 스텝의 오차와 T_bptt 시간부터 시작하는 다음 시간 스텝의 기울기를 사용하여 교차 엔트로피 기준으로부터 기울기를 역전파
  > - 4. Tbptt 타임스텝에 대한 네트워크 파라미터 (가중치) 의 기울기를 누적하고 가중치를 업데이트
  > - 각 서브시퀀스를 처리한 후 메모리 셀의 상태가 다음 서브시퀀스를 위해 저장.
  > - 여러 입력 시퀀스에서 여러 서브시퀀스를 처리할 때 일부 서브시퀀스는 T_bptt보다 짧을 수 있음
  > -  다음 서브시퀀스의 일괄 처리에서는 이러한 서브시퀀스를 새 입력 시퀀스의 서브시퀀스로 교체하고 이들에 대한 셀의 상태를 재설정

## (3) EXPERIMENTS
- 대규모 어휘 음성 인식 작업인 Google 영어 음성 검색 작업에서 DNN, RNN 및 LSTM 신경망 아키텍처의 성능 평가 및 비교
  
### Systems & Evaluation
- dataset
  > - 3백만 개의 발화로 구성된 데이터 세트
  > - 약 1900시간의 오디오 데이터
  > - 각 발언은 40차원 로그 필터 뱅크 에너지 특성의 25ms 프레임을 사용하여 표현되며, 이는 10ms마다 계산됨

- train setting
  > - 발화를 해당 트랜스크립트와 일치시키기 위해 14247개의 컨텍스트 종속 (CD) 상태를 갖는 9천만 파라미터 피드포워드 신경망 (FFNN)  사용
  > - input state: 14247개의 컨텍스트 종속 (CD) 상태
  > - output state: (126, 2000, 8000) 3가지 상태로 출력 -> 유사한 상태를 함께 그룹화하는 등가 클래스를 사용하여 수행
  > - 예: 126 상태 출력:  음성 상태 수 (3) 에 음성 클래스 수 (42) 를 곱하여 구한 컨텍스트 독립 (CI) 상태를 나타냄
  > - 훈련 전에 모든 네트워크의 가중치를 무작위로 초기화


- train
  > - 학습률: 각 네트워크 아키텍처 및 해당 구성에 맞게 설정-> 안정적으로 수렴하는 가장 큰 값으로 설정, 훈력 중 지수 감소

- evaluation
  > - 음성 프레임의 분류 정확도(음향 프레임의 라벨링 정확도)를 평가
  > - test set:  20만 프레임으로 구성된 set
  > - 테스트 세트에 있는 2만 3000개의 수동 전사된 발화에 대한 음성 인식 시스템에서 훈련된 모델들을 평가하고 단어 오류율(WER)을 보고
  > - 디코딩에 사용된 언어 모델의 어휘 크기: 260만

- DNN model
  > - 미니배치 크기가 200프레임인 SGD (확률적 경사하강법) 를 사용하여 훈련
  > - DNN 네트워크: fully connected layer 
  > - hidden lyaer : 로지스틱 시그모이드 활성화 함수 사용
  > - output layer : 전화 HMM (Hidden Markov Model) 상태를 출력,  소프트맥스 활성화 함수 사용

- LSTM 및 RNN model
  > - 24개 스레드가 포함된 ASGD (비동기 스토캐스틱 그래디언트 디센트) 를 사용하여 학습
  > - 각 스레드는 서로 다른 데이터 파티션을 처리하고 서로 다른 발화에서 나온 4개 또는 8개의 하위 시퀀스에 대해 그래디언트 스텝을 계산
  > - truncated BPTT (시간을 통한 역전파) 학습 알고리즘을 사용 :  기울기의 순방향 전파 및 역방향 전파에 타임 스텝 20 (T_bptt) 사용
  > - RNN의 hidden layer :  로지스틱 시그모이드 활성화 함수 사용
  > - 순환 투영 계층 아키텍처를 갖춘 RNN:  투영 계층에서 선형 활성화 단위 사용
  > - LSTM :  셀 입력 및 출력 단위에 쌍곡선 탄젠트 활성화 (tanh) 를 사용하고 입력, 출력 및 포겟 게이트 단위에 로지스틱 시그모이드를 사용
  > - LSTM의 순환 투영 계층과 선택적 비순환 투영 계층:  선형 활성화 단위를 사용
  > - LSTM과 RNN 모두에 대한 입력은:  40차원 로그 필터 뱅크 에너지 특성으로 구성된 25ms 프레임
  > - 미래 프레임의 정보를 활용하여 현재 프레임에서 더 나은 의사 결정을 내리기 위해 출력 상태 레이블이 5프레임 지연

### Results
- 용어
  > - c_N : LSTM의 메모리 셀 수(N)와 RNN의 숨겨진 레이어의 유닛 수
  > - r_N : LSTM 및 RNN의 재귀적 프로젝션 유닛 수
  > - p_N : LSTM의 비재귀적 프로젝션 유닛 수
  > - 

![fig.2](/assets/img/nlp/8_LSTM_fig2.png){: width="70%" height="70%"}
-  RNN :  훈련 초반에 매우 불안정하며 수렴을 달성하기 위해 폭발적인 그래디언트 문제로 인해 활성화 및 그래디언트를 제한
-  LSTM : 수렴 속도가 더 빠르면서 RNN 및 DNN보다 훨씬 더 좋은 프레임 정확도를 제공 -> LSTM_c512
-  논문에서 제공한 LSTM : 동일한 매개변수 수를 가진 표준 LSTM RNN 아키텍처보다 훨씬 더 나은 정확도 제공 -> LSTM_c2048_r512, LSTM_c2048_r256_p256

![fig.3](/assets/img/nlp/8_LSTM_fig3.png){: width="70%" height="70%"}
- LSTM_c512 / LSTM_c1024_r256 비교
- 재귀 및 비재귀 프로젝션 레이어를 모두 갖춘 LSTM 네트워크는 일반적으로 재귀적 프로젝션 레이어만 있는 LSTM 네트워크보다 더 좋은 성능을 보임

![fig.6](/assets/img/nlp/8_LSTM_fig6.png){: width="70%" height="70%"}
- 컨텍스트 독립 126 , 컨텍스트 종속 2000 출력일 때  WER 비교
-  LSTM_c512 / LSTM_c1024_r256 비교
