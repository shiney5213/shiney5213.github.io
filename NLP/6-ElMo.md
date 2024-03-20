---
layout: page
title: ELMo :Deep contextualized word representations 
description: >
  ELOm paper review
hide_description: true
sitemap: false
---


- Title: ELMo :Deep contextualized word representations 
- paper: [download]
- journal: 
- year : 2018
- Subjects:	Computation and Language (cs.CL) 
- Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer
- summary
  > - ELMo : Embeddings from Language Models
  > - 단어의 복잡한 특성(구문 및 의미)을  고려하여 다양한 맥락에서 정확한 의미의 단어 사용 모델링
  > - -> 기존의 모델: 단어가 embedding 된 후, 변하지 않음.
  > - -> ELMo : embedding을 사용할 때 주변 맥락 정보에 따라 가변적일 수 있도록 만듦.->Deep contextualized word representations 
  > - 기존에는  SQuAD(질의응답), SNLI(문장 사이의 모순 파악), SRL(의미역 결정), Coref(Entity 찾기, Blog), NER(Entity 인식), SST-5(문장 분류)에 특화된 각 모델이 존재
  > - 이 논문에서는 하나의 Language Model을 통해서 6개의 task를 모두 수행
  > - BiLM 모델의 레이어마다 다른 정보를 포함 -> softmax 정규화 가중치를 두어 embedding을 만들어 task에 따라 사용 
- 사용 기법
  > - 전이학습 -> 기 학습된 언어 모델의 지식이 임베팅으로 전이 ( 대규모 데이어에 사전 학습 된 모델에 biLM 추가)
  > - LSTM의 최상위 레이어를 사용하지 않고, 모든 레이어의 값을 연산하여 사용-> a linear combination of the biLM layers.
  > - biLM : 양방향 순환 신경망
  > - 각 task별로 최적화를 위해 γ값 조절
  
[download]: https://arxiv.org/abs/1802.05365



1. this unordered seed list will be replaced by toc as unordered list
{:toc}
<br>

## (0) Abstract
- 연구 목표 
> - 단어 사용의 복잡한 특성(구문 및 의미 등) 포착
> - 이러한 사용이 언어적 맥락에서 어떻게 변하는지 고려하여 다원성(단어의 여러의미) 모델링

- model
  > - 대규모 텍스트 말뭉치에서 사전 훈련 된 깊은 양방향 언어모델(biLM)의 내부 상태의 학습 

- 결과 및 의의
  > - 단어 벡터:  BiLM의 내부 상태를 사용하여 문장의 앞 단어와 다음 단어를 모두 고려하여 단어의 문맥 정보를 파악
  > - 유용한 단어 사용 및 문맥에 대한 중요한 정보를 파악 가능
  > - 질문에 대한 답변, 텍스트 관련, 감정 분석을 비롯한 6가지 난이도 높은 NLP 문제에 대한 모델의 성능이 크게 향상

## (6) Conclusion
- NLP task 성능 개선
> - ELMo : 심층적 상황에 맞는 단어 표현
  > - 기존 모델에 통하하면 자연어 처리 작업에서 성능이 크게 향상
  > - 단어 벡터가 단어 사용과 다중성의 복잡성을 포착하는데 효과적 

- BiLM 레이어의 중요성
> - 다양한 유형의 구문 및 의미 정보를 인코딩에 효과적
> - 모든 레이어를 사용하면 전반적인 nlp task 성능 향상

## (1) Introduction
- 이전 연구
  > - Word2Vec : 많은 신경망 모델의 중요한 기반이지만 고품질의 표현 학습은 어려움
  > - 각 토큰이 전체 입력 문장의 함수로 할당되는 단어 유형의 임베딩 방법

- 연구 과제
  > - 단어의 복잡한 특성(구문 및 의미)을  고려하여 다양한 맥락에서 단어의 사용(다의성) 모델링

- 연구 과정: ELMo(Embeddings from Language Models) 
  > - 대규모 텍스트 말뭉치에서 결합된 언어 모델(LM)로 양방향 LSTM 에서 파생된 벡터 사용
  > - 모든 biLM의 내부 레이어의 함수로 연산
  > - 각 엔드 태스크에 대해 각 입력 단어 위에 쌓인 벡터의 선형 조합을 학습( LSTM의 최종 레이어를 사용하는 것보다 성능 좋음)
  > - LSTM의 내부 상태(레이어)를 결합하여  풍부한 단어 표현 가능
  
- 연구 결과
> - 단어 의미가 문맥에 따라 달라지는 표현을 포착
> - 문맥을 기반으로 정확한 의미의 단어를 결정 -> 단어 의미 명확화 task에 적합
> -  LSTM 계층의 하위 수준에 있는 숨겨진 상태를 이용하여 단어의 품사 태깅 task 가능
> - LSTM 계층의 상위, 하위 수준의 상태를 모두 이용하면 단어의 의미 및 구문 정보를 파악
> - semi-supervision : 학습시 부분적으로 레이블이 지정된 데이터를 사용하는 것-> 다양한 NLP task에 적용

## (5) Analysis
### Alternate layer weighting schemes (레이어의 가중치 설정의 대안)
- 양방향 언어 모델 (BiLM)의 layer 결합 방법
  > - 일반적으로 컨텍스트 표현에 BiLM의 마지막 계층 또는 MT 인코더만 사용-> 
  > - ELMo: 모든 layer 사용( LSTM 레이어별로 가중치를 두고 더함)
  > - 더할 때 softmax 레이어로 정규화된 가중치 사용-> task에 따라 조절
  > - 정규화 파라미터 (λ) : λ이 클수록 layer의 단순 평균으로 줄이는 효과,  λ=0.001같이 작으면 가중치를 다양해짐
  > - task별로 λ값 조정하여 사용

- SQuAD, SNLI , SRL 비교
  > - BiLM의 모든 레이어(정규화 강도 λ의 선택 항목이 다름)를 최상위 레이어와 비교
  > - SQuAD(질의응답), SNLI(문장 사이의 모순 파악), SRL(의미역 결정)
  > - 모든 layer의 표현을 포함하면 마지막 계층만 사용할 때보다 전체 성능 향상

![table2](/assets/img/nlp/6_ELMo_table2.png){: width="70%" height="70%"}



### Where to include ELMo?
- SRL : BiRNN의 입력 계층에만 word Embedding을 포함할 때 성능 높음
- SNLI , SQuAD : BiRNN의 입력, 출력 계층에 word Embedding 사용할 때 성능 높음
- task에 따라 달라짐

### What information is captured by the biLM’s representations?
- biLM의 문맥 표현
  > - 단어의 의미를 주변 단어를 고려하여 명확하게 구분
  > - 문장에서의 말할 때의 어조와 단어의 뤼앙스를 명확하게 구분
  > - 품사와 단어의 의미를 명확하게 구분

#### Word sense disambiguation
- BiLM의 단어 의미 예측 방법
  > - simple 1-nearest neighbor approach 방법 사용
  > - BiLM으로 모든 단어에 대한 표현을 계산하고 각 의미에 대한 평균 표현을 취하여 대상 단어의 의미를 예측

#### POS tagging
- 기본적인 구문을 파악하기 위해 컨텍스트 표현을 사용하여 선형 분류기에 입력으로 사용
- BiLM의 여러 계층이 서로 다른 유형의 정보를 파악
- pos tagging에서는 첫번째 레이어(하위 계층)에 있는 표현 사용
- 하위 계층: 일반적인 정보 파악
  
#### Implications for supervised tasks
- biLM의 다른 레이어가 서로 다른 유형의 정보를 나타내고 있음
- 첫번째 레이어는 구문(syntactic) 정보 포함
- 두번째 레이어는 의미(semantic) 정보 포함

### Sample efficiency
- 기존 모델에 ELMo를 추가하면 파라미터 업데이트 (epoch) 수 감소
- 더 작은 규모의 훈련 세트를 효율적으로 사용
  
### Visualization of learned weights



## (2) Related work
### 사전 훈련된 단어 벡터
- 기존 연구
  > -  대규모의 레이블되지 않은 텍스트에서 단어의 문법적 및 의미적 정보를 포착하는 능력 우수
  > -  한계: 각 단어 벡터는 하나의 context 단어만 허용
  > - Word representations: A simple and general method for semi-supervised learning(2010)
  > - Distributed representations of words and phrases and their compositionality(2013)
  > - Glove: Global vectors for word representation.(2014)

- 단어 벡터 개선
  > - 서브 워드 정보를 이용 (Wieting et al., 2016; Bojanowski et al., 2017) 
  > - 각 단어의 의미에 대해 별도로 학습(Neelakantan et al., 2014) 
  
- 본 논문
  > -  character convolutions 이용하여 subword의 이점 활용
  > - 통합된 multi-sense(다의어) 정보를 이용해 미리 정의된 sense(의미) class 예측

### 문맥에 따라 다른 vector 학습
- 기존 연구
  > - context2vec (Melamud et al., 2016) : 양방향 LSTM 사용하여 중심단어 주변의 context 인코딩
  > - CoVe( McCann et al., 2017) : 중심 단어 자체를 표현에 포함하여 지도형 신경 기계 번역(MT) 시스템의 인코더 계산
  > - (Peters et al., 2017): 비지도 언어 모델의 인코더 계산
  > - 한계: 큰 데이터셋을 사용한다는 이점이 있지만, MT 접근 방식은 병령 망충치 크기의 제한이 있음.

- 본 논문
  > - 풍부한 단일 언어 데이터에 대한 액세스를 통해 양방향 언어 모델(biLM)을 약 3000만 개의 문장으로 된 말뭉치에서 학습
  > - 깊은 문맥적 표현으로 일반화 -> 다양한 NLP task에서 우수한 성능

### deep biRNN 모델의 각 레이어가 다른 정보 인코딩
- 기존 연구
> - Hashimoto et al., 2017; Søgaard and Goldberg, 2016 : 깊은 LSTM의 첫번째 레이어에서 다중 작업 구문 지도 (예: 품사 태그)를 도입하면 높은 수준의 작업(예: 의존성 구문 분석, CCG 슈퍼 태깅)의 전반적인 성능을 향상
> -  Belinkov et al. (2017) : LSTM 인코더의 첫 번째 레이어에서 학습된 표현이 두 번째 레이어보다 더 나은 품사 태그를 예측
> - Melamud et al., 2016: 단어 문맥을 인코딩하는 LSTM의 최상위 레이어는 단어 의미의 표현을 학습

- 본 연구
  > - task에 따라 ELMo 표현을 수정할 수 있으며, 다양한 유형의 semi-supervisision downstream task에 적용 가능

### task에 따라 ELMo 조정
- 기존 연구
  > - Dai and Le (2015) and Ramachandran et al.(2017) : 언어 모델 및 시퀀스 오토인코더를 사용하여 인코더-디코더 쌍을 사전 훈련한 후 작업 특정 supervision을 사용하여 미세 조정

- 본 연구
  > - 레이블되지 않은 데이터로 biLM을 사전 훈련한 후 가중치를 고정하고  task에 적합한 모델을 추가
  > - task데이터가 적은 경우에도 풍부하고 범용적인 biLM 표현을 얻을 수 있음.


## (3) ElMo : Embeddings from Language Models
- ELMo word representation
  > - 전체 입력 문장의 함수
  > - character convolution 레이어를 갖는 biLM의 두 개의 레이어의 top에서 선형 함수로 계산
  > - biLM을 대규모로 사전 훈련하고, 기존의 NLP 아키텍처에 통합되어 semi-supervised 학습 수행

###  Bidirectional language models
- forward language model
  > - 히스토리 (t_1,..., t_k-1) 가 주어진 각 토큰 t_k의 확률을 모델링하여 토큰 시퀀스 (t_1, t_2,..., t_N) 의 확률을 계산
  > 1. x_k_LM : 문맥에 독립적인 토큰 벡터 생성 ( 토큰 임베딩 결과/ chsacter 기반 CNN 결과)
  > 2. LSTM의 L layer에 통과
  > 3. h_k,j^LM : LSTM 레이어의 output (문맥에 종속적인 벡터 ),  j = 1, . . . , L
  > 4. LSTM의 최상위 레이어의 output(h_k,L^LM)에 softmax 적용하여 다음 토큰(t_k+1) 예측
![formula1](/assets/img/nlp/6_ELMo_formula1.png){: width="70%" height="70%"}

- backward LM
  > - 미래 context 토큰( (t_k+1, t_K+2,..., t_N) 을 이용해 이전 토큰 t의 확률을 모델링
![formula2](/assets/img/nlp/6_ELMo_formula2.png){: width="70%" height="70%"}

- biLM
  > - 순방향 언어 모델 +  역방향 언어 모델
  > - log likelihood 최대화
![formula3](/assets/img/nlp/6_ELMo_formula3.png){: width="70%" height="70%"}

- parameter
  > - 토큰 표현 (x) 과 소프트맥스 계층 (s) 의 파라미터 : 순방향, 역방향 모델간 공유
  > - LSTM에 대해서는 각 방향에서 별도의 파라미터 관리

###  ELMo :  a linear combination of the biLM layers.
- biLM layers with L-layer
> - 각 token t_k에 대해 (2L + 1)개 vector
> - h_k,o^LM : token layer
> - h_k,j^LM : each biLSTM layer
![formula4](/assets/img/nlp/6_ELMo_formula4.png){: width="70%" height="70%"}

-  모든 레이어를 단일 벡터로 축소
> - 1. E(Rk) = h_k,L^LM : 최상위 레이어를 선택
> - 2. task에 따라 모든 biLM 레이어에  특정 가중치를 적용하여 계산
> - task 별 가중치 :  각 계층의 소프트맥스 정규화 가중치 (stask) 를 사용하여 계산
![formula5](/assets/img/nlp/6_ELMo_formula5.png){: width="70%" height="70%"}
> - s^task: softmax-normalized weights -> 여러 레이어의 정규화
> - γ^task: scalar parameter -> downstream 모델에서 ELMo 모델의 중요도 제어-> task별 최적화에 중요한 요소

###  Using biLMs for supervised NLP tasks
- supervised architecture에 ELMo 추가
  > - supervised architecture의 가장 아래 layer에 ELMo 추가
  > - 토큰 시퀀스 (t1, . . . , tN)가 주어지면
  > - 사전 훈련된 단어 임베딩(선택적으로 character 기반 표현)을 사용하여 각 토큰 위치에 대한 문맥 독립적인 토큰 표현 x_k를 형성
  > - 양방향 RNN, CNN 이용하여 문맥에 민감한 h_k 생성
  
- 다양한 방법
> - BiLM의 가중치를 고정하고 eLMO 벡터를 토큰 표현과 결합한 다음 작업 RNN에 전달하여 supervision 모델에 eLMO를 추가
> - task RNN의 출력에도 ELMo를 포함하여 추가 개선 가능
> - biLSTM 뒤에 bi-attention 레이어 추가
> - biLSTM 위에 클러스터링 모델을 추가
> -  ELMo에 적당한 드롭아웃을 추가
> -  ELMo 가중치를 정규화하기 위해 손실에 λ\|\|w\|\|_2^2를 추가


### Pre-trained bidirectional language modelarchitecture
- Jozefowicz et al. (2016)와 Kim et al. (2015)의 architecture에 양방향 학습과 LSTM 레이어간 residual 연결 추가
- 모델
  > - CNN-BIG-LSTM에서 모든 임베딩 및 은닉 차원을 절반으로 감소 -> task의 모델 크기과 계산 복잡도를 위해
  > - 모델은 2개의 biLSTM 레이어를 사용
  > - 첫번째 레이어: 4096 개의 유닛, 두번째 레이어로의 잔차 연결
  > - 두번째 레이어:512 차원의 유닛
  > - 문맥에 덜 민감한 유형은 2048 문자 n-gram 컨볼루션 필터를 사용후, 2 개의 highwaty layer와 선형 투영으로 512개의 표현
  > - 각 입력 토큰에 대해 3개의 표현 레이어 제공

## (4) Evaluation
- 기존 모델에 ELMo를 추가하는 것만으로도 task별 성능 향상
- 단어의 복잡한 특성과 문맥이 달라짐에 따라 사용되는 단어의 의미를 포착
![table1](/assets/img/nlp/6_ELMo_table1.png){: width="100%" height="100%"}

### Question answering
- baseline : Clark and Gardner, 2017 - Seo et al.의 Bidirectional Attention Flow 모델(BiDAF; 2017)의 개선 버전
- 베이스라인 모델에 ELMo를 추가한 후 테스트 세트의 F1은 81.1%에서 85.8%로 4.7% 향상
  
### Textual entailment
-  전제가 주어졌을 때 가설이 참인지 결정
-  베이스라인: ESIM 모델
-  ESIM 모델: 전제와 가설을 인코딩하기 위해 biLSTM을 사용한 뒤, 행렬 어텐션 레이어, 지역 추론 레이어, 다른 biLSTM 추론 구성 레이어, 마지막으로 출력 레이어 앞의 풀링 연산을 사용
-  베이스라인에 ELMo를 추가하면 다섯 개의 무작위 시드에 걸쳐 평균 0.7% 정확도가 향상
  
### Semantic role labeling
- 문장의 술어-인자 구조를 모델링하며, 종종 "누가 무엇을 누구에게 했는가"를 대답하는 것으로 설명
- 베이스 라인 : He et al. (2017)
-  He et al. (2017): SRL을 BIO 태깅 문제로 모델링하고, Zhou and Xu (2015)를 따라 전방과 후방 방향이 교차된 8개의 층으로 구성된 깊은 biLSTM
-  베이스라인에 ELMo를 추가하면 F1 점수가 81.4%에서 84.6%로 3.2% 증가

### Coreference resolution
- 텍스트에서 동일한 기본 실제 세계 개체를 참조하는 멘션들을 클러스터링하는 작업
- 베이스라인 모델:  Lee et al. (2017)의 엔드-투-엔드 스팬 기반 신경 모델
- ELMo를 추가함으로써 평균 F1을 67.2에서 70.4로 3.2% 향상

### Named entity extraction
- 개체명 추출
- 베이스라인 모델: 사전 훈련된 단어 임베딩, 문자 기반 CNN 표현, 두 개의 biLSTM 레이어 및 조건부 랜덤 필드(CRF) 손실(Lafferty et al., 2001)을 사용
- 다섯 번의 실행에 걸쳐 평균 92.22%의 F1을 달성
- 마지막 레이어 대신 모든 레이어를 사용하는 것이 여러 작업에서 성능을 향상
  
### Sentiment analysis
- 감성 분류
- 베이스라인 모델: McCann et al. (2017)의 biattentive classification network (BCN)
- CoVe를 ELMo로 교체하면 최고 성능 대비 1.0%의 절대 정확도 향상

## Reference
- 전이학습 기반 NLP(1): ELMo : https://brunch.co.kr/@learning/12