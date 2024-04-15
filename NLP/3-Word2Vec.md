---
layout: page
title: Word2Voc : 
description: >
  Word2Vec paper review
hide_description: true
sitemap: false
---


- Title: Efficient Estimation of Word Representations in Vector Space
- Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
- paper: [download]
- journal: 
- year : 2013
- Subjects:	Computation and Language (cs.CL)
- summary
  > - 단어의 vector Representation 품질에 대한 연구<br>
  > - 목표 : 구문 및 의미론적 언어 작업을 위한 단어 벡터를 생성할 때 다양한 모델 아키텍처의 성능을 비교<br>
  > - 모델 : CBoW와 Skip-Gram 모델을 통해 단순한 모델 사용
  > - 결과 :  단순한 아키텍처가 널리 사용되는 신경망 모델에 비해 고품질의 단어 벡터를 생성할 수 있음을 밝힘
  > - 의미 :  벡터에 단어의 의미정보 저장-> 단어간의 관계 파악 + 벡터 공간상 유사한 단어 군집화 가능-> 의미 정보 표현
- 사용 기법
  > - DistBelief 분산 프레임워크
  > - 계층적 소프트맥스 : 모델의 출력을 계산하기 위해 시간 많이 소요 -> 이진 트리를 이용해 출력할 단어 수 감소
  > - 네거티브 샘플링(Negative Sampling) : skip gram 모델에서 data pair를 만들 때 중심단어를 통해 만든 데이터는 1(정답), 네거티브 샘플링을 통해 만든 데이터는 0(오답)으로 라벨링하여 다중 분류 문제를 이진 분류로 바꾸어 학습


[download]: https://arxiv.org/abs/1301.3781



1. this unordered seed list will be replaced by toc as unordered list
{:toc}
<br>
## (0) Abstract
- 대규모 데이터 세트에서 고품질의 word vector를 생성하기 위한 두가지 모델(CBoW,  Skip-Gram) 제안
- 품질 측정 및 비교
   > 단어 유사성 task에서 이전 기술에 비해 낮은 컴퓨팅 cost로 정확도 크게 향상
- 학습의 효율성
    > 고품질 word vector를 빠르게 학습<br>
  > 16억 개의 단어를 포함하는 대규모 데이터 세트에서 하루 이내에 고품질 워드 벡터 학습
  
- 최첨단 성능
  > 단어의 의미와 구문 측면에서 단어 간의 관계와 유사성을 정확하게 포착<br>
  > 언어의 미묘한 차이와 복잡성을 포착하는 단어 표현을 생성하는 데 효과적


## (6)Conclusion
- 연구 소개
  > 단어의 vector Representation 품질에 대한 연구<br>
  > 목표 : 구문 및 의미론적 언어 작업을 위한 단어 벡터를 생성할 때 다양한 모델 아키텍처의 성능을 비교<br>
  > 결과 : CBoW와 Skip-Gram 모델을 통해 단순한 모델 아키텍처가 널리 사용되는 신경망 모델에 비해 고품질의 단어 벡터를 생성할 수 있음을 밝힘
- 신경망 모델과의 비교
   > 단순한 모델을 사용하면 <br>
   > 신경망 모델에 비해 계산 복잡도가 낮기 때문에 학습에 필요한 계산 리소스와 시간이 감소-> 대용량의 데이터셋에서 고품질의 word vector 생성
- DistBelief 분산 프레임워크
   > 대규모 말뭉치에서 CBoW와 Skip-Gram 모델 학습 가능<br>
   > 이전 모델에 비해 성능 크게 향상
- 활용
  > SemEval-2012 Task 2, 감성 분석, 의역어 감지, 기계 번역 등<br>
  > 지식 기반 확장 및 사실 검증에 응용


## (4) Results
### Task Description
- test set
   > 단어 벡터의 품질을 측정하기 위해  <br>
   > 5가지 유형의 의미론적 질문과 9가지 유형의 구문론적 질문으로 구성된 포괄적인 테스트 세트를 정의<br>
   > 8869개의 의미적 질문과 10675개의 구문적 질문으로 구성<br>
   > 질문 만드는 과정<br>
        >  유사한 단어 쌍의 목록을 수동으로 만든 다음 두 단어 쌍을 연결하여 큰 질문 목록을 형성<br>
        >  예:  68개의 큰 미국 도시와 그들이 속한 주의 목록을 만들고 무작위로 두 단어 쌍을 선택하여 약 2.5천 개의 질문 생성<br>
        >  단일 토큰 단어만 포함(뉴욕과 같은 복합어는 포함X)<br>
-  평가 항목: 단어 간의 의미적 관계와 구문적 유사성을 평가<br>
    > 각 문제 유형에 대하 전반적인 정확도 평가<br>
    > 계산된 벡터에 가장 가까운 단어가 문제의 정답과 정확히 일치하는 경우에만 정답으로 간주<br>
    > 단어 형태에 대한 정보가 없으므로 완벽한 정확도를 달성하기 어려움. 이러한 정확도는 단어 벡터의 유용성과 긍정적으로 관련될 것으로 예상<br>

### Maximization of Accuracy
- data
  > 6B token을 사용한 Google News corpus에서 가장 많이 사용하는 단어 100만개 사용
  > train data 중 가장 빈번하게 사용되는 단어 30k개로 훈련
- data 양,  차원 과 정확도 관계
  > 데이터 양을 추가하거나 차원이 높아지면 정확도 향상
  > 그러나 어느 시점 이후에는 차원과 데이터양을 함께 늘려야 정확도 향상됨<br>
  > 훈련 데이터의 양을 두 배로 늘리면 벡터 크기를 두 배로 늘리는 것과 거의 동일한 계산 복잡도가 증가

![table2](/assets/img/nlp/word2vec_table2.png){: width="100%" height="100%"}

- train
  > epoch : 3
  > learning rate: 0.025 -> 0 (선형적 감소)

### Comparison of Model Architectures
- 조건
  > 동일한 학습 데이터와  640차원의 동일한 차원 사용<br>
  >학습 데이터 :  여러 LDC 말뭉치로 구성 ((3억 2천만 단어, 82K 어휘)
- train process
  > DistBelief parallel training 사용<br>
  > 8개 이전 단어 사용<br>
  > projection layer: 640 * 8<br>
- 성능 
  > RNN : 구문 질문에서 좋은 성능<br>
  > NNLM : RNNLM의 word vector가 비선형 hidden layer에 직접 연결 -> RNN보다 더 좋은 성능<br>
  > CBOW : 구문 task에서는 NNLM보다 성능 좋고, 의미 task에서는 비슷<br>
  > Skip-Gram :  구문 task에서는 NNLM보다는 성능이 좋고, CBOW 보다 약간 성능이 떨어짐<br>
  >              의미론적 task에서는 CBOW보다 우수함.<br>
  
### Large Scale Parallel Traing of Models
- DistBelief 
  > 분산 프레임워크 사용
- training time
  > CBOW와 Skip-gram이 NNLM보다 학습 시간이빠름
  > 하나의 CPU만 사용<br>
  > CBOW: Google News data의 subjet에 대해 약 하루만에 학습<br>
  > Skip-gram : Google News data의 subjet에 대해  약 3일 소요<br>
  > 추가 실험: 1 epoch로 학습<br>
  > 1 epoch, 2배 data 의 성능 = 3 epoch, 1배 data의 성능<br>

![table6](/assets/img/nlp/word2vec_table6.png){: width="100%" height="100%"}


### Microsoft Research Sentence Completion Challenge
- Skip-gram 성능
   > LSA 유사성보다 성능이 좋지는 않지만  RNNLM 상호 보완적인 점수를 제공하여 58.9%  정확도  

## (1) Introduction
- 기존의 연구의 한계
    > 현재(논문 발표 당시 2013년) NLP시스템에서 단어를 atomic unit으로 다루었고, 개별 단어를 표현할 뿐 단어 간의 유사성이나 관계를 고려하지 않음 <br>
    > N-gram모델 : 대표적으로 통계 언어 모델링에 사용되는 모델로, 단순성, 견고성, 그리고 방대한 데이터를 대상으로 학습한 단순한 모델이 작은 데이터로 학습한 복잡한 모델보다 성능이 우수하다는 장점이 있다.<br>
    > 그러나 특정 영역의 데이터 양이 제한되어 있는 경우, 성능은 고품질의 데이터셋의 크기에 의존함. 단순한 scale up만으로 큰 진전으로 이어지지 않으므로, 보다 발전된 기술이 필요<br>
- 단어의 분산 표현의 필요성
   >  큰 데이터 세트에서 복잡한 모델 훈련 가능 
   > 단어의 분산 표현을 사용하는 신경망 기반 언어 모델이 N 그램 모델보다 좋은 성능을 보이고 있으므로, 이러한 고급 기술에 집중해야 함.<br>

###  Goals of the Paper
- word vector representation techniques
  > 방대한 데이터세트에서 고품질의 단어 벡터를 표현하는 기술 소개<br>
  > 단어 오프셋 기술로 유사한 단어 찾기 가능
- word vector 품질 평가
  >  테스트 set 설계를 통해 구문 및 의미 규칙성 측정<br>
  > 단어 간의 유사성뿐만 아니라, 단어가 갖는 다양한 유사성을 평가<br>
  > 단어의 선형회귀를 보존하는 새로운 아키텍쳐를 개발->벡터연산의 정확성을 최대화
- training time   
  > 훈련 시간과 정확도가 단어 벡터의 차원과 훈련 데이터의 양에 어떻게 의존하는지 <br>

### Previous Work
- neural network language model (NNLM)
  > 단어 벡터 학습에 관한 다양한 NNLM 모델이 사용<br>
  > 그러나 대부분의 이전 연구는 훈련 과정에서 많은 컴퓨팅 자원 소모
- A neural probabilistic language model 
  > - 선형 투사 계층과 비선형 은닉 계층을 가진 피드포워드 신경망이 사용되어 단어 벡터 표현과 통계적 언어 모델을 동시에 학습
- Language Modeling for Speech Recognition in Czech / y. Neural network based language models for higly inflective languages
  > - 단어 벡터를 먼저 단일 은닉 계층을 사용하여 신경망으로 학습한 후, 학습된 단어 벡터를 사용하여 NNLM을 훈련
-  Language Modeling for Speech Recognition in Czech
  > - 서로 다른 모델 아키텍처를 사용하여 다양한 말뭉치에서 훈련하여 단어 벡터 추정
  > -  훈련에 대해 상당히 더 많은 계산 비용 
  
## (2) Model Archetectures
- 다양한 word vector 표현 모델, 
   > 잠재 의미 분석 (LSA) : 대규모 dataset에서 비쌈<br>
   > 잠재 디리클레 할당 (LDA)
- 신경망에 의해 학습된 단어의 분산 표현에 집중
  > 단어간의 선형 규칙성 보존에 LSA보다 성능이 좋음.
- 모델의 계산 복잡도
  > O = E × T × Q <br>
  > E:  훈련 epoch 수, T: 훈련 세트의 단어 수, Q:  각 모델 아키텍처의 특정 파라미터<br>
  > 일반적으로 E = 3에서 50, T는 최대 10억까지 선택

### Feedforward Neural Net Language Model(NNLM)
- A neural probabilistic language model(Y. Bengio, 2003)
   > 입력 계층, 투영 계층, 은닉 계층, 출력 계층으로 구성<br>
   > 투영 계층과 은닉 계층 사이의 계산이 복잡<br>
  > 계산복잡도(Q) = N × D + N × D × H + H × V<br>
  > (N: 입력 단어 개수, D : 투영 계층 unit 수, H: 은닉 계층 크기, V: 출력 계층 크기)<br>
  > H × V : 지배적인 항 -> 계산 복잡도 커짐
-  solution : 계층적 소프트맥스([Hierarchical Softmax, HS]) 사용
  > 어휘를 Huffman 이진 트리로 표현 <br>
  > Huffman 트리: 빈번한 단어에 짧은 이진 코드 할당-> 평가해야 하는 출력 단위의 수 감소<br>
  > 효과 : 어휘 크기가 백만 단어인 경우, 평가 속도가 약 두 배 빨라짐


[Hierarchical Softmax, HS]: https://uponthesky.tistory.com/15

### Recurrent Newral Net Language Model(RNNLM) : 순환 신경망 (RNN) 기반 언어 모델
- NNLM의 한계
  > 컨텍스트 길이를 지정<br>
  > RNNLM : 컨텍스트 길이를 지정하지 않아도 더 복잡한 패턴을 효율적으로 표현<br>
- RNNLM
  > 입력 계층, 은닉 계층, 출력 계층으로 구성<br>
  > 순환 행렬 : 은닉 계층 상태가 현재 입력과 이전 시간 단계의 은닉 계층 상태를 기반으로 업데이트 ->  모델이 일종의 단기 기억을 가질 수 있음<br>
  > Q = H × H + H × V <br>
  > 대부분의 복잡도는  H × H와 관련<br>
- 계층적 소프트맥스
   > H × V  -> H × log2 (V) : 계산복잡도 감소

### Parallel Training of Neuralnetworks
- DistBelief : 대규모 분산 프레임워크 -> 대규모 데이터 세트에 대한 효율적인 학습
  > 동일한 모델의 여러 복제본(일반적으로 100개이상)을 병렬로 실행-> 모델의 여러 인스턴스 동시에 학습 가능<br>
  > 각 복제본은 중앙 집중식 서버를 통해 그레이디언트 업데이트를 동기화하며 모든 매개변수 유지-> 모든 복제 모델이 최신 정보로 업데이트 됨<br>
  > 각 복제본은 데이터 센터의 여러 머신에 있는 다중 CPU 코어를 사용<br>
- Adagrad -> 더 빠른 수렴과 최적화에 도움<br>
   > 적응형 학습률 절차를 사용하는 미니 배치 비동기적 그레이디언트 하강법을 사용<br>
   > 적응형 학습률 절차 : 과거 기울기를 기반으로 각 파라미터의 학습률을 조정하는 적응형 학습률 알고리즘<br>
   > 훈련 데이터를 작은 배치로 나누고 각 복제본은 다른 복제본의 업데이트가 완료될 때까지 기다리지 않고 그래디언트를 비동기적으로 업데이트


## (3) New Log-Linear Nodels
- Architecture 목표 :  모델을 단순화하여 계산 복잡도 최소화
  > 신경망 : 비선형 관계를 처리할 수 있다는 장점이 있지만, 은닉 계층이 계산 복잡도를 높임<br>
  > -> 데규모 데이터 셋에서 효율적으로 훈련할 수 있는 간단한 모델 생성
- training 단계
  > 1. 간단한 모델을 사용하여 연속형 단어 벡터를 학습<br>
  > 2. 분산된 단어 표현을 기반으로 N-gram 신경망 언어 모델 (NNLM) 을 훈련

### Continuous Bag-of-Words Model(CBoW)
- 피드포워드 신경망 언어 모델 (NNLM)과 차이
   > 비선형 은닉 계층이 제거되고 모든 단어에 대해 투영 계층 공유<br>
   > 모든 단어를 같은 위치에 투영하고 해당 벡터의 평균 구하기
- CBoW
  >  문맥에 따라 과거와 미래의 단어를 모두 고려하여 중심 단어 예측<br>
  > 단어 순서는 고려하지 않음 -> `Bag-of-Words`<br>
  > input:  네 개의 미래 단어와 네 개의 과거 단어<br>
  > model: 로그-선형 분류기 구축<br>
  > 목표: 현재(가운데) 단어를 올바르게 분류
- 훈련 복잡도
   > Q = N × D + D × log2(V)<br>
   > (N은 단어 수, D는 단어 벡터의 차원, V는 어휘 크기)<br>
   > CBOW : 맥락의 연속 분산 표현을 사용
  - []

### Continuous Skip-gram Model
- Skip-gram
   > 현재 단어에 기반하여 컨텍스트 예측<br>
   > input: 현재 단어<br>
   > model: 연속 투사 계층을 가진 로그 선형 분류기<br>
   > 목표: 현재 단어의 앞과 뒤의 특정 범위 내에 있는 단어를 예측<br>
   > 학습 : 각 훈련 단어에 대해 범위 < 1; C >에서 임의의 숫자 R을 선택한 다음, 현재 단어의 과거와 미래의 R개의 단어를 정확한 레이블로 사용<br>
   > 출력: R+ R 단어 -> R × 2 단어 분류가 수행
- 현재 단어로부터 멀리 있는 단어
  > 먼 단어 사용시 target 단어 백터의 품질은 향상되지만 계산 복잡도 증가
  > 현재 단어와 관련이 적음<br>
  > 멀리 있는 단어로부터 표본을 추출하는 횟수를 줄임으로써 해당 단어에 가중치를 덜 부여
- Q = C × (D + D × log2(V))
   > D는 단어 벡터의 차원이고 V는 어휘의 크기<br>
   >  C: 예측 범위를 결정, 단어들 사이의  최대 거리<br>
   > C가 클수록 단어 벡터의 품질이 향상되지만, 계산 복잡성도 증가<br>
   > 논문에서는 C= 10으로 설정

![Figure1](/assets/img/nlp/word2vec_CBoW_Skipgram.png){: width="100%" height="100%"}


## (5) Examples of the Learned Relationships
- 단어의 다양한 관계 파악
  > 예: Paris - France + Italy = Rome
  > 관계의 정확도: 약 60% 성능
- 정확도 향상 방법 제안
   > 더 큰 데이터셋과 더 큰 차원을 가진 단어 벡터로 훈련
   > 관찰된 여러 관계의 예를 제공

## (7) Follow-Up Work
- CBow와 Skip-gram 모델을 사용하여 단어 벡터를 계산하는 단일 기계 다중 스레드 C++ 코드를 발표
- 1000억 개 이상의 단어로 훈련된 명명된 엔터티를 나타내는 140만 개 이상의 벡터를 게시