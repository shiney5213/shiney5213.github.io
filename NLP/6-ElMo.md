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
  > - BiLM 모델의 레이어마다 다른 정보를 포함 -> 가중치를 두어 embedding을 만들어 task에 따라 사용 
- 사용 기법
  > - 전이학습 -> 기 학습된 언어 모델의 지식이 임베팅으로 전이
  > - LSTM의 최상위 레이어를 사용하지 않고, 모든 레이어의 값을 연산하여 사용
  > - biLM : 양방향 순환 신경망
  
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

![table2](/assets/img/nlp/6_ELMo_table2.png){: width="100%" height="100%"}



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
- > - simple 1-nearest neighbor approach 방법 사용
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
- embedding 만드는 방법
> - 


## (2) Related work


## (3) ElMo : Embeddings from Language Models

## (4) Evaluation



## Reference
- 전이학습 기반 NLP(1): ELMo : https://brunch.co.kr/@learning/12