---
layout: page
title: fastText
description: >
  fastText paper review
hide_description: true
sitemap: false
---


- Title: fastText: Enriching Word Vectors with Subword Information
- paper: [download]
- journal: 
- year : 2016
- Subjects:	Computation and Language (cs.CL),  Machine Learning (cs.LG)
- Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov
- summary
  > - skip gram model을 기반으로 문자 단위의 n-gram 벡터들의 표현을 합치는 방법
  > - 목표 :  
  > - 모델 : 
  > - 결과 :  
  > - 의미 : train set에서 없던 OOV(out of vocabulary) 단어나 희소 단어(rare word)들에 대해서, 오탈자가 있는 단어에 대해서도 학습 가능
- 사용 기법
  > -

[download]: https://arxiv.org/abs/1607.04606



1. this unordered seed list will be replaced by toc as unordered list
{:toc}
<br>

## (0) Abstract
- 연구 배경
> - 레이블이 지정되지 않은 대형 코퍼라에 대해 학습된 연속 단어 표현은 자연어 처리 작업에서 일반적으로 사용되지만 단어 형태를 무시하는 경우가 많음.
  
- 연구 목표 및 방법
> - 각 단어를 문자 n-gram의 집합으로 나타내는 `스킵그램 모델`을 기반으로 한 새로운 접근법 제안
> - 벡터 표현은 각 문자 n-gram과 연관되고, 단어는 이러한 표현들의 합으로 표현 -> 희귀 단어나 어휘량이 많은 언어 표현 가능

- 결과 및 의의
> - 모델의 속도가 빠름
> - 대규모 corpus에 대해 빠른 학습 속도
> - 학습 데이터에 없는 단어 vygus rksmd
> - 9개의 언어에 대해 단어 유사성 및 비유 task에서 좋은 성능

## (7)Conclusion
-  model
> - 서브워드 정보를 통합하여 단어 표현을 학습
> - 문자 n-gram을 스킵그램 모델에 통합하는 접근 방법

- 결과
> - 모델이 간단하므로 학습 시간이 빠르고 전처리나 감독 불필요
> - 서브워드 정보를 고려하지 않는 기준선 및 형태학적 분석을 기반으로 하는 방법보다 성능이 좋음.


## (1) Introduction
### word embedding
- distributional semantics(분포 의미 체계)
> - 일반적으로 레이블이 지정되지 않은 대규모 코퍼스에서 동시 발생 통계를 사용하여 생성
> - 최근에는 feed-forward neural network, simple log-bilinear models(Word2Vec)을 사용

- 최근 연구의 한계
  > -  각 어휘의 단어를 별도의 벡터로 표현하며 매개 변수 공유를 하지 않음.
  > - 단어의 내부 구조를 무시

- 개선 사항
  >  터키어, 핀란드어 등은 형태론적으로 풍부한 정보를 가지고 있음.
  > 단어 형성이 규칙적이므로, 문자 수준이 정보를 통합하면 벡터표현 개선 가능

- 연구 방법
  >  - 문자 n-gram에 대한 표현을 학습하고, 단어를 n-gram 벡터의 합으로 나타내기
  > - 서브워드 정보를 고려하는 연속 스킵그램 모델(Mikolov 등, 2013b) 확장

## (5) Results
### Human similarity judgement
- 훈련 데이터에 없는 단어(Out-Of-Vocatulary, OOV)에 따른 모델
> - OOV를 null vector로 표현한 경우 -> sisg-모델
> - OOV를 n-gram 벡터를 합하여 계산한 경우 -> sisg모델
> -  sisg 모델이 sisg-모델보다 성능이 좋음 -> character 기반 n-gram형태의 서브워드 정보를 사용 결과

- character 기반 n-gram을 사용했을 때 효과
  > - 아랍어, 독일어, 러시아어 > 영어, 프랑스어
  > - 격이 많은 문법적 변화를보이거나 복합 단어가 많은 경우 성능이 더 좋음.

- 일반적인 단어 : 서브워드 정보를 사용하지 않고도 좋은 벡터 추출 가능
- 희귀 단어: word 간의  character 수준의 유사성 사용하는 것이 좋음.

### Word analogy tasks
- 형태적 정보
> - 구문 task에 성능 개선
> - 의미론적 질문에는 도움 안됨
> n-gram의 길이에 따라 성능이 달라짐-> 5.5 섹션 확인 
  
### Comparison with morphological representations
- 형태론적 표현에 대한 다양한 모델과 비교
  > - 재귀신경망, 
  > - 형태소 cbow
  > -형태 변환
  > - log-bilinear language model 

- 성능 비교
  > - 형태론 분절기에서 얻은 서브워드 정보를 기반으로 한 기술에 비해 잘 수행
  > - 두사와 접미사 분석을 기반으로 한 방법이 fastText보다 좋음

- OOV 단어: character n-gram 표현 합계

### Effect of the size of the training data
- sisg (OOV에 대해 n-gram 표현 합산한 경우) 성능 우수
- cbow는 데이터가 많아지면 성능이 증가하지만, sisg는 데이터셋이 적어도 빠르게 학습 가능
- 훈련 데이터셋이 적어도 좋은 단어 벡터 추출
- OOV에 대해서도 성능이 좋음.

  
### Effect of the size of n-grams
- 적절한 n-gram 개수: 3~6개
> - n-gram의 character 개수가 성능에 영향을 미침
> - 만족스러운 성능 but task과 언어에 따라 다르며 적절하게 조정 필요
>  문자 n-gram을 계산하기 전에 단어의 시작과 끝을 나타내는 특수 위치 문자를 앞뒤에 추가해야 함.

### Language modeling
- model
> - 650개의 LSTM 유닛을 가진 순환 신경망
> - dropout 으로 정규화( probability : 0.5)
> - weight decay(regularization parameter of 10^−5)
> - Adagrad algorithm ( learning rate : 0.1, clipping the gradients if norm larger than 1.0.)
> - 가중치 초기화 범위:  [−0.05, 0.05]
> - batch size: 20
>
- result
  > - 서브워드 정보를 사용하여 훈련된 단어 표현:  일반적인 skipgram 모델보다 더 나은 성능
  > - 언어 모델링 작업에서 서브워드 정보의 중요성을 보임

## (2) Related work

## (3) Model
### General model
### shbword model

## (4) Experimental setup
### Baseline
### Optimization
### Implementation details
### Datasets

## (6) Qualitative analysis
### Nearest neighbors
### Character n-grams and morphemes
### Word similarity for OOV words







![figure](/assets/img/nlp/4_GloVE_figure2.png){: width="100%" height="100%"}


## Reference
- [논문 리뷰] GloVe: Global Vectors for Word Representation: https://imhappynunu.tistory.com/14