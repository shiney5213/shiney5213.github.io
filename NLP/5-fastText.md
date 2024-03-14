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
  > - 목표 : 형택학정 정보를 이용해  단어 임베딩 학습
  > - 모델 : character 기반 n-gram을 스킵그램 모델에 통합하는 접근 방법
  > - 결과 :  train set에서 없던 OOV(out of vocabulary) 단어나 희소 단어(rare word)들에 대해서, 오탈자가 있는 단어에 대해서도 학습 가능
  > - 의미 : 하위 단어를 공유하는 단어끼리는 같은 정보를 공유 -> 비슷한 단어끼리는 비슷한 임베딩 갖게 됨 ->  단어 간 유사도 높임
  > - 한국어와 같은 언어는 형태적 주조를 갖고 있기때문에 하위단위로 나누어 임베딩 학습하여 OOV 문제 해결 가능
- 사용 기법
  > - character 기반의 n-gram 방법
  > - embedding 방법 : 각 하위 단어의 벡터를 구하고 이를 합산하여 단어의 임베딩으로 사용
  > - `sisg` : train data에 없는 단어(OOV)에 대해  n-gram 벡터를 합하여 계산
  > - 단어의 시작, 끝에 특수기호 `<`, `>` 사용 -> 단어의 하위 문자열 고려하는데 중요한 역할

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


## (3) Model
- model
> - 형태학적 정보를 고려한 단어 표현 학습
> - 서브워드 단위를 고려하여 단어를 character n-gram의 합으로 표현
> - General model(단어 벡터를 학습하기 위함) -> subword model(character n-gram 사용)

### General model
#### skip-gram model(Mikolov 등(2013b))
(1) 로그 우도 최대화 함수
  > - W: 주어진 단어 어휘의 크기
  > - w : 개별 단어 , w ∈ {1, ..., W}
  > - 목표: 각 단어 w에 대한 벡터 표현 학습 + 중심 단어가 주어졌을 때 주변 단어를 잘 예측
  > - 단어 w1, ..., wT의 시퀀스로 표현된 대형 훈련 말뭉치가 주어졌을 때, 다음 단어의 로그 우도를 최대화하는 것
![formula1](/assets/img/nlp/5_fastText_formula1.png){: width="70%" height="70%"}

(2) 단어 w_t가 주어졌을 때 주변단어 w_c가 나타날 확률
> - C_t: 단어 W_t 주변의 단어들의 집합
> - w_t : 중심 단어
> - w_c : 주변 단어
> - 단어 w_t가 주어졌을 때 컨텍스트 단어 w_c를 관찰할 확률:
> - 함수 s : (w_t, w_c) 쌍을 점수로 매핑하는 함수
![formula2](/assets/img/nlp/5_fastText_formula2.png){: width="70%" height="70%"}

- skip gram model의 한계
  > - 단어 w_t에 대해 오직 하나의 주변단어 w_c를 예측
  > - 단어 확률 정의 -> softmax 함수 사용

(3) negative sampling : w_c 예측 문제 -> 독립적인 이진 분류 task
- 목표 : 컨텍스트 단어의 존재 (또는 부재)를 독립적으로 예측하는 것.
- 위치 t에 대한 모든 context 단어를 1(양성)으로 가정하고, 무작위로 음성(0)예제 샘플링
- 선택된 context 위치 c에 대해  binary logistic loss 사용
![formula3](/assets/img/nlp/5_fastText_formula3.png){: width="70%" height="70%"}
- negative log-likelihood
![formula4](/assets/img/nlp/5_fastText_formula4.png){: width="70%" height="70%"}

- s (scoring)함수 : 단어 벡터 사용
> - 단어 w에 대한 벡터: u_w, v_w라면
> - 각각 단어 w_t와 w_c에 해당하는 벡터 u_wt와 v_wc 정의
> - s함수: 단어와 컨텍스트 벡터 사이의 스칼라 곱으로 계산 -> s(w_t, w_c) = u_wt^T * v_wc


### shbword model
- skip gram 모델 한계
  > - 단어의 내부 구조 무시
  > - 다른 scoring 함수 제안

- 논문의 제안
  > - 단어의 시작과 끝에 special boundary symbols `<` and `>` 추가 -> 접두사와 접미사를 다른 문자 시퀀스와 구분
  > - 각 단어의 n-그램 집합에 해당 단어 w 자체도 포함하여 단어 표현 학습

- 단어 where에 character 기반 3-gram을 취했을 때
![example1](/assets/img/nlp/5_fastText_example1.png){: width="70%" height="70%"}

- scroing function
> -  크기가 G인 n-그램 vocavulary에서
> -  단어 w가 주어지면 w에 나타나는 n-그램의 집합을 Gw ⊂ {1, . . . , G}로 표시
> - 각 n-그램 g에 벡터 표현 z_g를 연결하면
![formula5](/assets/img/nlp/5_fastText_formula5.png){: width="70%" height="70%"}



## (2) Related work
### Morephological word representations
#### 형태학전 정보를 단어 표현에 사용하기 위한 다양한 방법
- Alexandrescu와 Kirchhoff(2006): 드문 단어를 더 잘 모델링하기 위해, 단어를 특징들의 집합으로 표현하는 팩토라이즈드 신경 언어 모델 제안 -> 형태학적 정보 사용
- Sak 등(2010):  형태론적으로 다양한 언어에 성공적 적용
- Lazaridou 등, 2013; Luong 등, 2013; Botha와 Blunsom, 2014; Qiu 등, 2014: 형태소에서 단어 표현을 유도하기 위한 다양한 합성 함수를 제안-> 형태학적 분해 사용
- Chen 등(2015): 은 중국어 단어와 문자의 임베딩을 동시에 학습-> 형태학적으로 유사한 단어가 유사한 표현을 가지도록 제약을 가하기
- Soricut과 Och(2015):  형태학적 변형의 벡터 표현을 학습하는 방법을 설명하여, 이러한 규칙을 적용하여 보지 않은 단어의 표현을 얻을 수 있음
- Cotterell과 Schütze(2015): 형태론적으로 주석이 달린 데이터를 기반으로 한 단어 표현
- Schütze(1993):  특이값 분해를 통해 문자 4-그램의 표현을 학습-> 4-그램의 표현을 합하여 단어의 표현을 유도 -> 본 논문과 가장 유사
- Wieting 등(2016): 단어를 문자 n-그램 카운트 벡터를 사용하여 표현하는 방법 제안 

### Character level festuress for NLP.
#### 단어를 세분화하지 않고 character에서 직접 언어 표현을 학습
- recurrent neural networks
> - 언어 모델링 (Mikolov 등, 2012; Sutskever 등, 2011; Graves, 2013; Bojanowski 등, 2015)
> - 텍스트 정규화 (Chrupała, 2014)
> - 품사 태깅 (Ling 등, 2015)
> - 파싱 (Ballesteros 등, 2015)

- character에 대해 학습한 합성곱 신경망
> -  품사 태깅 (dos Santos와 Zadrozny, 2014)
> - 감성 분석 (dos Santos와 Gatti, 2014)
> - 텍스트 분류 (Zhang 등, 2015) 
> -  언어 모델링 (Kim 등, 2016)

- 단어를 문자 n-그램의 집합으로 인코딩하는 제한된 볼츠만 머신을 기반으로 한 언어 모델
> -  Sperr 등(2013)

- 최근의 기계 번역 작업
> -  드문 단어의 표현을 얻기 위해 서브워드 단위를 사용 (Sennrich 등, 2016; Luong과 Manning, 2016).





![figure](/assets/img/nlp/4_GloVE_figure2.png){: width="100%" height="100%"}


## Reference
- [논문 리뷰] GloVe: Global Vectors for Word Representation: https://imhappynunu.tistory.com/14