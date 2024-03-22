---
layout: page
title: Language Model
description: >
  How you install Hydejack depends on whether you start a new site,
  or change the theme of an existing site.
hide_description: true
sitemap: false
---



1. this unordered seed list will be replaced by toc as unordered list
{:toc}

## 단어 표현(word Representation) 종류
### Local Representation (국소 표현) = Discrete Representation (이산 표현)
  - 해당 단어만 보고, 특정값을 매핑하여 단어 표현하는 방법
#### 국소 표현 방법
- One-hot Vector
- N-gram
- Count Based : `Bag og Words`, DTM

### Continuous Representation( 연속 표현) = `Distributed Reprsentation(분산 표현)`
#### 이론적 토대: `분포 가설(Distributional hypothesis)`
#### 분산 표현 방법
##### Count Based(통계 기반) 
- 동시발생행렬 (TF-IDF, PMI, PPMI, SVD)
- Full Document(LSA)
- Windows(Glove): 통계 기반 + 추론 기반

##### Prediction Based(추론 기반) 
- Word2Vec
- FastText

## Bow(Bag of Words) 
단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하여 표현하는 방법이다.<br>
각 단어가 등장한 횟수를 수치화하는 방법으로, 주로 어떤 단어가 얼마나 등장했는지를 기준으로 문서의 성격을 판단할 때 사용한다.

### DTX(Document-Term Matrix, 문서 단어 행렬)
다수의 문장에 등장하는 각 단어들의 빈도를 행렬로 표현한 것을 말한다.<br>
간단하고 구현하기 쉽지만, 대부분의 값이 0인 희소 행렬(sparse matrix)이므로 많은 저장공간과 높은 계싼 복잡도를 요구한다.<br>
또한 단순 빈도 수 기반이므로, 불용어(stopwords)와 중요한 단어에 대한 구분이 어렵다.<br>


#### TF-IDF(Term Frequency-Inverse Document Frequency, 단어 빈도-역 문서 빈도)
- TF(Term Frequency, 단어 빈도) : 특정 문서(d) 내에서 특정 단어(t)의 빈도 수를 나타내는 값
  > TF(dt) = count(t,d) 
  > TF값이 높으면 특정 문서에서 자주 사용되는 단어-> 전문 용어나 관용어로 간주
  > TF값이 낮으면 특정 문서에서 적게 사용되는 단어 -> 

- DF(Document Frequency, 문서 빈도) : 한 단어(t)가 얼마나 많은 문서(D)에 나타나는지 계산

  > DF(t,D) = count(t∈d: d∈D)<br>
  > DF값이 높으면 많은 문서에 등장 -> 널리 사용되어 중요도 낮은 단어<br>
  > DF값이 낮으면 적은 수의 문서에 등장 -> 특정 문맥에서만 사용되는 중요도 높은 단어<br>

- IDF(Inverse Doument Frequency, 역문서 빈도) : 특정문서 내에서 특정 단어(t)의 중요도 표현

  > IDF(t, D) = log( count(D) / 1 + DF(t, D))
  > DF가 높으면 해당 단어(t)가 일반적이 중요하지 않다는 의미 -> 역수를 취해 단어 빈도가 적을 수록 IDF가 커지도록 보정
  > IDF값이 높으면 특정 문서에만 자주 등장하는 단어 -> 중요도 높음.
  > IDF값이 낮으면 모든 문서에 자주 등장하는 단어-> 중요도 낮음

- TF-IDF
  > TF-IDF(t, d, D) = TF(t, d) * IDF(t, d)
  > TF-IDF값이 크면 : 문서 내에 단어가 자주 등장하지만, 전체 문서 내에는 해당 단어가 적게 등장 ->중요도 높음.
  > TF-IDF 값이 작으면 전체 문서 내 해당 단어가 자주 등장 (예: 관사, 관용어) -> 중요도 낮음


\<Reference\> <br>
- 파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터비전 심층학습(위키북스, 윤대희, 김동화, 송종민, 진현두 지음, 2023) : 06 임베딩<br>
- 밑바닥부터 시작하는 딥러닝2(한빛미디어, 사이토 고키 지음, 2019) : chapter 2 자연어와 단어의 분산 표현
- 딥 러닝을 이용한 자연어 처리 입문(위키독스, 유원준, 안상준 지음, 2024) 

<br>
<br>
<br>
Continue with [](2-embedding.md){:.heading.flip-title}
{:.read-more}


[upgrade]: upgrade.md
