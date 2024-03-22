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


## 임베딩을 만드는 세 철학

|구분|Bag of Words 가정|언어 모델|분포 가설|
|:---:|:---:|:---:|:---:|
|기준|단어 빈도|단어 순서|같이 사용한 단어|
|대표 통계량|Term-Document Matrix, TF-IDF| - |Word-Context Matrix, PMI, PPMI|
|통계기반 모델| LDA|N-Gram  |[단어 수준] Glove(추론 함께 사용), Swivel|
|추론 기반 모델(NN)| Deep Averaging Network | [문장 수준] ELMo, GPT, BERT | [단어 수준] Word2vec, FastText



### 언어 모델(Language Model)
입력된 문장으로 각 문장을 생성할 수 있는 확률을 계산하는 모델이다.<br>
이를 위해 주어진 문장을 바탕으로 문맥을 이해하고, 문장 구성에 대한 예측을 수행한다.

#### 자기회귀 언어 모델(Autoregressive Language Model)
입력된 문장(단어들의 시퀀스)의 조건부 확률의 연쇄법칙([Chain rule for conditional probability])을 통해 다음에 올 단어를 예측한다.
이를 위해 이전에 등장한 모든 토큰의 정보를 고려하며, 문장의 문맥 정보를 파악하여 다음 단어를 생성한다. 

[Chain rule for conditional probability]: https://en.wikipedia.org/wiki/Chain_rule_%28probability%29

#### 통계적 언어 모델(Statistical Language Model, SLM)
언어의 통계적 구조를 이용해 문장이나 단어의 시퀀스를 생성하거나 분석한다.<br>
마르코프 체인(Markoc Chain)은 `빈도` 기반의 조건부 확률 중 하나로, 주어진 데이터에서 각 변수가 발생할 빈도수를 기반으로 확률을 계산한다.<br>  
이 방법은 단어의 순서와 빈도에만 기초해 문장의 확률을 예측하므로 문맥을 제대로 파악하지 못하면 불완전하거나 부적절한 결과를 생성할 수 있으며<br>
한번도 관측한 적이 없는 단어나 문장에 대해서는 예측하지 못하는 `데이터 희소성(Data sparsity)`문제가 발생할 수 있다.<br>  
하지만 기존에 학습한 텍스트 데이터에서 패턴을 찾아 확률 분포를 생성하므로,<br>  
이를 이용해 새로운 문장을 생성할 수 있으며 다양한 텍스트 데이터를 학습할 수 있다.<br>  


최근 연구되는 자연어 처리 기법은 언어 모델을 이용해 가중치를 사전 학습한다.<br>
예를 들어 GPT(Generative Pre-trainedTransformer)나 BERT(Bidirectional Encoder Representations from Transformers)는 문장 생성 기법을 이용해 모델을 사전 학습한다.
{:.note}
<br>

##### N-gram Language Model
이전에 등장한 모든 단어를 고려하는 것이 아닌, 일부 단어(n개)만 고려하여 다음 단어를 예측하는 모델이다.
n-gram은 n개의 연속적인 단어 나열을 의미한다. 코퍼스에서 n개의 단어를 하나의 토큰으로 간주하여 다음에 올 단어를 예측한다.
- unigrams: n이 1 일 때
- bigrams: n이 2 일 때 앞의 1개 단어만 고려
- trigrams: n이 3 일 때 앞의 2개 단어만 고려
- 4-grams n이 4일 때 앞의 3개 단어만 고려<br>  
  

n을 선택할 때 trade-off가 존재한다.<br>
n을 크게 하면 언어 모델의 성능을 높일 수 있지만, 코퍼스에서 n개의 단어를 선택할 때, 카운트할 수 있는 확률이 적어지므로 희소 문제가 발생한다. 또한 n이 커지면 모델 사이즈가 커진다는 문제가 있다.<br>  
n을 작게 하면 코퍼스에서 카운트를 잘 할 수 있지만, 정확도는 떨어질 수 있다. 일반적으로 trade-off문제로 인해 정확도를 높이려면 n은 최대 5를 넘게잡아서는 안된다고 권장한다.<br>  
{:.note}
<br>

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
