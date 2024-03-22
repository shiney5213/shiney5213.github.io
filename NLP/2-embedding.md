---
layout: page
title: Embedding
description: >
  How you install Hydejack depends on whether you start a new site,
  or change the theme of an existing site.
hide_description: true
sitemap: false
---



1. this unordered seed list will be replaced by toc as unordered list
{:toc}


## 텍스트 벡터화(Text Vectorization)
앞서 토큰화에 대해 알아보았다. 단어나 문장을 형태소로 토큰화 한 후, 이를 다시 숫자로 변환하여 컴퓨터가 이해할 수 있도록 해야 한다.<br>
텍스트를 숫자로 변환하는 과정을 `텍스트벡터화`라고 한다.

### 원-핫 인코딩(Ohe-Hot Encoding)
문서에 등장하는 각 단어를 고유한 색인값으로 매칭한 후, 해당 색인 위치를 1, 나머지는 0으로 표시하는 방법이다.<br>

### 빈도 벡터화(Count Vectorization)
문서에서 단어의 빈도 수를 세어 해당 단어의 빈도를 벡터로 표현하는 방법이다.<br><br>

원-핫 인코딩과 빈도 벡터화는 단어나 문장을 벡터 형태로 변환하기 쉽고 간단하다는 장점이 있지만, <br>
벡터의 희소성(Sparsity)가 크기 때문에 컴퓨팅 비용의 증가와 차원의 저주 같은 문제를 겪을 수 있다. <br>
또한 벡터가 텍스트의 의미를 내포하고 있지 않으므로, 두 문장이 의미적으로 유사하다고 해도 벡터가 유사하지 않을 수 있다.
{:.note}
<br>

### 임베딩
(Word Enbedding)
단어의 의미를 학습해 고정된 길이의 실수 벡터로 표현하는 방법으로 <br>
벡터 공간에 두 단어의 의미를 상대적 위치로 표현해 단어의 관계(유사도 등)를 추론할 수 있다.<br>

#### 동적 임베딩(Dynamic Embedding)
워드 임베딩은 고정된 임베딩을 학습하기 때문에 다의어나 문맥정보를 다루기 어렵기 때문에 <br>
인공신경망을 활용해 동적 임베딩 기법을 사용한다.

임베딩은 단어나 문장을 숫자로 변환하는 과정을 말하며, 
여기에서는 텍스트 벡터화와 같은 의미로 사용되었다.
{:.note}
<br>

## 임베딩 기법의 역사
### 룰(rule) (~1990년대)
언어학적 지식을 활용하여 사람이 직접 입력값(feature)을 추출한다. <br>
예를 들어 모든 단어에 대한 유의어 집합을 만들고, 각 단어들의 관계를그래프로 표현하는 방법으로 `시소러스(thesaurus, 유의어 사전)`를 사용한다.<br>

### end to end(딥러닝 모델)(2000년대 중반~)
입력(input)과 출력(output)의 관계를 잘 근사할 수 있는 모델을 설계하여 데이터를 모델에 통째로 넣고, 사람의 개입없이 모델 스스로 처음부터 끝까지 이해하도록 유도한다. <br>
기계번역(machine translation)에 널리 쓰였던 시퀀스투시퀀스(sequence-tosequnece) 모델이 엔드투엔드의 대표 사례다

### pretrain/fine tunig(2018~)
우선 대규모의 말뭉치로 임베딩을 만들어 말뭉치의 의미적, 문법적 맥락을 포함시킨다. (pretrain) <br>
이후 임베딩을 입력으로 사용하는 새로운 딥러닝 모델을 만들고, task에 적합한 소규모 데이터를 사용하여 모델 전체를 업데이트한다(fine tuning)

실제 해결해야 하는 task를 Downstream Task이라고 한다. 예를 들어 품사 판별(Part-of-speech tagging), 개체명 인식(Named Entity Recognition), 의미역분석(Semantic Role Labeling) 등이 있다. 이에 반해 Upstream Task는 Downstream에 앞서 해결해야 하는 과제로, 단어/문장 임베딩을 pretrain하는 작업을 의미한다.
{:.note}

## 임베딩 기준: 단어 or 문장

| 구분 |단어 수준 모델|문장 수준 모델|
|:---:|:---:|:---:|
| 의미 |각 벡터에 해당 단어의 문맥적 의미 함축|단어 시퀀스(sequence) 전체의 문맥적 의미 함축|
| 특징 | 동음이의어(homonym) 구분 어려움| 단어 임베딩 기법보다 전이학습(Transfor Leanring) 효과가 좋음.|
|  예  | NPLM | ELMo|
|        | Word2Vec | BERT|
|        | GloVe    | GPT |
|        | FastText | XLNet |


## 임베딩 기법의 종류
### 통계 기반 : 잠재의미분석(Latent Semantic Analysis, LSA)
- 말뭉치의 통계량 정보(단어 사용 빈도 등)를 나타내는 행렬에 특이값 분해(Singular Value Decomposition, SDA)를 이용하여 벡터들의 차원을 축소하는 방법이다.
- 대상 행렬
  - 단어-문서 행렬(Term-Document Matrix),
  - TF-IDF 행렬(Term Frequency–Inverse Document Frequency),
  - 단어-문맥행렬(Word-Context Matrix),
  - 점별 상호정보량 행렬(Pointwise Mutual Information Matrix) 등

### 신경망(Neural Netqork,NN) 기반
- 인공신경망을 사용하여 임베딩을 진행한다.

## 임베딩 벡터 생성 방법
### 행렬 분해(Factorization) 기반 임베딩 : : GloVe, Swivel 등
말뭉치 정보가 들어있는 행렬을 두 개 이상의 작은 행렬로 분해하는 임베딩 기법이다.
분해한 이후에는 둘 중 하나의 행렬만 사용하거나, 둘을 더하거나(sum), 이어 붙여(concatenate) 임베딩으로 사용한다.

### 예측 기반 임베딩 : 단어 수준 임베딩(Word2Vec, FastText), 문장 수준 임베딩(BERT, ELMo,GPT, XLNet 등)
중심 단어를 보고 주변 단어를 예측하거나, 반대의 경우를 예측하는 과정에서 학습이 진행된다.

### 토픽 기반 임베딩: 잠재 디리클레 할당(Latent Dirichlet allocation, LDA)
주어진 문서의 주제(latent topic)를 추론하면서 임베딩을 진행하는 기법이다. LDA로 학습이 완료되면, 각 문서가 어떤 주제 분포를 가지고 있는지 확률 벡터형태로 반환하므로, 임베딩 기법의 일종으로 이해할 수 있다.


## 임베딩을 만드는 방법

|구분|Local Representations|Distributed Reprsentations|contextualized representations|
|:---:|:---:|:---:|:---:|
|기준|단어 출현 빈도|단어의 분포(문맥)|단어 순서|
|가정| Bag Of Word | 분포 가설 | 언어 모델 |
|대표 통계량|TDM, TF-IDF| Word-Context Matrix, PMI, PPMI | - |
|통계기반 모델| N-gram, LDA|  [단어 수준] Glove(추론 함께 사용), Swivel| -|
|추론 기반 모델(NN)| Deep Averaging Network | [단어 수준] Word2vec, FastText | [문장 수준] ELMo, GPT, BERT


### Local Representations (국소 표현) = Discrete Representations (이산 표현)
  - 해당 단어만 보고, 특정값을 매핑하여 단어 표현하는 방법
#### 국소 표현 방법
- One-hot Vector
- N-gram
- Count Based : Bag og Words, DTM

### Continuous Representations( 연속 표현) = `Distributed Reprsentations(분산 표현)`
  - 해당 단어와 주변의 문맥을 고려하여 단어 표현하는 방법 -> 단어의 의미, 뉘앙스 표현 가능
  -  단어를 고차원 벡터 공간에 매핑하여 단어의 의미를 담는 것
  - 단어의 의미를 문맥상 분포적 특성을 통해 나타남 -> 유사한 문맥에서 등장하는 단어는 비슷한 벡터 공간상 위치를 가짐
  
#### 이론적 토대: `분포 가설(Distributional hypothesis)`
> - 단어의 의미는 그 단어가 사용된 맥락(주변 단어)에 의해 형성한다는 가정 
> - 자연어 처리에서 분포(distribution):  특정 범위(window) 내에 등장하는 문맥의 집합
> - 분포 가설의 전제 : 개별 단어의 분포는 그 단어가 문장 내 위치와 이웃 단어 등에 따라 달라지므로, 어떤 단어 쌍(pair)이 비슷한 문맥 환경에 자주 등장한다면 그 의미도 유사할 것

#### 분산 표현 방법
##### Count Based(통계 기반) 
- 동시발생행렬 (TF-IDF, PMI, PPMI, SVD)
- Full Document(LSA)
- Windows(Glove): 통계 기반 + 추론 기반

##### Prediction Based(추론 기반) 
- Word2Vec
- FastText

### `contextualized representations`
- 기존 연구 : 한 단어의 의미가 항상 고정되어 있다는 한계 -> 단어의 '형태'가 같으면 같은 의미로 파악
- 문맥에 따라 단어를 여러가지 의미로 보고, 의미를 반영한 벡터 표현
  
#### 이론적 토대 : `언어 모델(language model)`
- 입력된 문장으로 각 문장을 생성할 수 있는 확률을 계산하는 모델
- 이를 위해 주어진 문장을 바탕으로 문맥을 이해하고, 문장 구성에 대한 예측 수행

#### 언어 모델 종류
##### 자기회귀 언어 모델(Autoregressive Language Model)
입력된 문장(단어들의 시퀀스)의 조건부 확률의 연쇄법칙([Chain rule for conditional probability])을 통해 다음에 올 단어를 예측한다.
이를 위해 이전에 등장한 모든 토큰의 정보를 고려하며, 문장의 문맥 정보를 파악하여 다음 단어를 생성한다. 

[Chain rule for conditional probability]: https://en.wikipedia.org/wiki/Chain_rule_%28probability%29

##### 통계적 언어 모델(Statistical Language Model, SLM)
언어의 통계적 구조를 이용해 문장이나 단어의 시퀀스를 생성하거나 분석한다.<br>
마르코프 체인(Markoc Chain)은 `빈도` 기반의 조건부 확률 중 하나로, 주어진 데이터에서 각 변수가 발생할 빈도수를 기반으로 확률을 계산한다.<br>  
이 방법은 단어의 순서와 빈도에만 기초해 문장의 확률을 예측하므로 문맥을 제대로 파악하지 못하면 불완전하거나 부적절한 결과를 생성할 수 있으며<br>
한번도 관측한 적이 없는 단어나 문장에 대해서는 예측하지 못하는 `데이터 희소성(Data sparsity)`문제가 발생할 수 있다.<br>  
하지만 기존에 학습한 텍스트 데이터에서 패턴을 찾아 확률 분포를 생성하므로,<br>  
이를 이용해 새로운 문장을 생성할 수 있으며 다양한 텍스트 데이터를 학습할 수 있다.<br>  

#### contextualized representations 모델
- ELMo(Embeddings from Language Models)
- ULMFiT(Universal Language Model Fine-tuning)
- Transformer
- GPT(Generative Pre-Training Transformer)
- BERT(Bidirectional Encoder Representation from Transformer)


최근 연구되는 자연어 처리 기법은 언어 모델을 이용해 가중치를 사전 학습한다.<br>
예를 들어 GPT(Generative Pre-trainedTransformer)나 BERT(Bidirectional Encoder Representations from Transformers)는 문장 생성 기법을 이용해 모델을 사전 학습한다.
{:.note}
<br>








<!-- ##### N-gram Language Model
이전에 등장한 모든 단어를 고려하는 것이 아닌, 일부 단어(n개)만 고려하여 다음 단어를 예측하는 모델이다.
n-gram은 n개의 연속적인 단어 나열을 의미한다. 코퍼스에서 n개의 단어를 하나의 토큰으로 간주하여 다음에 올 단어를 예측한다.
- unigrams: n이 1 일 때
- bigrams: n이 2 일 때 앞의 1개 단어만 고려
- trigrams: n이 3 일 때 앞의 2개 단어만 고려
- 4-grams n이 4일 때 앞의 3개 단어만 고려<br>  
  

n을 선택할 때 trade-off가 존재한다.<br>
n을 크게 하면 언어 모델의 성능을 높일 수 있지만, 코퍼스에서 n개의 단어를 선택할 때, 카운트할 수 있는 확률이 적어지므로 희소 문제가 발생한다. 또한 n이 커지면 모델 사이즈가 커진다는 문제가 있다.<br>  
n을 작게 하면 코퍼스에서 카운트를 잘 할 수 있지만, 정확도는 떨어질 수 있다. 일반적으로 trade-off문제로 인해 정확도를 높이려면 n은 최대 5를 넘게잡아서는 안된다고 권장한다.<br>  
{:.note} -->
<br>


<br>
<br>
<br>
---

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
