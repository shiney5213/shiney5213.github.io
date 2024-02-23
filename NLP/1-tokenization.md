---
layout: page
title: Tokenization
description: >
  How you install Hydejack depends on whether you start a new site,
  or change the theme of an existing site.
hide_description: true
sitemap: false
---

How you install Hydejack depends on whether you [start a new site](#new-sites), 
or change the theme of [an existing site](#existing-sites).

0. this unordered seed list will be replaced by toc as unordered list
{:toc}


## 자연어의 특성
- 모호성 (Ambiquity) : 맥락에 따라 여러 의미를 가짐
- 가변성 (variability) : 사투리(Dialects), 강세(Accent), 신조어(Coinedword), 작문 스타일 -> 다양하게 표현
- 구조 (Structure) : 문장의 구조와 문법적 요소 이해 -> 의미 추론, 분석

이러한 자연어의 특성을 고려하여 구조, 의미, 맥락을 분석하고 이해할 수 있는 알고리즘과 모델을 개발해야 한다.
{:.note}
<br>

> 이러한 자연어의 특성을 고려하여 구조, 의미, 맥락을 분석하고 이해할 수 있는 알고리즘과 모델을 개발해야 한다.


 
## 토큰화(tokenization)
컴퓨터가 자연어를 이해 및 분석, 처리할 수 있도록 말뭉치(Corpus)를 작게 나누는 과정이다. <br>
이때 `말뭉치`는 뉴스기사, 사용자 리뷰, 저널이나 컬럼 등 목적에 따라 구축되는 대규모의 텍스트 데이터이며, <br>
토큰화를 통해 개별 단어나 문장 부호 같은 테스트로 나뉘어진 의미 단위를 `토큰`이라 한다. <br>
말뭉치를 토큰화로 나누는 알고리즘 또는 소프트웨어를 `토크나이저(Tokenizer)`라고 한다.<br>


## tokenization 방법
- 공백 분할 : 공백(띄어쓰기) 단위로 분할
- 정규표현식 적용 : 정규표현식으로 특정 패턴을 식별해 분할
- 어휘 사전(Vocabulary) 사용: 사전에 정의된 단어 집합을 토큰으로 사용해 분할
- 머신러닝 활용 : 데이터세트 기반으로 토근화하는 방법을 학습한 머신러닝 적용 (예: SpaCy)

가장 많이 사용하는 방법은 `어휘사전(Vocabulary)`을 활용하는 방법이다. <br>
직접 사전을 구축하기 때문에 OOV(Out Of Vocab, 사전에 없는 단어)에 대한 고려가 필요하다.<br>
또한, 어휘사전이 커지면 학습 비용 증대는 물론 [차원의 저주(Curse of dimensionality)]에 빠질 수 있으므로 주의해야 한다. <br>
{:.note}
<br>

[차원의 저주(Curse of dimensionality)]: https://ko.wikipedia.org/wiki/%EC%B0%A8%EC%9B%90%EC%9D%98_%EC%A0%80%EC%A3%BC


## tokenization 기준
### 단어 토큰화(Word Tokenization)
공백(띄어쓰기) 기준으로 분할<br>
문장을 의미있는 단위로 나눠 표현할 수 있음<br>
접사, 문장 부호, 띄어쓰기 오류 등에 취약 <br>

### 글자 토큰화(Character Tokenization)
공백(띄어쓰기)과 글자 단위로 분할<br>
비교적 작은 단어 사전 구축하여, 컴퓨터 자원을 아낄 수 있음. 말뭉치를 학습할 때 각 단어를 자주 학습할 수 있음.<br>
언어 모델링 같은 시퀀스 예측 작업에 활용<br>

### 자소 토큰화
'ㄱ,ㄴ,ㅏ,ㅑ,...' 등의 의미상 구별할 수 있는 가장 작은 단위로 분할<br>
작은 크기의 단어 사전으로 OOV를 획기적으로 줄임<br>
개별 토근의 의미가 없으므로 토큰의 의미를 조합하여 결과를 도출해야 함<br>

### 형태소 토큰화(Morpheme Tokenization)
실제 의미를 가지고 있는 최소의 단위인 형태소 단위로 분할<br>
문장 내 각 형태소의 역할을 파악할 수 있어 문장 이해 및 처리 가능<br>

형태소 토큰화는 언어의 문법과 구조를 고려해 단어를 분리하고 의미있는 단위로 토큰을 분류할 수 있다.<br>
한국어는 어근에 다양한 접사와 조사가 조합하여 낱말을 이루기 때문에, 이 방법을 가장 많이 사용한다.<br>
{:.note}

## 형태소 토큰화
### 형태소 어휘 사전 (Morpheme Vocabulary)
각 단어의 형태소 정보를 포함하는 사전<br>
구성: 형태소 + 품사(Part Of Speech,POS) + 품사의 뜻 <br>
품사 태깅(POS tagging)을 통해 문맥을 고려하여 정확한 분석 가능<br>

### 형태소 분석기
- KoNLPy
  
한국어 자연어 처리를 위한 라이브러리<br>
명사 추출, 형태소 분석, 품사 태깅 등 지원 <br>
텍스트 마이닝, 감성분석, 토픽 모델링에 활용<br>
형태소 분석기 : Okt(Open Korean Text), 꼬꼬마(Kkma), 코모란(Komoran), 한나눔(Hannanum), 메캅(Mecab) 등<br>


|            |Okt|꼬꼬마||
|------------|:---:|:---:|
|기능|명사,구문,형태소 추출, 품사 태깅|명사, 문장, 형태소, 품사 추출, 구문 추출 불가|
|품사 종류   |19개|56개|
|장점|자세한 분석 불가|더 자세한 단위로 분석|
|단점|품사 태깅 소요 시간 ↓|품사 태깅 소요 시간 ↑, 모델 성능 저하|

- NLIK
  
영어, 네덜란드어, 프랑스어, 독일어 등 자연어 처리를 위한 라이브러리<br>
토큰화, 형태소 분석, 구문 분석, 개체명 인식, 감성 분석 등 지원<br>


- SpaCy
  
사이썬(Cython)기반의 자연어 처리 라이브러리<br>
NLIK보다 빠른 속도, 높은 정확도가 장점이지만 더 크고 복잡한 모델 사용<br>


## 하위 단어 토큰화(Subword Tokenization)
- 기존의 형태소 분석기
신조어의 발생, 오탈자, 축약어 등장<br>
모르는 단어를 적절한 단어로 나누는 데 취약 -> OOV 증가 ->어휘 사전의 크기 증가<br>

- 하위 단어 토큰화(Subword Tokenization)
하나의 단어를 빈번하게 사용되는 하위 단어의 조합으로 나누어 토큰화<br>
예: Reinforcement -> Rain + force + ment<br>
단어의 길이가 줄어들어 처리 속도 빠름 + OOV 문제, 신조어, 은어, 고유어 문제 완화<br>


### 바이트 페어 인코딩(Byte Pair Encoding,BPE) (= 다이그램 코딩(Digram Coding)
글자의 [빈도수][BPE] 기준<br>
가장 많이 등장한 단어는 하나의 토큰으로 토큰화 + 덜 등장하는 단어는 여러 토큰의 조합으로 표현<br>
<br>

[BPE]: https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt


### 워드피스
빈도가 아닌 확률 기반으로 글자 쌍 병합<br>
하위 단어를 생성할 때 이전 하위 단어와 함께 나타날 확률을 계산해 높은 확률의 하위 단어 선택<br>

### 유니그램(Unigram)


### sentencepiece  라이브러리 <!-- 나중에 실습해보자 -->
구글에서 개발한 하위단어 토크나이저 라이브러리<br>
BPE 방법과 워드피스, 유니그램 등 다양한 알고리즘 지원<br>

<br>
<br>
<br>

---

\<Reference\> <br>
- 파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터비전 심층학습(위키북스, 윤대희, 김동화, 송종민, 진현두 지음, 2023)<br>

<br>
<br>
<br>
Continue with [embedding](2-embedding.md){:.heading.flip-title}
{:.read-more}


[upgrade]: upgrade.md
