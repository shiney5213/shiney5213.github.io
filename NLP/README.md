---
layout: page
title: NLP
description: >
  Here you should be able to find everything you need to know to accomplish the most common tasks when blogging with Hydejack.
hide_description: true
sitemap: false
permalink: /nlp/
---

자연어(National Language)는 프로그래밍 언어와 구별되는 사람들이 사용하는 언어를 의미한다.<br>
자연어 처리(National Language Processing, NLP)는 컴퓨터가 인간의 언어를 이해하고, 해석 및 생성하기 위한 기술을 말한다.

NLP task
- 자연어 추론(National Language Inference, NLI): 두 개 이상의 문장이 주어졌을 때 두 문장 간이 관계 분류
- 질문 응답
- 감정 분석
- 텍스트 함의(Textual entailment): 전제가 주어졌을 때 가설이 참인지 결정
  
- SQuAD(질의응답), 
- SNLI(문장 사이의 모순 파악)
- SRL(의미역 결정)
- Coref(Entity 찾기, Blog)
- NER(Entity 인식) 
- SST-5(문장 분류)
- 


## Text Preprocessing
* [Tokenization]{:.heading.flip-title} --- Tokenization의 개념과 다양한 알고리즘 소개
* [└Sentencepeice]{:.heading.flip-title} --- Sentencepeice 라이브러리에서 BPE(Bite Pair Encoding) 알고리즘 구현
* [└Tokenizers]{:.heading.flip-title} --- Huggingface의 Tokenizers 라이브러리에서 Wordpiece 알고리즘 구현

[Tokenization]: 1-tokenization.md
[└Sentencepeice]: 1-sentencepiece.md
[└Tokenizers]: 1-tokenizers.md


## Word Embedding : 단어를 임의의 벡터로 표현하는 방식 , 분포 가설에 기반
* [Embedding]{:.heading.flip-title} --- Embedding의 기본 개념과 다양한 알고리즘 소개
* [└Bag_Of_Words]{:.heading.flip-title} --- Local Representation 방법 중 Bag Of Words 소개
<!-- * [└Distributional_hypothesis]{:.heading.flip-title} --- Continous Representation 방법의 가정인 `분포 가설' 소개-->
<!-- * [└Language_Model]{:.heading.flip-title} --- Continous Representation 방법의 가정인 '언어 모델' 소개  -->
* [Word2Vec]{:.heading.flip-title} --- Efficient Estimation of Word Representations in Vector Space (2013)
* [└CBoW]{:.heading.flip-title} --- Word2Vec 논문의 CBoW모델 구현
* [GloVe]{:.heading.flip-title} --- Global Vectors for Word Representation (2014)
* [fastText]{:.heading.flip-title} --- Enriching Word Vectors with Subword Information (2016)
<!-- * [└fastText]{:.heading.flip-title} --- gensim library를 이용하여 fastText 구현 -> 라이브러리 import 실패-->
* [ELMo]{:.heading.flip-title} --- Deep contextualized word representations (2018)
{:.related-posts.faded}


[Embedding]: 2-embedding.md
[└Bag_Of_Words]: 2-1-BagOfWords.md
[└Distributional_hypothesis]: 2-2-Distributional-hypothesis.md
[└Language_Model]: 2-3-Language-Model.md



[Word2Vec]: 3-Word2Vec.md
[└CBoW]: 3-Word2Vec-CBoW.md
[GloVe]: 4-GloVe.md
[fastText]: 5-fastText.md
<!-- [└fastText]: 5-fastText-code.md -->
[ELMo]: 6-ELMo.md



## RNN Series Models --- 언어 모델에 기반, 평가 척도 : perplexity 사용
* [RNN]{:.heading.flip-title} --- Recurrent neural network based language model (2010) (처음 등장: 1986)
* [└RNN]{:.heading.flip-title} --- pytorch tutorial에 있는 RNN 코드 구현(character 기반 사람 이름 분류)
* [LSTM]{:.heading.flip-title} --- Long Short Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling (2014) (처음 등장: 1997)
* [└LSTM]{:.heading.flip-title} --- 삼성전자 주식의 종가를 예측하는 LSTM 코드 구현
* [GRU]{:.heading.flip-title} --- Learning Phrase Representation using RNN Encoder-Decoder for Stistical Machine Translation (2014)
* [Seq2Seq]{:.heading.flip-title} ---Sequence to Sequence Learning with Neural Networks (2014)
{:.related-posts.faded}

[RNN]: 7-RNN.md
[└RNN]: 7-RNN-code.md
[LSTM]: 8-LSTM.md
[└LSTM]: 8-LSTM-code.md
[GRU]: 9-GRU.md


##  Attention Mechanism
9. Attention : Neural Machine Translation by Jointly Learning to Align and Translate (2015)
10. Transformer : Attention is All You Need (2017)
{:.related-posts.faded}

## Pre-trained language models based on the Transformer architecture
11. GPT-1 : Improving Language Understanding by Generative Pre-Training (2018)
12. BERT : Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)
13. GPT-2 : Language Models are Unsupervised Multitask Learners (2018)
14. RoBERTa : RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)
15. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (2019)
16. ELECTRA : Pre-training Text Encoders as Discriminators Rather Than Generators (2020)
17. XLNet : Generalized Autoregressive Pretraining for Language Understanding (2019)


## Large Language Model
18. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2019) - T5 논문 
19. GPT-3 : Language Models are Few-Shot Learners (2020)
20. Training language models to follow instructions with human feedback (2022) - InstructGPT 논문 
21. FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS (2022) - FLAN 논문
22. LLaMA: Open and Efficient Foundation Language Models (2023) - LLaMA 논문 
23.  Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)
24. GPT-4 Technical Report (2023)

## Parameter-Efficient Fine Tuning (PeFT)
25. LoRA: Low-Rank Adaptation of Large Language Models (2021)
26. GPT Understands, Too (2021) : Prefix Tuning
+) P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks : P-tuning v2 논문 
27. Towards a Unified View of Parameter-Efficient Transfer Learning (2022) : Adapter 
+) LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models (2023)

## Quantization 논문 
26. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)
27. QLoRA: Efficient Finetuning of Quantized LLMs (2023)


## Other
* [LICENSE]{:.heading.flip-title} --- The license of this project.
* [NOTICE]{:.heading.flip-title} --- Parts of this program are provided under separate licenses.
* [CHANGELOG]{:.heading.flip-title} --- Version history of Hydejack.
{:.related-posts.faded}





[ELMo] : 6-ELMo.md




[install]: install.md
[upgrade]: upgrade.md
[config]: config.md
[basics]: basics.md
[writing]: writing.md
[scripts]: scripts.md
[build]: build.md
[advanced]: advanced.md
[LICENSE]: ../LICENSE.md
[NOTICE]: ../NOTICE.md
[CHANGELOG]: ../CHANGELOG.md
