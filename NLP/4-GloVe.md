---
layout: page
title: GloVe
description: >
  GloVe paper review
hide_description: true
sitemap: false
---


- Title: GloVe: Global Vectors for Word Representation
- paper: [download]
- journal: 
- year : 2014
- Subjects:	Computation and Language (cs.CL)
- Jeffrey Pennington, Richard Socher, and Christopher Manning
- summary
  > - GloVe = Gloval Veacors의 줄임말 -> 모델이 global corpus 통계를 포착 가능
  > - count 기반의 LSA(Latent Semantic Analysis)와 예측 기반의 Word2Vec의 한계를 보완하여 두 가지 모두 사용
  > - 목표 :  vector Representation 생성
  > - 모델 :  global log-bilinear regression model
  > - 결과 :  단어 유추 작업, 유사성 작업 및 명명된 개체 인식에 대한 성능이 향상
  > - 의미 :  카운트 기반 방법의 이점과 예측 기반 방법 모두 사용
- 사용 기법
  > - global word-word co-occurrence Matrix
  > - negative sampling



[download]: https://aclanthology.org/D14-1162.pdf



1. this unordered seed list will be replaced by toc as unordered list
{:toc}
<br>

## (0) Abstract
- 연구 배경
> -  최근에 제안된 단어의 벡터 공간 표현 방법이 벡터 산술을 사용하여 세세한 의미론적 및 구문론적 규칙을 포착하는 데 성공
> - 그러나 규칙의 원천은 여전히 불투명함.

- 연구 목표 및 방법
> - 목표: 이러한 규칙이 단어 벡터에서 나타나기 위해 필요한 모델 속성을 분석하고 명시화
> - 모델: 전역 행렬 인수 분해 및 로컬 컨텍스트 윈도우 방법의 장점을 결합한 새로운 global log-bilinear regression model 을 제안

- 결과 및 의의
> - 의미 있는 하위 구조를 갖춘 벡터 공간이 생성
> - 단어 유추 작업에서 75%의 성능을 보임
> - 단어 유추 작업, 유사성 작업 및 명명된 개체 인식에 대한 성능이 향상
> 

## (5)Conclusion
- 연구 취지
  > 단어의 분포 표현이 카운트 기반보다 예측 기반 방법이 지지 받음<br>
  > 두 가지 방법은 둘 다 말뭉치의 기본 동시 발생 통계에 의존하기 때문에 근본적으로 다르지 않음.
  > 카운트 기반 방법은 글로벌 통계를 효율적으로 수집할 수 있다는 이점이 있음.

-  GloVE
> - 모델: 카운트 기반 방법의 이점과 예측 기반 방법의 의미 있는 선형 하부 구조를 결합한 새로운 글로벌 로그-쌍선형 회귀 모델
> - 단어 표현의 비지도 학습을 위한 새로운 global log-bilinear regression model

- 결과
> - 단어 유추, 단어 유사성 및 명명된 개체 인식 작업에서 우수한 성능을 보임


## (1) Introduction
### sementic vector space model
> - 각 단어를 실수 벡터로 표현 -> 정보 검색, 문서 분류, 질문 응답, 명명된 개체 인식, 구문 분석 등에서 활용
> - 기본적인 품질 평가 방법: 기본적인 두 벡터 간의 거리(또는 각도)를 측정
> - Word2vec 
    > -  단어 유사성에 기반한 평가-> 다양한 차원을 조사함으로써 단어 벡터 공간의 미세한 구조를 탐구
    > - 의미의 차원을 생성하는 모델-> 분산 표현의 다중 클러스터링 개념 

### two main model family
- 글로벌 매트릭스 인수분해 방법 
> - 예:잠재 의미 분석(Latent Semantic Analysis, LSA) 
>  - 통계 정보를 효율적으로 활용하지만, 단어 유추 작업에서 상대적으로 성능이 낮음 

-  로컬 컨텍스트 윈도우 방법 
> - 예: 스킵그램 모델
> -  단어 유추 작업에서 더 나은 결과를 보임
> -  global co-occurrence counts가 아닌 개별 local context window에서 훈련되기 때문에 말뭉치의 통계를 잘 활용하지 못함

### 연구
- 단어 의미의 선형 방향을 생성 -> global log-bilinear regression models
- global word-word co-occurrence counts 기반 학습 -> a specific weighted least squares model 
- 결과: 의미 있는 하위 구조를 갖는 단어 벡터 공간을 생성-> 단어 유추 데이터셋에서 75%의 정확도 보임

## (4) Experiments
### Evaluation methods
#### Word analogies
-  "a가 b와 같은 관계이면 c는 무엇인가?"의 질문
-  의미적 질문 : 주로 사람이나 장소에 관한 유추 (예: "아테네는 그리스와 같은 관계이면 베를린은 무엇과 같은가?"와 같은 것)
-  구문적 질문 : 주로 동사 시제나 형용사 형태에 관한 유추( 예:  "춤은 춤추기와 같은 관계이면 날다는 무엇과 같은가?")
-  코사인 유사도에 따라 W_b - W_a + W_c에 가장 가까운 표현 W_d를 찾아 d라는 단어를 결정 (W_a - W_b = W_c - W_d)

#### Word Similarity
- WordSim-353, MC, RG, SCWS, RW 등 다양한 단어 유사성 과제에 대해 평가
  
#### Named Entity Recognition
- CoNLL-03 훈련 데이터에서 모델을 훈련하고 세 가지 데이터셋에서 테스트 진행

### Corpora and training details
#### dataset: 54.9 billion tokens
- a 2010 Wikipedia dump with 1 billion tokens
- a 2014 Wikipedia dump with 1.6 billion tokens
- Gigaword 5 which has 4.3 billion tokens
- the combination Gigaword5 + Wikipedia2014, which has 6 billion tokens;
- 42 billion tokens of web data, from Common Crawl5. 

#### preprocessing
- 각 말뭉치를 Stanford 토크나이저로 토큰화 + 소문자로 만들기
- 상위 400,000개의 가장 빈번한 단어로 어휘를 구축
- context windows: decreasing weighting function 사용-> d 단어가 떨어져 있는 단어 쌍이 전체 카운트에 1/d를 기여하도록 함

#### hyperparams
- xmax = 100
- α = 3/4
- AdaGrad (Duchi et al., 2011)
- X:stochastically sampling nonzero elements 
- init learing rate: 0.05
- epoch: 50 (if d < 300) else 100
      
#### model output 
-  W, W~ -> W + W~을 사용
-  과적합, noise을 줄이고, 성능 높임

### Results
#### word analogy task 
- 계층적 소프트맥스를 사용한 Word2Vec 보다 negative sampling을 사용한 GloVE가 더 우수
- SVD-L모델은 큰 말뭉치에서 감소 -> SVD 모델은 대규모 데이터에는 적합하지 않음
- GloVE는 CBoW의 1/2 말뭉치 데이터를 사용했음에도 성능이 더 우수함
- ![table2](/assets/img/nlp/4_GloVE_table2.png){: width="50%" height="50%"}

- 
#### Named Entity Recognition
- GloVe 모델은 모든 평가 지표에서 다른 모델보다 우수

### Model Analysis
#### Vecot Length and Context size
- Context Windows
  > - 대칭적: 문맥 단어를 좌, 우로 확장되는 경우 ( co: 비대칭: 왼쪽으로만 확장 되는 경우)
  > - 벡터의 차원이 200 이상인 경우: 정확도 증가율 감소
  > - 윈도우 사이즈가 작고 context window가 비대칭인 경우: 구문 task 성능이 더 좋음 -> 구문 정보가 주로 주변 문맥에서 파생되고 단어 순서에 강력하게 영향을 받기 때문
  > - 윈도우 사이즈가 큰 경우: 의미 정보 task 성능이 더 좋음.

![figure](/assets/img/nlp/4_GloVE_figure2.png){: width="100%" height="100%"}

#### Corpus Size
- 구문 task
  > - 말뭉치 크기가 커질 수록 성능 증가
  > - 큰 말뭉치가 더 나은 통계를 생성하기 때문에

- 의미 task
  > - 말뭉치 크기가 작은 Wikipedia 데이터에서 성능이 더 좋음
  > - Wikipedia가 대부분의 해당 위치에 대해 상당히 포괄적인 문서를 갖고 있기 때문으로 추정


#### Run-time
- 데이터 X 만드는 시간
> -  context window 크기, 어휘 크기 및 말뭉치 크기를 포함한 여러 요소에 의존
> - 단일 스레드의 듀얼 2.1GHz Intel Xeon E5-2658 머신을 사용하여 10 단어 대칭 문맥 창, 400,000 단어 어휘 및 60억 토큰 말뭉치로 X를 채우는 데 약 85분 소요

- 모델 훈련 시간
  > - 300차원 벡터를 사용하는 경우 (그리고 위의 머신의 모든 32개 코어를 사용하는 경우), 1 epoch당  약 14분 소요

#### Comparison with Word2Vec
- 동일한 말뭉치, 어휘, 창 크기 및 훈련 시간에 대해 GloVe가 일관되게 word2vec을 능가
- Word2Vec : negative sample의 개수가 10을 넘어가면 정확도 감소


## (2) Related Work
### Matrix Factorization Methods
- 저차원 단어 표현을 생성하기 위한 방법 ( 예: LSA, HAL 등)
- 단점: 가장 빈번한 단어가 유사도 측정에 많은 영향을 미침
- 해결: COALS 방법(Rohde et al., 2006) -> 동시발행행렬을 정규화된 상관관계나 엔트로피를 통해 변환
- 최근 연구: 양의 점별 상호 정보(PPMI,  Bullinaria and Levy ,2007),  Hellinger PCA(HPCA) (Lebret and Collobert, 2014) 
  
### Shallow Window-Based Methods
-  local context window 내 문맥 단어 표현을 학습
-  유용한 단어 표현을 학습하기 위해 전체 신경망 구조 사용
> - Bengio et al. (2003): 언어 모델링을 위한 간단한 신경망 아키텍처의 일부로 단어 벡터 표현을 학습하는 모델 소개
>  Collobert and Weston (2008): 단어 벡터 훈련을 하향식 훈련 목표와 분리
> Collobert et al. (2011):  언어 모델의 경우와 같이 단어 표현을 학습하기 위해 단어의 전체 문맥 사용

-  단어 표현을 위한 간단한 모델 사용
  > - Mikolov et al. (2013a): 스킵-그램과 CBOW 모델->  두 단어 벡터 사이의 내적을 기반으로 한 간단한 단일층 아키텍처를 제안
  > - Mnih와 Kavukcuoglu (2013):  관련된 벡터 로그-이중 선형 모델인 vLBL 및 ivLBL를 제안
  > - Levy et al. (2014): PPMI 지표를 기반으로 한 명시적 단어 임베딩 제안
  > - 단어 벡터 간의 선형 관계로 언어적 패턴을 학습할 수 있음을 보임

- 단점
  > -  동시발생 행렬을 직접 다루지 않음
  > - 전체 말뭉치를 대상으로 context windows를 스캔하지만 많은 데이터를 반복하여 사용하지 못함


## (3) The GloVe Model
### notaion
> - X: 단어-단어 동시 발행 빈도
> - X_ij : 단어 i가 문맥 단어 일 때 단어 j가 나타나는 횟수
> - X_i = ∑ k X_ik -> 단어 i가 맥락단어일 때 어던 단어가 나타나는 횟수
> - P_ij = P(j|i) = X_ij / X_i -> 맥락 단어 i가 있을 때 단어 j가 나타날 확률
> - F() : model
> - W_i : 중신 단어의 임베딩 벡터
> - W_k~: 주변 단어의 임베딩 벡터

- 동시 발생 확률로부터 의미를 추출하는 예
> - i = ice, j = steam 일 때
> - i, j와의 관계를 알기 위해서 다양한 증거 단어인 k와의 동시 발행 확률의 비율을 조사
> - k = solid라면(ice와는 관련이 있지만 steam과는 관련이 없음)
  >> - P_ik / P_jk = P(solid|ice)/ P(solid|steam) = 큰 값
> - k = gas 라면(ice와는 관련이 없지만 steam과는 관련이 있음)
  >> - P_ik / P_jk = P(gas|ice)/ P(gas|steam) = 작은 값 
> - k = ice, steam과 모두 관련이 없다면
  >> - P_ik / P_jk = 약 1
> - 관련된 단어/ 무관한 단어와의 차이 구별 가능
> 
![table1](/assets/img/nlp/4_GloVE_table1.png){: width="100%" height="100%"}

  
### model
(1) Pik / Pjk 비율이 세 단어 i, j 및 k에 의존하므로 
![formula1](/assets/img/nlp/4_GloVE_formula1.png){: width="70%" height="70%"}

(2) F 함수에서 단어 벡터 공간에서 P_ik / P_jk  비율의 정보를 인코딩 하기 위해
-  선형 구조인 벡터 공간의 특성 이용 -> 벡터 차이 이용
![formula2](/assets/img/nlp/4_GloVE_formula2.png){: width="70%" height="70%"}

(3)  vector 연산을 위해 
- 2의 식에서 F의 인수는 벡터, 오른쪽 항은 스칼라이므로 dot product 적용
![formula3](/assets/img/nlp/4_GloVE_formula3.png){: width="70%" height="70%"}

(4) Relabeling에 의한 symmetry를 만족하기 위해 그룹 (R,+)와 (R>0, ×)에 대해 group homomorphism([준동형성])이 필요
- 단어-단어 동시 발생 행렬에서 단어와 맥락 단어의 구별은 임의적임
-  w ↔ ˜w를 교환하는 것뿐만 아니라 X ↔ X^T도 교환 가능해야 함.
![formula4](/assets/img/nlp/4_GloVE_formula4.png){: width="70%" height="70%"}
[준동형성]: https://ko.wikipedia.org/wiki/%EC%A4%80%EB%8F%99%ED%98%95

(5) 3의 방정식에 의해 4를 풀면
![formula5](/assets/img/nlp/4_GloVE_formula5.png){: width="70%" height="70%"}

(6) F = exp이면
- F(x) = exp(x)이면 4, 6의 뺄셈이 나눗셈으로 변환되어 각 단어들 사이의 연관성을 자유롭게 표현 가능
- F(x)= exp일 때 log식으로 표현하면
![formula6](/assets/img/nlp/4_GloVE_formula6.png){: width="70%" height="70%"}

(7) 6의 연산을 간편하게 하기 위해
- 오른쪽항의 log(Xi) 때문에 교환대칭성이 나타나지 않음
- bias를 추가해서 log(Xi) = bi로 대체, ˜wk에 대한 추가적인 바이어스 b˜k를 추가하여 대칭성 복원
![formula7](/assets/img/nlp/4_GloVE_formula7.png){: width="70%" height="70%"}

### loss function
![formula8_1](/assets/img/nlp/4_GloVE_formula8_1.png){: width="70%" height="70%"}

(8) log(X_ik)에서 X_ik가 0이 되는 경우 log값이 발산하므로 -> 정보량에 가중치(f(X_ij))를 추가
- 동시발생 행렬에서 동시 등장 빈도의 값 X_ik가 굉장히 낮은 경우에는 정보에 도움이 되지 않음
- X_ik의 값에 영향을 받는 가중치 함수(Weighting function)을 손실 함수에 도입
![formula8](/assets/img/nlp/4_GloVE_formula8.png){: width="70%" height="70%"}

-  가중치 함수(Weighting function)
> - 발생 확률이 적은 단어에 대해서는 가중치 비중을 빈도에 따라 줄일 수 있음.
> - 임계값: xmax = 100, α = 3/4 으로 설정
![formula9](/assets/img/nlp/4_GloVE_formula9.png){: width="70%" height="70%"}





## Reference
- [논문 리뷰] GloVe: Global Vectors for Word Representation: https://imhappynunu.tistory.com/14