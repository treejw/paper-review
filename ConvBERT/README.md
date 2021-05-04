# ConvBERT: Improving BERT with Span-based Dynamic Convolution `2021.2`

## Abstract

- BERT는 Global 관점에서  Self-Attention을 하기 때문에 전체입력에 대한 attention map을 생성한다. 이것은 local dependency가 적용되지 않아 self attention head는 전체를 학습하게 되는데 여기에서 많은 메모리와 계산비용이 낭비되고 있다.

- (electra나 roberta등도 이를 문제 삼았던 기억 추가하기)

- 따라서 local dependency를 반영하는 모델을 만들기 위해서 기존의 self-attention head 대신 새로운 attention head인 span-based dynamic convolution을 제안한다. 


- span-based dynamic convolution = Convolution Heads(논문제안) + Self-Attention Head(기존 BERT)
  - Local & Global Context를 모두 효과적으로 학습할 수 있다.

- 실험 결과 ConvBERT는 기존의 BERT와 ELECTRA 모델보다 성능과 비용면에서 이점을 보여줌.
  - 성능 : GLUE 86.4  /  비용 : 1/4 미만
 
<br>

## Introduction
- BERT와 같이 Comprehensive Representations을 학습하기 위해 Multi-head Self-Attention에 지나치게 의존하는 구조는 계산 비용면에서 낭비가 있다.
  - BERT의 Self-Attention Heads는 non-local operator이지만, 자연어의 고유한 특성상 Local Dependency를 학습해야함.
  - 저자들의 실험 결과 BERT를 Downstream Task를 위한 Fine-Tuning하는 과정에서 Self Attention head 몇 개를 지워도 성능 저하가 일어나지 않았다고 함.

- 아이디어 : Local Operation으로 Attention Head 몇 개를 대체한다면 낭비를 자연적으로 줄일 수 있지 않을까? & Convolution이 Local Feature Extraction에서 우수한 면을 보이므로 Local Operator로 잘 동작하지 않을까?

- 방법 : Convolution을 Self-Attention에 통합한 Mixed Attention Mechanism을 제안

- 아래 그림의 (a)를 보면 Self-Attention은 모든 토큰을 가지고 Global Dependency를 계산하지만, 저자는 현재 토큰의 local span을 가지고 Local Dependency를 포착하고자 함

- 이를 위해 모든 토큰에 대해 동일한 파라미터를 공유하는 standard convolution을 사용하는 것보다 Dynamic Convolution을 사용하는 것이 토큰 사이의 local dependency를 보다 유연하게 포착할 수 있음.

- 하지만 Dynamic Convolution은 서로 다른 컨텍스트에 위치한 같은 토큰을 구분하지 못하고 동일한 Kernel을 만들어낸다는 문제가 있다. 예를 들어, 그림(b)에서 can에 해당하는 커널은 모두 동일하다.

- 이를 해결하기 위해 하나의 토큰을 가지고 커널을 만들지 않고 Span을 사용하여 보다 adaptive convolution kernel을 만들 수 있는 Span-based dynamic convolution을 제안하였다. 그림(c)를 보면 3 개의 can을 통해 만들어진 Kernel이 모두 다른 것을 볼 수 있다.

- 이렇게 만들어진 Span-based dynamic convolution을 사용하여 global information과 local information 모두 더 잘 capture할 수 있는 Convolutional Self-attention (Mixed Attention)을 만들었다.

- Convolutional Self-Attention으로 만들어진 ConvBERT를 통해 더 적은 비용과 파라미터로 GLUE 벤치마크에서 
BERT-base보다 5.5 높고, ELECTRA-base보다 0.7 높은 점수를 달성하였다고 한다.


## Method
### 0. 기존 BERT
![bert구조](https://user-images.githubusercontent.com/43063980/116991405-77be3180-ad0f-11eb-9bb3-b940e5eb599a.png)

d : hidden dimension

n : token 개수

input X를 3개의 linear transformation을 사용하여 Key, Query, Value의 세개로 임베딩

dk : d/H-dimensional segments

![bert수식](https://user-images.githubusercontent.com/43063980/116988489-696e1680-ad0b-11eb-9ac6-bce9db1b383a.png)


![image](https://user-images.githubusercontent.com/43063980/116988411-4a6f8480-ad0b-11eb-824a-e81980c0bf96.png)

- bert와 convbert의 average attention map을 시각화 해놓은 모습
- attention head의 많은 부분이 local dependency을 학습한다는 것을 의미한다고 합니다.
- 기존의 bert에서 모든 토큰 쌍 간의 attention weight를 계산하는 것은 불필요한 계산 및 모델 이중화(redundancy)를 가져온다. 

### 1. Light-weight and dynamic convolution

**[Light-weight convolution]**
: local dependency를 모델링하고, 파라미터 size를 k -> d로 줄여줌.

- convolution kernel : 
![image](https://user-images.githubusercontent.com/43063980/116994456-b35afa80-ad13-11eb-8967-22764680528c.png)
                                                           
- 위치가 i, 채널깊이가 c일 때의 (depth-wise)convlution output : 
![image](https://user-images.githubusercontent.com/43063980/116994554-d2598c80-ad13-11eb-816b-f8b66617f751.png)


channel dimension에 따라 weight를 묶고, 정리하면 결론적으로 이런 수식이 나온다. 
![image](https://user-images.githubusercontent.com/43063980/116988516-75f26f00-ad0b-11eb-9f21-46d2f36418c1.png)

**[dynamic convolution]**

: local depenedency가 적용된 convolution
현재 입력 토큰에 대한 가중치와 함께 인근 토큰애 대해서 새로운 representation embedding을 생성한다.

![image](https://user-images.githubusercontent.com/43063980/116998749-a3461980-ad19-11eb-92a9-c86171302512.png)



![image](https://user-images.githubusercontent.com/43063980/116988533-7d197d00-ad0b-11eb-8e86-e51f59cf131b.png)


- dynamic tonvlution을 통해 local dependency는 적용할 수 있었으나 convolution kernel은 압력하는 하나의 토큰에만 의존하는 문제가 있었다.
  - 문맥을 이해하지 못하고 동일한 토큰일 때, 동일한 kernel를 만들어 냄.
  - 이로 인해 다음에서 span의 범위를 적용시킴




<br>


### 2. Span-based dynamic convolution
- 앞선 문제점 때문에 input으로 들어갔던 단일 토큰 대신 깊이별 분리 가능한 convolution과 query의 계산을 사용하여 local context정보를 포함하여 convolution kernel을 생성할 수 있게 한다.

![image](https://user-images.githubusercontent.com/43063980/116998768-ad681800-ad19-11eb-99ec-978d7a3046ad.png)

Conv : 깊이별 분리 가능한 convolution

- 분리 가능한 convolution과 query의 계산
![image](https://user-images.githubusercontent.com/43063980/116988622-97535b00-ad0b-11eb-9437-459537888436.png)

- 이후 구한 convolution kernel
![image](https://user-images.githubusercontent.com/43063980/116988646-a0442c80-ad0b-11eb-90f2-906e8ab55069.png)


### 3. ConvBERT architecture 

![image](https://user-images.githubusercontent.com/43063980/116988593-8dc9f300-ad0b-11eb-8f7d-6103cba4bb94.png)

**[Mixed Attention]**
- 기존의 bert와 Span-based dynamic convolution을 concat하여 사용함.
![image](https://user-images.githubusercontent.com/43063980/116988712-b520c000-ad0b-11eb-9396-1de6329e0a89.png)

**[Bottleneck design for self-attention]**
- linear layer가 input decention 보다 output demention이 작은 모양이다. (d/r)

**[Grouped feed-forward module]**
- 논문에서는 기존의 transformer의 feed-forward module에서 파라미터를 사용하는거 같은데 이때 파라미터 수와 계산을 줄이기 위해 Grouped Linear(GL)을 사용한다. 

![image](https://user-images.githubusercontent.com/43063980/116988747-c1a51880-ad0b-11eb-9790-419d0bf42f96.png)


<br>



### 참고
- paper : [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/pdf/2008.02496.pdf)
- [Gwig's Research Blog](https://ankle96.tistory.com/58)
- [machinecurve](https://www.machinecurve.com/index.php/question/what-is-convbert-and-how-does-it-work/)
- [Github](https://github.com/yitu-opensource/ConvBert/blob/5b4546ced2af2f7cd5332ba25330879ff9365f42/model/modeling.py#L767)
