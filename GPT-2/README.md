# OpenAI GPT-2 - Language Models are Unsupervised Multitask Learners `2018`

## Abstract

- 기존 자연어 처리 과제들은 대게 과제에 특화된 dataset과 지도학습을 통해 학습이 이뤄졌다.
- GPT-2는 어떤 명시적인 지도학습이 없이 이러한 과제들을 처리할 수 있는 능력을 보인다.

<br>

## 1. Introduction

### ◼ 지금 학습체계의 문제점

1. 문제점
   - 학습의 주된 방법은 목표 과제에 대한 dataset을 모은 후 학습하여 IID로 성능을 측정한다.
   - 이 방법은 좁은 범위의 과제에서는 효과적이지만 범용적인 이해를 필요로 하는 과제에서는 높은 성능을 내지 못한다.
   - 최근 넓은 범위의 dataset과 여러 과제에 대한 GLUE benchmark 등이 제안되기 시작했다.
2. 해결을 위한 모델 출현
   - Multitask learning의 출현
      - 단일 과제 학습에 있어서 수십만 개의 data를 요구하는데 Multitask learning에서는 그 몇 배가 필요한다.
   - task-specific한 parameters를 최대한 줄이고, pretrained parameters를 fine-tuning하기만 하는 방법 (예) [GPT-1](https://github.com/treejw/Study_NLP/tree/main/GPT-1)
      - 이런 방법은 여전히 지도학습을 필요로 한다.
      - 지도학습을 최소한으로 하거나 전혀 필요치 않다면 추론이나 감정 분석과 같은 특정 과제에 있어서 큰 발전이 있을 것이다.


<br>

### ◼ 이 논문의 contributions
- **모델이 어떤 parameter나 모델 구조의 변화가 없어도 zero-shot setting에서 downstream task를 수행할 수 있음을 보여줌**
   - 이전 모델들과 달리 zero-shot으로 다양한 과제를 수행할 수 있음을 보여줌
- **GPT-2는 8개 중 7개의 NLP task들에서 zero-shot으로 SOTA 달성함.**


<br>

## 2. Approach
### ◼ NLP 모델링
1. 보통 NLP 모델링은 각 원소가 일련의 symbol로 구성된 예제에서 비지도분포 추정을 하는 것으로 정의
2. 언어는 자연적으로 연속된 순서를 갖기 때문에 조건부확률의 곱에 따른 합동확률로 구해진다.
![image](https://user-images.githubusercontent.com/41243762/103394185-4b7c8380-4b6a-11eb-9d54-75ba699f787d.png)
3. 이런 조건부확률을 매우 잘 계산하는 모델 Transformer가 있다.
4. 범용으로 사용하기 위해서는 `p(output | input)`이 아닌 `p(output | input, task)`로 표현되어야 한다.
5. [The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730)에서 언어는 과제/입력/출력 모두를 일련의 symbol로 명시하는 방법이 가능하다고 밝혔다.
   - ex) 번역 : (프랑스어로 번역, 영어 텍스트, 프랑스어 텍스트)로 표현된다.
   - ex) 독해 : (질문에 대한 대답, 문서, 질문, 대답) 
6. NLP 모델링은 어떤 symbol이 예측할 출력인지에 대한 명시적인 지도가 없어도 된다.
   - 시퀀스의 부분집합에 대해서만 평가해도 지도목접함수는 비지도목적함수와 같다.
   - 따라서 global minimum은 비지도학습과 지도학습에서 같은 값을 갖는다.
   - 즉, 비지도목적함수에서 수렴하게 할 수 있다.
<br>

### ◼ Training Dataset

- 기존 연구 대부분에서 사용된 dataset은 뉴스와 같은 한 영역에서만 가져온 data로 구성되어 있다.
- GPT-2는 다양한 출처로부터 가져온다.
- Reddit에서 3 karma 이상을 받은 글에 포함된 외부링크의 글을 가져왔다.
   - 결과적으로 45M개의 링크를 가져왔다.
- Dataset 이름은 Web Text
- 텍스트 추출은 [Dragnet](http://www2013.w3c.br/companion/p89.pdf)과 [Newspaper 내용추출기](https://github.com/codelucas/newspaper)를 사용했다.
- 2017년 12월 이후의 글과 위키피디아 글은 제거했으며, 중복제거 등을 거쳐 8M개의 문서, 40GB의 텍스트를 확보하였다.
   - 위키피디아는 다른 dataset에서 흔하고, 학습과 측정 단계에서의 데이터가 겹치는 문제로 인해 분석이 복잡해질 수 있어 제외했다.
<br>

### ◼ Model
- Transformer가 기본 구조이며, GPT-1의 구조를 대부분 따른다.
   - Layer norm이 각 sub-block의 입력으로 옮겨졌다.
   - Layer norm이 마지막 self-attention block 이후에 추가되었다.
   - 모델 깊이에 따른 residual path의 누적에 관한 초기화 방법이 변경되었다.
      - N이 residual layer의 수라 했을 때, residual layer의 가중치에 1/sqrt(N)을 곱했다.
   - 사전이 50,257개로 확장되었다.
   - 문맥고려범위가 512~1024개로 늘어났으면 batch size도 512로 증가했다.

<br>

## 3. Experiments
모델은 크기가 각각 다른 4개를 만들어 실험했다. 각 모델의 크기는 다음과 같다.
|**Parameters**|**Layers**|**d_{model}**|
|:---:|:---:|:---:|
|117M|12|768|
|345M|24|1024|
|762M|36|1280|
|1542M|48|1600|

### Language Modeling
![image](https://user-images.githubusercontent.com/41243762/103394961-09097580-4b6f-11eb-996f-3fe43dbdea77.png)

#### ◼ LAMBADA
텍스트의 장거리 의존성을 평가한다.
- ACC 는 19% → 52.66%
- perplexity는 99.8 → 8.6
으로 향상시켰다.

#### ◼ Children’s Boot Test (CBT)
품사에 따른 언어 모델 성능 측정 dataset
![image](https://user-images.githubusercontent.com/41243762/103395008-41a94f00-4b6f-11eb-9d1c-c56d0ae163a9.png)

#### ◼ Winograd Schema Challenge
텍스트의 중의성을 해석하는 능력을 측정하여, 일반상식 추론능력을 평가한다.
![image](https://user-images.githubusercontent.com/41243762/103395104-b3819880-4b6f-11eb-8890-c0baaab0bd22.png)

##### 나머지 실험들은 생략.



<br><br><br>

---
### 참고
- paper : [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [github](https://github.com/openai/gpt-2) | [Jeong Ukjae's Blog](https://jeongukjae.github.io/posts/1-gpt2-revivew/) | [Jay Alammar's blog](http://jalammar.github.io/illustrated-gpt2/?fbclid=IwAR2-9C2kKU-mObfA89Th47SqsA2kniJUkdXvjwtzK14DvnwB_iApYaIwsP0) | [Gorio Learning blog](https://greeksharifa.github.io/nlp(natural%20language%20processing)%20/%20rnns/2019/08/28/OpenAI-GPT-2-Language-Models-are-Unsupervised-Multitask-Learners/)
