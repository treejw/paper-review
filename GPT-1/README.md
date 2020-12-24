 
# Improving Language Understanding by Generative Pre-Training (GPT-1) `2018.6`

## Abstract

- `unlabeling된 데이터`는 많지만, 특정 task를 학습시키기 위한 `labeling된 데이터`는 부족한 경우가 많음.

- **이 논문에서는,**
   - Step 1) `unlabeling된 데이터`를 language model로 pre-training 진행.
   - Step 2) 특정 task에 대한 fine-tuning 진행.

<br>

## 1. Introduction
- **원본 그대로의 텍스트를 활용하여 비지도 학습을 통해 학습하는 모델을 만들기 어려운 이유 2가지 (pre-training)**

   1. 어떤 `optimization objective`가 전이학습에 효과적인 representation을 배우는데에 효과적인지 알 수 없다.
   2. 모델에서 학습된 representation을 `다른 NLP task로 transfer하는데 가장 효율적인 방법`이 정해지지 않았다.

- 이 논문은 pre-training 시 위와 같은 어려운 문제들에 대해 새로운 방안을 제시.

<br>

## 3. Framework

### 3.1. Unsupervised pre-training

- unsupervised corpus tokens U={u<sub>1</sub>,…,u<sub>n</sub>}가 주어지면, 아래의 likelihood를 최대화하는 standard language modeling objective가 적용. (아래는 pre-training의 목적함수)

   > <img height="45;" src="https://user-images.githubusercontent.com/42428487/103034914-63bc3380-45a9-11eb-9ee3-817250bbc174.png">  ... 식(1) 

   즉, k(context windows size) 만큼의 unsupervised corpus tokens을 이용해서 현재 token u<sub>i</sub>를 예측하는 것

<br>

- 모델은 **multi-layer Transformer Decoder**를 사용하여 Language Modeling에 활용.

   > <img height="80;" src="https://user-images.githubusercontent.com/42428487/103034893-543cea80-45a9-11eb-8c36-0ed0ce2bbe7f.png"> ... 식(2)

   > - **U**: input context token
   > - **W<sub>e</sub>**: token embedding
   > - **W<sub>p</sub>**: positional embedding
   > - **n**: layer 개수



<br>

### 3.2. Supervised fine-tuning
- 식(1)로 모델을 학습한 후, supervised target task에 대해 parameters를 조정.
   - softmax를 통해 input tokens *x<sub>1</sub>, ... , x<sub>m</sub>*에 해당하는 label *y*를 예측하는 방식
      > <img height="35;" src="https://user-images.githubusercontent.com/42428487/103034961-7c2c4e00-45a9-11eb-8ec8-61d9125c3dfe.png"> ... 식(3)
   
   - P(U)`(pre-training의 결과)`를 입력으로 하는 linear layer를 추가. (아래는 위 task의 목적함수, C는 labeled dataset)
      > <img height="60;" src="https://user-images.githubusercontent.com/42428487/103036503-412c1980-45ad-11eb-84b8-25b18c83f794.png"> ... 식(4)

<br>

- 또한, fine-tuning 목적함수에 auxiliary objective(보조 목표함수)로 language modeling을 포함하면, 아래와 같은 장점이 있음.
   1. improving generalization of the supervised model
   2. accelerating convergence
      > <img height="30;" src="https://user-images.githubusercontent.com/42428487/103036613-84868800-45ad-11eb-9025-45393f122747.png"> ... 식(5)

<br>

### 3.3. Task-specific input transformations

![image](https://user-images.githubusercontent.com/42428487/103037323-e85d8080-45ae-11eb-9851-64f75aba51c1.png)

<p align="center"> ▲ <strong>(left)</strong> Transformer architecture and training objectives used in this work. <strong>(right)</strong> Input transformations for fine-tuning on different tasks </p>

<br>

- **Text Classification**
   - 위 그림처럼 바로 fine-tuning 가능.

<br>

- **Textual entailment (언어적 인과관계 유추)**
   - 전제(premise)와 가설(hypothesis) 토큰 시퀀스를 연결하고 그 사이에 구분 토큰 사용

<br>

- **Similarity**
   - 유사성 task의 경우, 두 문장은 고유한 순서가 없기 때문에, 구분 토큰을 포함하여 2개의 입력 시퀀스를 2가지 방식으로 연결. 
   
     각각을 독립적으로 처리한 후, element-wise로 더한 후 linear layer로 보냄.

<br>

- ** Question Answering(질의 응답) and Commonsense Reasoning(상식 추론)**
   - `Context text Z`, `질문 q`, `가능한 답변 set {a_k}`가 주어진다면,
   - 구분 토큰`($)`을 포함하여 `Context`와 `질문`에 `가능한 각 답변`과 연결. → [z; q; $; ak]
   - 각각 linear layer을 통과한 값들은 softmax layer를 통해 정규화되어, 가능한 답변에 대한 출력 분포 생성. 
   
<br>

## 4. Experiments

### ▪ GPT-1 vs. SOTA
#### Table 2: natural language inference 

<img height="200;" src="https://user-images.githubusercontent.com/42428487/103039209-9703c000-45b3-11eb-9569-0ccca5d5ec8a.png">


#### Table 3: question answering and commonsense reasoning tasks

<img height="160;" src="https://user-images.githubusercontent.com/42428487/103039395-29a45f00-45b4-11eb-9073-870091ae1390.png">


#### Table 4: Semantic similarity and classification tasks

<img height="240;" src="https://user-images.githubusercontent.com/42428487/103039536-81db6100-45b4-11eb-8f0f-37be1cc9fa16.png">


<br>


## 5. Analysis

#### Table 5: Analysis of various model ablations on different tasks (mc= Mathews correlation, acc=Accuracy, pc=Pearson correlation)
<img height="180;" src="https://user-images.githubusercontent.com/42428487/103039664-c1a24880-45b4-11eb-95b8-84d0359ad9cd.png">

- **Transformer w/o pre-training**: pre-training 없이, transformer 모델을 바로 supervised target tasks에 대해 학습
- **Transformer w/o aux LM**: fine-tuning시 auxiliary LM objective 사용 안함.
- **LSTM w/ aux LM**: Transformer 대신 LSTM 사용.


<br><br>

---
### 참고
- paper : [Improving Language Understanding by Generative Pre-Training (GPT-1)](http://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Young-Geun Kim's Blog](https://medium.com/@eyfydsyd97/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-improving-language-understanding-by-generative-pre-training-gpt1-c65bed865990) | [Jeong Ukjae's Blog](https://jeongukjae.github.io/posts/gpt-review/) | [hipgyung's Blog](https://hipgyung.tistory.com/24)

