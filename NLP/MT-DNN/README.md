# Multi-Task Deep Neural Networks for Natural Language Understanding `ACL 2019`

## Abstract

- **BERT**와 다양한 NLU task를 학습하는 **Multi-task learning** 통해 기존 BERT의 성능을 능가하는 MT-DNN 모델 제안. (NLU: Natural Language Understanding)

- 다양한 task의 많은 데이터를 이용하여 보다 더 general representation을 학습하도록 함.    
   > 이러한 결과는 새로운 task와 domain에 적용하는데 도움이 됨.

- NLU의 10개 task, GLUE의 8개 task에 대해 SOTA 달성.

- SNLI와 SciTail datasets을 이용한 실험을 통해 domain 내 label 수가 적은 데이터에 대해서 MT-DNN이 BERT보다 domain adaptation 에도 성능이 좋음을 입증함. 

<br><br>

## 1. Introduction

#### Representation을 배우는 방법
> - Multi-task learning
> - Language model pre-training
- 이 논문에서는 위 2가지 방법을 잘 융합한 모델을 제안. 

<br>

#### MTL(multi-task learning)의 효과 및 필요성
- MTL은 이전 task에서 배운 지식을 적용하여 새로운 task를 학습하는 데 도움이 되는 사람의 학습 활동에서 영감을 받음.
   > "사람이 스키타는 법"을 배운다고 하자. 
   > 이때 스케이팅을 타본 사람과 아닌 사람 두 명을 비교해보면, 스케이팅 탄 사람이 배우기 쉽다 

- 또한, Supervised 학습은 많은 양의 task-specifc labeled data를 필요로 함.

- MLT은 특정 task에 overfitting 되어 학습되는 것을 regularization하는 효과가 있음. 이를 통해 universal representation 학습이 가능하게 함.

<br>

#### BERT vs MT-DNN
- BERT와 유사한 점 : pre-training과 fine-tunning 두 가지 step으로 학습
- BERT와 다른 점 : fine-tunning할 때, MTL 방법을 사용.

<br><br>


## 2. Tasks

- MT-DNN은 GLUE의 9개의 task를 MTL에 활용. 
- 이 task들을 4가지 분류로 나누어 설명.

#### Single-sentence classification
- 하나의 문장이 주어졌을 때 문장의 class를 분류하는 Task
- Task
   - CoLA - 문장이 문법적으로 맞는지 분류 (True / False)
   - SST-2 - 영화 리뷰 문장의 감정 분류 (Positive / Negative)

#### Text Similarity
- 문장 쌍이 주어졌을 때, 점수를 예측하는 Regression Task
- Task
   - STS-B - 문장 간의 의미적 유사도를 점수로 예측

#### Pairwise Text Classification
- 문장 쌍이 주어졌을 때, 문장의 관계를 분류하는 Task
- Task
   - RTE, MNLI - 문장 간의 의미적 관계를 3가지로 분류 (Entailment / Contradiction / Neutral)
   - QQP, MRPC - 문장 간 의미가 같음 여부를 분류 (True / False)

#### Relevance Ranking
- 질문 문장과 지문이 주어졌을 때, 지문 중 정답이 있는 문장을 Ranking을 통해 찾는 Task
- Task
   - QNLI - 질문과 해당 지문 중 한 문장이 쌍으로 주어졌을 때 해당 지문 문장에 질문의 답이 있는지 여부를 분류 (True/False)
      > MT-DNN에서는 이를 pairwise-ranking task 방식으로 처리 (모든 지문 문장에 정답이 있을 가능성을 Scoring 하여 가장 Score가 높은 지문 문장만을 True로 분류하는 방식)

<br><br>

## 3. The proposed MT-DNN Model
### MT-DNN 모델 구조

![image](https://user-images.githubusercontent.com/42428487/110679375-c1f8ca00-821a-11eb-8b44-6a55a748c5c1.png)

- **Shared layers**와 **Task specific layers**로 나뉨
- Shared layers의 경우, BERT처럼 Transformer encoder 사용

- 모델 학습시, 무작위로 특정 Task의 Data를 Batch로 뽑아서 학습
   - 이때, shared layers는 계속 학습
   - Task-specific layers는 해당 Task의 Data로 학습시에만 학습 

- Task-specific layer는 각각 1개의 layer로, 각 Task-specific layer마다 적합한 Loss 함수로 구성

<br>

### Lexicon Encoder
- (BERT와 동일)
- Token Embeddings, Segment Embeddings, Position Embeddings -> 3가지 Embedding을 통해 Input 구성
- 문장 맨 앞에 [CLS] 토큰 추가, 문장 쌍인 경우 문장 사이에 [SEP] 토큰 추가

<br>

### Transformer Encoder
- Pre-training을 하는 부분
- (BERT와 다른 점) fine-tunning 할 때, task 별로 fine-tunning을 하지 않고 MTL로 fine-tunning 하게 된다.

<br>

### _Single-Sentence Classification Output_
- Eq.1
   - <img height="150" src="https://y-rok.github.io/assets/img/Untitled-b505ff79-fd58-441a-9cc2-7b9194c4820b.png">

<details><summary> BERT output 중 [CLS] 토큰을 사용</summary>

![image](https://user-images.githubusercontent.com/42428487/110680873-7e9f5b00-821c-11eb-884e-870b47692aff.png)

</details>

- [CLS] 토큰과 Task specific parameter의 곱에 Softmax를 취하여 Output을 추출

<br>

### _Text Similarity Output_
- Eq.2
   - <img height="150" src="https://y-rok.github.io/assets/img/Untitled-6ed73731-4502-40fa-83e8-b84ff65c7325.png">

- BERT output 중 [CLS] 토큰을 사용
- Regression Task이기 때문에,  토큰과 Task specific parameter의 곱에 Sigmoid를 사용하여 score 예측

<br>

### _Pairwise Text Classification Output_
- 두 문장의 의미 관계를 분류하는 작업

- 이 논문에서는 BERT와 다르게 [SAN(Stochastic Answer Network)](https://arxiv.org/pdf/1804.07888.pdf)를 이용함.
   - BERT의 경우, linear layer + softmax 구조 이용

<br>

-  **SAN(Stochastic Answer Network)**
   -  _( 이 논문(MT-DNN) 저자가 전에 발표한 모델 )_
   - NLI(natural language inference) task의 기존 SOTA 네트워크

   - 이 네트워크를 사용한 이유 (예 - MNLI dataset)
      ```
        If you need this book, it is probably too late unless you are about to take an SAT or GRE.
        (이 책이 필요하다면, SAT나 GRE를 받으려고 하지 않는 한 너무 늦을 것이다.)

        It’s never too late, unless you’re about to take a test.
         (시험을 치러야 할 때가 아니라면, 결코 늦지 않다.
      ```
   
      - 두 문장의 관계를 파악하려면 적어도 2단계 추론이 필요함. <br>
         (SAT와 GRE가 test인것을 유추, 두 문장이 비슷한지 여부를 판단)

   - SAN : 주어진 문장들에 대한 **Multi-step Reasoning**을 모델링하는 구조

      <p align="center"><img height="300" src="https://y-rok.github.io/assets/img/2019-05-19-18-10-16.png"></p>

      - M_h, M_p : Hypothiesis 문장과 Premise 문장을 [CLS] H [SEP] P [SEP] 형태로 BERT에 입력으로 넣어 얻은  H와 P 각각에 속하는 Token들의 Token Vector. 

      - input x : 이전 Hidden State(Hypothesis 문장의 Embedding)를 고려한 Premise 문장의 Embedding Vector
      - hidden state : Input값(Premise 문장 Embedding)을 고려하여 변형한 Hypothesis 문장의 Embedding

      <br>
      
      - Eq.3
         - <img height="150" src="https://y-rok.github.io/assets/img/2019-05-19-18-23-14.png">
         - 각 time step 마다 문장 간 관계를 고려하여 classification 예측을 함.
      - Eq.4
         - ![image](https://user-images.githubusercontent.com/42428487/110741347-da4b0200-8277-11eb-968b-c8a9404a0fe0.png)  
         - 위 과정을 k번 반복했다면 (k-step reasoning), k번 결과의 평균 값을 통해 최종 결과 예측.

<br>

### Relevance Ranking Output
- Eq.5
   - <img height="150" src="https://y-rok.github.io/assets/img/2019-05-19-18-42-32.png">
- Question과 Sentence Pair를 Input으로 넣어 생성한 [CLS] Token에 Sigmoid 취한 후,
- Sentence 별로 score를 구하고, 가장 높은 score인 Sentence만 Question에 해당하는 정답이 있다고 예측하는 방식으로 Classification 수행.

<br><br>

### 3.1 The Training Procedure

#### The training procedure of MT-DNNN consists of two stages:
1. Lexicon encoder과 Transformer encoder의 unsupervised 학습 (BERT 방식)
   - Masked language modeling
   - Next sentence prediction

2. MTL 과정을 SGD로 학습. 
   - 모든 shared layers와 task-specific layers의 parameters 학습
   - 모든 task(9개의 GLUE)의 datasets를 모아두고(D), 매 epoch마다 mini-batch(b_t)를 선택하는 방식.

   ![image](https://user-images.githubusercontent.com/42428487/110738890-69095000-8273-11eb-8146-f5bb4b1a86c1.png)

- Eq.6 (**classification tasks** - objective)

   ![image](https://user-images.githubusercontent.com/42428487/110739048-beddf800-8273-11eb-8243-3d04bd4addfb.png)
   - tasks : single-sentence or pairwise text classification
   - (참고) [eq.1](https://github.com/treejw/temp/blob/main/README.md#single-sentence-classification-output)
   - **cross-entropy loss** 사용 

- Eq.7 (**similarity tasks** - objective)

   ![image](https://user-images.githubusercontent.com/42428487/110739095-d74e1280-8273-11eb-8a59-e5a9713c2dc3.png)
   - tasks : STS-B
   - (참고) [eq.2](https://github.com/treejw/temp/blob/main/README.md#text-similarity-output)
   - **MSE loss** 사용

- Eq.8 (**relevance ranking tasks** - objective)

   ![image](https://user-images.githubusercontent.com/42428487/110739127-e8971f00-8273-11eb-9494-da859bb0f1f7.png) 
   ![image](https://user-images.githubusercontent.com/42428487/110740518-8855ac80-8276-11eb-8d78-28ad20198b51.png)
   - tasks : QNLI
   - (참고) [eq.5](https://github.com/treejw/temp/blob/main/README.md#relevance-ranking-output)
   - Query가 주어졌을 때, 후보 A 리스트가 주어짐.
   - A = [A+, A-] // Positive sample A+ 1개, Negative examples A- (|A|-1)개
   - |A|개 sample에 대해 softmax 식을 통해 Negative loglikelihood loss를 정의
   - γ는 tuning factor로 본 논문에서는 1로 지정.


<br><br>

## 4. Experiments

- 3개의 NLU benchmark에서 MT-DNN 실험 진행.
   > GLUE, SNLI, SciTail
- GLUE는 BERT에 비해 MT-DNN의 MTL이 효과적임을 보여주는 데이터로 사용
- SNLI와 SciTail운 Domain adaptation에서의 MTL의 효과를 보여주는데 사용.

### 4.1 Datasets
![image](https://user-images.githubusercontent.com/42428487/110751648-80eacf00-8287-11eb-9932-f49cad3bbefd.png)

<br>

### 4.3 GLUE Main Results
#### ◼ GLUE datasets에서의 성능
![image](https://user-images.githubusercontent.com/42428487/110752221-50576500-8288-11eb-8b29-47272091fecb.png)


#### ◼ BERT_base vs Single-Task DNN vs Multi-Task DNN
![image](https://user-images.githubusercontent.com/42428487/110751909-dd4dee80-8287-11eb-8cff-7d08b6928958.png)

<br>

### 4.4 Domain Adaptation Results on SNLI and SciTail 
![image](https://user-images.githubusercontent.com/42428487/110752960-44b86e00-8289-11eb-9278-d4568a322a2a.png) ![image](https://user-images.githubusercontent.com/42428487/110752945-3ff3ba00-8289-11eb-8d2b-f80d6dff7667.png)

#### ◼ Results on SNLI & SciTail
![image](https://user-images.githubusercontent.com/42428487/110753846-6cf49c80-828a-11eb-90d8-e7ec1e51822e.png)


<br><br>

---
### 참고
- paper : [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf)
- [AI Information](https://ai-information.blogspot.com/2019/02/multi-task-deep-neural-networks-for.html) | [y-rok's blog](https://y-rok.github.io/nlp/2019/05/20/mt-dnn.html) 
- [Air's Big Data's Blog](https://airsbigdata.tistory.com/202)
