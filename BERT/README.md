# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding `2018.10`

## Abstract

- BERT 모델 제안.
   - 기존의 language representation model과는 달리(예를들어 [ELMo](https://github.com/treejw/Study_NLP/tree/main/Word_Embedding#elmoembeddings-from-language-model)), 
   - BERT는 unlabeled text에서 right, left context를 모두 고려하게 만드는 **deep bidirectional representations**을 pretrain하기 위해 설계됨.

- pre-trained BERT model에서 하나의 output layer을 더한 뒤 다양한 task에 대해 fine tuning 한 결과, SOTA 달성.


<br><br><br>


## 1. Introduction

### ◼ pre-trained language representations을 downstream task에 적용하는 방법 2가지

1. **Feature-based**
   - pre-trained representation을 추가적인 feature 정도로만 활용을 하고, task-specific architecture를 사용하는 방식. (예) [ELMo](https://github.com/treejw/Study_NLP/tree/main/Word_Embedding#elmoembeddings-from-language-model)

2. **fine-tuning**
   - task-specific한 parameters를 최대한 줄이고, pretrained parameters를 fine-tuning하기만 하는 방법 (예) [GPT-1](https://github.com/treejw/Study_NLP/tree/main/GPT-1)

#### ▪ 기존 모델) ELMo와 GPT 모두 unidirectional LM을 사용.
- GPT 같은 경우,  left-to-right architecture이기 때문에 Transformer의 self-attention layer에서 모든 token이 이전의 token들만 활용할 수 있음

#### ▪ BERT) unidirectionality constraint를 masked LM을 사용하면서 해결
-  masked ML 이란
   - 입력 문장에서 임의로 특정 토큰을 가리고 해당 문장만을 주고 가려진 토큰를 추측하게 만드는 방법.
   - 기존의 LM과는 다르게 양쪽의 context를 전부 활용할 수 있다.
- pre-trained 시, masked LM task와 next sentence prediction task로 학습.


<br>

### ◼ 이 논문의 contributions
- **language representations에 대한 bidirectional pre-training의 중요성을 보여줌.**
   - pre-training에 unidirectional language models을 사용하는 [GPT-1](https://github.com/treejw/Study_NLP/tree/main/GPT-1)과 달리, 
   - BERT는 **masked language models**을 사용하여 pretrained deep bidirectional representations을 가능하게 함.

- **pre-trained representations이 많은 task-specific architecture들의 필요성을 줄여준다는 것을 보여줌.**
   - BERT는  기존의 많은 task-specific architecture보다 좋은 성능을 보이는 최초의 fine-tuning 기반 representation model임.

- **BERT는 11개의 NLP task들에서 SOTA 달성함.**


<br><br><br>


## 3. BERT
<p align="center"><img height="300;" src="https://user-images.githubusercontent.com/42428487/103376755-dfc1f880-4b20-11eb-9dc4-9bc1456ee978.png">
<p align="center"><strong> ▲ Overall pre-training and fine-tuning procedures for BERT </strong></p>

#### ◼ BERT 학습 STEP
1. **pre-training**: unlabeled data으로 학습
2. **fine-tuning**: pre-training에서 학습한 모델 parameter로 초기화한 다음에 labeled data로 다시 학습

<br>

#### ◼ Model Architecture

- Multi layer bidirectional Transformer encoder 사용. (참고) [Transformer](https://github.com/treejw/Study_NLP/tree/main/Transformer(Attention%20is%20all%20you%20need))

- **BERT<sub>BASE</sub>** 와 **BERT<sub>LARGE</sub>** -> 사이즈가 다른 2개의 모델 발표.
   - **BERT<sub>BASE</sub>**: (L=12, H=768, A=12, Total Parameters=110M) 
   - **BERT<sub>LARGE</sub>**: (L=24, H=1024, A=16, Total Parameters=340M)

<br>

#### ◼ Input/Output Representations

<p align="center"><img height="250;" src="https://user-images.githubusercontent.com/42428487/103379608-8a3e1980-4b29-11eb-8e09-39bfdaa513b9.png"></p>

<p align="center"><strong> ▲ BERT input representation </strong></p>


- BERT에서는 30,000개의 단어로 [**WordPiece**](https://github.com/treejw/Study_NLP/tree/main/Tokenizer#-subword-tokenization-%EB%AC%B8%EC%9E%A5--subword) embedding을 사용

- **input embedding**은 token embedding, segmentation embeddings, position embeddings의 합. [(참고)](https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a)
   - token embedding: 단어의 벡터 표현
   - segmentation embeddings: 입력이 sentence pair일 때, 문장을 구분하기 위한 벡터 표현
   - position embeddings: 입력이 sentence pair일 때, 문장의 순서를 고려해주기 위한 벡터 표현


- 항상 첫번째 토큰은 `[CLS]`를 사용
   - 이 토큰에 해당하는 final hidden state는 classification task에서 사용할 수 있음.

- single sentence가 아닌, sentence pair는 special token(`[SEP]`)을 넣어 하나의 sequence로 만듬. 

<br><br>

### 3.1 Pre-training BERT
- 2가지 unsupervised task로 pre-training 진행.

#### ▪ Task #1: Masked LM

<img height="70;" src="https://user-images.githubusercontent.com/42428487/103381480-3d5d4180-4b2f-11eb-904d-7fe20bf6484c.png">

-  deep bidirectional representaion을 학습하기 위해, 각 sequence에서 15% 정도를 random으로 mask 씌운 뒤 원래 그 자리에 있던 token을 예측하는 방식으로 학습.

- 근데, pre-training 할 때는 항상 입력에 mask가 있고 fine-tuning을 할 때는 mask가 전혀 없기 때문에, 그 자리를 항상 mask로 바꾸는게 아니라 다른 단어로 채움 
   - Randomly 80% of tokens, gonna be a [MASK] token
   - Randomly 10% of tokens, gonna be a [RANDOM] token(another word)
   - Randomly 10% of tokens, will be remain as same. But need to be predicted.

###### 상세 학습과정) BERT 모델의 출력으로 sequence 길이만큼이 나오고 그 중에서 mask가 있는 자리의 출력값만 사용한다. fully connected layer를 사용해서 vocab 크기 만큼의 softmax 출력을 내보낸다.


<br>

#### ▪ Task #2: Next Sentence Prediction (NSP)

<img height="200;" src="https://user-images.githubusercontent.com/42428487/103381959-b610cd80-4b30-11eb-8c17-fa7f082fc8a8.png">

- 학습 방식
   - A, B 문장이 있을 때 50%의 확률로 B는 실제로 A의 다음 문장이도록 하고 IsNext로 레이블링. 
   - 나머지 50% 확률로 corpus에서 랜덤하게 아무 문장이나 들고 온 후 NotNext로 레이블링.
   
- 이 task를 진행하는 이유
   - pretrain model을 token-level의 task에만 활용할 것이 아니라 sentence-level에도 활용하기 위해서.
   - 이 task를 통해 BERT 모델은 문장 사이의 연관성에 대해서 학습할 수 있다.


<br>

#### ▪ Pre-training data
- 사용한 학습 데이터: BooksCorpus (800M words) , English Wikipedia (2,500M words)
- Masked LM loss와 Next sentence loss를 더해서 학습.



<br><br>

### 3.2. Fine-tuning BERT
![image](https://user-images.githubusercontent.com/42428487/103382694-418b5e00-4b33-11eb-804a-034b0a2e34d0.png)

<p align="center"><strong> ▲ Illustrations of Fine-tuning BERT on Different Tasks.</strong></p><br>


- BERT 모델은 token-level의 task에도 sentence-level의 task에도 활용 가능.

   - token-level task로는 question answering, Named entity recognition 등이 존재.
   
   - sentence-level task로는 sentence classification 등이 존재.
   
      > classification을 할 때는 맨 첫번째 자리의 transformer의 output을 활용


<br><br><br>

## 4. Experiments
![image](https://user-images.githubusercontent.com/42428487/103383077-bca14400-4b34-11eb-8531-c38dbda8b350.png)

###### 나머지 실험들은 생략.



<br><br><br>

---
### 참고
- paper : [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Jeong Ukjae's Blog](https://jeongukjae.github.io/posts/bert-review/) | [dnddnjs's Blog](https://dnddnjs.github.io/nlp/2019/05/08/BERT/) 
