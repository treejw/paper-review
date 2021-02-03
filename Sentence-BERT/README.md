# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks `2019.8`


## Abstract
[BERT]

- BERT와 RoBERTa는 Semantic Textual Similarity(STS) 같은 sentence pair regression task에서 SOTA를 찍었다.
    
        STS : '텍스트 의미유사도평가'
        sentence pair regression task : 두 쌍의 문장의 연관성을 다루는 task

- 하지만, 두문장이 네트워크를 거쳐야 했기에 연산량이 컸고, 그로인해 시간이 오래걸린다는 단점이 있었다. 

<br>


두 문장을 함께 input하여 계산하는 것은 

문장의 의미유사도를 추출하는 task와 unsupervised tasks에는 적절하지 않으며

이에 따라, BERT의 pretraining 네트워크를 바꿔야 한다고 주장한다.

<br>

[Sentence-BERT]
- pretraining 부분에서 **siamese** and **triplet network** 구조를 가지는 **Sentence-BERT**를 제안

siamse network : 두 개의 input을 각각 하나의 네트워크에 집어 넣는 것
        
    이러한 구조는 
        
        효과적인 sentence embedding 결과를 가져옴.
        코사인 유사도로 계산될 수 있음.
        정확도는 유지한 채로 걸리는 시간 단축.
        :만개의 문장 중 가장 유사한 문장찾기에 걸리는 시간 (65h > 5s) 
        
- SBERT는 STS에서 SOTA의 성과를 냈으며 다른 transfer learning tasks들의 sentence embedding 방식들보다 뛰어남.
    


<br>

## Introduction

- Sentence-BERT : Siamese & triple network를 사용하여 BERT를 수정한 것
이를 통해 semantic(의미있는)한 sentence embedding을 할 수 있다.

- 기존에 BERT로는 적용할 수 없던 task에서도 사용가능함.
(ex.large-scale semantic similarity comparison, clustering, and information retrieval)

- 기존의 BERT는 cross-encoder를 사용했었음.
   
   cross-encoder : context(sentence A)와 candidate(sentence B)을 결합하여 encoder를 거칠 때, context 다음에 올 문장인지 계산하는 방식.
   (pair regression task에서 너무 많은 조합들이 생겨 정확도는 높지만 너무 오래걸림. 부적절함.)

- BERT의 embedding방식과 CLS토큰을 사용하는 것은 오히려 안 좋은 결과를 만들기에 SBERT를 만들게 되었다.


<br>


## 3. Model

SBERT는 각각의 문장을 기존의 Bert에 넣어주고
나온 output에 추가로 pooling작업을 해여 embedding 사이즈를 정해준 후,
각 task에 맞는 objective Function을 거친다. 

아래와 같은 구조를 가짐.

![image](https://user-images.githubusercontent.com/43063980/106695395-d810da80-661d-11eb-8324-01fcf153bdc4.png)

### [Pooling]

SBERT는 BERT의 출력에 Pooling 작업을 추가하여 고정된 크기의 sentence embedding을 만듦.

이때 pooling방식은 3가지가 있었음.
CLS-token을 출력하는 방식과 Mean, Max pooling으로 output vector를 처리해주는 방식 중 
**Mean pooling**이 가장 좋은 결과를 내서 default로 사용하고 있다.

<br>


### [Objective Functions]
각각 task마다 특성이 다르기 때문에 가장 Objective Function을 사용하고자 함.
Objective Function은 Pretraining 과정에서 사용되며, 각각의 loss는 fine-tunning 과정에서 사용되는 loss를 말함.

![image](https://user-images.githubusercontent.com/43063980/106693905-cc6fe480-661a-11eb-8033-e3476cdcaf7b.png)

- **Classification Objective Function**

  - element-wise difference(벡터차의 크기 = 거리)로 두 벡터를 concat 해줌.
  - |u−v|에 Weight를 곱해준다. softmax를 통해 얼마나 유사한지 확률로 표현.
  이때 n : 임베딩사이즈, k : class 개수(왜 3인지는 모르겠음)
  - cross-entropy loss 사용

![image](https://user-images.githubusercontent.com/43063980/106696001-1fe43180-661f-11eb-850b-6f25f9b1765e.png)

- **Regression Objective Function**

  - 두 문장의 임베딩벡터 U, V 간 코사인유사도 계산
  - MSE loss 사용
  
![image](https://user-images.githubusercontent.com/43063980/106696070-3d190000-661f-11eb-908a-78b65c9c0132.png)

- **Triplet Objective Function**

  - Anchor sentence : a / positive sentence : p / negative sentence : n 라고 할 때
  - a와 p는 가까워지고, a와 n는 멀어지도록 하는 Triplet loss이며, 이 둘의 차이는 1로 default함.
  - 유클리드 거리측정법을 사용함.
  - 아래의 loss function를 minimize되도록 학습시킴.
  
![image](https://user-images.githubusercontent.com/43063980/106696022-2a063000-661f-11eb-94b9-5a316e05cb7e.png)
  



<br>


### 3.1 Training Details

- SBERT는 SNLI, Multi-Genre NLI의 두 dataset을 조합하여 fine-tuning시켰으며
SNLI : 57만개의 sentence pairs (문장관계에 따라 contradiction, entailment, neutral로 라벨링됨.)
MNLI : 43만개의 sentence pairs (여러장르의 spoken and written text)
- 한 epoch당 3분류 softmax classifier objective funtion 사용함.
- Batch size = 16 / Adam optimizer / Learning rate = 2e-5 / Linear learning rate warm-up over 10% of training data / Pooling strategy = MEAN

<br>


## 4 Evaluation - Semantic Textual Similarity

### 4.1 Unsupervised STS
- STS를 fine tunning 하지 않았을 때


각각 2012,13,14,15,16년도 STS task, STS benchmark, SICK 관련 datast사용하여 평가함.
각 모델로부터 나온 sentence embedding으로 구한 코사인 유사도와 실제 라벨의 값과 비교해 점수 냄.
두 문장의 유사도를 0~5의 레이블로 제공

![image](https://user-images.githubusercontent.com/43063980/106701676-61c6a500-662a-11eb-9696-790c5901f1a6.png)

- 대부분 SBERT의 성능이 좋은 걸 알 수 있음.

### 4.2 Supervised STS

- STS를 fine tunning 했을 때
![image](https://user-images.githubusercontent.com/43063980/106701701-6db26700-662a-11eb-8404-93ed8cec58b6.png)
- 하기 전보다 더 향상된 걸 볼 수 있음.


### 4.3 Argument Facet Similarity
![image](https://user-images.githubusercontent.com/43063980/106701717-773bcf00-662a-11eb-86a6-faef496b2739.png)

### 4.4 Wikipedia Sections Distinction
![image](https://user-images.githubusercontent.com/43063980/106701726-7d31b000-662a-11eb-9654-dcb89fc8e66e.png)

<br>


## 5 Evaluation - SentEval
SentEval : sentence embedding의 품질을 평가하기 위한 라이브러리

![image](https://user-images.githubusercontent.com/43063980/106703613-eff05a80-662d-11eb-8d82-fa75d547d013.png)

- SBERT의 sentence embedding의 성능이 더 좋음을 알 수 있다.

<br>


## 6 Ablation Study
- 여러가지의 Pooling, concat방식 중, 각각 MEAN, |u−v|의 방식이 가장 잘나왔다.

![image](https://user-images.githubusercontent.com/43063980/106703940-94729c80-662e-11eb-9754-dda5b49323c4.png)



### 참고
- paper : [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf)
- 블로그 : https://roomylee.github.io/sentence-bert/ | https://blog.naver.com/djtnrud123/222031646168
- 영상 : https://www.youtube.com/watch?v=izCeQOOuZpY&list=LL&index=33
