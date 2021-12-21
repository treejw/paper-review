# CLIP: Learning Transferable Visual Models From Natural Language Supervision `2021.02`

## 1. Abstract
Image와 Text를 pair로 학습함으로써 Domain에 robustness가 높은 pretained model을 제안한다.
-> Zero Shot Transfer을 가능하게 하였다. 

<br><br>

## 2. Model
![image](https://user-images.githubusercontent.com/41942097/146925344-497d70b5-7fc0-48e3-bcbc-0270fa3c340c.png)

- (1) 이 pretraining 과정으로 (Text, Image)를 쌍으로 학습한다. 
- N의 배치에서 그림과 같이 N * N 의 (Text, Image)쌍을 예측하도록 학습힌다.
- 대각선에 대한 cosine 유사도의 경우 높아지도록 학습하고 나머지는 낮아지도록 학습한다. 

<br>

### 학습 코드(간략)

<center><img src="https://user-images.githubusercontent.com/41942097/146925983-588cbb2a-af6c-441b-9245-c59f8e15b67c.png" width="500" height="500"></center>

<br>

### Image Encoder
- ResNet50
  ResNet50-D의 global average pooling을 attention pooling으로 대체하였다.
  
- ViT (Vision Transformer)
  layer normalization을 추가하였다.
  
### Text Encoder
- Transformer를 사용하였다.

<br><br>

## 3. Experiments

[Experiment 참고자료](https://daeun-computer-uneasy.tistory.com/39?category=996323)

<br>

### 1) Zero Shot Transfer

* 파인튜닝(Fine tuning) : 다운스트림 태스크에 해당하는 데이터 전체(그림에서는 data2) 를 모두 사용
* 제로샷 러닝(Zero-shot learning) : 다운스트림 태스크의 데이터를 전혀 사용하지 않고 pretrain 모델로 다운스트림 태스크를 바로 수행
* 원샷 러닝(One-shot learning) : 다운스트림 태스크의 데이터를 한 건만 사용해 어떻게 수행되는지 참고한 뒤 바로 다운스트림 태스크 수행
* 퓨샷 러닝(Few-shot learning) : 다운스트림 태스크의 데이터를 몇 건만 사용하고, pre-train 모델을 몇개의 데이터에 맞게 업데이트

<center><img src="https://user-images.githubusercontent.com/41942097/146926680-6396e4e5-82b1-4633-b904-bae0797d4982.png" width="500" height="500"></center>

<br>

### 2) Representation Learning

<center><img src="https://user-images.githubusercontent.com/41942097/146926768-07281522-6ce6-46cc-9adf-8a80733b4b60.png" width="650" height="500"></center>

<br>

### 3) Domain shift & Domain Generalization
- Train set 및 Test set에서 Domain shift 존재
- 다른 테스크에 대한 학습이 없으므로 (Zero shot) 모든 Domain에 대한 Generalization 필요
-> Text와 Image를 쌍으로 학습하였기 때문에 Generalization 우수

<center><img src="https://user-images.githubusercontent.com/41942097/146927133-b8523c49-ae52-4503-9412-33ace1e0cd0d.png" width="800"></center>
<center><img src="https://user-images.githubusercontent.com/41942097/146927159-12e17c18-4939-49c3-8477-5e7f8194e978.png" width="700"></center>

