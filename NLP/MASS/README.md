# MASS: Masked Sequence to Sequence Pre-training for Language Generation `2019`


## 1. Abstract
Language generation분야에서 데이터가 적을때도 효과적으로 pre-train 할 수 있는 방법을 제시한다. (unsupervised-learning)
> - Encoder-Decoder 형태로 sequence를 적절하게 masking하여 문장의 뜻을 잘 학습하도록 한다.

## 2. METHOD
  ### 2.1 모델 구조 및 학습방법
   #### - ***기본 모델 구조***

   ![image](https://user-images.githubusercontent.com/41942097/118096856-e173c500-b40c-11eb-9fbb-4648fd7d0f2d.png)
   > _는 mask된 token을 의미

<br>
<br>

   #### - ***loss function***

   ![image](https://user-images.githubusercontent.com/41942097/118097094-2d266e80-b40d-11eb-9c4a-fabd9f449938.png)
   > - P의 경우 mask된 input에 대하여 올바른 token이 나올 확률을 의미
   > - likelihood로 정의됨. 이후는 chian rule에 따라 정리

<br>
<br>

   #### - ***BERT, Standard language modeling(GPT)와의 비교를 통한 예시***

   ![image](https://user-images.githubusercontent.com/41942097/118098335-bab68e00-b40e-11eb-99a7-020d041a93d4.png)
   > - BERT의 경우 k 즉, 마스크 할 token의 개수를 1로 설정한 것과 같다. (이때, BERT의 다중 masking 기법은 bidirectional하게 학습하기 위한 조건이 아닌 빠른 학습을 위한 조건)
   > - GPT의 경우 k를 m 즉, 전체 sequence token의 길이로 설정한 것과 같다.

<br>
<br>

  ### 2.2 Contribution 및 분석

  - BERT의 경우 encoder, decoder를 따로 학습시켜 language understanding에는 적합하나 language generation에는 부적합  

  - MASS는 encoder, decoder를 같이 학습시킴으로써 3가지 효과를 본다
     * sequence to sequence framework으로 하여금 mask된 토큰만 예측하게 함으로써 encoder가 unmask된 token의 의미를 학습하도록 하고, decoder가 encoder에서 더 유용한 정보를 얻을 수 있게 한다.
    
     * decoder가 연속된 mask된 토큰을 예측하게 함으로써 언어적으로 더 유용한 모델을 학습 가능하다.
   
     * encoder에서 unmask된 부분만 decoder에서 mask하여 input으로 활용함으로 input 토큰의 의존도를 줄인다.

<br>
<br>

## 3. Experiment
 > encoder 6개, decoder 6개, embedding 1024 로 English-French, English-German, English-Romanian 수행(Pre-train)

### 3.1 NMT

![image](https://user-images.githubusercontent.com/41942097/118116291-112ec700-b425-11eb-89b4-bf60a7072ad6.png)

<br>
<br>

### 3.2 Text Summarization

![image](https://user-images.githubusercontent.com/41942097/118116676-84d0d400-b425-11eb-8c14-753cc161174d.png)

<br>
<br>

### 3.3 Conversational Response Generation

![image](https://user-images.githubusercontent.com/41942097/118116797-b0ec5500-b425-11eb-99c2-2229a4f1839f.png)

<br>
<br>

### 3.4 k의 의미

![image](https://user-images.githubusercontent.com/41942097/118116886-d2e5d780-b425-11eb-85c1-b01b442462e9.png)
