# ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS `ICLR 2020`

## 1. Abstract
기존 BERT처럼 input을 masking 하여 사용하는것은 효율적이려면 많은 연산량을 요구한다.
> - eg The chef cooked the meal -> [MASK] chef [MASK] the meal

따라서 해당 논문은, 
> - [MASK]토큰을 사용하는 대신 generator network를 통해 token을 적절한 대안으로 대체하여 input을 손상.
> - token의 ID를 예측하는 model대신 discriminator network를 통해 각 token이 generator sample로 대체되었는지 여부를 예측.

그 결과 기존의 MLM보다 효율적이며 동일한 model size, data, compute에서 BERT 및 XLNet과 같은 방법보다 실질적으로 우수함을 보여준다.

<br><br>

## 2. METHOD
### 2.1 모델 구조
<p align="center"><img src="https://user-images.githubusercontent.com/41942097/107421438-4c3b0900-6b5d-11eb-8e91-ff12d8ef2ec1.JPG"></p>

- Generator는 output token들의 분포를 나타내는 어떤 모델이든 상관없지만, 해당 논문에서는 작은 MLM모델을 사용함

> Generator, Discriminator 둘다 기본적으로 Transformer의 encoder를 


### 2.2 모델 수식
#### - ***Generator***

![image](https://user-images.githubusercontent.com/41942097/107425332-33812200-6b62-11eb-8a19-dd67d3eb86e8.JPG)
-  position t에 대해 generator는 softmax layer로 token x(t)를 generation할 확률을 출력함
- (x는 input token, h는 f contextualized vector representations을 의미함)

<br>

   #### - ***Discriminator***

   ![image](https://user-images.githubusercontent.com/41942097/107425348-37ad3f80-6b62-11eb-9881-cc68e460075d.JPG)
   - e는 token embedding. 주어진 position t에 대해 discriminator는 token x(t)가 “fake”인지, 즉 data distribution이 아닌 generator에서 나온것인지 예측

<br>

   #### - ***Loss Function***

   ![image](https://user-images.githubusercontent.com/41942097/107425359-3a0f9980-6b62-11eb-9440-da27710831d1.JPG)

   <center><img src="https://user-images.githubusercontent.com/41942097/107425355-3845d600-6b62-11eb-981e-d7186fef852f.JPG" width="450" height="50"></center>

   - 기존의 GAN과 다르게 Generator는 likelihood를 maximize하는 방법을 배움.(참고 [likelihood maximize](https://ko.wikipedia.org/wiki/%EC%B5%9C%EB%8C%80%EA%B0%80%EB%8A%A5%EB%8F%84_%EB%B0%A9%EB%B2%95))

<br>

   #### - ***기존 GAN loss***

   ![image](https://user-images.githubusercontent.com/41942097/107430051-07689f80-6b68-11eb-82d5-38c948595119.JPG)

   > 기존 GAN loss와 비교해보게 되면 Discriminator의 수식은 같지만 Generator의 output으로 학습되는것이 확인 가능하고, 

   > Generator는 기존과 다르게 likelihood maximization수식이 들어간것을 확인할 수 있음.

<br>

   #### - ***최종 loss***

   ![image](https://user-images.githubusercontent.com/41942097/107425368-3bd95d00-6b62-11eb-8b40-938c4f78e840.JPG)
   
   > 학습 후 Generator는 사용하지 않음. fine-tune시 Discriminator만 사용
   
<br><br>
## 3. Experiment 

> 기본적인 모델은 BERT의 파라미터를  그대로 사용함. 
GLUE 데이터셋은 ELECTRA에 간단한 linear classifier 추가, SQuAD는 XLNet의 questionanswering 모듈 ELECTRA에 추가(BERT에서 SQuAD 2.0을 위해 만들어진 classifier보다는 살짝 복잡함)

<br>

### 3.1 Model Extentions
#### ***Weight Sharing***

> Generator, Discriminator 크기가 똑같을 시 같은 weight를 공유하지만, 실험결과 Generator가 1/4 ~ 1/2 정도 작은게 더 효율적이므로 embedding(token, positional embedding)만 공유.
이때 Discriminator의 hidden state 사이즈로 embedding 사이즈 사용. 

#### ***training algorithm***

> two-stage, adversarial으로 학습시켜봤지만 성능 향상은 없었음.

![image](https://user-images.githubusercontent.com/41942097/107435127-14d55800-6b6f-11eb-8cb3-2560efc47424.JPG)

<br>

### 3.2 Small Models

![image](https://user-images.githubusercontent.com/41942097/107436449-e3f62280-6b70-11eb-8a16-f86fdf98ed55.JPG)

> sequence length (512 => 128), e batch size (256 => 128), hidden dimension size (768 => 256), token embeddings (768 => 128)

<br>

### 3.3 Large Models

![image](https://user-images.githubusercontent.com/41942097/107436839-6f6fb380-6b71-11eb-8639-893dc317b93a.JPG)

![image](https://user-images.githubusercontent.com/41942097/107436845-70a0e080-6b71-11eb-8278-3e88d66ed125.JPG)

![image](https://user-images.githubusercontent.com/41942097/107437262-0fc5d800-6b72-11eb-96d3-5ffc93f2a0d1.JPG)

> 400k steps (ELECTRA-400K; roughly 1/4 the pre-training compute of RoBERTa)
1.75M steps (ELECTRA-1.75M; similar compute to RoBERTa)

<br>

### 3.4  Efficientcy Analysis

![image](https://user-images.githubusercontent.com/41942097/107438014-37697000-6b73-11eb-9525-f7ff0d6343f6.JPG)


<br><br>

---
### 참고
- paper : [https://arxiv.org/pdf/2003.10555.pdf)
- [jeonsworld](https://jeonsworld.github.io/NLP/electra/)
