# You Only Learn One Representation: Unified Network for Multiple Tasks `2021`

## 1. Abstract
하나의 네트워크를 통하여 다양한 task를 수행할 수 있도록 하는 학습방법을 제안한다.
> - 함축적, 직관적 학습 데이터를 한번에 학습할 수 있는 모델을 제안한다.

<br><br>

## 2. 모델 개요

<p align="center"><img src="https://user-images.githubusercontent.com/41942097/124111304-1a8ef400-daa4-11eb-9897-cefdf0b156ca.png"></p>

기본 모델은 yolo v4를 기반으로 한다.

<br><br>

## 3. 함축적 지식 학습 기법

### 3.1 Manifold space reduction
<p align="center"><img src="https://user-images.githubusercontent.com/41942097/124111869-b456a100-daa4-11eb-89f2-358d8f81186b.png"></p>

> 대표성이 잘 나타난 데이터의 예시

- 그림처럼 대표성이 잘 표현된다면 사영벡터의 내적을 취함으로써 함축적 벡터의 대표 차원을 줄일 수 있도록 한다. 이로인해 더욱 효율적으로 다양한 task에 적용시킬 수 있다. 

<br>
### 3.2 Kernel space alignment

- multi task와 multi-head neural networks에서는 kernel space가 일치하지 않는 문제가 발생한다.

<p align="center"><img src="https://user-images.githubusercontent.com/41942097/124113014-e6b4ce00-daa5-11eb-948d-73c6c07793ff.png"></p>

- 이는 FPN(Feature Pyramid Network)와 같은 scale 별로 object들이 나타는 모델에서 발생하는 문제이다.
- 이를 해결하기 위하여 output featrue과 함축적 대표 벡터에 추가로 곱셈을 함으로써 해결한다.

<br>

### 3.3 More functions

<p align="center"><img src="https://user-images.githubusercontent.com/41942097/124113802-cf2a1500-daa6-11eb-867b-26da398b843c.png"></p>

- 사진과 같은 함수에도 함축적 대표 벡터는 추가될 수 있다. 
> - 예시 : NN으로 하여금 중심 좌표의 offset을 추가하여 예측하게 하거나, 자동으로 anchor의 hyperparameter를 예측할 수 있도록 한다.

<br><br>

## 4. 제안된 모델에서의 내재적 지식

### 4.1 함축적 지식의 함수화

<p align="center"><img src="https://user-images.githubusercontent.com/41942097/124114638-c38b1e00-daa7-11eb-94e0-de8e70823851.png"></p>

> - x : 관측값, θ : 모델 파라미터, fθ : 모델, e: 에러, y: 특정 task의 target 값

<p align="center"><img src="https://user-images.githubusercontent.com/41942097/124115344-a73bb100-daa8-11eb-8834-8333463e7d9b.png"></p>

- (a) 의 그림처럼 e를 최소화 한다면 현재의 task에서만 가능하다는 단점이 있다.
- (b) 의 그림처럼 e를 재배열하여 공간을 줄이게 된다면 여러가지 task를 같은 manifold space에서 계산이 가능하게 하지만, 이는 고전적인 계산방법을 쓸수없다. ( Euclidean distance, maximum value of one-hot vector)
- 따라서 (c)의 그림과 같이 e를 모델링 해야한다. 

#### 4.1.1 unified networks

<p align="center"><img src="https://user-images.githubusercontent.com/41942097/124116671-2b426880-daaa-11eb-9eee-5c27c23a370c.png"></p>

- e에 함축적, 직관적 지식을 같이 넣을 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/41942097/124116716-39908480-daaa-11eb-8037-6edc822d532f.png"></p>

- (2)의 식을 미분하면 다믕과 같은 식을 얻을 수 있다. 이때 fθ(x) 는 전체 대표성 백터을 의미하고,  gΦ(z)는 task 별 대표성 벡터를 의미한다. 
- dΨ 는 task별 완전히 다른 discriminator를 의미한다.

<br><br>

## 5. Experiments

![image](https://user-images.githubusercontent.com/41942097/124118887-cf2d1380-daac-11eb-9b07-091d1f77a228.png)
---
![image](https://user-images.githubusercontent.com/41942097/124119024-f1bf2c80-daac-11eb-81de-31ec08067098.png)
---

![image](https://user-images.githubusercontent.com/41942097/124118911-d522f480-daac-11eb-8c6f-e47b351a7fea.png)



