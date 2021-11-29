# Combination of Multiple Global Descriptors for Image Retrieval `(CGC)` `2020.04`

## 1. Abstract
- 최근 Image Retrieval 분야에서 여러 "모델의 앙상블(Ensemble)"이나, "Global Descriptor의 결합"을 통해 좋은 성능을 보여왔다.
   > Global Descriptor : SPoC, MAC, GeM 등을 의미함

   그러나 개별 모델을 따로 학습한 후 앙상블 하는 것은 Time과 Memory측면에서 비효율적이다.

- 본 논문은 앙상블 효과를 위해, Multiple Global Descriptor를 사용하며, End-to-End 학습하는 모델 CGD를 제안한다.
- CGD 의 장점
   - **유연성 및 확장성** : CNN backbone, Global Descriptor, Loss, Dataset에 대해서, 자유롭게 사용 가능 + End-to-End 학습
   - 결합된 Multiple Global Descriptor 성능확인: Single Descriptor 보다 더 **뛰어난 성능**을 보임
   - 여러 Image Retrieval 분야에서 **SOTA를 기록**

</br>

## 2. Introduction
- 기존의 많은 Image Retrieval 연구에서는 Convolution layer들 뒷 단에 FC(Fully connected) layer를 붙여서 이미지의 Dimension을 줄인 Global Descriptor로 사용해왔다.

- 이어 다양한 연구들이 진행되고 Global Descriptor에도 여러가지 변형 및 발전이 있어왔다.  
   - Global pooling method: Convolution layer의 activation을 활용한 기법
      - SPoC(Sum Pooling of Convolution): `큰 영역`
      - MAC(Maximum Activation of Convolution): `초점이 맞춰진 영역`
      - GeM(Generalized mean Pooling)
      - Global Pooling method에 성능을 더 높이기 위해서 변형을 주기도 했다.
     </br>
   > 각 Global Descriptor는 다른 성질을 가지고 있어서, 데이터셋에 따라서 성능 차이가 있다는 단점이 있다.

</br>

- 비교적 최근 연구에서는 global descriptor 모델을 각각 학습 시켜서 합치는 기법(ensemble)을 사용한다.  
  
   > 시간과 메모리 소모가 크다는 단점이 있다.

- 이런 문제를 해결하기 위해 retrieval 모델을 end-to-end로 학습시키는 ensemble 기법도 시도되고 있다.

   > 다양한 데이터셋에 보편적으로 적용할 수 있게 만들기 위해서는 특별한 loss 함수가 필요하다. 설계가 까다롭고, 학습시키는 과정도 복잡할 것이다.
   
</br>

- **논문에서는 데이터의 다양성을 고려X, 여러 global descriptor를 활용해 ensemble 같은 효과를 낼 수 있는 기법에 초점을 맞추었다.**

- **CGD** (Combination of multiple Global Descriptors)
   - end-to-end로 global descriptor를 결합하게 만들어준다.
   - global descriptor에 맞춰 조작하거나 ensemble 모델을 명시하지 않아도 ensemble 같은 효과를 낼 수 있다.
   - 매우 유연하고 global descriptor, CNN backbone, loss, dataset에 따라 확장할 수 있다.

- 이 프레임워크를 사용해서 다른 기법들과 큰 차이를 보이며 SOTA(state-of-the-art)를 달성했다. 
  > (CARS196, CUB200, SOP(Standard Online Products), In-shop(In-shop Clothes))

</br>

## 3. Proposed Framework

CGD 프레임워크를 사용해 여러 개의 global descriptor들을 concatenate 시킨 combine descriptor를 만들어 학습시킨다.

이 논문에서 제안한 프레임워크는 CNN backbone 네트워크와 두 개의 모듈로 이루어져있다.

- 주 모듈: ranking loss를 이용해 2개 이상의 global descriptor의 결합으로 이루어진 image representation을 학습한다.
- 보조 모듈: classification loss를 이용해 CNN을 fine-tune 하도록 돕는다.

CGD를 사용해 학습을 시킬 때, final loss(ranking loss(주 모듈)+classification loss(보조모듈))를 사용한다.

</br>

<p align="center"><img src="https://user-images.githubusercontent.com/53847442/143822053-a5eb42c0-f06f-4985-a288-82be036d3f7d.png"  width="80%" height="80%"/></p>

### 3.1 Backbone Network
ShuffleNet-v2, ResNet 등의 CNN backbone 네트워크를 사용할 수 있는데, 위 그림의 프레임워크에서는 ResNet-50을 사용했다.

마지막 단의 feature map의 정보를 보존하기 위해서 네트워크의 Stage3 - Stage4 사이에 downsampling 부분을 제거했다. 

Stage4에서는 224x224 크기(Image Retrieval Task 대비 낮은 해상도)를 입력으로 받아 14x14 크기의 feature map을 출력한다.

> down sampling을 안해서 feature map 사이즈가 큰 편 > 정보 보존

</br>

### 3.2 Main Module: Multiple Global Descriptors
마지막 convolutional layer에서 여러 global descriptors를 사용하여 image representation을 출력하는 branch들로 구성되어 있다. 

논문에서는 가장 대표적인 global descriptor인 SPoC, MAC, GeM 세 가지를 사용한다.

Image I 가 주어졌을 때, 마지막 convolutional layer의 output은 3차원 텐서(C X H X W)이다.

Xc가 H X W차원의 activation들의 집합이라 할 때, global descriptor는 X를 input으로하여 벡터 f를 pooling과정을 거쳐 만들어낸다. 

이러한 pooling과정을 식으로 나타내면 다음과 같다.

![image](https://user-images.githubusercontent.com/53847442/143825193-cf1c8c7e-e6cf-4bb9-b19a-9199ff009bef.png)

> Pc 가 1일 때가 SPoC(Sum Pooling of Convolutions), 
> 
> Pc 가 inf일 때가 MAC(Maximum Activation of Convolutions), 
> 
> 그 외 케이스(논문에서는 3)가 GeM


- Mulitple Global Descriptor 처리과정

      1. Global Descriptor를 활용한 Feature 추출

      2. FC Layer를 활용한 차원축소(Dimemsionality Reduction)

      3. l2-normailzaion을 활용한, Nomalization 진행

      4. 각 descriptors에 대해 3단계까지 진행한 Feature들을 Concat, Nomalizaion

      5. Ranking Loss를 활용하여 학습 (여러 Ranking Loss를 사용할 수 있으나, Batch-hard Triplet을 사용)


**여러 개의 global descriptor를 결합함으로써 얻어지는 장점**

- parameter를 몇 개만 추가하면서도 ensemble 같은 효과를 볼 수 있다.

- end-to-end 로 학습하게 만들기 위해서 단일 CNN backbone 네트워크에서 여러 개의 global descriptor를 추출하고 결합했다.
   - diversity control 없이 각 branch의 output으로 얻어지는 다양한 성질을 사용할 수 있다.


</br>

### 3.3 Auxiliary Module: Classification Loss

보조 모듈은 auxiliary classification loss를 이용해 주 모듈에서 나오는 첫 번째 global descriptor를 기반으로 CNN backbone을 fine-tune 한다.

보통 이런 접근법을 사용할 때는 다음 두 가지 과정을 거쳐 진행된다.

      1. convolutional filter의 성능을 높이기 위해 classification loss를 통한 CNN backbone을 fine-tune 시키기
      
      2. global descriptor의 성능을 높이기 위해 network를 fine-tune 시키기

하지만 본 논문에서는 이 방법을 변형시켜 end-to-end 로 가능하게끔 위의 두 단계를 하나의 단계로 합쳤다.

auxiliary classification loss 를 이용하면, 네트워크를 더 빠르게 학습시킬 수 있고, 주 모듈의 ranking loss로만 학습시키는 것 보다 안정적이다.

softmax cross-entropy loss에서의 temperature scaling과 label smoothing이 classification loss를 학습시키는데 도움을 준다.
이를 활용한 softmax loss는 아래와 같다. 

![image](https://user-images.githubusercontent.com/53847442/143828853-319dbc83-2256-42ec-bb15-c3792a32ce87.png)

```
where N, M, and yi are the batch size, the number of classes, and the corresponding identity label of i-th input, respectively. 
W, and b are trainable weight, and bias, respectively. f is a global descriptor from the first branch, where τ is a temperature parameter with default value 1.
```

 - τ(temperature parameter): 1 보다 낮으면, 학습하기 어려운 Example의 Gradient를 크게 만들어 학습에 도움을 줌
    >  더 어려운 예시에 대해 더 큰 gradient를 지정하고 class내에서 데이터가 compact하게 만들고 class사이는 벌어지게 만든다

- Label Smoothing : overfitting 방지하고, Genalization 향상시킴  + Learn Better Embedding 

</br>

## Experiment

논문에서 진행한 실험에서는 CUB200-2011, CARS196, Standard Online Products, In-shop Clothes 데이터셋을 사용

- [Table1]Ranking Loss만 사용한 것과 Classification Loss를 같이 사용한 것을 비교했을 때는 같이 사용한 것의 성능이 좋았다.

- [Table2]Trick(Label smoothing, Temperature scaling)을 사용하지 않은 것과 하나씩 사용한 것, 둘 다 사용한 것을 비교했을 때
   - ``둘 다 사용 > Temperature scaling 만 > Label smoothing 만 > 둘 다 사용 안함`` 순으로 높은 성능


![image](https://user-images.githubusercontent.com/53847442/143831269-22de4561-9661-4715-9a7c-eb46ac7156de.png)

</br>

- A : 개별 Ranking loss / B : Concat → FC → Raking / CGD: FC → Concat → Ranking
   - B 구조와 같이 Concat 후, FC Layer를 쌓으면,  각 Global Descriptor의 특성이 섞여버린다.
   - ``CGD > B > A`` 순으로 높은 성능

- [Table3] : CGD의 성능이 가장 좋음 
- [Table4] : Feature 통합 시, Concat > Sum  ,

  
 <p align="center"><img src="https://user-images.githubusercontent.com/53847442/143831966-2c99aeb4-23cf-4369-b9cb-4394d2382b9c.png"  width="80%" height="80%"/></p>

</br>

***
## 참고
[Paper](https://arxiv.org/pdf/1903.10663.pdf)

[simonezz's Blog](https://simonezz.tistory.com/96) | [kmhana's Blog](https://kmhana.tistory.com/19)
