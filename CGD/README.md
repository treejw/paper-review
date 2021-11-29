# Combination of Multiple Global Descriptors for Image Retrieval `(CGC)` `2020.04`

## 1. Abstract
- 최근 Image Retrieval 분야에서 여러 "모델의 앙상블(Ensemble)"이나, "Global Descriptor의 결합"을 통해 좋은 성능을 보여왔다.
   > Global Descriptor : SPoC, MAC, GeM 등을 의미함

   그러나 개별 모델을 따로 학습한 후 앙상블 하는 것은, 어렵고 Time과 Memory측면에서 비효율적이다.

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

- 주 모듈: ranking loss를 이용해 여러 global descriptor의 결합으로 이루어진 image representation을 학습한다.
- 보조 모듈: classification loss를 이용해 CNN을 fine-tune 하도록 돕는다.

CGD를 사용해 학습을 시킬 때, final loss(ranking loss(주 모듈)+classification loss(보조모듈))를 사용한다.

</br>

<p align="center"><img src="https://user-images.githubusercontent.com/53847442/143822053-a5eb42c0-f06f-4985-a288-82be036d3f7d.png"  width="80%" height="80%"/></p>

