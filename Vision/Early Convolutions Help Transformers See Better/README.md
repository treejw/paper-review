# Early Convolutions Help Transformers See Better 
`(2021)` `Facebook AI`

## 0. Abstract

- **Vision Transformer `(ViT)` 는 표준 이하의 성능을 보인다.**

   - 특히 optimizer 선택과 optimizer params에 민감하다.
   - lr을 작게 하거나, Optimizer SGD를 사용하면 수렴이 안되고 weight decay 계수에도 민감하다.
   - 또한 ImageNet 에서 CNN의 성능을 능가하지 못한다.
   - ViT 와 비교하면, **CNN은 optimize 하기 쉽다.**
  
- **본 논문이 주장하는 VIT 를 optimize 하기 어려운 이유**
 
   - ViT의 초기 image를 patch 단위로 자를 때 사용하는 stride-p와 p x p kernel size를 지닌 Conv 연산 때문에 어려운것이다.
   - 일반적으로 `p = 16`을 사용하는데, 보통 CNN을 설계할때 이렇게 큰 커널 사이즈와 큰 스트라이드의 조합은 사용하지 않는다.
  
- 위 주장의 정당성을 확인하기 위해 original ViT VS `stride-2, 3x3 conv`를 이용한 ViT를 비교해 보겠다. 
   - 그 결과, 후자의 모델이 안정성이 크게 향상되고 최대 성능을 향상 시키면서 동시에 연산량과 런타임은 유지하는것을 확인했다.
   - 이 효과는 모델 연산량과 데이터셋 규모에 상관 없이 적용된다.
   - 이러한 결과를 통해 ViT 모델에 standard lightweight conv를 사용할것을 권고하게 되었다.

</br>

## 1. Introduction
- CNN을 대체할 방법으로 ViT를 많이 연구하고 있다. 하지만 ViT는 표준 이하의 최적화 가능성을 보이고 있다.
   - ViT는 optimizer과 데이터셋, 학습 hyperparameter , 네트워크 깊이 등에 굉장히 민감하다.

- 반면에, Conv는 최적화가 쉽고 강력하다. 
   - SGD, basic data argumentation, standard hyperparameter 값 기반의 학습이 수년간 널리 사용되어 왔다.
   
- 이 둘의 차이는 `초기의 이미지 처리`에 있다.
   - ViT는 입력 이미지를 pxp의 겹치지 않는 패치로 패치화 하여 인코더의 입력으로 사용한다.
   - stride-p와 p x p kernel size (p=16 / 일반적인 CNN 설계보다는 훨씬 큰 값)
     <img src="https://user-images.githubusercontent.com/53847442/154908132-74fdbb3e-e943-48e9-bf28-8309decfb8f4.png" width="50%" height="50%">

   - 그래서 저자는 아래 둘을 비교하고자 한다.
   
     ![image](https://user-images.githubusercontent.com/53847442/154908454-e9996561-28af-44d0-82b8-8b36c91da58f.png)
 

    
      - ViT의 패치화 시스템을 거의 동일한 복잡성을 가진 conv network로 교체한다.
      - transformer block 수를 하나 줄인다. (동일한 연산량을 갖도록)
      - 결과 모델을 ViTC, original을 ViT라고 하면, ViTC 가 이김
</br>

```   
  <결과>  
         -  converges faster
         - 처음으로 큰 정확도 저하 없이 AdamW 또는 SGD를 사용할 수 있었음
         - lr 과 weight decay 선택에 있어서 안정적임
         - ImageNet top1에서의 CNN의 정확도를 능가
```
</br>

## 2. Related Work
> Convolutional neural networks (CNNs)

> Self-attention in vision models

> Vision transformer (ViT)

> ViT improvements

</br>

## 3. Vision Transformer Architectures
### The vision transformer (ViT)
입력 이미지를 겹치지 않는 pxp 패치로 분할하고 D 차원의 특징 벡터에 선형 투영한다.
이 임베딩은 classification head가 뒤에 붙어 변압기 인코더의 입력으로 쓰인다.

### ViTp models (P: patch)
선행 연구에서는 ViT-Tiny, ViT-Small, ViT-Base 등과 같은 다양한 크기의 ViT 모델을 제안했다.
일반적으로 1기가플롭(GF), 2GF, 4GF, 8GF 등으로 표준화된 CNN과의 비교를 용이하게 하기 위해 원래 ViT 모델을 수정하여 이러한 복잡성에 대한 모델을 얻는다. 

![image](https://user-images.githubusercontent.com/53847442/155050175-2cdce4f9-ed18-41aa-b91e-cc2732147e3d.png)

### Convolutional stem design
3x3 conv를 쌓고 마지막에transformer encoder의 d 차원 입력과 size를 맞추기 위해 1x1 conv를 사용한다. 
224x224 이미지를 3x3 conv를 이용하여 다운샘플링하여 ViTp의 입력과 동일한 size로 만든다.
(3x3 stride 2, output channels 수를 두배로 하거나, stride 1, output channel 수를 유지하거나)

ViTp 대신 ViTc를 사용할때, transformer 블럭 하나를 제거하여 연산량을 제어한다. (transformer block 1 = cnn)
이 모델은 최고 성능을 내도록 설계된것이 아니라 단순화하여 주장의 정당성을 보이기 위한것이다.


### ViTc models 
![image](https://user-images.githubusercontent.com/53847442/155055340-06ba9f66-25a3-4cc1-8065-b2b9307080e6.png)

이 표는 transformer 블럭 하나를 빼고 cnn을 이용했을때 복잡도에 거의 차이가 없음을 보인다. 
논문의 목표는 하이브리드 ViT 설계 공간을 탐색하는 것이 아니라, 패치화 스템을 표준 CNN 설계 관행을 따르는 최소 cnn으로 단순히 교체하는 경우의 최적화 효과를 연구하는 것임을 강조.

## 4. Measuring Optimizability
ImageNet의 수렴 속도를 참고하여, 400 Epoch에서의 top-1의 정확도를 대략적인 점근적 결과로 정의한다

### Optimizer stability
선행 연구들은 AdamW를 사용하여 최적화 했고, 이 외의 optimizer에 대한 성능은 제시하지 않았다.
하지만 ImageNet에 대한 정확도가 하락했다는 report가 있다.
CNN은 SGD 또는 AdamW 모두 최적화가 잘 되며, SGD를 일반적으로 사용한다. 
   - SGD는 하이퍼 파라미터가 적고(예: AdamW의 β2 튜닝이 중요할 수 있음) 
   - 최적화 상태 메모리가 50% 적게 필요하므로 확장이 용이하다.
   
우리는 최적기 안정성을 AdamW와 SGD 사이의 정확도 격차로 정의한다

### Hyperparameter (lr, wd) stability
lr, weight decay는 SGD와 AdamW의 최적화에 가장 중요한 하이퍼 파라미터들이다. 
실험에서는 lr과 wd의 다양한 선택으로 훈련된 모델의 오차 분포 함수(EDF)를 비교하여 매개 변수 안정성을 알아볼 것
(모델에 대한 EDF를 생성하기 위해 lr 및 wd의 값을 랜덤하게 샘플링하고 그에 따라 학습을 진행할 것)


### Peak performance
우리는 최고 성능을 가진 최적화 도구와 인위적으로 조정된 lr 및 weight decay를 사용하여 400 epoch 학습 시킨 모델을 평가한다.

## 5. Experinment
![image](https://user-images.githubusercontent.com/53847442/155059421-d92b95ea-a179-4bc2-9774-12a1fa3b059a.png)
![image](https://user-images.githubusercontent.com/53847442/155059925-24391184-799f-47d2-979d-5992a501586e.png)
> Figure 4: Hyperparameter stability for AdamW (lr and wd)

![image](https://user-images.githubusercontent.com/53847442/155060634-0ee225bb-a952-4add-beb0-d9b89e1fc44f.png)
> Figure 5: Hyperparameter stability for SGD (lr and wd)


> 임의의 lr 및 wd(각 모델에 대한 최적 값 주변의 고정된 폭 간격에서)를 사용하여 50 Epoch 동안 모델의 64개 인스턴스를 훈련
> EDF 그래프가 가파를수록 안정성이 우수한것


![image](https://user-images.githubusercontent.com/53847442/155060846-25a70d7a-9929-4b32-a6cb-d630811c7d68.png)
> Figure 6: Peak performance
