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
   - 
   <img src="https://user-images.githubusercontent.com/53847442/154907920-a0b60c12-9126-42fa-9c7c-c67d9ae71596.png" width="50%" height="50%">



## 2. Related Work
## 3. Vision Transformer Architectures
## 4. Measuring Optimizability
