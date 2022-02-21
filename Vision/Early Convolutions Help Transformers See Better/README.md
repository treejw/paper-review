# Early Convolutions Help Transformers See Better `(2021) Facebook AI`

## 1. Abstract

- **Vision Transformer `(ViT)` 는 표준 이하의 성능을 보인다.**

   - 특히 optimizer 선택과 optimizer params에 민감하다.
   - lr을 작게 하거나, Optimizer SGD를 사용하면 수렴이 안되고 weight decay 계수에도 민감하다.
   - 또한 ImageNet 에서 CNN의 성능을 능가하지 못한다.
   - ViT 와 비교하면, **CNN은 optimize 하기 쉽다.**
  
- **본 논문이 주장하는 VIT 를 optimize 하기 어려운 이유**
 
   - ViT의 초기 image를 patch 단위로 자를 때 사용하는 stride-p와 p x p kernel size를 지닌 Conv 연산 때문에 어려운것이다.
   - 일반적으로 `p = 16`을 사용하는데, 보통 CNN을 설계할때 이렇게 큰 커널 사이즈와 큰 스트라이드의 조합은 사용하지 않는다.
  
- 위 주장의 정당성을 확인하기 위해 original ViT VS stride-2, 3x3 conv를 이용한 ViT를 비교해 보겠다.
 


## 2. Related Work
## 3. Vision Transformer Architectures
## 4. Measuring Optimizability
