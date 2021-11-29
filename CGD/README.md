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
