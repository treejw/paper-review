# ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision `2021.06`

<br>

## 1. Abstract
- VLP(Vision and Language Pre-Training)에 대한 선행 연구들은 Text 혹은 Image의 특징을 추출 방식에 의존했다.
- 이는 효율성과 속도 측면에서 큰 영향을 끼친다.
- ViLT는 이런 부분을 크게 단순화하여 다운스트림 테스크에서 빠른 속도향상을 보여준다.

![image](https://user-images.githubusercontent.com/41243762/157249187-0f4649e8-bfe3-460a-a6c5-aebdf58f549f.png)



#### Contribution
- CNN과 같은 특별한 visual embedder를 사용하지 않음으로서 모델 단순화
- 그럼에도 불구하고 경쟁력 있는 성능을 보임
- whole word masking과 image augmentation이 VLP에 효과가 있음을 보임
<br>

## 2. Model
![image](https://user-images.githubusercontent.com/41243762/157250516-758a84d2-49b2-4094-88e9-ff25d69fcc3f.png)

### 2.1 Pre-Train Objective

- ViT와 같이 이미지를 patch단위로 받아 linear projection 후 사용
- Image Text Match과 Masked Language Modeling을 활용하여 pre-training 수행
- IPOT(Inexact Proximal point method Transports) 기법을 이용하여 Word Patch Alignment objective 설계

<br><br>

### 2.2  whole word masking
- “giraffe” -> ["gi", "##raf", "##fe"] -> ["gi", "[MASK]", "##fe"]
- “giraffe” -> ["gi", "##raf", "##fe"] -> ["[MASK]"]

## 3. Experiments
### 3.1 Pre-Training data
![image](https://user-images.githubusercontent.com/41243762/157255586-5030e246-1d41-4e49-baa3-5b88b6dd9b9c.png)

### 3.2 Runtime
![image](https://user-images.githubusercontent.com/41243762/157255675-32694694-e311-44a6-b8c9-71684e496651.png)
![image](https://user-images.githubusercontent.com/41243762/157255813-8ef712d3-3732-453d-8126-f309d23213b7.png)

### 3.3 Down stream Task
![image](https://user-images.githubusercontent.com/41243762/157255925-50c965ba-61e5-4843-97b5-7a3a28f1ac13.png)

### 3.4 Ablation Study
![image](https://user-images.githubusercontent.com/41243762/157256193-f68fae85-b40b-4c8c-83dc-ecfeb53dd83e.png)
![image](https://user-images.githubusercontent.com/41243762/157256237-d90bfeef-5d05-41e1-b379-3813e3f729e7.png)
![image](https://user-images.githubusercontent.com/41243762/157256339-b558b6a9-128c-4b4b-81a0-1b0b20894045.png)

---

