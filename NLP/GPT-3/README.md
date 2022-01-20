# Language Models are Few-Shot Learners `arXiv.org [cs.CL] 2020`

## 1. Abstract
- NLP 모델은 많은 발전을 이루었지만 task에 사용하려면 fine-tuning이 필요하다.
- 일반적으로 모델의 매개변수가 기술과 지식을 습득하기 때문에 규모가 커질수록 성능이 좋아진다.
- 우리는 175B의 매개변수를 가진 모델을 만들어서 다음 3가지 조건에서 위 가설을 테스트한다.
> 1. few-shot
> 2. one-shot
> 3. zero-shot

![image](https://user-images.githubusercontent.com/41243762/108952535-fada6000-76ac-11eb-9494-ffb9a3328307.png)
그림은 우리가 연구 한 조건에서 모델이 단어에서 관련없는 기호를 제거해야하는 간단한 작업에 대한 몇 번의 학습을 보여준다.
- 자연어 작업 설명을 추가하면 모델 성능이 향상되고
- 모델 컨텍스트의 예제 수가 많아지면 성능이 향상되고
- 모델 크기에 따라 성능이 향상된다.

<br><br>

## 2. Aproach
### 2.1 Zero, One, Few shot & fine-tuning
![image](https://user-images.githubusercontent.com/41243762/108952977-c1562480-76ad-11eb-8f6e-ac10e68a4c22.png)



### 2.2 Model
![image](https://user-images.githubusercontent.com/41243762/108953023-d6cb4e80-76ad-11eb-8984-b3625731de64.png)
![image](https://user-images.githubusercontent.com/41243762/108953406-7d175400-76ae-11eb-960c-727cf1544ee2.png)

- GPT-2 같다. 
- the modified initialization, pre-normalization,  and reversible tokenization (추가 및 수정)
- alternating dense and locally banded sparse attention patterns in the layers of the transformer
- 모든 모델은 3,000억 토큰에 대해 학습함.



### 2.3 Dataset
![image](https://user-images.githubusercontent.com/41243762/108953386-738dec00-76ae-11eb-837f-4d686cbaa83b.png)

<br><br>
## 3. Result
### 3.1 train-graph
![image](https://user-images.githubusercontent.com/41243762/108953802-22cac300-76af-11eb-8f45-d56ad5af0959.png)

### 3.2 Language Modeling, Cloze, and Completion Tasks
#### 3.2.1 PTB Language Modeling
![image](https://user-images.githubusercontent.com/41243762/108954022-7b01c500-76af-11eb-8af7-4e40abd805ec.png)
#### 3.2.2 LAMBADA, HellaSwag, StoryCloze
![image](https://user-images.githubusercontent.com/41243762/108954200-ca47f580-76af-11eb-876a-c041e30e22ff.png)
#### 3.2.3 Closed Book Question Answering
![image](https://user-images.githubusercontent.com/41243762/108954478-375b8b00-76b0-11eb-971f-dedb5b769d87.png)
#### 3.2.4 Translation
![image](https://user-images.githubusercontent.com/41243762/108954622-7093fb00-76b0-11eb-954a-18d90bc1912d.png)
#### 3.2.5 Arithmetic
![image](https://user-images.githubusercontent.com/41243762/108955799-17c56200-76b2-11eb-8f8f-173bbbc224fa.png)
#### 3.2.6 News Article Generation
![image](https://user-images.githubusercontent.com/41243762/108956133-97533100-76b2-11eb-8191-41880b04360d.png)
---기타 실험 다수---

## 4. Measuring and Preventing Memorization Of Benchmarks
![image](https://user-images.githubusercontent.com/41243762/108956469-134d7900-76b3-11eb-9c98-f524180eb31c.png)

## 5. Limitations
- 성능이 좋다고는 할 수 없다.
- 긴 글을 생성했을 때 동어가 반복되어 가독성이 떨어지며 모순적인 문장 또는 관련없는 문장을 만들기도 한다.
- 너무 많은 양의 데이터를 학습한다. 효율성이 떨어진다.
- 너무 큰 스케일의 모델로 비용이 많이 든다.

## 6. Broader Impacts
- 가짜 정보의 생산
- threat actor의 위험성
- 성별, 인종, 종교 등 특정 분야에 대한 편향이 존재한다.
- 학습 중 에너지 사용량
---
### 참고
- [paper](https://arxiv.org/abs/2005.14165)
- [littlefoxdiary](https://littlefoxdiary.tistory.com/44)
