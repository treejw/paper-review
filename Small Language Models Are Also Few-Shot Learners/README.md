# [It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners](https://arxiv.org/abs/2009.07118) `2021.04`

## Abstract
- 기존 사전 훈련된 언어 모델은 엄청난 양의 컴퓨팅이 요구된다.
- 하지만 GPT-3와 유사한 성능을 훨씬 더 적은 매개변수 수로 만들 수 있다.
- 텍스트 입력을 Cloze 질문으로 변환하면 된다.
- 우리는 작은 언어 모델로 성공적인 자연어 이해를 하기 위한 핵심 요소를 식별했다.
<img src='https://user-images.githubusercontent.com/41243762/150129411-d91d2b65-e070-434e-abaa-3f01d73de944.png' width='50%'>


## contribution
- 우리는 PET 및 iPET를 제안하고 ALBERT와 결합하여 0.1%의 매개변수와 32개의 trainset만을 사용해 SuperGLUE에서 GPT-3보다 높은 성능을 달성했다.
-  또한 PET은 높은 하이퍼파라미터 최적화가 필요 없고 단일 GPU에서 몇시간 만에 성능을 낸다.
- 마지막으로 라벨이 없는 데이터에 대해서도 유사한 성능을 얻을 수 있다는 것을 보여주고, PET의 성능에 대한 분석을 제공한다.
- 데이터 및 LM의 특성과 PET의 '친환경'적인 속성을 감안할 때 우리의 결과물은 환경적으로 건전한 NLP에 중요한 기여를 했다.
<br><br>

## Pattern-Exploiting Traning
![image](https://user-images.githubusercontent.com/41243762/150135390-fde104b0-7f71-4575-8d04-204d31fb877f.png)

- 인풋 x를 아웃풋 y로 매핑하기 위해 PET 알고리즘은 pattern-verbalizer pairs(PVPs)의 집합이 필요하다.
   * a pattern P: x → T* 는 인풋을 마스크 하나를 포함한 Cloze 질문으로 매핑
   * a verbalizer v : y→ T는 아웃풋을 패턴 내에서 태스크의 의미를 나타낼 수 있는 하나의 토큰으로 매핑

- x와 y에 대해 cloze 질문을로 변환할 PVP 만들기
   1. 주어진 few examples에 대해 MLM 학습하기
      - 다양한 PVP에 있는 각각의 패턴 에 대해 x, y를 사용해 MLM fine-tuning 
      - 실제 정답 y와 모델이 예측한 v(y)가 빈칸에 올 확률 사이의 cross-entropy를 최소화

2. 라벨링 되지 않은 데이터셋 annotate 하기
  - 이제 학습된 모델들을 앙상블 하여 라벨링되지 않은 데이터들의 라벨을 생성
  - 각각의 데이터는 확률분포에 따라 soft-labeling을 통해 annotation
  - 확률은 각각의 패턴들이 1단계의 학습을 거치기 전에 few example들에 대해 가지는 정확도를 가중치로 하여 예측값을 가중합하여 계산한다. 

<br><br>

## PET with Multiple Masks
- 하나의 토큰으로 출력하면 발생할 수 있는 에러 ex) goodness = #good + #ness

- v : y→ T*는 아웃풋을 패턴 내에서 태스크의 의미를 나타낼 수 있는 토큰으로 매핑

 ### inference
![image](https://user-images.githubusercontent.com/41243762/150140247-c423efb8-d6f1-47d3-ad39-cc9adc645b89.png)


### Training
![image](https://user-images.githubusercontent.com/41243762/150140303-64133f84-ed53-4eb7-a397-bf4f72bde36d.png)

![image](https://user-images.githubusercontent.com/41243762/150140326-bde77ad0-05b5-4109-88de-e353053ef4fe.png)


## Experiments
![image](https://user-images.githubusercontent.com/41243762/150140881-e75d48fb-fc43-4a4a-93c9-9a11fc0d0e27.png)


---
- paper [It’s Not Just Size That Matters:
Small Language Models Are Also Few-Shot Learners] | [Exploiting Cloze Questions for Few Shot Text Classification and Natural
Language Inference](https://arxiv.org/pdf/2001.07676.pdf)
(https://arxiv.org/pdf/2009.07118.pdf)
- 참고 사이트: [youtube](https://www.youtube.com/watch?v=q5FGZBqK-vc&t=859s) 
- [아기여우's blog](https://littlefoxdiary.tistory.com/62)
