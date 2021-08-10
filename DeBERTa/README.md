# [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/pdf/2006.03654.pdf) `2021.3`

## Abstract
- 이 논문에서는 2가지 새로운 방법을 사용한 모델 DeBERTa를 제안하며, BERT와 RoBERTa의 성능을 향상켰다.
- 첫 번째는 disentangled attention mechanism이다.
- 단어를 content와 position 두 가지의 vector로 표현하며, 단어 간의 attention weight는 content와 relative position의 disentangled matrices로 계산한다.
- 두 번째는 Enganced mask decoder이다.
- 사전 훈련의 과정에서 마스크 토큰을 예측하기 위해 absolute position을 decoding layer에서 통합한다. 
- 추가적으로 일반화 성능을 개선하기 위해 새로운 virtual adversarial 훈련 방법을 사용하여 fine-tuning 한다.
- 우리는 이러한 기술이 사전훈련의 효율성과 NLU, NLG의 다운스트림 Task에 대해 성능을 크게 향상시킨다는 것을 보여준다.
- RoBERTA-L와 비교하여 훈련 데이터 절반으로 학습한 DeBERTa 모델은 다양한 Task에서 일관적으로 더 나은 성능을 보인다.
- 특히 더 큰 버전으로 확장한 DeBERTa는 최초로 SuperGLUE에서 사람의 성능을 뛰어넘었다.


<br><br>


## 2. Architecture
### 2.1 Disentangled Attention
- i번째 단어와 j번째 단어의 Attention Value A_ij는 다음의 식으로 표현된다.
![image](https://user-images.githubusercontent.com/41243762/128858344-3bb07109-72df-4f6f-b6d9-6e48ec497d1a.png)

- 하지만 위치 벡터 간의 연산은 유의미한 정보를 갖지 않기 때문에 계산에서 제외한다.
- 이는 이전의 Relative positional embedding과 다르게 content-to-content, content-to-position, position-to-content 세가지 연산을 포함한다.
- 위 식의 실제 연산은 다음과 같다.
![image](https://user-images.githubusercontent.com/41243762/128858858-e712432e-a204-4794-a55c-22c9e58cf2ca.png)

- 이 때 델타는 다음 식으로 표현되며 이는 Relative Position의 index를 의미한다.
![image](https://user-images.githubusercontent.com/41243762/128859159-15f7e3ad-e4c4-4751-99f8-e92d1cbb264a.png)


<br>

### 2.2 Enhanced Mask Decoder
- Relative Position만 사용할 경우 문제가 발생한다.
- "A `new store` opened beside the `new mall`"
- 위 문장에서 `store`와 `mall`은 앞의 `new`가 동일하다.
- 하지만 문장의 메인은 `store`이기 때문에 순서가 중요하다.
- 이를 해결하기 위해 Absolute Position을 고려해야 한다.
- Enhanced Mask Decoder의 경우 BERT와 다르게 마지막 softmax 직전에 정보를 합쳐준다.

<br>

## 3. Scale Invariant Fine-Tuning(SiFT)
- Fine-Tuning할 때 모델의 일반화 성능을 올리기 위한 정규화 방법
- model의 input에 대해 작은 Perturbation을 만든다.

<br>

## 4. Experiment
### 4.1 Performance on large model
![image](https://user-images.githubusercontent.com/41243762/128861614-843fe2b8-62db-4c6f-bc6f-d593ad98c48c.png)

![image](https://user-images.githubusercontent.com/41243762/128861710-e7c9ea24-c799-442a-8a7f-c4827558064f.png)


<br><br>

### 4.2 Performance on base model
![image](https://user-images.githubusercontent.com/41243762/128861960-778d56e0-5d49-4228-b94d-2d7fdd74c22a.png)

<br><br>

### 4.3 Ablation study
![image](https://user-images.githubusercontent.com/41243762/128861746-83c9e23f-82a6-4147-8bcf-6771c3314057.png)


<br><br>

### 4.4 Scale up to 1.5B params
![image](https://user-images.githubusercontent.com/41243762/128862045-b3d08cde-e66b-4847-850b-60ee3299e533.png)

<br><br>

---

### 참고

- paper: [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/pdf/2006.03654.pdf)
- [Wiki by Beomi](https://wiki.beomi.net/deberta.html) | [YouTube - Beyond Bert](https://www.youtube.com/watch?v=f2__p05aY2I)
