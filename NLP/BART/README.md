# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension `2019.10`

## Abstract
- Masked language model과 denoising auto-encoder가 좋은 성능을 보인다.
- 하지만 기존은 방법들은 특정 end task에 집중하여 활용성이 떨어진다.
<br>

## 1. Introduction
- BERT는 bidirection encoder로 noise된 token을 예측하지만, generation task에서 사용이 어렵다. 
- GPT는 autoregressive하게 다음 token을 예측해 generation에 사용이 가능하지만 bidirectional 정보를 얻지 못한다. 
- BART는 손상된 text를 입력받아 bidirectional 모델로 인코딩하고, 정답 text에 대한 likelihood를 autoregressive 디코더로 계산한다. 

<br>

### ◼ Contribution
1. BART는 noising이 자유롭다는 장점이 있다. 
2. 문장 순서를 바꾸거나 임의 길이의 토큰을 하나의 mask로 바꾸는 등의 여러 noising 기법을 평가한다

<br>

## 2. Noise Task
![image](https://user-images.githubusercontent.com/41243762/123260610-b662b200-d530-11eb-9c28-0877c176f050.png)
- Token Masking: BERT와 동일
- Token Deletion: 랜덤 토큰을 삭제하고 이를 복구하는 방식이다. Masking과의 차이점은 어떤 위치의 토큰이 지워졌는지 알 수 없다는 점이다.
- Text Infilling: 포아송 분포를 따르는 길이의 text span length 생성해서 이 길이만큼 이어진 토큰들을 하나의 mask 토큰으로 masking, 0인 경우에는 mask 토큰만 추가될 수도 있다. 모델이 얼마나 많은 토큰이 빠졌는지 예측하게 한다.
- Sentence Permutaion: Document를 문장 단위로 섞는 방법이다.
- Document Rotation: 토큰 하나를 정해서 문장이 그 토큰부터 시작하게 한다. 모델이 document의 시작을 구분하게 한다.

## 3. Fine-Tuning
- Sequence Classification Tasks: 디코더의 final hidden state가 새로운 linear classifier로 전달
- Token Classification Tasks: 디코더의 top hidden state를 각 단어에 대한 representation으로 사용
- Sequence Generation
- Machine Translation:
  - 새로운 인코더를 추가해서 인코더-디코더를 fine-tuning 한다. 
  - 새로운 인코더는 외국어를 BART가 학습한 언어로 mapping 한다.
  - 처음에는 대부분의 BART 파라미터는 그대로 두고 인코더와 BART의 position embedding, BART 인코더의 첫번째 레이어 self-attention input projection matrix만 학습시킨다. 두번째 단계에서는 모든 파라미터를 학습시킨다.
  - ![image](https://user-images.githubusercontent.com/41243762/123261689-068e4400-d532-11eb-90e0-dda0f799aa2e.png)

## 4. Result
### ◼ Experiments setting
- Language Model: cross-attention이 빠진 BART 디코더와 같다. (GPT와 비슷)
- Permuted Language Model: XL-Net을 기반으로 한다. 1/6 토큰을 샘플링하고 이것을 랜덤한 순서로 auto-regressive하게 생성한다.
- Masked Language Model: BERT처럼 15% 토큰을 mask 토큰으로 바꾸고 독립적으로 이 토큰을 예측하게 한다.
- Multitask Masked Language Model: UniLM처럼 self-attention mask를 추가해서 masked language model을 학습한다.
- Masked Seq-to-Seq: MASS와 비슷하다. 토큰의 50%를 포함하는 span에 mask를 하고 mask된 토큰을 예측하는 seq-to-seq 모델을 학습시킨다.

### ◼ Model test
![image](https://user-images.githubusercontent.com/41243762/123262458-dbf0bb00-d532-11eb-973f-6b3b8baa0283.png)

### ◼ Discriminative Tasks
![image](https://user-images.githubusercontent.com/41243762/123262657-18241b80-d533-11eb-8632-d6ccf31137fe.png)

### ◼ Generation Tasks
![image](https://user-images.githubusercontent.com/41243762/123262689-23774700-d533-11eb-82a3-2cbf77f76a6e.png)

### ◼ Qualitative Analysis
![image](https://user-images.githubusercontent.com/41243762/123262787-3db12500-d533-11eb-963e-dee978073e35.png)

---
### 참고
- paper : [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [dladustn95's blog](https://dladustn95.github.io/nlp/BART_paper_review/) | [Youtube](https://www.youtube.com/watch?v=VmYMnpDLPEo)
