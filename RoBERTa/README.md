# RoBERTa: A Robustly Optimized BERT Pretraining Approach `2019.7`

## Abstract
- BERT는 undertraining 되어 있다.
- BERT가 잘 training 되면 BERT 이후에 발표된 모델의 성능과 일치하거나 향상될 수 있다.
- 우리는 하이퍼파라미터나 데이터의 크기가 모델에 얼마나 큰 영향을 미치는지 측정한다.

<br>

## 1. Introduction
- ELMO, GPT, BERT, XLM, XLNet 등 많은 모델이 성능 향상을 가져왔지만 이들 방법 중 어떤 측면이 큰 기여를 했는지는 잘 모른다.
- 학습은 시간과 비용이 들기 때문에 수행할 수 있는 조정 횟수를 제한하고, 종종 다양한 크기의 개인 data로 학습되기 때문이다.
- 또한 우리는 BERT가 매우 undertraining 되어 있다는 것을 확인했다.
- 그래서 우리는 다음과 같은 방법을 사용하여 Robustly Optimized BERT Pretraining Approach를 제안한다.
  1. 더 길게, 더 큰 배치와, 더 큰 데이터로 학습한다.
  2. NSP loss를 이용한 pretrain을 하지 않는다.
  3. 더 긴 문장으로 학습한다.
  4. [MASK] 토큰을 다이나믹하게 적용한다.
  5. 새로운 대규모 데이터셋 CC-NEWS를 수집한다.

<br>

### ◼ Contribution
1. 우리는 BERT에 있어서 중요한 설계와 훈련 전략을 제시하여, 더 좋은 Downstream Task 성능을 유도하는 방법을 제시한다.
2. 새로운 데이터세트인 CC-NEWS를 사용하여 pretraining에 있어서 더 많은 데이터를 사용하는 것이 Downstream Task의 성능을 향상시킨다는 사실을 확인한다.
3. 설계가 잘 된 MLM 사전학습은 최근 발표된 모든 다른 방법과 competitive하다는걸 보여준다.

<br>

## 2. Experimental Setup
### ◼ Implementation
- 우리는 BERT를 큰 배치사이즈로 학습시킬 때, Adam Optim에서 β2를 0.98로 하는게 가장 좋다는걸 알아냈다.
- 우리는 기존 버트와 다음 두가지를 다르게 했다.
  - Randomly inject short sequences를 사용하지 않았다.
  - 업데이트의 처음 90% 동안 reduced sequences length로 훈련 하지 않았다.
- 우리는 오직 full-length sequences로만 학습했다.

<br>

### ◼ Data
- BERT는 16GB데이터(BookCorpus + Wikipedia)를 사용
- 우리는 다양한 사이즈와 도메인을 가진 5개 corpora를 사용한다. (160GB)
  1. BookCorpus
  2. Wikipedia
  3. CC-NEWS
  4. OpenWebText
  5. Stories 

<br>

## 3. Training Procedure Analysis
- 모델이 성공적으로 사전학습 할 수 있는 요소를 정량화 한다.
- 여기서는 BERT_BASE와 동일한 구성을 가진 BERT 모델을 교육하는 것으로 한다.
  - L = 12
  - H = 768
  - A = 12
  - 110M params

<br>

### ◼ Static VS. Dynamic Masking
- Static (BERT)
  - BERT는 데이터 사전 처리 중 마스킹을 한 번 수행하여 단일 static mask를 사용한다.
  - 모든 시기의 각 훈련에 대해 동일한 마스크를 사용하지 않기 위해 데이터를 10배 복제하여 40epoch에 동안 10번 다른 방식으로 마스크되도록 했다.
  - 따라서 4번 같은 데이터를 본다.
- Dynamic Masking (RoBERTa)
  - Dynamic Masking은 sequence를 입력할 때 마다 마스크를 생성한다.
  - 이는 더 많은 step과 데이터로 학습할 때 중요하다.
- ![image](https://user-images.githubusercontent.com/41243762/104411945-b9836a80-55ae-11eb-9147-86073a54ddfc.png)

<br>

### ◼ Model Input Format and Next Sentence Prediction
- NSP loss 는 이미 많은 논문에서 의문을 가졌다.
- 따라서 우리는 모델 입력 포멧과 NSP loss 의 필요성을 실험한다.
  1. Segment Pair + NSP - BERT와 동일한 설정입니다.
  2. Sentence Pair + NSP - 각 Segment가 하나의 문장으로만 구성됩니다. 각 segment가 매우 짧기 때문에 batch size를 늘려 다른 구성들과 한번에 optimize되는 토큰 수를 비슷하게 설정하였습니다.
  3. Full Sentence - 각 input은 하나 이상의 문서들에서 연속적으로 샘플링 됩니다. 하나의 문서가 끝나면, 다음 문서를 그대로 연결(특수 토큰으로 분리)하여 총 토큰 갯수가 최대 길이를 최대한 채우도록 구성하였습니다.
  4. Doc Sentence - 3번 설정과 유사하지만, 하나의 문서가 끝나면 다음 문서를 이용하지 않습니다. 문서가 끝난 경우 토큰 갯수는 최대 길이보다 작게 되는데, 이를 보정하기 위해 batch size를 dynamic하게 조절하여, 한번에 optimize되는 토큰 갯수를 일정하게 유지했습니다.
- ![image](https://user-images.githubusercontent.com/41243762/104412476-ab821980-55af-11eb-857f-d261b43f3c6e.png)

<br>

### ◼ Training with large batches
- BERT는 256 batch size로 1M 학습하는데 이는 2k batch size로 125k 학습한 것과 같다.
- ![image](https://user-images.githubusercontent.com/41243762/104412606-f13ee200-55af-11eb-8cdb-c95d7919f6e1.png)

<br>

### ◼ Text Encoding
- BERT는 BPE와 휴리스틱을 사용하여  30k 개의 vocab을 만들었다.
- 우리는 일체의 휴리스틱과 전처리 없이 BPE로 50k 개의 vocab을 만들어 사용한다.
  - 이는 일부 task에서는 성능이 떨어지지만 범용성 향상이 더 중요하다고 생각했기 때문이다.

<br>

## 4. RoBERTa
- 우리는 task 성능을 향상시키는 사전학습 방법을 제안한다.
    1. Dynamic Masking을 사용
    2. NSP loss 제거
    3. large size batch
    4. big BPE vocab
- 추가적으로 이전에 비교에서 무시되었던 중요한 요인 2가지를 제시한다.
    - 사전학습 데이터 사이즈
    - 데이터를 학습하는 횟수
    - ex) XLNet은 BERT보다 많은 데이터로 더 많이 학습되었다. 이는 BERT에 비해서 사전훈련 때 4배 더 많은 sequence를 볼 수 있다. 
- ![image](https://user-images.githubusercontent.com/41243762/104413489-c05fac80-55b1-11eb-93d0-27c4730d91a4.png)

<br>

### ◼ GLUE
![image](https://user-images.githubusercontent.com/41243762/104413579-e71de300-55b1-11eb-90f0-2344434932f2.png)

<br>

### ◼ SQuAD
- BERT와 XLNet은 추가적인 QA data를 사용하여 학습했지만 우리는 오직 SQuAD에서 제공한 학습데이터로만 학습했다.
- ![image](https://user-images.githubusercontent.com/41243762/104413672-15032780-55b2-11eb-9c23-b65b05009df7.png)

<br>

### ◼ RACE
![image](https://user-images.githubusercontent.com/41243762/104413707-2a785180-55b2-11eb-8f6b-4811e9d28928.png)


---
### 참고
- paper : [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
- [jeongukjae's blog](https://jeongukjae.github.io/posts/3-roberta-review/) | [Yeongmin's blog](https://baekyeongmin.github.io/paper-review/roberta-review/)
