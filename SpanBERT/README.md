# SpanBERT: Improving Pre-training by Representing and Predicting Spans `2020`

## Abstract
◼ 무작위 MASK가 아닌 span MASK를 하여 성능을 향상시킨다.
◼ 예측 또한 토큰 하나에 대한 예측이 아닌 span boundary representations을 예측하여 범위 전부를 예측하는 방식으로 성능을 향상시킨다.

<br>

## 1. Introduction
◼ BERT와 같은 모델은 word, subword를 마스킹해 pretrain을 하여 성능을 크게 향상시켰음. 그러나 많은 NLP task에는 text span간 relationship 추론이 필요하다.
◼ 따라서 random contiguous span과 그에 따른 학습 방법을 제안한다.

<br>

## 2. Method
![image](https://user-images.githubusercontent.com/41243762/120059725-14b18780-c08e-11eb-9b6c-12cf6765c2d3.png)

### 2.1 Span Masking
◼ BERT와 달리 contiguous span에 대해 마스킹을 진행
   -  geometric distribution(l∼Geo(p))으로 마스킹 길이를 선택
   -  마스킹 시작 위치는 uniformly 하게 
   -  span의 길이는 subword tokens이 아닌 complete word단위


![image](https://user-images.githubusercontent.com/41243762/120059892-424b0080-c08f-11eb-8bea-bc133400544f.png)

### 2.2 Span Boundary Objective (SBO)
◼ Span Boundary의 token과 position embedding 값을 학습에 사용
![image](https://user-images.githubusercontent.com/41243762/120059943-a5d52e00-c08f-11eb-8179-1626860ed7a0.png)
◼ MLM과 SBO를 합하여 최종 Loss로 사용한다.
![image](https://user-images.githubusercontent.com/41243762/120060000-12e8c380-c090-11eb-9ccc-01a6814c5301.png)

### 2.3 Single-Sequence Training
◼ NSP 없이 한 개의 Sequence를 입력으로 사용한다.

## 3. Experiments
◼ BERT - 기본 BERT
◼ Our BERT - spanbert와 같은 데이터로 학습
◼ Our BERT-1seq - 2.3 방법을 적용하여 NLP 없이 1개의 full-sequence로 학습
◼ SpanBERT

### RESERT
![image](https://user-images.githubusercontent.com/41243762/120060126-e84b3a80-c090-11eb-9df4-13efb12bfb63.png)
![image](https://user-images.githubusercontent.com/41243762/120060139-fbf6a100-c090-11eb-8499-c83f9f2e3151.png)
![image](https://user-images.githubusercontent.com/41243762/120060165-1597e880-c091-11eb-8226-a3a733c2dc83.png)
![image](https://user-images.githubusercontent.com/41243762/120060147-07e26300-c091-11eb-8301-9f0045399d2f.png)



---
### 참고
- paper : [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf)
- [mlgalaxy blog](https://mlgalaxy.blogspot.com/2020/10/spanbert-improving-pre-training-by.html) | [jeonsworld blog](https://jeonsworld.github.io/NLP/spanbert/)
