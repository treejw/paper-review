# XLNet: Generalized Autoregressive Pretraining for Language Understanding `2019.6`

## Abstract

- BERT는 양방향 문맥을 모델링하는 denosising autoencoding기반의 모델로 autoregressive 접근 방식을 가진 모델보다 더 좋다.
- 하지만 mask된 토큰간의 의존성을 학습할 수 없고,
- fine-tuning에는 mask가 없어서 pretrain-finetune 간의 불일치가 발생한다.
- 우리는 다음 방법을 제안한다.
   1. factorization order의 모든 permutation을 다 계산하여 양방향 문맥을 학습할 수 있고,
   2. autoregressive 방식을 사용하여 BERT의 한계를 극복한다.
   3. Transformer-XL의 아이디어를 통합한 XLNet을 제안한다.

<br>

## 1. Introduction

### ◼ AR (AutoRegressive)
- ELMO, GPT 등의 학습 방법으로 이전 단어를 보고 다음 단어를 예측하는 방식
- 수식은 다음과 같다.
![image](https://user-images.githubusercontent.com/41243762/103730093-4feee380-5025-11eb-8688-52ed60d57126.png)
- 문제점
   - 방향성이 정해져야 한다.
      - 한 쪽 정보만 이용 가능(왼쪽에서 오른쪽 or 오른쪽에서 왼쪽)
      - ELMO는 둘 다 따로 한 다음 합치는 형태로 양방향이라 하기에는 무리가 있다.

<br>

### ◼ AE (AutoEncoding)
- 주어진 input에서 input을 그대로 예측하는 방식
- BERT의 경우 Denosing autoencoing으로 input + noise에서 input을 예측하는 방식이다.
- 수식은 다음과 같다.
![image](https://user-images.githubusercontent.com/41243762/103730274-d4416680-5025-11eb-80f6-afa95e6b958b.png)
- 문제점
   - Independent assumption
      - [MASK] 토큰간의 독립을 가정한다.
      - ex) New York is a city -> [MASK] [MASK] is a city 의 경우 New와 York의 의존성을 알 수 없다.  
   - pretrain-finetune discrepancy
      - [MASK] 토큰의 유무

<br>

### ◼ Contribution
- 기존 모델 AR과 AE의 장점을 모두 활용하는 방법을 제안
   - 가능한 모든 Factorization order를 계산해 AR의 일방향성 문제를 해결한다.
   - 의도적인 noise같은 것에 의존하지 않기 때문에 DAE의 한계인 independent 가정과 discrepancy를 해결한다.
- 사전 교육을 위한 아키텍처 설계 개선
   - Transformer-XL의 매커니즘과 relative encoding을 사전 교육에 통합하여 긴 텍스트 task의 성능을 개선한다.
   - permutation-based language modeling(XLNet의 방식)은 Transformer-XL model에서 동작하지 않기 때문에 Transformer-XL 모델을 reparameterize할 것을 제안한다.

<br>

## 2. Proposed Method
### ◼ Permutation Language Modeling
- 모델 수식
![image](https://user-images.githubusercontent.com/41243762/103731795-54b59680-5029-11eb-9ea0-8b1806b389d3.png)
- pretrain 개념 예시
   - AR : 나는 밥을 맛있게 먹는다 ==> input(나는, 밥을) pred(맛있게)
   - AE : 나는 밥을 맛있게 먹는다 ==> input(나는, 밥을, [MASK], 먹는다) pred(맛있게)
   - PLM : 나는 밥을 맛있게 먹는다 ==> input([나는, 밥을, 먹는다],[먹는다, 나는, 밥을] ... [나는]) pred(맛있게)
- 그림 예시 (단어가 4개일 때 3번 단어를 예측할 경우) - mem은 이전 세그먼트 메모리
   - permutation [**3**, 2, 4, 1]
      - ![image](https://user-images.githubusercontent.com/41243762/103733047-2e452a80-502c-11eb-9aae-4fc99a560791.png)
   - permutation [2, 4, **3**, 1]
      - ![image](https://user-images.githubusercontent.com/41243762/103733183-795f3d80-502c-11eb-9c00-50c309509280.png)
   - permutation [1, 4, 2,  **3**]
      - ![image](https://user-images.githubusercontent.com/41243762/103733195-811ee200-502c-11eb-8c51-d9deb7ff29d8.png)
   - permutation [4, **3**, 1, 2]
      - ![image](https://user-images.githubusercontent.com/41243762/103733207-88de8680-502c-11eb-8e8a-5cc098898e51.png)
- Attention mask를 이용한 예시 (행: 쿼리, 열: 키, 빨간색: 사용, 회색: 0처리)
    - 단어가 4개라고 했을 때
       - AR
          - ![image](https://user-images.githubusercontent.com/41243762/103732484-c8a46e80-502a-11eb-8ad0-3c691f5d0e4e.png)
       - PLM
          - ![image](https://user-images.githubusercontent.com/41243762/103732512-dbb73e80-502a-11eb-972c-342a003003b1.png)
- 문제점
   - 동일한 입력을 받았을 때 다른 출력을 내야하는 모순 발생
      - [3, 2, **4**, 1]
      - [3, 2, **1**, 4]
   - 이를 해결하기 위해 Two-stream self attention 사용

<br>

### ◼ Two-stream self attention
![image](https://user-images.githubusercontent.com/41243762/103732927-d4446500-502b-11eb-9a4d-31491855117e.png)
#### ◾ Content stream
![image](https://user-images.githubusercontent.com/41243762/103733341-e541a600-502c-11eb-8ead-265618e0ff34.png)
- 수식
   -  ![image](https://user-images.githubusercontent.com/41243762/103733295-c93e0480-502c-11eb-9de1-aba17c543bd0.png)
   - h: 컨텐트 스트림
   - m: layer 번호
   - z: 하나의 순열 ex) [4, 3, 1, 2]
   - t: z의 요소 번호
   - z_t: z의 t번째 요소
 #### ◾ Query stream
![image](https://user-images.githubusercontent.com/41243762/103733611-7c0e6280-502d-11eb-9998-e7036e7c4d31.png)
- 수식
   - ![image](https://user-images.githubusercontent.com/41243762/103733629-87fa2480-502d-11eb-9e1c-4dbad68a32e3.png)
   - g: 쿼리 스트림
   - m: layer 번호
   - z: 하나의 순열 ex) [4, 3, 1, 2]
   - t: z의 요소 번호
   - z_t: z의 t번째 요소
   - w: 위치 정보
- 특징
   -  지금 맞춰야할 단어의 임베딩 정보가 들어가지 않는다. (X_z_t)

 #### ◾ 문제점 해결
- 결과적으로  동일한 입력을 받았을 때 다른 출력을 내야하는 모순이 해결된다.
- [3, 2, **4**, 1] : 여태까지 [맛있게, 밥을] 이라는 단어를 봤는데. 이번에 맞춰야 할 단어는 원래 문장에서 네번째에 있었어. 이 단어는 뭘까?
- [3, 2, **1**, 4] : 여태까지 [맛있게, 밥을] 이라는 단어를 봤는데. 이번에 맞춰야 할 단어는 원래 문장에서 첫번째에 있었어. 이 단어는 뭘까?

<br>

### ◼ Incorporating Ideas from Transformer-XL
- Relative Positional Encoding
   - 기존의 고정된 포지셔널 인코딩은 토큰 간 포지션 정보를 활용할 수 없기 때문에 이를 사용한다.
- Segment Recurrence Mechanism
   - 이전 세그맨트에 대한 정보를 캐싱하여 사용한다. 

<br>

## 3. Discussion and Analysis
- BERT와 가장 큰 차이 [New, York, is, a, city]
   - BERT와 XLNet에서 모두 [New, York] 을 타겟으로 선택했다고 가정.
   - XLNet의 factorization order는 [is, a, city, New, York] 이라고 가정.
   - ![image](https://user-images.githubusercontent.com/41243762/103734351-2470f680-502f-11eb-984a-0d9003ce32c1.png)
   - BERT와 달리 XLNet은 [New, York]간의 의존성을 포착
- GPT의 경우 (New → York) 의 의존성은 잡아내지만 (York → New) 의 의존성은 잡아내지 못함.
- 언어 모델링에 있어서  “Thom Yorke is the singer of Radiohead” 라는 context와 “Who is the singer of Radiohead” 라는 질문에 대해 기존의 AR로는 “Thom Yorke” 와 “Radiohead” 간의 의존성을 잡아낼 수 없지만 XLNet은 가능

<br>

## 4. Experiments
![image](https://user-images.githubusercontent.com/41243762/103734674-da3c4500-502f-11eb-8786-6a525550892d.png)
- 나머지 실험 추가 해야함..
---
### 참고
- paper : [XLNet: Generalized Autoregressive Pretraining
for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf)
- [ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2019/09/11/xlnet/) | [novdov's blog](https://novdov.github.io/machnielearning/nlp/2019/07/13/XLNet-%EB%A6%AC%EB%B7%B0/) | [PINGPONG blog](https://blog.pingpong.us/xlnet-review/)
