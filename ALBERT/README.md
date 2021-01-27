# ALBERT: A Lite BERT for Self-supervised Learning of Language Representations `2019.9`


## Abstract
일반적으로 모델의 크기가 커짐에 따라 성능이 향상되는 경향 존재한다. 그러나 아래와 같은 문제가 발생할 수 있다.
>   - GPU/TPU Memory Limitation
>   - Longer Training Time

이 논문에서는,
   - (적은 메모리 소모, 학습 속도 향상)을 위한 2가지 parameter reduction technique을 제안.
   >   - **Factorized embedding parameterization**
   >   - **Cross-layer parameter sharing**

   - 또한, Next Sentence Prediction(NSP) 대신 **Sentence order prediction(SOP)** 을 사용하여 학습.

그 결과, BERT-large에 비해 파라미터 수가 적으면서 더 좋은 성능을 보이는 모델인 ALBERT 제안.



<br>

## 3. THE ELEMENTS OF ALBERT

### 3.1 MODEL ARCHITECTURE CHOICES

ALBERT는 GELU를 사용하는 transformer encoder를 사용한다는 점에서 BERT와 유사하다. 또한, BERT의 표기법을 따른다.
> `E` : Vocabulary embedding size  `L` : The number of encoder layers  `H` : the hidden size

- BERT 모델과 마찬가지로, feed-forward/filter size는 *4H*,  attention heads 수는 *H/64*로 세팅함.

#### ◼ Factorized embedding parameterization

<p align="center"><img src="https://user-images.githubusercontent.com/42428487/105883781-96df6000-604a-11eb-88fc-dd7ce806ee04.png"></p>

- BERT, XLNet, RoBERTa에서 WordPiece Embedding Size(E)와 Hidden Layer Size(H)는 항상 서로 같게 설정되었다. (예: 768)

   → 위의 설정은 modeling 측면과 practical 측면에서 볼 때, 최선의 선택이 아니다.
   
   - modeling 측면
      > - WordPiece Embeddings : context-independent representation 학습
      > - Hidden-layer Embeddings : context-dependent representations 학습
      - 담고있는 정보의 양이 다르므로, E << H로 설정하여 모델 파라미터 작게 만들 수 있음.
      
   - Practical 측면
      - Vocab size(V)는 큰 값임. 
      - 이때 E와 H를 같게할 경우 → Embedding Matrix의 크기가 V×H(E)가 됨. (파라미터 개수가 매우 많음)


- 따라서, 이 논문에서는 Embedding parameter을 2개의 작은 matrix로 분해함.
   - 즉, 중간에 임베딩 Layer를 하나 추가하는 방식으로 볼 수 있음. (ALBERT에서는 E 값을 128로 설정함)
   - 기존 `(V×H)`인 파라미터 개수를 `(V×E + E×H)`로 감소시킨 것.
   - (예시) BERT(V=30000, E=H=768) => 약 2300만 // ALBERT(V=30000, E=128, H=768) => 약 390만 (대략 5배 감소)

<br>

#### ◼ Cross-layer Parameter Sharing

<p align="center"><img src="https://user-images.githubusercontent.com/42428487/105895471-e75dba00-6058-11eb-932b-038b67ed4fd7.png"></p>

- 기존에 이미 **Feed Forward Network(FFN) 파라미터만 공유** 또는 **Attention 파라미터만 공유**하는 여러가지 방법이 존재함.

- 그러나, ALBERT는 layer의 모든 파라미터를 공유하는 방식임. 
- 즉, BERT가 transformer block 1번~12번을 거쳤다면, ALBERT는 1개의 transformer block을 12번 거치는 방식.

![image](https://user-images.githubusercontent.com/42428487/105889180-3dc6fa80-6051-11eb-893a-98020de4e742.png)
- 모든 레이어를 공유한 결과, 각 레이어의 input, output embedding의 L2 distance랑 cosine similarity이 BERT에 비해 smooth 해짐. 

   → 이 말은 weight 공유가 Network 파라미터를 안정화하는데 영향을 미친다는 의미.

<br>

#### ◼ Inter-sentence coherence loss

- BERT에서 사용된 Next Sentence Prediction(NSP)는 downstream task 성능을 높이기 위해 만들어졌지만,
- 이 논문의 저자들은 NSP가 비효율적이다라고 판단하고, 새로운 방법을 만든 것이 Sentence order prediction(SOP)이다.

#### NSP 
> - 두번째 문장이 첫문장의 다음 문장인지를 맞추는 방식으로 학습.
> - 학습 데이터 구성 시 두 번째 문장은 실제 문장(positive example) 혹은 임의로 뽑은 문장(negative example)으로 구성.
- 문장 쌍 간의 관계에 대한 추론이 필요한 NLP task에 대한 성능을 향상시키기 위해 사용된 것.
- 하지만, 이러한 NSP는 임의로 뽑은 문장은 첫 문장과 완전히 다른 Topic의 문장일 확률이 높으므로 ,
- 문장 간 연관 관계를 학습한다기 보다는 두 문장이 **같은 Topic에 대해 말하는지를 판단하는 Topic Prediction**에 가깝다.

#### SOP
> - 문장의 순서가 옳은지 여부를 예측하는 방식으로 학습.
> - 실제 연속인 두 문장(Positive Example)과 두 문장의 순서를 앞뒤로 바꾼 것(Negative Example)으로 구성.
- 같은 Topic 내에서 2개의 문장의 연속여부를 보는 것이므로, NSP보다 두 문장 간의 연관 관계를 보다 더 잘 학습할 것임.

![image](https://user-images.githubusercontent.com/42428487/105890489-c4c8a280-6052-11eb-89e0-a505443d51fa.png)
- 표5에서 볼 수 있듯이, NSP로 학습 시 → NSP는 높지만, SOP는 상당히 낮음.
- 그러나 SOP로 학습 시 → NSP, SOP 모두 도 괜찮은 성능임.


<br>

### 3.2 MODEL SETUP

![image](https://user-images.githubusercontent.com/42428487/105892524-546f5080-6055-11eb-81ad-e0aa57917040.png)

<br>

## 4. EXPERIMENTAL RESULTS

### OVERALL COMPARISON BETWEEN BERT AND ALBERT
![image](https://user-images.githubusercontent.com/42428487/105894946-37884c80-6058-11eb-80cb-c244e8ba91bb.png)

### CROSS-LAYER PARAMETER SHARING
![image](https://user-images.githubusercontent.com/42428487/105895048-5ab2fc00-6058-11eb-9dc8-aee628dcc7b8.png)

### WHAT IF WE TRAIN FOR THE SAME AMOUNT OF TIME?
![image](https://user-images.githubusercontent.com/42428487/105895105-6e5e6280-6058-11eb-9a3e-9b680caf25a9.png)


<br><br>

---
### 참고
- paper : [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf)
- [y-rok's Blog](https://y-rok.github.io/nlp/2019/10/23/albert.html) | [jeongukjae's Blog](https://jeongukjae.github.io/posts/4-albert-review/)
