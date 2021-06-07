# MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices `2020.04`


## Abstract
- 기존 NLP 모델들은 성능이 뛰어나지만 모델의 크기가 매우 커 mobile device에서 사용이 어려움.
- 본 논문에서는 **knowledge transfer**을 적용한 **MobileBERT**을 제안.
   - knowledge transfer : Teacher Model(IB-BERT) → Student Model(MobileBERT)
   - 모델의 depth를 줄이려는 기존의 방식들과 달리, 본 논문은 width를 줄이는 방향으로 모델 크기 감소. 

- MobileBERT는 BERT_base 모델과 유사한 성능을 가지며, 크기가 4.3배 작고 속도는 5.5배 더 빠름.

<br><br>


## 3. MobileBERT

### 3.1 Bottleneck and Inverted-Bottleneck

![image](https://user-images.githubusercontent.com/42428487/120944240-67660000-c76e-11eb-954f-12d09039cae2.png)

**Figure 1. (a) BERT (b) IB-BERT: Inverted-Bottleneck BERT (c) MobileBERT**

<br>

![image](https://user-images.githubusercontent.com/42428487/120955794-29c49f80-c78d-11eb-8798-a449325dc033.png)

<br>

#### MobileBERT (Bottleneck Structure)

- MobileBERT는 BERT-large처럼 24개의 layer를 가진 깊은 구조이지만, 각 block의 hidden 차원(h_input)은 128로 작게 설계되어 있음.
- 본 논문에서는 input/output 차원을 512로 조정하기 위해 2개의 linear transformation을 추가함. → **bottleneck structure**
- 위와 같은 deep & thin한 구조의 network를 학습시키는 것은 어려운 일이므로, Teacher Model을 학습시킨 후 Student Model(MobileBERT)로 Knowledge transfer을 수행.

<br>

#### IB-BERT (Inverted-Bottleneck Structure)
- IB-BERT는 MobileBERT의 Teacher Model로, BERT-large와 같은 MHA(h_output)과 FFN을 갖지만, MobileBERT로 knowledge transfer를 할 수 있도록 input/output을 512로 만드는 linear layer를 포함한다.
- 위와 달리, input이 핵심 모델보다 차원이 작으므로 **Inverted Bottleneck Sturcture**이다.


<br><br>


### 3.2 Stacked Feed-Forward Networks

- Transformer에서 MHA와 FFN은 서로 다른 역할을 수행하고 있음
   > MHA: 모델이 서로 다른 공간에서 오는 정보에 동시에 attend할 수 있도록 함. <br>
   > FFN: 모델의 비선형성을 증가시킴.

- 기존 BERT 모델에서는 MHA와 FFN의 파라미터 개수의 비율이 항상 1:2로 유지되어 있었음.
- 그러나, Bottleneck 구조를 도입하면서 Multi-Head Attention (MHA) 모듈과 Feed-Forward Network (FFN) 모듈 사이의 균형이 무너진다는 단점이 존재.
   > Bottlenneck 구조를 적용하면, MHA의 input은 더 넓은 feature map(inter-block 크기)에서 오는 반면, FFN의 input은 더 얇은 bottleneck (intra-block 크기)에서 옴.

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Frv4vG%2FbtqQefAV1u8%2FLtM5IRCx13qQyzfWyNtQ2k%2Fimg.png)
- MHA는 inter-block flow(빨간 선)에 따라 512차원을 input으로 받아 4개의 attention head 적용 → 128차원의 output 생성
- FFN은 intra-block flow(파란 선)에 따라 128차원의 MHA output을 입력으로 받아 512차원의 intermediate hidden을 거쳐 128차원의 output 생성
▶ MHA가 FFN에 비해 상대적으로 더 많은 파라미터를 보유하므로, 본 논문에서는 FFN을 단순히 여러층 쌓은 구조인 **stacked FFN**을 적용하여 위와 같은 문제를 해결.


<br><br>


### 3.3 Operational Optimization

- 본 논문에서 모델의 latency를 분석한 결과, Layer Normalization와 GELU activation이 latency의 상당 부분을 차지한 것을 알아냄.
- MobileBERT에서는 아래와 같이 MobileBERT에 새로운 operation로 대체함.

#### [Layer Normalization](https://wikidocs.net/31379#9-residual-connection-layer-normalization) 제거
- n-channel의 hidden state h에 대한 layer normalization은 element-wise linear transformation으로 대체
   > ![image](https://user-images.githubusercontent.com/42428487/120969696-6ef4cb80-c7a5-11eb-9ff2-a37a7bd3d80d.png) <br>
   > - γ, β는 n차원 벡터 <br>
   > - ◦ 는 Hadamard product(두 행렬의 각 성분을 곱하는 연산)를 의미

#### ReLU activation 사용
- GELU 연산보다 간단한 연산인 ReLU를 사용.


<br><br>


### 3.4 Embedding Factorization
- Embddding 부분은 모델 크기의 상당 부분을 차지하므로, MobileBERT에서는 Embedding layer를 압축하기 위해 Embedding size를 128로 설정함.
- 이후, 1D convolution (with kernel size: 3)을 이용하여 raw token embedding을 512 차원의 output으로 만들어줌.


<br><br>


### 3.5 Training Objectives
- 본 논문에서는 2가지 knowledge transfer objectives를 사용.

#### 1. Feature Map Transfer (FMT)
- 기존 BERT의 각 layer는 이전 layer의 output을 input으로 사용하므로, 각 layer가 반환하는 feature map이 Teacher Model과 최대한 비슷하게 나오는 것이 중요함.
- 따라서 Teacher Model과 MobileBERT의 feature map 사이의 Mean Squared Error loss를 목적함수로 사용.
   > ![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc5IOJf%2FbtqQstxWh9L%2FIGi6qxgZUo4cPm09hj1Dk1%2Fimg.png) <br>
   > _l_은 layer index, _T_는 sequence 길이, _N_은 feature map size

<br>

#### 2. Attention Transfer (AT)
- 각 attention head에서 Teacher Model과 MobileBERT의 self attention 분포간의 KL-divergence를 활용
   > ![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbhP1xj%2FbtqQu3MexEh%2F05Xrp2KJBYEFfULJKxXOHk%2Fimg.png) <br>
   > _A_는 attention head 개수 <br>
   > 각 layer에서 attention transfer를 위해 Teacher Model과 Student Model 모두 attention head 개수를 4로 설정. 

<br>

#### 3. Pre-training Distillation (PD)
- layer간의 knowledge transfer와 더불어 MobileBERT를 pre-training 할 때에도 knowledge distillation loss를 사용함.
- original MLM(masked language modeling) loss와 NSP(next sentence prediction) loss를 조합하여 **새로운 MLM KD(knowledge distillation) loss**를 사용함.
   > ![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FC0mqN%2FbtqQstrhrpH%2F5Iz8gpI7qyrKpHokvYHh2K%2Fimg.png) <br>
   > α: 0 ~ 1 사이의 hyperparameter <br>
   > L_KD 는 teacher model과 student model 사이의 MLM에 대한 knowledge distillation loss


<br><br>


### 3.6 Training Strategies
![image](https://user-images.githubusercontent.com/42428487/120973001-538bbf80-c7a9-11eb-9040-26a49fb7703a.png)
- MobileBERT는 위에서 정의한 objective function을 통해 학습하는데, 이때 사용할 3가지 전략을 제안.

#### Auxiliary Knowledge Transfer (AKT)
- 각 layer마다 수행하는 knowledge transfer와 전체 BERT의 MLM에 대해 수행하는 knowledge distillation를 결합해 하나의 loss로 사용하는 방식

#### Joint Knowledge Transfer (JKT)
- 각 layer마다 knowledge transfer를 우선적으로 수행하고, 모두 완료되고 나면 그 뒤에 knowledge distillation을 수행

#### Progressive Knowledge Transfer (PKT)
- MobileBERT는 IB-BERT Teacher Model을 완벽하게 따라하지 못할 수도 있고, 이전 layer에서 잘못된 결과가 뒤의 layer의 knowledge distillation을 방해할 수 있음.
- 따라서, knowledge transfer를 각 layer마다 순차적으로 수행. 단, 이전 layer에 대해서는 freezing한 후 학습 
- (실제로는 아예 freezing이 아닌, 더 이전 layer일수록 더 작은 learning rate을 사용하는 방식으로 조절)


<br><br>

## 4. Experiments

### 4.1 Model Settings
- IB-BERT Teacher Model과 MobileBERT Model의 최적의 setting을 찾기위해 SQuAD에 대해 F1 score를 비교.
- batch size는 2048로 125k steps 학습. (기존 BERT의 1/2 수준)


### Architecture Search for IB-BERT
![image](https://user-images.githubusercontent.com/42428487/120974937-a9616700-c7ab-11eb-91d6-52d2ff9689d7.png)
- h_inter: embedding 크기, h_intra: model 크기
- 저자는 h_inter값이 512이 되는 시점부터 감소할수록 성능이 감소하는 것을 확인함. → 따라서 IB-BERT의 h_inter를 512로 정함.
- 마찬가지로 h_intra도 1024부터 감소시켜 성능 확인함 
   - h_intra 값이 감소할수록 모델의 성능이 급격히 감소하는 것을 확인. → 따라서 h_intra 값은 BERT-large와 동일한 1024로 정함.
   - (h_intra는 BERT 모델의 representation power를 의미하는 것이므로 값을 줄일수록 성능이 하락하는 것으로 보임)

<br>

### Architecture Search for MobileBERT
![image](https://user-images.githubusercontent.com/42428487/120975806-9d29d980-c7ac-11eb-8f2c-33ac89ffea1f.png)
- MHA와 FFN 사이의 적절한 비율을 찾기 위한 실험 진행.
- 대체로 MHA와 FFN의 파라미터 수 비율이 0.4 ~ 0.6일 때 모델의 성능이 가장 좋은것을 확인.
- 따라서 본 논문에서는 h_intra는 128, FFN은 4로 정함.
- 추가로 multi-head 개수는 성능에 큰 영향을 미치지 않는다는 사실을 발견. (Table 2의 (c)외 (f)로 확인 가능)


<br>

### 4.3 Results on GLUE
![image](https://user-images.githubusercontent.com/42428487/120976585-70c28d00-c7ad-11eb-9bbf-45cf6850080a.png)
- MobileBERT_TINY: MHA의 크기를 줄여 FFN을 stacking하지 않은 model
- MobileBERT_w/o_OPT: latency를 줄이기 위해 도입한 operational optimization을 제거한 model

<br>

### 4.4 Results on  SQuAD
![image](https://user-images.githubusercontent.com/42428487/120977215-0b22d080-c7ae-11eb-806f-fda0d28f6e9a.png)
- SQuAD에서는 BERT-base 모델의 성능보다 우수함.

<br>

### 4.6 Ablation Studies

### 4.6.1 Operational Optimizations
![image](https://user-images.githubusercontent.com/42428487/120977631-7ff60a80-c7ae-11eb-9a86-10ed57afd613.png)

### 4.6.2 Training Strategies
![image](https://user-images.githubusercontent.com/42428487/120977424-4a512180-c7ae-11eb-83f9-2528c34a4543.png)

### 4.6.3 Training Objectives
![image](https://user-images.githubusercontent.com/42428487/120977755-a9169b00-c7ae-11eb-99cf-d9942a90af87.png)




<br><br>

---
### 참고
- [paper](https://arxiv.org/pdf/2004.02984.pdf)
- [brunch](https://brunch.co.kr/@choseunghyek/4) | [Hansu Kim's Blog](https://cpm0722.github.io/paper-review/mobilebert-a-compact-task-agnostic-bert-for-resource-limited-devices) | [LittleFox's Blog](https://littlefoxdiary.tistory.com/65)
