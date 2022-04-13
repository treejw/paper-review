# [MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video](http://staff.ustc.edu.cn/~hexn/papers/mm19-MMGCN.pdf)  `2019`

<br>

## 1. Background Knowledge
**Multimodal**
- 인간은 살아가는 데 필요한 정보를 학습하기위해 대표적으로 5개의 감각 기관으로 부터 수집되는 데이터를 바탕 학습
- 인간의 인지적 학습법을 모방하여 다양한 형태(modality) 데이터로 학습하는 방법
- 즉, **특징 차원이 다른 데이터**를 동시에 학습하는 방법
<br>

**Multimodal learning**
- **Joint representaion**
   - 각각의 modality가 서로 같은 공간으로 합쳐져서 하나의 모델을 통과하는 방법
   - x = f(x1,x2,...)
- **Coordinated representations** 
   - 각각의 modality마다 개별 모델을 통과하는 방법  
   - x = f(x1)g(x2)

<br><br>

## 2. Introduction
- 그래프를 활용하여 새로운 아이템을 추천하는 방법을 제안한다.
- i1은 u1,u2가 공통으로 시청한 영화
- u1에게 새로운 영화를 추천할때
   - u2가 시청한 영화를 추천
   - 이때, u2의 정보를 추가적으로 이용(포스터, 줄거리 등)하면 결과가 달라진다.
   - 아래 그림에서 i1은 Visual Spcae에서i2와 가깝게 맵핑이 되지만, Textual Space에서는i3와 가깝게 맵핑
- 즉,  u1이 Visual modality와 Textual modality 중 어느 것을 더 선호하는지 구분하는 것이 중요하다.
- 아이템에 대한 modality의 정보가 적절히 반영하는것이 중요하다.

![image](https://user-images.githubusercontent.com/53847442/163123936-1f119faa-f0d2-47db-9bc7-4351e3755e5f.png)

<br>

### **Contribution**
1. 다양한 modality의 정보 교환이 어떻게 사용자 선호도를 반영하고, 추천의 성능에 미치는 영향을 연구
2. MGCN이라는 새로운 모델을 제안해 유저-아이템 이분형 그래프에서 각 modality의 정보가 전파되어 아이템 컨텐츠 정보를 활용한 더 나은 유저의 repsentations을 계산
3. Tiktok, Kwai, MovieLense 세 가지 데이터 셋을 활용해 최신 SOTA 모델들과 비교하여 제안 모델의 우수성을 증명

<br><br>

## 3. MODEL FRAMEWORK
![image](https://user-images.githubusercontent.com/53847442/163125750-077deea0-a1a6-46e0-88ff-42915a0a3eb4.png)
1) 각 modality 별로 이분형 그래프(bipartite graph)로 나타내고 
2) Aggregation Layer와 Combination Layer를 통과
3) 중심 노드로부터 연결된 modality의 정보를 표현할 수 있는 최종 노드를 생성
4) 최종적으로 유저 노드를 기준으로 합쳐진 modality와, 아이템 노드를 기준으로 합쳐진 modality가 하나의 벡터를 만들어 행렬 연산을 통해 유저에게 새로운 아이템을 추천한다.

```
참고로, 각 modality마다 feautres vector로 만들어 주기 위해

Visual modality:pre-trained ResNet50
Acoustic Modality:VGGish
Textual Modality:Sentence2Vector
```

### 3.2 Aggregation layer
- 중심 노드로부터 이웃이 되는 노드들의 정보들을 합쳐주는 역할
- hm = f (Nu)
- 중심 노드로부터 연결된 이웃 modality의 기여도를 수집
- Mean Aggregation, Max Aggregation
<br>

![image](https://user-images.githubusercontent.com/53847442/163156494-b9d52e0e-b1b6-460f-a2f5-91271bc418c4.png)
![image](https://user-images.githubusercontent.com/53847442/163156565-bdcf7670-5af8-42be-87d4-17deb0874424.png)

```
Nu는 유저 노드(중심이 되는 노드)의 이웃, 즉 유저 노드(중심 노드)를 기준으로 상호작용 있는 이웃 노드를 의미
W1,m은 이웃 노드들의 정보를 얻기 위한  trainable transformation matrix를 의미
jm은 modality m에 속하는 아이템을 의미
```

### 3.3 Combination layer
- aggregation layer를 수행하여 얻은 노드의 구조적 정보 hm과 유저 노드가 해당 modality에서 가지고 있었던 본래의 정보 um 그리고 유저 노드에서 각 modal에 속하는 아이템의 연결 정보를 알려주는 uid를 통해 하나의 통합된 representation으로 만든다.
- 이때 각 modality의 features vector는 서로 다른 차원을 가지게 되는데, ID 임베딩 차원과 같아질 수 있도록 
W2 trainable weight matrix를 곱해주어 모든 modality가 같은 공간에 표현될 수 있도록 만든다.
<br>

![image](https://user-images.githubusercontent.com/53847442/163229111-0f6c697e-32d1-439d-978d-993473d2f950.png)
<br>

- Combination 방법
  - Concatenation Combination: Aggregated 된 정보와 modality의 고유 정보를 독립된 정보로 가정하고 concat
  - Element-wise Combination: 두 정보 간 상호작용을 고려


  ![image](https://user-images.githubusercontent.com/53847442/163229515-45f69cf7-2d96-414a-b204-31ed801412b7.png)
<br>

### 3.4 Model Prediction
- 각 modality 별로 aggregation layer와 combination layer를 여러 번 쌓아서 유저-아이템 그래프 간의 고차연결성(high-order connectivity) 구조를 만든다.
- 아이템 노드에서도 위와 유사한 방식으로 representations을 만든다.
- modality마다 구한 representations을 합쳐서 아래와 같이 modality의 특성들이 반영한 user vector와 item vector를 구한다.
- u(m)L 은L번째 multi-modal의 combination layer의 출력값을 의미
<br>

![image](https://user-images.githubusercontent.com/53847442/163230109-b1af2ca9-c937-4042-a592-ef9dcbdc71ef.png)
(최종 유저, 아이템 representation)
<br>

- update는 추천시스템에서 흔히 사용되는 Bayesian Personalized Ranking loss를 사용
- loss는 실제 유저가 관찰한 아이템(i)과 관찰하지 않은 아이템(i′)의 차이를 계산하여 i와i′의 점수가 극대화되도록 모델 파라미터를 업데이트
- 
<br><br>

## 4. Experiments
- 데이터 셋은 Tiktok, Kwai, MovieLens 데이터를 사용해서 실험을 진행하였으며 
- Kwai 데이터 셋을 제외하고 모두 Visual, Acousitc, Textual modalities를 활용

<br>

![image](https://user-images.githubusercontent.com/53847442/163233324-79cf0a5d-de18-4629-ae29-b34891806733.png)

<br>

```GraphSAGE, NGCF는 Multi-modal을 적용한 모델이 아니기 때문에 공평성을 위해 방법을 바꿔 실험 
- GraphSAGE: multi-modal features를 노드의 features로 통합하여 모델을 학습
- NGCF: multi-modal features를 side-information으로 간주하여 모델을 학습
  (이에 대한 정확한 설명은 없지만 NGCF의 Embedding Layer에서 modality 정보를 반영했다고 판단함)
```

<br>

![image](https://user-images.githubusercontent.com/53847442/163233527-2e424554-6ea0-43cf-b575-97dbd9edf7aa.png)

<br>

- 유저 5명을 랜덤 샘플링해서 다른 modality에 대해 선호도의 차이가 있는지 t-SNE를 통해 시각화
- 노란색으로 표현된 유저1의 경우 Visual Modality에서 뚜렷한 차이를 보이지 않고 분산됨, Textual Modality에서는 전쟁과 로맨스의 특성들로 나누어짐
- 파란색으로 표현된 유저2의 경우 Visual Modality에서 고전주의 포스터와 애니메이션 포스터로 특성들이 나누어졌지만, Textual Modality에서는 특징 없음 
- 이를 통해서 유저들이 서로 다른 modality에서 각각 다른 선호도를 갖고 있다는 것을 증명함

<br><br>

## 5. Conclusion
4. Conclusion
- micro-video에서 좋은 추천을 제공하기 위해 유저와 아이템의 연결성을 고려하는 것이 중요할 뿐만 아니라 아이템 컨텐츠의 다양한 modality들을 고려하는 것이 중요
- 각각의 modality의 representation을 학습하기 위해 유저-아이템 간의 그래프 구조를 활용한 MMGCN 모델을 제안
- 해당 모델의 경우 aggregation layer와 combination layer를 통해 modality의 특성들을 반영할 수 있었으며, Tiktok, Kwai, MovieLens 데이터에서 제안 모델의 우수성을 보임
