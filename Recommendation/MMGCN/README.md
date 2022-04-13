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

참고로, 각 modality마다 feautres vector로 만들어 주기 위해 
```
Visual modality:pre-trained ResNet50
Acoustic Modality:VGGish
Textual Modality:Sentence2Vector
```

<br><br>

## 4. Experiments

<br><br>
