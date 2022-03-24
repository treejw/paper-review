## 전통적인 추천 시스템 [🔗](https://lsjsj92.tistory.com/563)

#### 📌 콘텐츠 기반 필터링 (content based filtering)
- 초기에 많이 사용되던 방법

- 사용자가 특정 아이템을 선호하는 경우 그 아이템과 비슷한 콘텐츠를 가진 다른 아이템을 추천해 주는 방식
```
(ex) ItemA(장르 a) ItemB(장르 b) ItemC(장르 a)
     User1이 ItemA에 높은 평점을 줬을 경우, 비슷한 장르인 ItemC를 추천
```

<br>

#### 📌 협업 필터링 (collaborative filtering)
- 사용자의 **행동 양식(user behavior)** 기반으로 추천해 주는 방식 (예. 구매 이력, 평점 등)

- **memory based (nearest neighbor) collaborative filtering**
   > - **User-Item matrix에서 사용자가 아직 평가하지 않은 아이템을 예측하는 것이 목표**
   > 
   > - 오른쪽 표와 같이 `User-Item` 행렬로 변환해주어야 함  <br>
   >   <img src="https://user-images.githubusercontent.com/42428487/159769982-901c9b28-0849-47ba-ad31-4317184634fb.png" width="600"><br>
   > - `사용자 기반` / `아이템 기반`으로 나뉘어짐 _(이중에서는 대체로 `아이템 기반`을 많이 사용한다고 함)_ 

<br>

- **latent factor collaborative filtering**
   > - 행렬 분해(matrix factorization)을 기반하여 사용됨
   >
   > - **대규모 다차원 행렬을 SVD와 같은 차원 감소 기법으로 분해하는 과정에서 latent factor을 찾아내어 추출해내는 방법** <br>
   >   (이때의 latent factor는 comedy, action 같은 장르가 될 수도 있음)
   >   <img src="https://user-images.githubusercontent.com/42428487/159777805-ee06171b-cb38-4952-8f28-71e84ae55049.png" width="600"><br>
   >  ```
   >  - User-Item 행렬 데이터를 이용해 Latent factor을 찾아냄.
   >  - 즉, User-Item`(R) 행렬을 `User-Latent`(P)와 `Item-Latent`(Q) 행렬로 분해 → R = P X Q.T
   >  - 이렇게 분해된 P와 Q를 통해 기존 R에서 몰랐던 값에 대해 구할 수 있게됨. 
   >    (ex) User1-ItemB에 대한 정보가 없을 때, P와 Q를 통해 예측값을 구할 수 있고 이 값이 높은 경우 User1에게 ItemB를 추천할 수 있게됨
   >  ```
   >  - 이 방식은 nearest neighbor collaborative filtering 보다 파라미터 절약이 된다는 장점이 존재하여 아직까지 많이 사용되는 방법임


<br><br><br> 


# [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) `2016.06`

- 2016년도에 구글이 발표한 추천랭킹 알고리즘으로, Google Play의 앱 추천에 사용됨.
- 사용자의 검색어(query)를 바탕으로 추천앱을 정렬하는데 적용됨.

## Abstract

- Memorization과 Generalization은 추천 시스템에서 중요한 요소임.
- Memorization
     - regression이나 classification 문제를 풀때, 일반적으로 linear 모델을 사용함. 이때 feature간의 cross-product는 데이터의 특징을 기억하는데 효과적임. 
     - 그러나, 일반화를 위해서는 Manual feature engineering이 추가적으로 필요함. <br><br>
- Generalization
     - 반면 임베딩을 활용한 deep neural networks의 경우, 추가적인 feature engineering이 덜 필요하며, feature간의 combination을 학습하기에도 좋음.
     - 그러나, 이 방법은 지나친 일반화로 인해, User-Item Interactions이 Sparse하고 High-Rank한 경우, 관련성이 적은 Item을 추천해줄 수 있음.
- 따라서 본 논문에서는 Memorization와 Generalization의 장점을 결합하기 위해 Wide 선형모델과 Deep Neural Networks를 공동으로 학습시키는 Wide & Deep Learning을 제안.

<br><br>

## 1. Introduction
- 추천 시스템에서 memorization와 generalization를 동시에 달성하는 것은 어려움.
- 특징 1
> - **memorization**은 `과거 데이터`에서 사용가능한 상관관계를 사용하고, Item과 Feature의 동시 빈발도를 학습함으로써 정의될 수 있음. 
> - **generalization**은 과거에 거의 발생하지 않았던 새로운 Feature 조합을 탐색함.
- 특징 2
> - **memorization** 기반 추천은 보통 더 topical이며, 사용자가 이미 작업을 수행한 Item에 직접 관련이 있음.
> - **generalization** 기반 추천은 추천된 아이템의 다양성을 향상시키려는 경향이 있음.
- 한계
> - 즉, (cross-product transformation을 이용한) **memorization** 한계점은 학습 데이터에 없는 Query-Item Feature Pair를 일반화하지 못한다는 단점이 존재.
> - **generalization**은 특이한 데이터의 경우를 고려하지 못하고, 잘못된 추천을 할 수 있음.
- 예시
>```
> (ex) [Wide] memorization   -> "참새는 날 수 있다." "독수리는 날 수 있다." "펭귄은 날 수 없다."
>                       └ 비둘기에 대한 정보 X
>      [Deep] generalization -> "날개가 있는 동물은 날 수 있다"
>                       └ 예외) 펭귄
>      [Wide & Deep] generalization + memorizing exceptions -> "날개가 있는 동물은 날 수 있으나, 펭귄은 날 수 없다."
> ```

- 따라서, 아래 그림과 같은 Deep Neural Networks와 Linear Model을 joint training 하여 하나의 모델에서 memorization와 generalization를 모두 성취하는 Wide&Deep Learning Framework를 제안.

![image](https://user-images.githubusercontent.com/42428487/159800414-3be55143-9957-41c3-a64a-e9845492f8e2.png)


<br><br>

## 2. Recommendation System Overflow

<img src="https://user-images.githubusercontent.com/42428487/159800653-b92c23e5-87fc-4bcf-8a3f-ec3901bc8682.png" width="500">

1. User의 검색 **Query** 입력
2. DB로부터 (combination of machine-learned models and human-defined rules 을 통해) 후보 앱들을 **Retrieval**
3. **🔸Ranking🔸(Wide & Deep Learning Framework)** 을 통해 후보 앱들의 점수를 메겨 정렬
   - 점수는 user 정보 x가 주어졌을 때, user가 y앱에 action할 확률인 P(y|x)를 의미함.

<br><br>

## 3. Wide & Deeo Learning

### 3.1 The Wide Component
![image](https://user-images.githubusercontent.com/42428487/159812449-d5f71635-5bdd-41ed-ac49-7feac3b2142e.png)

- wide 모델은 `user_installed_app`과 `impression_app` 2개의 feature을 [cross-product](https://leehyejin91.github.io/post-wide_n_deep/#supplement)한 결과 `x`를 input으로 사용.
- 사용한 선형 모델 : y = w.T*x + b
   - x = [x1, x2, ... , xd] : d Features 벡터 (raw input features와 transformed features를 포함함)
   - w = [w1, w2, ... , wd] : model parameters
   - b = bias

<br>

### 3.2 The Deep Component
![image](https://user-images.githubusercontent.com/42428487/159811923-eae04c2b-9b6a-4bbc-aaf1-d97dc5e86e61.png)

- continuous feature와 Embedding된 categorical feature을 concat한 결과 `a`를 input으로 사용.
- 이때 l_th layer는 아래와 같음 <br>
   <img src="https://user-images.githubusercontent.com/42428487/159810817-4c1ef424-f3bb-4914-8de9-07a59c16602d.png" width="380">
   - f = activation function (ReLU)
   - W = weight matrix
   - b = bias

<br>

### 3.3 Joint Training of Wide & Deep Model
![image](https://user-images.githubusercontent.com/42428487/159811522-b37f232c-ece7-4caf-87ee-90e60d4a1f36.png)
- 앙상블과달리, output의 gradient를 wide와 deep 모델에 동시에 backpropatation하여 학습
- wide & deep model의 prediction은 아래 수식과 같이 구해짐. <br>
   <img src="https://user-images.githubusercontent.com/42428487/159814710-8b0969fe-2eba-441b-83e0-2f89738877c9.png" width="380">
   - 각 모델에서 나온 결과를 더하고 sigmoid 함수를 통과시킨 결과가 최종 output

<br><br>

## 5. Experimnet Results

<img src="https://user-images.githubusercontent.com/42428487/159815006-4c6407c7-bf48-4885-945c-54dc1ccae937.png" width="500">

- 3주간 A/B Testing 프레임워크에서 실시간 온라인 실험을 수행함.
   >  A/B Test: 기존 서비스(A)와 새로 적용하고 싶은 서비스(B)를 통계적인 방법으로 비교하여 새로운 서비스가 기존 서비스에 비해 정말 효과가 있는지를 알아보는 방법
   - **대조군 (control)** 그룹의 경우, User의 1%가 무작위로 선택됨. / Wide-Only logistic regression 모델
   - **실험군** 그룹의 경우, 동일한 Features 집합으로 훈련된 Wide & Deep 모델이 생성한 추천이 제시됨.
- 

<br><br>

--- 
#### 참고
- [lsjsj92's blog](https://lsjsj92.tistory.com/563) | [soobarkbar's blog](https://soobarkbar.tistory.com/131) | [youtube](https://www.youtube.com/watch?v=hKoJPqWLrI4) | [leehyejin91's blog](https://leehyejin91.github.io/post-wide_n_deep/)


