# [BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) `2021.02`

## Abstract
- NLP BERT와 같이 Mask 방식으로 학습하는 방법 제안 <br><br>

### Contributions
-  self-supervised 방식으로 비전 트랜스포머를 사전 훈련하기 위한 마스크 이미지 모델링 작업을 제안하고, Variational Autoencoder의 관점에서 이론적 설명을 제공.
- BEIT를 사전 훈련하고 image classification, semantic segmentation 같은 다운스트림 작업에 대한 광범위한 fine-tune 수행
- 우리는 self-supervised BEIT의 self-attention 메커니즘이 label을 사용하지 않더라도 semantic regions과 경계를 구별하는 방법을 배운다는 것을 보여준다.

<br><br>

## Method

![image](https://user-images.githubusercontent.com/41243762/165305346-879d9e95-9cd5-48b7-8995-32d85ae977c8.png)

patched image: 14 x 14 x (16 x 16 x 3) <br>
visual token: 14 x 14

<br>

## Masked Image Modeling(MIM)
![image](https://user-images.githubusercontent.com/41243762/165308571-7dbec1bd-a123-4845-93a9-bcce73597253.png)

### 학습 방법
1. **discrete VAE**로 visual token 생성
2. 40%의 patch를 randomly block-wised masking
3. masked image에 대한 visual token 예측
4. ![image](https://user-images.githubusercontent.com/41243762/165310653-d9fdd3f4-56f2-4014-90dd-8c85681878b3.png)


<br>

##  Experiment
### pre-training setup
1. VIT-Basse
   - 12-layer, 768 hidden, 12 attention heads, 16x16 patch
3. ImageNet-1k
   - random resized cropping, horizontal flipping, color jittering
   - randomly mask
   - 224 x 224 (14x14 patched image)
5. Hyperparameter
   - 500k step
   - 2k batch size

### Image classification
![image](https://user-images.githubusercontent.com/41243762/165311612-63b1da22-50ea-4546-8040-14bf83c2f5be.png)

### Semantic Segmentation
![image](https://user-images.githubusercontent.com/41243762/165311831-fd44f820-2723-4e6c-aef2-77947f919597.png)


### Ablation Studies
![image](https://user-images.githubusercontent.com/41243762/165311975-26468ab6-0b69-4d25-a6ca-e3b871107cae.png)

###  Analysis of Self-Attention Map
![image](https://user-images.githubusercontent.com/41243762/165312011-1516b4ff-05fb-474c-9eac-e1e3df773b69.png)




<br><br>

---

- 참고 사이트: [youtube](https://www.youtube.com/watch?v=uCWhUayAwOY) | [blog](https://velog.io/@rucola-pizza/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0BEIT-Pre-Training-of-Image-Transformer)
