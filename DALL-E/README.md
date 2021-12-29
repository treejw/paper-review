# [DALL·E: Zero-Shot Text-to-Image Generation](https://arxiv.org/pdf/2102.12092.pdf) `2021.02`

## Abstract
- 기존 Text-to-Image Generation 분야는 fixed dataset에 대한 아닌 더 좋은 모델, 알고리즘을 찾는 것에 초점을 두고 연구되어 왔음.
   - 이에따라 더 복잡한 구조, 손실 함수, 추가 라벨 등을 필요로 하곤 했음. <br><br>
- 본 논문은 복잡한 가정들을 필요로 하지 않는 간단한 접근법을 제안
   - Text 및 이미지 토큰을 single stream으로 하여 auto-regressive하게 모델링하는 Transformer를 기반으로 함.
   - 충분한 데이터와 모델 scale이 주어지면, 이 접근법은 zero-shot 스타일로 평가했을 때에도 domain-specific model들과도 비슷한 성능을 보임.

<br><br>

## Method

![image](https://user-images.githubusercontent.com/42428487/147643409-cc3713d6-9740-429d-addd-ce906f5b3408.png)

<br>

## Data Collection (Text-Image Pairs)

- 1.2-billion parameters를 학습했을 때, Conceptual Cations 데이터 3.3M를 사용했었음
- 12-billion parameters로 모델 사이즈를 늘리기 위해, JFT-300M와 비슷하게 250M개의 데이터쌍을 수집함.
```
- Conceptual Captions (3.3M) 
- filtered subset of YFCC100M -> including MS-COCO validation images
- text-image pairs from Wikipedia
```

- 필터링 된 samples
```
- too short cations
- non-English cations
- "photographed on <date>" 같은 caption인 경우
- 이미지 비율이 [1/2, 2]에 포함되지 않는 경우 (즉, 한쪽으로 너무 긴 이미지인 경우 제외)
```

<br>

## 2-Stage Training

## Stage 1: Learning the Visual Codebook
- DALL-E의 목표: Text와 이미지 토큰을 "single stream"으로 하여 Transformer를 학습시키는 것.
- 이 때, 이미지 토큰으로서 픽섹을 그대로 활용하게 되면 고해상도의 이미지일 경우 엄청난 메모리가 소요됨.
- 따라서 VQ-VAE를 통해 이미지를 압축하여 이미지 토큰으로서 사용.
- **gumble-softmax relaxation** 사용.

#### Downsampling of Input Image 

<img src="https://user-images.githubusercontent.com/42428487/147645999-6afcc8b1-373c-4c37-9955-3fcf2eba280f.png" width="500">

- **discrete VAE** 를 활용해 256x256 RGB image를 32x32 grid of image tokens 으로 압축.
   - 이때, 각 token은 8,192가지 값을 가질 수 있음 (각 토큰의 정보 벡터 사이즈 = 8,192)
- 이러한 압축을 통해 transformer가 처리해야 하는 context 크기를 192(8x8x3)배 압축하면서도 visual quality는 어느정도 유지


<br>
<details><summary>(참고) VQ-VAE</summary>
<img src="https://user-images.githubusercontent.com/42428487/147646789-c2924cc8-ade2-4405-979e-7d1ce55e5c3f.png">
</details>




<br><br>


## Stage 2:  Learning the Prior
![image](https://user-images.githubusercontent.com/42428487/147652232-f038bc1b-3bfc-4016-b3e8-a06ebe2876a6.png)
- 텍스트와 이미지 쌍이 주어질 때, 
   - 캡션은 소문자화한 후 16,384개 단어 사전을 사용해 BPE-encode한다. (최대 256 토큰)
   - 이미지는 32x32 = 1024 토큰으로 인코딩 <br><br>
- 최대 256 BPE-encoded text tokens + 1024(32x32) image tokens를 concat하여 Transformer의 입력으로 사용


<br>

### Sparse Attention Masks
- 여기서 사용되는 Transformer 모델은 Decoder-only model로, 이미지 토큰은 64개의 self-attention layer에서 모든 텍스트 토큰에 접근 가능.
- self-attention mask를 어떻게 사용하느냐에 따라 세 가지 변형이 가능 <br><br>
   - text-to-text attention : 항상 casual mask 적용
   - **image-to-image attention : row / column / convolutional attention mask 적용 가능**

![image](https://user-images.githubusercontent.com/42428487/147659831-350919e9-dce5-44f1-94e4-c108dffc7558.png)

> - 최대 텍스트 길이가 6개 토큰, 이미지 길이가 16(4x4 그리드)개 토큰일 경우, 3가지 유형의 attention masks
> - (d) causal convolutional attention pattern with a 3x3 kernel (본 논문에서는 11x11 kernel 사용)
- convolutional attention mask는 마지막 block에서 사용
- 연속된 4개의 block에서 row-row-column-row masks가 순차적으로 사용.

<br>

#### Transformer 학습 방식
- transformer decoder는 다음 토큰을 예측하는 방식으로 학습됨.

<img src="https://user-images.githubusercontent.com/42428487/147657989-39d5c902-20de-43b3-86c7-a6259606422f.png" width="300">


<br>

### Training Objective
- Text와 이미지 토큰에 대한 결합 확률 분포를 학습

#### 학습 방식은 ELB(Evidence lower bound)를 최대화하는 것과 유사

<br>

- text / image tokens의 결합 분포 모델링

<img src="https://user-images.githubusercontent.com/42428487/147652375-ecdf85df-eaa8-4799-bca6-42645c9bb8bd.png" width="400">

```
x: 이미지
y: 캡션 (Text)
z: 인코딩된 이미지 토큰
```

<br>

- 이 모델의 lower bound
<img src="https://user-images.githubusercontent.com/42428487/147656701-1684b1c0-d9de-490a-853b-4790afd6f98b.png" width="450">

```
q_ϕ : RGB image x를 dVAE로 인코딩한 32×32 image token의 분포
p_θ : image tokens를 dvAE로 디코딩한 RGB images의 분포
p_ψ : 트랜스포머로 모델링되는 text tokens, image tokens의 결합분포
```


<br><br>

## Experiments
- 기존 방법론 - AttnGAN, DM-GAN, DF-GAN
- DALL-E - zero-shot 셋팅에서 평가

<br>

### Comparison of samples from our model to those from prior approaches on captions from MS-COCO
> - 트랜스포머로부터 샘플링한 결과물을 pretrained contrastive model을 사용하여 rerank
> - 이미지 캡션에 대한 후보 이미지가 주어졌을 때, contrastive model은 이미지가 캡션과 얼마나 잘 매칭 되는지에 기반하여 점수를 매겨줌

![image](https://user-images.githubusercontent.com/42428487/147658802-297a3542-734a-4cba-afee-c2ebe53ccc3c.png)

<br>

### Human evaluation or DALL-E

<img src="https://user-images.githubusercontent.com/42428487/147658978-4cd5857b-0ce5-4ac5-ba55-47ae427b1522.png" width="550"> <img src="https://user-images.githubusercontent.com/42428487/147659048-84250d97-d728-48cc-81ee-8acb49ff3a4c.png" width="350">

<br>

### Quantitative results on MS-COCO and CUB

![image](https://user-images.githubusercontent.com/42428487/147659186-b9328af3-fa97-4478-bc6a-9cf6b549580b.png)

#### Evalutation Metric
- IS(Inception Score) 
> - 생성된 이미지의 질을 평가하기 위한 척도로, 특히 GAN 모델 평가에 사용된다. IS는 사전학습된 딥러닝 모델을 사용하여 생성된 이미지를 분류
> - <이미지의 질: 이미지는 어떤 특정 물체 같아 보이는가>와 <이미지의 다양성: 다양한 물체가 생성되었는가>로 평가
> -  IS는 최저 점수 1점 ~ 최고 점수 1000점을 가진다 (사전학습 모델이 분류할 수 있는 물체의 클래스 수)

- FID(Fréchet inception distance)
> - 실제 이미지와 생성된 이미지 사이의 feature vector간의 거리를 계산한 점수
> - FID가 낮을수록 좋은 모델




<br><br>

---

- [OpenAI 블로그](https://openai.com/blog/dall-e/)
- 참고 사이트: [youtube](https://youtu.be/az-OV47oKvA) | [blog](https://littlefoxdiary.tistory.com/74) 
- VQ-VAE 설명: [blog](https://greeksharifa.github.io/discrete%20representation/2021/11/07/VQVAE/) 
