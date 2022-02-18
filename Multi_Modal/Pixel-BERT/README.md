# [Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers](https://arxiv.org/pdf/2004.00849.pdf) `2020.06`

## Abstract
- 기존 multi-modal 방법론에서 image feature을 추출하기 위해 사용된 object detection은 language understanding을 하기에 충분하지 않음.
  - object들이 아닌 배경 정보, object들이 겹쳤을 때의 정보들이 제대로 활용되지 않을 수도 있음. <br><br>
- 따라서, image와 text 사이의 관계를 더 잘 학습하기 위해 **Pixel-BERT** 제안.
- Pixel-BERT : visual & language embedding을 같이 학습하는 single stream deep multi-modal transformers
   - language embedding : BERT와 동일
   - image embedding : image pixels로 부터 CNN으로 feature 추출

<br><br>

## Introduction
### Cross-modality learning in vision and language 연구 흐름
### ◼ Visual backbone: pre-trained CNN feature
> ![image](https://user-images.githubusercontent.com/42428487/154512259-9645c447-caa7-4a33-ae65-9b4b5fe6d55b.png)
> ###### MUTAN: Multimodal Tucker Fusion for Visual Question Answering (2017, ICCV) 

<br>

### ◼ Visual Genome Dataset → cross-modality learning 연구 발전에 기여
> <img width="600" src="https://user-images.githubusercontent.com/42428487/154513745-78015092-2d51-43dd-bfda-7164ea8997b5.png"> 
> 
> ###### Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations (2017, IJCV)

<br>

### ◼ Visual backbone: faster-RCNN
> <img width="300" src="https://user-images.githubusercontent.com/42428487/154525013-03701467-cf72-4fe4-9bba-3e67f2d835a0.png"> <img src="https://user-images.githubusercontent.com/42428487/154525841-14594ca8-c807-4316-bcfe-958c6f4920d5.png">
>
> ###### Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering (2018, CVPR)

<br>

### ◼ Transformer을 통해 2개의 modality dense coneection 학습
> <img width="600" src="https://user-images.githubusercontent.com/42428487/154526859-439a32c6-fc0d-46b3-a17d-e31b7d10b7ac.png"> 
> 
> ###### Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks (2020, ECCV)


<br><br>

## Approach
### ◼ Overall architecture of Pixel-BERT
<img width="800" src="https://user-images.githubusercontent.com/42428487/154528383-3b05e38c-b369-4aa9-9975-f0be413a946c.png"> 

#### Sentence Encoder (Sentece Feature Embedding)
```
◾ BERT의 Embedding 방식과 동일 (position, token, semantic embedding 합쳐서 사용)
```
#### CNN-based Visual Encoder (Image Feature Embedding)
```
◾ CNN Backbone(ex. ResNet, ResNeXt)으로 feature 추출
◾ (semantic embedding : language embedding과 구별하기 위해 추가)
◾ 2 × 2 max pooling
◾ Random Sampling (단, pre-training 때만 사용)
```
#### Cross-Modality Module
```
◾ Transformer에 들어가는 입력 형태: 
  [CLS] language tokens [SEP] image tokens
  = {[CLS], w1, w2, · · · , wn, [SEP], v1, v2, · · · , vk}
```

<br>

### ◼ Pre-training : (1) MLM (2) ITM (3) Pixel Random Sampling
#### (1) MLM (Masked Language Modeling)
```
BERT의 pre-training 중 MLM 방식과 거의 유사
◾ language tokens에서 0.15 확률로 랜덤으로 mask
◾ 나머지 non-masked tokens(language tokens)와 visual tokens을 통해 mask token을 예측
```
#### (2) ITM (Image-Text Matching)
```
BERT의 pre-training 중 NSP(Next Sentence Prediction) 방식과 거의 유사
◾ text와 image 쌍에서 매칭이 올바른 경우 = positive samples
                            잘못된 경우 = negative samples
◾ [CLS] token을 통해 binary classifier
```
#### (3) Pixel Random Sampling
```
◾ (Dropout에서 영감을 얻음)
◾ CNN과 Max Pooing의 결과로 나온 결과에서 랜덤으로 100개의 pixel을 추출하여 image feature로 사용
==> robustness 향상 
==> computation cost 감소 (100 x Dim)  // (참고. object detection 방식일 경우 (region N x Dim)
```

<br>

## Experiments
### Datasets
<img width="700" src="https://user-images.githubusercontent.com/42428487/154547781-f2752a3d-07ad-4b69-9b29-2a4a74f7daa3.png">

### Implementation Details
- Language tokens -> BERT와 동일하게 WordPiece tokenizer 사용
- Visual backbone -> ResNet-50 , ResNeXt-152 (pretrained model on ImageNet)
   -  ResNet-50 : image resize : 800 ~ 1333
   -  ResNeXt-152 : image resize : 600 ~ 1000 (GPU memory usage 고려)
 - 64 NVIDIA Tesla V100 GPUs에서 Pixel-BERT pre-training (epoch: 40, batch size: 4096)

<br>

## Downstream Task

### ◼ VQA (Visual Question Answering)

<img width="700" src="https://user-images.githubusercontent.com/42428487/154548799-24eeaba7-8f1f-4578-bc32-43c5828fd95f.png">

<img width="350" src="https://user-images.githubusercontent.com/42428487/154548872-866dd9c6-2807-485a-9e44-e9579451f74b.png"> <img width="300" src="https://user-images.githubusercontent.com/42428487/154549420-648bee1b-34a5-4396-a813-5bd05518228d.png">


<br>

### ◼ NLVR2 (Natural Language for Visual Reasoning for Real)
<img width="400" alt="pairex0" src="https://user-images.githubusercontent.com/42428487/154550015-e5118d1f-0a65-4f54-8ff4-8cf2975ef9c3.png"> <img width="400" alt="pairex1" src="https://user-images.githubusercontent.com/42428487/154550027-e5d1b6b0-f938-4fca-b7c9-1faea2ec4eef.png">
<img width="400" alt="pairex2" src="https://user-images.githubusercontent.com/42428487/154550035-d84355cc-6cae-4ac5-9ed4-3408c5178a62.png"> <img width="400" alt="pairex3" src="https://user-images.githubusercontent.com/42428487/154550044-d130173b-e8de-4037-b901-3b3bd559366b.png">
<img width="400" alt="pairex4" src="https://user-images.githubusercontent.com/42428487/154550054-f2a35e5b-046a-4984-8349-29be78c7c962.png">

<img width="350" src="https://user-images.githubusercontent.com/42428487/154550389-1270dff5-fe5f-4d3a-9eb2-472df4239748.png"> <img width="550" src="https://user-images.githubusercontent.com/42428487/154550719-8d1136ea-c60f-4232-95c6-6ed8901353f5.png">



<br>

### ◼ IR (Image to Text retrieval) / TR (Text to Image retrieval)
- **R@k** (Recall at k) : 모델의 검색 결과 상위 k개 안에 relavant item이 몇 개인지를 나타내는 measure
```
- 입력 : Query(Text)와 , Image
- 출력 : 둘 사이의 relevance score
- relevance score 기준으로 상위 k개 추천 
```
- 특히, IR task의 경우, image에 대한 global description을 이해하는 것이 필요. --> Pixel-BERT는 이러한 점에서 image pixel을 사용한 것이 장점으로 작용

<img width="700" src="https://user-images.githubusercontent.com/42428487/154554533-45cbaf75-2691-41d5-8723-fddc5752b7b4.png">  <br>
<img width="700" src="https://user-images.githubusercontent.com/42428487/154552797-f487b5d5-6364-4c88-bfdc-da882dfde3b3.png">


<br>

## Ablation Study
- 사용한 pre-training 기법들(MLM, ITM, Random Sampling)의 효과 검증
<img width="700" src="https://user-images.githubusercontent.com/42428487/154554422-7a8cfae2-957a-4f66-a3c4-711678524c65.png">



<br><br>

## 이 논문의 한계
- object(region)-based image feature을 사용할 때보다 더 많은 computation cost, memory 사용
   -  object(region)-based image feature : R x dim
   -  Pixel BERT :  100 x dim (fine-tuning 시에는 100 이상, dim=2048)
- 큰 규모의 모델로, pre-training 시 매우 많은 자원 필요. <br>
 (object-based image feature + transformer 이용한 논문보다 더 많은 GPU 사용했지만 성능이 낮음) <br>
   ㄴOscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks (Single Tesla P100 , R=4 or 6 , dim=2048)




<br><br>

---
- 참고
   - [Paper](https://arxiv.org/pdf/2004.00849.pdf) | [YouTube](https://www.youtube.com/watch?v=Kgh88DLHHTo)
