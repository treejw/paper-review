# DETR: End-to-End Object Detection with Transformers `2020`

## 1. Abstract
- object detection > direct set prediction
- 기존 detecion의 non-maximum suppression, anchor generation 등 직접 디자인해야 하는 요소들을 제거 > 파이프라인 간소화
- set 기반의 global loss 사용, Transformer 사용
- Faster R-CNN과 동급의 정확도 및 런타임 속도
</br>

## 2. Introduction
- object detecion의 목표: BBOX, label 예측하는것
- 최근 detector 모델들은 앵커 region proposal > BBOX 과정이 휴리스틱하기 때문에 NMS과정에 성능의 영향을 많이 받는다.
- Transformer 기반의 encoder-decoder 구조를 사용하여 nms를 대신한다.

   > Transformer의 self-attention: elements들 간의 모든 pairwise interactions를 통해 중복된 prediction을 제거한다.

- DETR은 모든 object들을 한번에 predict한다. 
  > 이는 end-to-end 특성인데, 실제값과 예측값 사이에서 set loss를 통해 bipartite matching(이분 매칭)을 훈련한다. 

</br>

### DETR 장점과 단점
<장점>
1. object detection에서 휴리스틱을 사용하기 때문에 복잡해지는 과정을 direct prediction 문제를 바꾸어 단순화함
2. CNN 이후 Transformer를 사용하면 되는 구조로 매우 단순한 구조를 취하게 된다.

<단점>
1. Transformer의 특성상 학습하는데 굉장히 많은 시간이 필요하다.
2. small object 탐지력이 약하다.


</br>


## 3. Proposed Method
### 3-1. Architecture
![image](https://user-images.githubusercontent.com/53847442/148017271-cd2ab421-0c2e-408f-81ea-6a8dabc86c96.png)


`DETR의 아키텍쳐는 feature 추출을 위한 CNN backbone, encoder-decoder 구조의 transformer, 최종 detection prediction을 수행하는 FFN(Feed Forward Network)로 이루어져 있다.`
</br>

1) `Backbone`: Image에 대해 CNN을 통해 feature map 추출 (주로 resnet-50과 resnet-101)
2) `Positional Encoding`: 시퀀스 데이터로 변형 (d x HW) + positional encoding (위치 정보)
     - 1x1 convolution으로 feature map의 channel을 압축하고 feature map의 size (HxW)만큼의 data 가 형성된다.
     - feature map의 spatial domain에 해당하는 부분이 vector화 되기때문에 position에 관한 정보를 잃게된다.
     - transformer 구조가 permutation-invariant (순서와 독립된 동일한 출력)이기에 각 unit들에 position 정보를 삽입하는 positional encoding 과정이 필요하다.

3) `Transformer encoder`: encoder는 앞선 positional encoded sequence data를 입력받아 attention machanism을 거친 data를 출력한다.
4) `Transformer decoder`: N개 object의 임베딩에 대해 디코딩한다
      - encoder의 출력과 앞선 positional encoding에 해당하는 object queries라고 명명된 embeddings 정보를 입력받아 Self-attention과 encoder-decoder 구조를 이용하여 중복되지 않는 N개의 object를 예측한다.

5) `Prediction feed-forward networks (FFNs)`
      - transformer를 거친 N개의 unit들은 디코딩된 임베딩값을 통해 중심 좌표, 높이, 폭으로 이루어진 4차원 벡터를 예측하고, linear layer는 softmax 함수를 사용하여 class를 예측한다.
      - <b>이 경우 기존의 OD처럼 영상 내 객체 몇개를 찾는 것이 아닌 무조건적인 N개의 객체를 찾으므로 실제 객체가 아닌 no object 또한 찾도록 학습이 이루어진다.</b>

</br>

### 3-2. Transformers
![image](https://user-images.githubusercontent.com/53847442/148019445-dc754df5-118d-4a7f-a96f-7f6d4a84bc91.png)

***
## 참고
[Paper](https://arxiv.org/pdf/2005.12872.pdf)

