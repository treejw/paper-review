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

`<Encoder>`
- Encoder는 attention machanism 기반으로 feature map의 pixel과 pixel간의 관계를 학습한다.
- Locality 중심의 CNN과 다르게 global 한 정보를 학습하여 이미지 내 객체의 위치나 관계 등을 학습 할 수 있다.

`<Decoder>`
- Decoder의 역할은 어떤 입력값을 받아 이미지 내에 존재하는 object 의 클래스 및 위치를 출력하는것
- Permutation invarient 한 trainsformer 특성때문에 입력값이 서로 달라야 서로 다른 출력값을 낼 수 있다.
   - 학습이 가능한 positional encoding(object query)를 랜덤하게 초기화하여 입력값으로 사용한다.
- Encoder-Decoder간의 attention을 통해 어느 query가 어느 위치에서 object를 찾을 수 있을지 학습한다.(물체가 어느 위치에 있을 확률이 높은지)
- Self-Attention을 통해 자신들의 역할을 어떻게 분배하여 최적의 일대일 매칭을 수행할 수 있을지 학습한다.

`<예시>`

![image](https://user-images.githubusercontent.com/53847442/148023525-1b4e70d5-fe08-4838-bc9d-2186aac936e9.png)


- Decoder의 각 슬롯들은 이미지에 대한 이해 + 각자의 관계를 학습하여 슬롯 개수(N)개 의 임베딩 값을 출력한다. (전체 이미지를 하나의 context로 이용)
- 각 임베딩 값을 FFN을 거쳐 예측한 물체의 유무 + 물체의 위치를 출력한다.
   - NLP와 다르게 순서적 관계가 존재x, set-prediction을 수행한다.

</br>

## 4. Experiment
### 4.1 Faster R-CNN과의 비교
- Dataset: COCO minival
- Backbone: ResNet(pre-train ImageNet), ResNet101(pre-train ImageNet)
- 훈련 epoch : 300 (Faster R-CNN의 경우 500)
- 훈련 시간 : 72 시간

![image](https://user-images.githubusercontent.com/53847442/148024167-78a241f0-ab66-4827-86d3-5e9d4cf37043.png)
</br>

> DETR은 Faster R-CNN에 비해 충분히 높은 성능을 보인다. 그러나 AP_L(큰 object)에 대해서는 높은 성능을 보이지만, AP_S(작은 object)에 대해서는 낮은 성능을 보인다.

> 연산량 측면에서 보면 DETR이, FPS에서는 Faster R-CNN이 좋은 성능을 보인다.

> 하지만 이 두 모델 모두 Real-time이나 경량화가 목표가 아니라 비교대상은 아니다.


</br>

![image](https://user-images.githubusercontent.com/53847442/148024487-c7f514ff-b7be-435b-bd53-5fb72f95e229.png)
</br>

> 인코더의 층수에 따른 영향을 나타낸 표. 인코더의 layer수가 늘어날 수록 AP가 증가하는걸 확인할 수 있다. 

> 실험결과를 통해 (인코더 layer 6개를 기준으로) 인코더 layer가 없을때는 AP의 경우 -3.9point , APL의 경우 -6.0까지 떨어 지는 것을 확인했고,  

> self-attention이 있는 인코더 layer의 개수가 영향을 끼치는 것을 알 수 있다.

>  아래  그림을 보면 인코더의 self-attention과정을 볼 수 있는데, 이렇게 인스턴스가 잘 나누어 진다면 디코더에서 object의 위치와 클래스를 예측하는건 매우 쉬운일이 된다.

![image](https://user-images.githubusercontent.com/53847442/148024669-6a0ac868-5d20-4e96-a7dd-62f3370490e4.png)

</br>

> 디코더 층수에 따른 영향을 나타낸 표. 인코더의 layer수가 늘어날 수록 AP가 증가하는걸 확인할 수 있다. 

> 특히, 1개와 이후간의 차이는 매우 큰데 이 것은 하나의 디코더만 있는 transformers는 하나의 object에 중복된 prediction이 생성되기 쉽기 때문이다.

</br>

## 5. Conclution
- Direct set prediction을 위해 transformers와 이분 매칭 loss를 기반으로 하는 object detector인 DETR을 제안한다.

- 해당 알고리즘은 COCO dataset에서 Faster R-CNN과 유사한 성능을 달성하고, 구현이 간편하여 확장 가능성이 높다.

- Self-attention을 통해 global processing을 수행함으로써 Faster R-CNN보다 큰 object에 대해 높은 성능을 달성하지만 작은 object detection에 대한 개선이 필요하다.


***
## 참고
[Paper](https://arxiv.org/pdf/2005.12872.pdf)

