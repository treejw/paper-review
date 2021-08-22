# USB: Universal-Scale Object Detection Benchmark  `2021.03`

## 1. Abstract
- 현재 Object Detection 분야의 benchmark로 통용되고 있는 COCO dataset은 scale에대한 다양성이 부족하다.
- 따라서 여러가지 domain 및 scale을 가지고 있는 데이터 셋 (COCO, Waymo, Manga109-s)를 통하여 제대로 된 성능비교 가능한 benchmark(USB)를 만들겠다.
- 또한 USB에서 뛰어난 성능을 자랑하는 UniverseNet을 제안한다.(COCO 54.1%AP)

<br><br>
## 2. Introduction
### COCO dataset의 문제점

- 다양한 scale의 이미지 및 domain이 부족하다
- COCO 평가방법이 제대로 되고있지 않다. 
 => 대부분의 detector들은 24epoch, lr 0.01 ~ 0.02, image size 1333 ~ 800 정도로 학습되고 있다. 이는 공평한 비교가 못된다.
- multi-scale의 분석방법으로 부족하다. 
 => Res2Net backbone, SEPC neck, ATSS head와 같이 구성된 multi-scale 모델이 유행하고 있는데 이는 COCO 외의 데이터에서는 효율성, 결합성, 특징 등에 대하여 분석이 부족하다.
 <br>
 
### main contribution

- Universal-Scale object detection Benchmark(USB)를 제안한다. (COCO, Waymo, Mango109-s)
- 공평한 학습 조건을 구성하기 위해 학습하는 epoch 및 테스트 image resolution의 다양한 thresholds을 제안함으로써 공평한 비교를 가능하도록 한다.
- YOLOv4보다 9.3점 뛰어난 UniversNet을 제안한다.

<br><br>
## 3. Benchmark Protocols of USB

![image](https://user-images.githubusercontent.com/41942097/130332823-225e9770-0ac8-4793-ba7c-c9481db7b513.png)

![image](https://user-images.githubusercontent.com/41942097/130332558-6fc110e9-58b8-4486-90b1-b0317773a4c9.png)

- COCO는 train2017, val2017을 기준으로 삼았다.
- WOD의 train은 798 sequences이고 val은 202 sequence이다. (1sequence는 ~20 frame, frame 당 5개의 카메라로 찍힌 각 이미지 존재)
- M109는 grayscale, 겹치는 경우가 많고, 작은 object와 큰 object의 차이가 크다.(큰 object를 축소화 시킨게 아닌 간략화한 얼굴 등의 object로 구성되기 때문)

<br>

### training protocol

![image](https://user-images.githubusercontent.com/41942097/130333043-cb7ecc25-0545-4c94-9f36-a5ad07bae37a.png)

- USB 1.0은 ~24 epochs, 2.0은 ~73 epochs, 3.0은 ~300 epochs로 구성되어있다.

![image](https://user-images.githubusercontent.com/41942097/130333141-e9049850-def1-40d7-876a-b28797f3e783.png)

<br>

### evaluation metric

![image](https://user-images.githubusercontent.com/41942097/130333185-96cce48c-14a9-48ce-a5a8-0fcaa8194070.png)

![image](https://user-images.githubusercontent.com/41942097/130333186-5b7c7f49-3669-4b72-a2ec-5c9209311a03.png)

![image](https://user-images.githubusercontent.com/41942097/130333200-f7a0c549-71b5-4f5a-8450-9af257eac850.png)

- M109s의 경우 maxDet을 300으로 증가시킴

<br><br>

## 4. UniverseNet

- single-stage detetor입니다.
- Baseline model은 RetinaNet으로 잡는다.(backbone은 ResNet-50-B, neck은 FPN, loss는 focal loss)
- UniverseNet은 RetinaNet을 기반으로 만들어졌고, ATSS와 SEPC를 사용한다. (backbone은  Res2Net-50-v1b, neck은 DCN, multi-scale training 사용)

<br><br>

## 5. Experiment

### Experiment Setting

![image](https://user-images.githubusercontent.com/41942097/130333374-14044733-dbae-44f9-898e-c73c07ff29a1.png)


![image](https://user-images.githubusercontent.com/41942097/130333383-e74c75a9-1900-418f-b335-480def875242.png)


![image](https://user-images.githubusercontent.com/41942097/130333403-96e2732b-7237-4db3-b0e9-35221e5bc40b.png)
