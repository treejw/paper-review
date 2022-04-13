# [GLocal-K: Global and Local Kernels for Recommender Systems](https://arxiv.org/pdf/2108.12184v1.pdf)  `2021.08`

<br><br>

## 1. Abstract

- Global Kernel과 Local Kernel을 사용하여 high-dimensional sparse한 user-item matrix를 뽑고자 한다. ML-100K, ML-1M, Douban에서 **SOTA**를 달성


<br><br>

## 2. GLocal-K
![image](https://user-images.githubusercontent.com/41942097/158418287-ffcc55df-6d89-402e-a4f4-a3f4f05d26f6.png)

* m은 items, n은 users

<br>

### 2.1 Pre-training with Local Kernel

![image](https://user-images.githubusercontent.com/41942097/158421779-46508219-2217-4863-86a3-4420fc81200f.png)
- f,g는 non-linear activation function

![image](https://user-images.githubusercontent.com/41942097/158421860-930fc105-a882-4e81-9fcf-ce353dc670e5.png)

* K()는 ABF Kernel function
* i는 행, j는 열을 의미 -> 즉, i는 items, j는 users

- Kernel Trick

> [Kernel Trick 설명](https://sanghyu.tistory.com/14)


![image](https://user-images.githubusercontent.com/41942097/158421927-2b8dd16b-46fb-4997-b1d2-929381843ba9.png)

- W'는 Weight와 Kernel function의 Hadamard-product 계산을 통하여 sparse한 weight matrix를 구한다.

<br>

### 2.2 Fine-tuning with Global Kernel

![image](https://user-images.githubusercontent.com/41942097/158421991-10fdb91d-8953-4789-8c15-949e9783993d.png)

* x 연산은 convolution 연산을 의미


<br><br>

## 3. Experiments

> experiment의 지표는 모두 RMSE를 사용

![image](https://user-images.githubusercontent.com/41942097/158422271-c4928ea2-d470-47b3-850a-b048248d0fc4.png)

![image](https://user-images.githubusercontent.com/41942097/158422317-351e4a90-dc13-4349-aa8f-331e12886b62.png)

![image](https://user-images.githubusercontent.com/41942097/158422379-2a0b2fb1-5e14-43d6-9c1d-958a52ffc237.png)
