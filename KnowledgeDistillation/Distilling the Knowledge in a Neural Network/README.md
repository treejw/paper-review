# Distilling the Knowledge in a Neural Network `2015.03`

# 1. Abstarct
- task들에 대하여 model들을 ensemble하여 예측하는 것은 computationally expensive하다.
- 학습된 모델들의 정보들은 작은 모델로 효과적인 방법으로 전파할 수 있는 방법을 제안한다. (Distillation)

<br><br>

# 2. Distillation

### 2.1 Softmax에서의 Temperature 의미

![image](https://user-images.githubusercontent.com/41942097/151275131-931c4c8c-0ac1-4638-9f6f-166541936686.png)

- Softmax란? : Output layer의 결과값인 logit(<img src="https://render.githubusercontent.com/render/math?math=z_i">)을 해당 layer의 다른 class에 대한 logit과 비교하여, 각 class의 확률(<img src="https://render.githubusercontent.com/render/math?math=q_i">)로 바꾸어줌.

- Temperature(T)의 의미
```
- 기본적인 softmax는 T=1의 값을 가진다.
- T가 커질수록 probability distribution이 작아진다. 즉, 각 class간의 probability의 차이가 작아진다.
```


- knowledge Distillation에서의 T
```
- Distillation의 가장 간단한 방법은 transfer set과 soft target distribution으로 Distillation model을 학습시키는 것이다. 
(이때, soft target distribution은 combersome model에서 높은 Temperature의 softmax로 생성된 값이다.)
- Distillation model을 학습할 때, training에서는 높은 Temperature을 갖도록 하고, test 및 prediction의 상황에서는 T=1로 고정한다.)
```

<br>

> Tip : 해당 논문에서 Distillation model에 대하여 실험하였을 때, transfer dataset의 soft tagert에 대하여 올바른 label을 주는 것은 효과가 있었지만,
        그보다 2가지 objective function을 average하여 학습시키는것이 더욱 효과적이였다고 한다.
 
<br>

### 2.2 2가지 Objective Function

<br>

#### 2.2.1 cross entropy with the soft targets (high Temperature)
- Cumbersome model과 Distillation model에 동일한 T(높은값)을 대입하여 생성된 soft target간의 cross entropy



#### 2.2.2  cross entropy with the correct labels
- 기본 모델 학습과 같이 hard target간의 cross entropy


#### 2.2.3 total loss

![image](https://user-images.githubusercontent.com/41942097/151281332-e6d73e9e-4d0a-4ce7-ac89-58277981f27d.png)

- cross entropy with the soft targets의 값과 cross entropy with the correct labels에 가중치를 준 값의 합

<br><br>

# 3. MNIST result

- Large model(single) : 60,000개의 training set에 대하여 2개의 hidden layer (1200개의 hidden units) 학습 (Dropout 및 regularization 사용)
> 67 test errors
- small model : 2 hidden layer (800 rectified linear hidden units), no regularization
> 146 test errors
- Distilled model : small model에 soft target loss만 추가 (T=20)
> 74 test errors

<br>

```
만약 Distilled model에 3에대한 데이터를 제외하고 학습해도 bias에 대한 조절만 잘 하면 뛰어난 성능을 가짐 (1010개의 3중에 14만 틀림)
```

<br><br>

# 4. Speech Recognition result
![image](https://user-images.githubusercontent.com/41942097/151293007-51f82d3c-aebd-489f-a933-25441b12ab0d.png)

<br><br>

# 5. Training ensembles of specialists on very big datasets
![image](https://user-images.githubusercontent.com/41942097/151293353-51393ad4-1660-4bad-b73a-6ba5c289a1d4.png)

