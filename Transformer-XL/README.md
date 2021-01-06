# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context `2019.1`

## Abstract

#### 기존 (vanilla) transformer의 한계
- context fragmentation problem 존재
- long-term dependency에 대한 한계

   - Transformer는 매우 긴 텍스트를 segment 단위로 잘라서 처리한다. 즉, 고정된 길이의 하나의 segment를 input으로 사용하는 것.
      - *(ex) BERT의 feature extraction 시 최대 길이 : 512*

   - 따라서, 연속된 segment들 간의 관련도를 계산할 수 없다는 한계점이 존재한다.

#### 이 논문에서는 Transformer-XL(eXtra Long) 구조를 새롭게 제안

- **segment-level recurrence** : 현재 segment를 처리할 때, 이전 segment를 처리할 때 계산된 hidden state들을 사용하는 방식.
- **relative positional encoding** : 위의 방식에 맞게 positional encoding을 변형함.

#### 그 결과,

- Transformer-XL은 RNN보다 80%, vanilla Transformer보다 450% 긴 dependency을 가지며, 속도도 최대 1,800배 더 빠름.
- wiki8, text 8, WikiText-103, One Billion Word, Penn Treebank에서 SOTA을 달성


<br><br><br>

## 3. Model

### 3.1. Vanilla Transformer Language Model

- a corpus of tokens : `X = [x1, ... , xT]`
![image](https://user-images.githubusercontent.com/42428487/103703359-c4f0f780-4fea-11eb-94f3-3f63cb84c05b.png)

<p align="center"><strong> ▲ Fig 1. vanilla model의 train 방식(left)과 evaluation 방식(right)</strong></p>


#### ▪ Vanilla Transformer LM의 학습 방식
- 위의 Fig 1(a) 그림처럼 segment 단위로 language modeling을 학습하게 되면, 

- segment 크기를 넘어서는 long-term dependency를 학습할 수 없고, **context fragmentation problem**이 발생함.
   - context fragmentation problem
   > 특정 symbol으로 segment을 나누는 것이 아닌 fixed-length로 단지 나누어서 학습하는 것. 
   >
   > 즉 previous segment에서의 정보를 가져다 쓰지 않는 학습 방법때문에 생기는 문제.

<br>

#### ▪ Vanilla Transformer LM의 Evaluation 방식

- 이러한 vaniila model은 last position만을 예측하는 형태이므로 evaluation 방법은 아래와 같음
   > 한 segment를 이용하여 segment의 맨 마지막 한 개의 위치에 올 token만 예측.
   > 
   > [x1, x2, ... xt-1]을 이용하여 xt를 예측 -> [x2, x3 .. xt]을 이용하여 xt+1을 예측

- 이러한 evaluation 방법은  training을 할 때보다 context fragmentation problem의 영향을 덜 받지만, Fig 1(b) 그림에서 표시된 것처럼 중복되는 연산량이 굉장히 많아진다.

   → **Transformer-xl은 이러한 연산량을 줄여 evaluation 속도도 빠르게 향상시킨다.**



<br><br>

### 3.2. Segment-Level Recurrence with State Reuse
![image](https://user-images.githubusercontent.com/42428487/103703712-637d5880-4feb-11eb-941f-6c927284ae1a.png)

<p align="center"><strong> ▲ Fig 2. Transformer-xl model의 train 방식(left)과 evaluation 방식(right)</strong></p>



#### ▪ Transformer-XL의 LM의 학습 방식
- Transformer 구조에 Recurrence 방식을 도입하여 길이 제한 한계를 해결함.

- segment 학습 시, 이전 segment에서 계산된 `hidden state sequence` 활용. (이전 segment의 정보를 캐시로 가지고 있어 속도가 빠름)

- 이전 segment의 hidden state들이 다음 segment를 처리하기 위해 사용될 때는 gradient에 따라 학습시키지 않고 고정.

   <img height="100;" src="https://user-images.githubusercontent.com/42428487/103708563-28335780-4ff4-11eb-8cf1-958eb7e36b27.png">

> - n : Layer의 층수
> - h_T : 길이가 L인 input [x_1, x_2 .. x_L]
> - h_T+1 : h_T의 뒤에 이어진 길이가 L인 input [x_L+1, x_L+2 .. x_L+L]
> - SG : stop-gradient
> - [h_u ㅇ h_v] : 두 hidden state를 concatenate.
> - 수식에 나와 있듯 self-attention의 key, value를 계산할 때 이전 segment의 hidden state와 현재 segment의 hidden state를 concatenate하여 얻은 벡터를 이용(h 위의 ~표시).

<details>
    <summary>(참고. vanilla transformer vs. Transformer-xl)</summary><br>

![image](https://machinereads.files.wordpress.com/2020/05/image-36.png?w=1024)

</details>

- 이론상으로, current segment의 processing할 때, GPU 메모리가 허락하는한 수 많은 previous segments의 cache을 사용할 수 있음.

<br>

#### ▪ Transformer-XL LM의 Evaluation 방식
- Recurrence를 활용하면 evaluation(prediction)이 훨씬 빨라진다.
   
   - 이전 segment의 계산 결과를 저장해놓고 활용할 수 있기 때문에 매번 다시 계산할 필요가 없기 때문.


<br><br>

### 3.3. Relative Positional Encodings

#### ▪ Vanillay Transformer의 Positional Encodings

   - 문장 내에서 단어의 순서에 따라 **0 ~ max_sentence_length 까지 position 정보**를 word embedding vector에 더해준다.
   
   - 수식
   
      **Q<sup>T</sup>K = (W<sub>q</sub>(E<sub>x<sub>j</sub></sub> + U<sub>i</sub>))<sup>T</sup> (W<sub>k</sub>(E<sub>x<sub>j</sub></sub> + U<sub>j</sub>))**
      
      <img height="120;" src="https://user-images.githubusercontent.com/42428487/103710264-fa501200-4ff7-11eb-90c7-21db57e5a6a6.png">
      
      - Query인 Q와 Key인 K 사이의 attention: `A = Q^T * K`
         - `Q = (E + U) * Wq` , `K = (E + U) * Wk`
         - `E`: 토큰 임베딩, `U`: 포지션 정보
      - 즉, U에 인코딩된 i번째, j번째 absolute 포지션 정보을 통해 두 단어 간의 위치 차이를 모델에 반영.
   
   
<br>


#### ▪ Transformer-XL의 Relative Positional Encodings
   
   - 위의 방식을 Transformer-XL에 그대로 적용하면, 이전 segment의 첫번쨰 토큰과 현재 segment의 첫번째 토큰의 Positional Encoding 값이 같게 된다는 문제가 발생.
   - 이 문제를 해결하기 위해 기존의 'absolute' 포지션 정보가 아닌, **'relative' 포지션 정보를 사용.**

   - (ex. 아래 그림) 단어 "I"를 기준으로 왼쪽으로 갈수록 위치에 `-1`을 더하고, 오른쪽으로 갈수록 `+1`를 더함
   
      <img height="350;" src="https://machinereads.files.wordpress.com/2020/05/image-48.png"><br>

   - 수식

      <img height="120;" src="https://user-images.githubusercontent.com/42428487/103710457-69c60180-4ff8-11eb-9ffa-af664f5e416c.png">
      
      - (b), (d)에서의 Key 벡터를 구할 때, `U_j`를 `R`로 대체.
         - R은 relative 파트를 의미. R을 기존에 Transformer에서 쓰던 식으로 sinusoid encoding matrix을 사용
     
      - (c), (d)의 `U_i` 는 R0와 같다고 볼 수 있음.
         - Query i-th을 기준으로 Key j-th와의 관계를 찾는 것에서 i-th positional encoding은 어떤 Query vector가 들어와도 똑같다
         - 여기서 두 개의 `U_i`를 같은 것으로 치환하지 않고 `U_i^T * W_q^T`를 (c)에서는 u∈R<sup>d</sup>, (d)에서는 v∈R<sup>d</sup>로 치환을 하여 training parameter을 만든다.

      - 마지막으로 W<sub>k</sub> 벡터를 W<sub>k,E</sub>와 W<sub>k,R</sub>로 나누어 처리.
      
      - 이렇게 바뀐 (a)~(d)를 논문에서는 아래와 같이 설명.
         - (a): represents context-based addressing
         - (b): captures a content-dependent positional bias
         - (c): governs a global content bias
         - (d) encodes a global positional bias
      
 <br><br>
   
### Transformer-XL 전체 수식 - N-layer + single attention head (n=1, ... , N)
![img](https://1.bp.blogspot.com/-QPFmafMPugo/XR2VV1jfbCI/AAAAAAAADwA/_aAH2qTi8b0qqZgsCRaZlcxiY4Y6v1g8gCLcBGAs/s1600/20190704_145736.png)

<br><br>

### Transformer-XL 전체 구조

   <img height="400;" src="https://machinereads.files.wordpress.com/2020/05/image-6.png">



<br><br><br>

## 4. Experiments

### 4.1. Main Results
![](https://1.bp.blogspot.com/-Hmx155Ayxyc/XSQ9OeG_k1I/AAAAAAAADwY/IY0cUkzfLlgZZTVECtfgNl1X4qQwSFmUACLcBGAs/s1600/20190709_160738.png)

- 여러 데이터를 이용한 word-level, character-level language modeling에서 sota를 기록
- (table 4) One Billion Word dataset의 경우, 문장의 shuffling된 long term dependency가 없는 데이터로 generalizable modeling to short sequences에 대한 성능도 좋다는 것을 보여줌.

<br><br>

### 4.2. Ablation Study
#### ▪ recurrence mechanism과 new positional encoding에 대한 실험
![](https://1.bp.blogspot.com/-Xusopub5MIU/XSRJKCicLBI/AAAAAAAADww/uY4xgEPh1YgWtXDQd-Da5L9nlA2FWFsBQCLcBGAs/s1600/20190709_165828.png)

<br>

#### ▪ context fragmentation에 관한 실험
![](https://1.bp.blogspot.com/-TUkRytojfYg/XSRQ6CHcOtI/AAAAAAAADw8/I5yvhH4nmDkoHze-ViSmZ9q7NmDkd6dTACLcBGAs/s1600/20190709_173007.png)
- recurrence 방법이 좋은 성능을 도출한다는 것을 보여줌.

<br><br>

### 4.3 Relative Effective Context Length
![](https://1.bp.blogspot.com/-r6fFWh8srco/XSVAauM0i4I/AAAAAAAADxg/lkxEzPiSrvkJvLAScCNxqyMmYROEZ7fPQCLcBGAs/s1600/20190710_103325.png)

- 이 논문에서 Relative Effective Context Length(RECL)을 제시.
   - RECL: Effective Context Length(ECL)을 evaluate 하는 방법.
      - ECL: context span이 threshold 보다 더 큰 gain을 도출할 수 있는 longest length을 뜻함.
   - 간단히 말하면, 얼마나 큰 context를 효율적으로 잘 학습하는지 확인하는 실험.

<br><br><br>

---
#### ※ 이 논문 특이점
- ICLR 2019에 reject된 논문이지만, Transformer-XL 구조의 잠재력(?) 때문에 주목받음.
   - ICLR 2019 reject된 이유 중 1가지: downstream task에서의 성능 결과 없음.
- 실제로, XLNet에서 Transformer-XL 구조가 사용되어 좋은 성능을 보여 Transformer-XL이 좋은 모델임이 입증되었다고 볼 수 있음.



<br><br><br>

---
### 참고
- paper : [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf)
- [YouTube](https://www.youtube.com/watch?v=lSTljZy8ag4&t=17s) | [harper's Blog](https://machinereads.com/2020/05/11/xlnet-generalized-autoregressive-pretraining-for-language-understanding-2-3/) | [RUNGJOO's Blog](https://ai-information.blogspot.com/2019/06/nl-040-transformer-xl-attentive.html) | [Myung Ha Kwon's Blog](http://mlgalaxy.blogspot.com/2019/07/transformer-xl.html)

