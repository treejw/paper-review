# Attention is all you need (Transformer)
## Model Architecture
![image](https://user-images.githubusercontent.com/41243762/100835929-bf673580-34b1-11eb-8d45-6a3904b8cbd3.png)

<br>

## Encoder
### 1. word embedding & positional encoding
- 단어의 위치정보를 반영하기 위한 방법
- sin, cos 함수를 이용 
   - |1|의 값을 갖는다.
   - 입력 문장의 길이에 상관없이 상대적인 위치정보를 줄 수 있다.

<br>
   
### 2.  Multi-head Attention
- 1의 결과에 weight(q, k, v)를 곱해서 각 워드의 Query, Key, Value를 계산
- Scaled dot-product attention
   - Q * K = attention score (단어의 연관성)
   - scale & softmax * V = attention value (문장 안에서 연관성을 포함한 워드의 벡터)
- Multi-head
   - 여러 관점에서의 관측으로 문장의 모호성 보완
   
   <img height="300;" src="https://wikidocs.net/images/page/31379/transformer12.PNG">
   
   <img height="150;" src="https://wikidocs.net/images/page/31379/transformer16.PNG">

   ![image](https://user-images.githubusercontent.com/41243762/100832684-afe4ee00-34ab-11eb-88eb-f81792eed60d.png)

<br>
   
### 3. Add & Norm
- positional encoding의 위치정보 손실 방지
- Norm을 통한 학습효과 향상

<br>
   
### 4. Feed Forward

<img height="200;" src="https://wikidocs.net/images/page/31379/positionwiseffnn.PNG">
<br>
   
## Decoder
### 1. Masked
- 학습을 할 때, 출력하지 않은 값은 가리기 위함.

<br>
   
### 2.  Multi-head Attention
- Q : decoder의 입력값
- K, V : encoder의 출력값 
- encoder의 K와 decoder 현재 워드의 Q의 연관성을 계산 후 encoder V를 곱하여 번역성능 상향



<br><br>



---
### 참고
- paper : [Attention is all you need (Transformer)](https://arxiv.org/abs/1706.03762)
- [wikidocs](https://wikidocs.net/31379) | 
   [harper's Blog](https://machinereads.com/2018/09/26/attention-is-all-you-need/) 
