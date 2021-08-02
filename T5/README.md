# [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) `2019.10`

## Abstract
- Transfer Learning은 NLP에서 효과적인 기술로 여겨지고 있으며, 다양한 접근방식 방법론들이 발표되어 왔음.
   - Transfer Learning : 많은 양의 데이터로 pre-training -> downstream task로 finetuning

- 이 논문에서는 모든 text기반 language problem들을 **text-to-text format**으로 변환하는 통합된 프레임워크를 도입하여 NLP를 위한 transfer learning 방법을 제안.

- 또한, pre-training objectives, architectures, unlabeled data sets, transfer approaches 등에 대해 비교하는 실험을 진행.

- Summarization, Question Answering, Text Classification 등에서 SOTA 달성.

<br><br>


## 2. Setup
- 학습 코퍼스인 C4(Colossal Clean Crawled Corpus)에 대한 설명 및 T5(Text-to-Text Transfer Transformer) 모델 설명

<br>

### 2.1 Model
- 기존 Transformer 설명

<br>

### 2.2 The Colossal Clean Crawled Corpus (C4)
- Common Crawl (매달 웹에서 크롤링한 대규모의 학습 데이터를 공개함, 매달 약 20TB)

- 다음의 heuristic을 거쳐 Cleaned Common Crawl 데이터를 제작.
   - 문장이 끝 부호(예: 마침표, 느낌표, 물음표 또는 끝 인용 부호)로 끝나는 줄만 유지
   - “List of Dirty, Naughty, Obscene or Otherwise Bad Words”의 단어가 포함된 모든 페이지를 제거
   - 코드를 포함하는 중괄호 "{}"가 포함된 페이지는 제거
   - 중복을 제거하기 위해 3-sentence span 이용 (데이터 셋에 똑같은 span이 있으면 제거)
   - 영어인 text에만 초점을 맞추므로 *langdetect* 를 이용하여 영어일 확률이 0.99보다 작으면 해당 페이지 사용안함.
- 위의 작업으로 750GB의 pre-training을 위한 data set을 얻었으며 C4라고 부름. ([TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/c4)에 공개되어 있음) 

<br>

### 2.3 Downstream Tasks
- 일반적인 language learning 능력을 측정하기 위해 다음의 downstream task에서 테스트 진행
- GLUE, SuperGLUE : text classifictaion
- CNN/Daily Mail : abstractive summarization
- SQuAD : Question Answering
- WMT : English to German, French, and Romanian translation

<br>

### 2.4 Input and Output Format
![image](https://user-images.githubusercontent.com/42428487/127610413-d6ae0315-6cf3-4042-b046-9a625eba9b87.png)

- 다양한 task들에서 하나의 모델을 학습하려면 통일된 입력 및 출력 형식이 필요함.
- 각 task를 구분하기 위해 이 논문에서는 원래 Input Text 앞에 task-specific prefix를 추가함

- **QQP**
   > <img width="700;" src="https://user-images.githubusercontent.com/42428487/127613375-b76f45d6-5637-4f56-adc3-5be89b89202e.png">
   > <img width="700;" src="https://user-images.githubusercontent.com/42428487/127613412-e6597d56-b1ed-4f70-a202-c8de07d41af9.png">
- **COLA**
   > <img width="700;" src="https://user-images.githubusercontent.com/42428487/127613806-09dc092f-8ea0-4293-8a31-2816c5dc8c14.png">
- **WMT English to French**
   > <img width="700;" src="https://user-images.githubusercontent.com/42428487/127613700-2eea21c8-9603-40f7-a557-aa84aaf8ba19.png">
- **STSB**
   > <img width="700;" src="https://user-images.githubusercontent.com/42428487/127614146-bf0d08e4-0ffa-47f7-ad79-7249649a805b.png">


<br><br>

## 3. Experiments

### 3.1 Baseline
- Standard encoder-decoder Transformer에 denoising objective로 pre-training
- 각 downstream tasks에 fine-tuning 


#### 3.1.1 Model
- Standard encoder-decoder Transformer 모델 사용.
- **BERT_base**를 기반으로 모델 세팅 (220 million parameters -> **BERT_base**의 2배)


#### 3.1.2 Training
- text-to-text tasks
- C4 데이터로 2^19 steps 만큼 pre-train
- 모든 task들에서 2^18 steps 만큼 fine-tune 


#### 3.1.3 Vocabulary
- SentencePiece 이용 (32,000 wordpieces)
- 번역 task를 위한 영어 이외의 언어에 대한 vocabulary도 필요함.
   - C4에 사용된 Common Crawl 페이지들 중 독일어, 프랑스어, 루마니아어로된 페이지를 분류
   - SentencePiece 학습 시 `영어:독일어:불어:로마어 = 10:1:1:1` 로 섞어서 학습


#### 3.1.4 Unsupervised Objective
<img height="200;" src="https://user-images.githubusercontent.com/42428487/127783486-a6002ab5-6b65-49ab-a592-19f2ecd1687b.png">

- BERT와 약간 다른 denoising 학습 방법을 사용
- 기존 BERT에서의 방식 : mask 처리된 단어를 \<mask\> 토큰으로 치환 후 masking된 토큰을 맞추는 방식
- 이 논문에서의 방식
   - input sequence에서 15%를 랜덤으로 drop out
   - 연속으로 drop-out된 토큰들(span)은 하나의 sentinal token으로 치환
      - sentinal token : \<X\>, \<Y\>와 같이 unique한 토큰으로 sentinal ID들에서 부여
      - Sentinal IDs : Vocabulary에 추가된 special token으로 어떠한 단어에도 해당되지 않음.
   - 출력: \<X\> 토큰은 _for inviting_ , \<Y\> 토큰은 _last_, \<Z\> 마지막이라는 것을 알리는 토큰


#### 3.1.5 Baseline Performance
![image](https://user-images.githubusercontent.com/42428487/127784023-eb3b808c-327d-495e-9ad6-9988edcaae75.png)

- 각 task에 대해 똑같은 step만큼 학습 진행 (평균, 표준편차, no-pretraining 시 성능)
- BERT_base 보다는 높은 성능, 그러나 SOTA는 아님.   
- pre-training은 성능 향상에 효과적임.
   
<br><br>

### 3.2 Architectures
- 다양한 Transformer 구조에 대해 실험

#### 3.2.1 Model Structure

#### Attention mask patterns
![image](https://user-images.githubusercontent.com/42428487/127784402-c47a1af3-8b2e-475c-bda5-3680211f3927.png)
   
- Fully-visible : 출력단어(Query)가 모든 입력단어(Key)에 attention 할 수 있음 (Transformer encoder, AE)
- Causal : 출력단어(Query)가 자신의 현재 포함 이전 타임 스텝의 입력단어(Key)에 attention 가능 (Transformer decoder, AR)
- Causal with prefix : 출력단어(Query)가 자신의 현재 포함 이전 타임 스텝의 입력단어(Key)와 일정 길이의 prefix단어(key)에 attention 가능

<br>

#### Architecture variants
<img height="300;" src="https://user-images.githubusercontent.com/42428487/127784733-84451857-ba28-4800-9b3f-cf58998e399e.png">

- Encoder-Decoder : Encoder(Fully-visible), Decoder(Causal)
- Language Model : Causal 
- Prefix LM
   - Text-to-Text 방식에서는 input sequence의 전체 정보를 활용
   - 입력 text만큼을 prefix로 여기고 입력 text에 대해서는 AE 방식을, 출력 text에 대해서는 AR 방식을 이용.


#### 3.2.4 Result
![image](https://user-images.githubusercontent.com/42428487/127785087-dfc0d86a-7652-4401-adcf-ce2421ec624e.png)

- Encoder-Decoder 구조에 Denoising objective 사용한 방식이 가장 성능이 좋음

<br><br>

### 3.3 Unsupervised objectives
- 다양한 기존의 unsupervised objective를 text-to-text format으로 변경
![image](https://user-images.githubusercontent.com/42428487/127785166-c0ed96f7-a4dc-4977-a1dd-791ed1635d5d.png)

- objective 종류가 너무 많으므로 중요 요소부터 가장 성능이 좋게 나오는 것을 차례로 고정으로하며 하위 요소들 테스트 진행
![image](https://user-images.githubusercontent.com/42428487/127785284-45544cfe-3c99-4de8-82ba-799bed7a6746.png)

#### 3.3.1 Disparate high-level approaches
- 크게 Prefix LM, BERT-style, Deshuffling 비교 : BERT-style이 가장 성능 좋음
![image](https://user-images.githubusercontent.com/42428487/127785464-13cdc902-ad8d-4d79-927b-5b488f67cfb1.png)


#### 3.3.2 Simplifying the BERT objective
- BERT-style에서 mask 하는 방식들 비교 : Replace corrupted spans가 가장 성능 좋음 (=Baseline에 적용된 masking 방식)
![image](https://user-images.githubusercontent.com/42428487/127785457-e879510f-97a3-490c-85de-ce018cb9c970.png)


#### 3.3.3 Varying the corruption rate
- corruption rate에 따른 성능 비교 : 15%가 가장 성능 좋음
![image](https://user-images.githubusercontent.com/42428487/127785451-6d47d827-4fa0-4bcf-bb3e-b6609e9a9ea7.png)


#### 3.3.4 Corruption Spans
- corruption span length에 따른 성능 비교
![image](https://user-images.githubusercontent.com/42428487/127785442-4f0980b4-8052-40b0-82ef-411e6957e053.png)



<br><br>


### 3.4 Pre-training dataset
- pre-training dataset에 따른 성능 비교

#### 3.4.1 Unlabeled datasets
![image](https://user-images.githubusercontent.com/42428487/127785992-227b8cd6-95b8-445d-ae1f-41036facc0e8.png)
- C4로 pre-training 시 전반적으로 성능이 좋음.
- in-domain unlabeled data로 pre-training -> in-domain downstream tasks 일 경우, 성능이 좋음 (ex. Wikipedia -> SQuAD_
- WebText의 경우, C4보다 데이터가 적지만 좋은 성능을 보임.

#### 3.4.2 Pre-training data size
![image](https://user-images.githubusercontent.com/42428487/127786008-76595d9f-e815-40e0-9a8c-736cd8f35fe7.png)
- data size가 충분이 크면, 같은 데이터로 여러번 학습하는것 보다 많은 데이터로 적은 횟수만큼 학습하는 것이 더 성능이 좋음.

<br><br>


### 3.5 Training Strategy
- downstream task를 위한 training strategy

#### 3.5.1 Fine-tuning methods
- Fine-tuning 시 데이터가 적을수록 모델의 모든 파라미터를 학습시키는 것은 좋지 않음
   - Pre-trained 된 부분은 고정, 뒷단의 Classifier layer들만 학습하는 형태
   - Encoder-decoder 방법에는 Decoder 파트에 적용이 어려우므로 본 논문에서는 대안으로 2가지 방법 제안

#### Adapter layers

<img height="400;" src="https://user-images.githubusercontent.com/42428487/127786296-e47d736b-c29d-41a6-a5ce-f5a37db7faa2.png">

- Transformer Laye의 Feed-Forward Layer 뒤에 “Adapter Layer”를 추가 해서 학습하는 방식
- Adapter Layer의 차원을 d로 두고 실험 진행



#### Gradual unfreezing
- 마지막 layer부터 천천히 fine-tuning 하는 방식

#### Results
![image](https://user-images.githubusercontent.com/42428487/127786410-fc56967e-b59d-4f25-88af-734a3e37d28e.png)

- downstream task의 데이터 적을 경우, d 값이 크지 않아도 됨.
- Gradual unfreezing 방식은 fine-tuning 속도 향상에 도움. 

<br>

#### 3.5.2 Multi-task learning

- text-to-text 에서의 multi-task learning은 여러 데이터를 한번에 학습하는 것
- 각 task의 데이터 비율을 얼마나 할지가 학습에 영향을 끼침 -> 본 논문에서는 아래 3가지 방법 사용 & 비교

-  **Example-Proportional mixing** : 각 테스크 별 데이터셋 사이즈에 비례해서 샘플링
-  **Equal mixing** : 각 task 데이터셋을 같은 확률로 샘플링 해서 학습하는 것.
-  **Temperature-scaled mixing**
   - multilingual BERT 에서 사용한 방법
   - 각 데이터를 T개로 나누고 하나씩 가져와 합치는 방식
   - `T=1` (Example-Proportional mixing) , `T=∞` (Equal mixing)

![image](https://user-images.githubusercontent.com/42428487/127786709-a03b0d8b-2262-4cba-b7ba-fa9b28c158f3.png)


#### 3.5.3 Combining multi-task learning with fine-tuning

- Multi-task pre-training + fine-tuning
   - Pre-Training : Example-proportional mixture (`K=2^19`, `K`: artificial data set size limit)
   - Fine-Tuning : 테스크 별로 독립적으로 학습

- Leave-one-out Multi-task training
   - Pre-Training : 하나의 테스크 데이터를 빼고 학습
   - Fine-Tuning : 위 하나의 테스크에 대해서 학습 진행

- Supervised multi-task pre-training
   - Pre-Training : Examples-proportional mixture (`K=2^19`)


![image](https://user-images.githubusercontent.com/42428487/127786739-789d72ce-32c2-4b4d-87f6-24bf212fd386.png)

<br><br>

### 3.6 Scaling
- layer size, batch size, training steps, ensemble의 scale에 따른 실험 진행

![image](https://user-images.githubusercontent.com/42428487/127786833-4c339930-38e2-4d14-bd37-6cbce57866bb.png)


<br><br>

### 3.7 Putting it all together
- Objective 
   - denosing objective
   - span corruption rate: 15%, span length:3
- Training 
   - 데이터 반복 학습이 필요없는 양인 C4 데이터로 학습
   - 1M steps (batch size: 2048 , seq_length: 512)
   - 즉, 1 Trillion tokens 만큼 학습 
- Model Size
   - Base : 220M
   - Small : 50M
   - Large : 770M 
   - 3B & 11B
- Fine-tuning 방식 : multi-task pre-trining -> fine-tuning

- Performance of T5 variants on every task
   ![image](https://user-images.githubusercontent.com/42428487/127787376-13872420-d815-4a97-ab6b-7e6b2bd82b5c.png)


<br><br>


---

### 참고

- paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [yhdosu's blog](https://yhdosu.github.io/2019/11/12/T5.html#323-objective) | [YouTube - PR12](https://www.youtube.com/watch?v=Acp17_is9zU)
