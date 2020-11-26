# Word Embedding

### ✔ Word Embedding
- 단어를 벡터(**Dense Representation**)로 변환하는 방법
- Word Embedding 종류: LSA, Word2Vec, FastText, GloVe 등


<br>

### ◼ **Sparse Representation** vs **Dense Representation**
- **Sparse Representation (희소 표현)**
   - **one-hot 벡터** - 표현하려는 단어의 인덱스 값만 1이고 나머지는 0인 벡터 `개 = [1 0 0 0 ... ]`

   - **단점**
      - 단어의 집합이 커질수록 차원이 커진다.
      - 단어의 의미를 담지 못하는 표현 방법이다.

- **Dense Representation (밀집 표현)**
   - 사용자가 설정한 값으로 단어의 벡터 길이가 고정된다.
   - 벡터의 값이 0과 1로만 표현되는 것이 아닌, 아래와 같이 **실수**로 표현.
   -  벡터 차원이 128일 때, `개 = [0.2 1.8 1.1 -2.1 1.1 ... 중략 ... ]`
 
   - **특징 및 장점**
      - Sparse Representation에 비해 상대적으로 적은 차원으로 단어를 표현할 수 있다.
      - Dense Representation를 워드 임베딩 과정을 통해 나온 결과라고 하여 임베딩 벡터(embedding vector)라고도 한다.


<br><br>


### ✔ Word2Vec
- **Distributed Representation (분산 표현)** 방법을 이용하여 단어간의 유사도를 계산할 수 있다.
- CBOW와 skip-Gram 2가지 방식이 존재
   - **[CBOW (Continuous Bag of Words)](https://wikidocs.net/22660#3-cbowcontinuous-bag-of-words)**
      - 주변에 있는 단어들로, **중간에 있는 단어를 예측**하는 방법
   - **[Skip-Gram](https://wikidocs.net/22660#4-skip-gram)**
      - 중간에 있는 단어로, **주변 단어들을 예측**하는 방법. (CBOW와 반대)

![](https://miro.medium.com/max/875/1*i-aWU_fjKblzRG4OTgmCkA.png)

<br><br>

### ✔ GloVe(Global Vectors for Word Representation)
- **카운트 기반**과 **예측 기반**을 모두 사용하는 방법론
   - 기존의 카운트 기반의 LSA와 예측 기반의 Word2Vec의 단점을 보완한다는 목적의 방법. 
- Word2Vec 만큼 성능이 좋음.
- **GloVe의 아이디어**
   - **임베딩 된 중심 단어와 주변 단어 벡터의 내적**이 **전체 코퍼스에서의 동시 등장 확률**이 되도록 만드는 것


![](https://miro.medium.com/max/875/1*2HuruOHvhP7_gnW2DKB2FQ.png)



<br><br><br>

## ELMo(Embeddings from Language Model) 
### ✔ Motivation
 - Word2Vec과 GloVe의 한계
   - 다양한 의미를 지닌 하나의 단어를 하나의 임베딩 벡터로 표현한다.
   - 예) Bank Account(은행 계좌) | River Bank(강둑)
- Idea: 단어를 임베딩하기 전, 전체 문장을 고려해서 임베딩을 하겠다.

   ▶ **문맥을 반영한 단어 임베딩 (Contextualized Word Embedding)**

<br>

### ✔ Proposed Method
#### 🔸 Bidirectional Language Model (biLM)

<img src="https://wikidocs.net/images/page/33930/forwardbackwordlm2.PNG">

- ELMo는 언어 모델로써 **양방향 LSTM(biLM)** 을 이용.
   - forward LM
      -  tk−1까지의 시퀀스 정보를 가지고 tk의 확률 계산
      
      <img height="65;" src="https://user-images.githubusercontent.com/42428487/100283235-721a2e00-2fb0-11eb-985c-929663d78157.png">

   - backward LM
      
      <img height="60;" src="https://user-images.githubusercontent.com/42428487/100283653-40ee2d80-2fb1-11eb-8a99-132ed4941471.png">

   - biLM은 forward LM과 backward LM 각각에 대한 log likelihood를 최대화하는 방식으로 학습.

      <img height="110;" src="https://user-images.githubusercontent.com/42428487/100285318-0934b500-2fb4-11eb-98f8-13470ad99553.png">


- biLM의 입력으로 `char CNN`이라는 임베딩을 사용.

<img src="https://wikidocs.net/images/page/33930/playwordvector.PNG">
<img height="70;" src="https://user-images.githubusercontent.com/42428487/100285794-da6b0e80-2fb4-11eb-8e39-e385bc923abc.png">

- `R` : representation 집합
- `x` : biLM의 입력으로 사용된 단어의 representation
- `h` : LSTM의 순방향/역방향의 representation
- `j` : 몇번째 layer 
- `k` : 몇번째 단어

<br>

#### 🔸  ELMo 임베딩 절차
1) 각 층의 출력값을 연결(concatenate)
<img src="https://wikidocs.net/images/page/33930/concatenate.PNG">

2) 각 층의 출력값 별로 가중치를 곱한다.
이 가중치를 여기서는 s<sub>1</sub>, s<sub>2</sub>, s<sub>3</sub>라고 가정.
<img src="https://wikidocs.net/images/page/33930/concatenate.PNG">

3) 각 층의 출력값을 모두 더한다.
<img src="https://wikidocs.net/images/page/33930/weightedsum.PNG">

4) 벡터의 크기를 결정하는 스칼라 매개변수(γ)를 곱한다.
<img src="https://wikidocs.net/images/page/33930/scalarparameter.PNG">

<img height="60;" src="https://user-images.githubusercontent.com/42428487/100285850-f4a4ec80-2fb4-11eb-92c8-fc5e0e14710a.png">

<br>

#### 🔸 특정 task에서의 ELMo 임베딩 
<img src="https://wikidocs.net/images/page/33930/elmorepresentation.PNG">


<br><br>



---

### 참고
- Word Embedding 정의 
   - [wikidocs](https://wikidocs.net/33520) | 
   [ckdgus1433 Bolg](https://blog.naver.com/PostView.nhn?blogId=ckdgus1433&logNo=222030454167&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView)
- Word2Vec
   - paper: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) 
   - [wikidocs](https://wikidocs.net/22660) | 
   [towards data science](https://towardsdatascience.com/word-embeddings-for-nlp-5b72991e01d4)
- GloVe
   - paper: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) 
   - [wikidocs](https://wikidocs.net/22885)
- ELMo
   - paper: [Deep contextualized word representaions (2018)](https://aclweb.org/anthology/N18-1202)
   - [wikidocs](https://wikidocs.net/33930) | 
   [Baek Kyun Shin Blog](https://bkshin.tistory.com/entry/NLP-12-%EA%B8%80%EB%A1%9C%EB%B8%8CGloVe) | 
   [Dos tacos Blog](https://dos-tacos.github.io/paper%20review/deep-contextualized-word-representations/) | 
   [Youtube](https://www.youtube.com/watch?v=6K3joYQ0DYE)
   
   
