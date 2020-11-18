# Text Tokenization
Tokenization: 주어진 corpus에서 token으로 나누는 작업. 보통 의미있는 단위를 token으로 정의.

<br>

# Tokenizer 종류
## ◼ Sentence Tokenization (문단 > 문장)
- 특징
   - 문장의 마지막을 뜻하는 기호(`마침표`, `개행문자(\n)`, `느낌표`, `물음표`) + 공백 기준으로 분리

- example 
   ```python
   from nltk.tokenize import sent_tokenize
   print(sent_tokenize("Hello World. It's good to see you. Thanks for buying this book."))
   ``` 
   
   - `Hello World. It's good to see you. Thanks for buying this book.`
   - `Hello World.` `It's good to see you.` `Thanks for buying this book.`






<br>

## ◼ Word Tokenization (문장 > 단어)


### ▪ word_tokenize 
- 특징
   - 단어 단위로 토큰화
   - 축약형(contractions) 단어 분리

- example 
   ```python
   from nltk.tokenize import word_tokenize
   print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
   ```
   
   - `Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.`
   - `Do` `n't` `be` `fooled` `by` `the` `dark` `sounding` `name` `Mr.` `Jone` `'s` `Orphanage` `is` `as` `cheery` `as` `cheery` `goes` `for` `a` `pastry` `shop` `.`



<br>

### ▪ WordPunctTokenizer
- 특징
   - 단어 단위로 토큰화
   - 모든 구두점 기준으로도 분리
   
###### ※ 구두점 : 쉼표(,), 마침표(.), 느낌표(!), 물음표(?), 콜론(:), 세미콜론(;), 큰 따옴표(""), 작은 따옴표(''), 하이픈(-) 등..

- example 
   ```python
   from nltk.tokenize import WordPunctTokenizer
   tokenizer = WordPunctTokenizer() 
   print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
   ```

   - `Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.`
   - `Don` `'` `t` `be` `fooled` `by` `the` `dark` `sounding` `name` `Mr` `.` `Jone` `'` `s` `Orphanage` `is` `as` `cheery` `as` `cheery` `goes` `for` `a` `pastry` `shop` `.`




<br>

### ▪ Tokenizing sentences using regular expressions (정규식 이용)
![reg](https://user-images.githubusercontent.com/42428487/99420771-f7fb0100-2940-11eb-9eb6-4a9aa4035c05.png)

- 단어 단위로

   ```python
   from nltk.tokenize.regexp import RegexpTokenizer
   tokenizer = RegexpTokenizer("[\w']+")
   print(tokenizer.tokenize("Can't is a contraction."))
   ```

   ```python
   from nltk.tokenize.regexp import regexp_tokenize 
   print(regexp_tokenize("Can't is a contraction.", "[\w']+"))
   ```
   
   - `Can't is a contraction.`
   - `Can't` `is` `a` `contraction`
 
 <br>

- Simple whitespace tokenizer (공백 기준으로)
   ```python
   from nltk.tokenize.regexp import RegexpTokenizer 
   tokenizer = RegexpTokenizer("\s+", gaps=True) 
   print(tokenizer.tokenize("Can't is a contraction."))
   ```
   - `Can't is a contraction.`
   - `Can't` `is` `a` `contraction.`

<br>

### ▪ Penn Treebank Tokenization
- 특징
   - 하이푼으로 구성된 단어는 하나로 유지
   - doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리

- example 
   ```python
   from nltk.tokenize import TreebankWordTokenizer
   tokenizer=TreebankWordTokenizer()
   text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
   ```
   - `"Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own.`
   - `Starting` `a` `home-based` `restaurant` `may` `be` `an` `ideal.` `it` `does` `n't` `have` `a` `food` `chain` `or` `restaurant` `of` `their` `own` `.`

<br>


## ◼ Subword Tokenization (문장 > subword)
**서브 워드 기반 토크나이즈**란 `자주 등장한 단어는 그대로 두고, 자주 등장하지 않은 단어는 의미있는 서브 워드 토큰들로 분절한다`라는 원칙에 기반을 둔 알고리즘.

### ▪ Byte-Pair Encoding (BPE)
- [참고 1](https://huffon.github.io/2020/07/05/tokenizers/#byte-pair-encoding) | [참고2](https://wikidocs.net/22592#1-bpebyte-pair-encoding)
- 특징
   - 빈도수에 기반하여 가장 많이 등장한 쌍을 병합

### ▪ WordPiece
- [참고1](https://huffon.github.io/2020/07/05/tokenizers/#wordpiece) | [참고2](https://wikidocs.net/22592#1-bpebyte-pair-encoding)
- 특징
   - 병합되었을 때 corpus의 우도(likelihood)를 가장 높이는 쌍을 병합   
   - 언더바(_)는 문장 복원을 위한 장치로, 복원 시 언더바 대신 띄어쓰기로 바꾸면 된다.
   - example)
      - `Jet makers feud over seat width with big orders at stake`
      - `_J` `et` `_makers` `_fe` `ud` `_over` `_seat` `_width` `_with` `_big` `_orders` `_at` `_stake`
   - BERT에서도 사용됨

### ▪ Unigram Language Model
- [참고1](https://huffon.github.io/2020/07/05/tokenizers/#unigram) | [참고2](https://wikidocs.net/22592#4-unigram-language-model-tokenizer)
- 특징
   - Unigram LM tokenizer는 각각의 서브워드들에 대해서 손실(loss)를 계산.
   - 이때, 손실이라는 것은 해당 서브워드가 단어 집합에서 제거되었을 경우, corpus의 우도(likelihood)가 감소하는 정도를 뜻함.
   - 측정된 서브워드들을 손실의 정도로 정렬 → 최악의 영향을 주는 10~20%의 토큰을 제거
   - 위 과정을 원하는 단어 집합의 크기에 도달할 때까지 반복.

### ▪ SentencePiece
- [참고1](https://huffon.github.io/2020/07/05/tokenizers/#sentencepiece) | [참고2](https://wikidocs.net/86657)
- 특징
   - 앞의 알고리즘과 달리, 입력 문장을 Raw Stream으로 취급해 공백을 포함한 모든 캐릭터를 활용해, BPE 혹은 Unigram을 적용하며 사전을 구축
   - Transformers 라이브러리가 지원하는 모델들 중 SentencePiece를 활용하는 모든 모델들의 토크나이저는 Unigram을 활용해 훈련됨.
   - ALBERT, XLNet 등에서도 사용됨.

<br><br>
---

### 참고
- nltk 관련 
   [EXCELSIOR Blog](https://excelsior-cjh.tistory.com/63) | 
   [wikidocs](https://wikidocs.net/21698) | 
   [Baek Kyun Shin Blog](https://bkshin.tistory.com/entry/NLP-2-%ED%85%8D%EC%8A%A4%ED%8A%B8-%ED%86%A0%ED%81%B0%ED%99%94Text-Tokenization) | 
   [kokirimambo Blog](https://mambo-coding-note.tistory.com/240)
- Subword Tokenization 관련
   [Huffon Blog](https://huffon.github.io/2020/07/05/tokenizers/) | 
   [Edward Ma Blog](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46) | 
   [wikidocs](https://wikidocs.net/22592)
