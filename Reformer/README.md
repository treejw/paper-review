# Reformer: The Efficient Transformer `2020.1`

## Abstract
- Transformer 모델은 긴 시퀀스에 대해 큰 비용을 요구한다.
- Transformer 모델의 효율성을 증가 시키기 위해  locality-sensitive hashing과 reversible residual layers을 사용한다.
- Transformer 모델과 동등한 성능을 발휘하면서 훨씬 더 메모리 효율적이고 긴 시퀀스에서 훨씬 더 빠릅니다.

<br>

## 1. Introduction
- Transformer는 자연어 분야에 널리 사용되며 여러 분야에서 SOTA를 달성한다.
- 이를 위해서 더욱 더 큰 Transformer 모델을 연구하고 훈련한다.
- 하지만 이러한 대형 Transformer 모델은 대규모 산업 연구 실험실에서만 현실적으로 훈련할 수 있으며
- 모델 병렬 처리로 훈련된 이러한 모델은 단일 훈련 단계에서도 메모리 요구 사항이 커 때문에 단일 GPU에서 미세 조정할 수도 없다.
- 메모리 사용량을 증가시키는 원인은 다음과 같다.
   - N 레이어가 있는 모델은 역전파를 위해 출력값을 저장해야 하기 때문에 단일 레이어 모델보다 N배 더 크다.
   - 레이어 중간 피드포워드 레이어의 깊이는 dmodel보다 훨씬 크기 때문에 메모리 사용의 많은 부분을 차지한다.
   - 길이 L의 시퀀스에 대한 Attention은 계산 및 메모리 복잡성 모두에서 O(L2)이므로 단일 시퀀스 64K 토큰의 경우에도 GPU 메모리를 고갈시킨다.
- 다음 방법을 적용하여 이를 효과적으로 연산 가능하게 한다.
   - Reversible layers
   - chunks & removes
   - locality-sensitive hashing  

<br>

### ◼ Contribution
 우리는 합성 작업, 길이 64K의 시퀀스가 있는 텍스트 작업 및 길이 12K의 시퀀스가 있는 이미지 생성 작업에서 리포머가 전체 Transformer로 얻은 결과와 일치하지만 특히 텍스트 작업에서 훨씬 더 빠르게 실행되고 훨씬 더 나은 메모리 효율성으로 실행됨을 보여준다.

<br>

## 2. Background
### ◼ Memory
- Transformer usage memory ≈ 모델 깊이 x 모델 넓이 x 문장 길이 x batch size
- 딥러닝 모델 Memory 사용 특징: 역전파하기 전까지 중간 결과물을 저장 해야 한다.
<img src='https://user-images.githubusercontent.com/41243762/145993741-ecf47dd2-7811-4cec-bda2-146f11c7a8aa.png' width='90%'>
<br>
<img src='https://user-images.githubusercontent.com/41243762/145994712-b056996c-74fa-473d-979f-ae0c6e0af0e5.png' width='90%'>
<br>

### ◼ 효율적인 메모리 사용법
- Transformer에서 memory를 효율적으로 사용하려면
   - 모델 깊이, 넓이, 문장 길이 줄이기 ❌ 
   - batch size 줄이기 ❌ 
   - 메모리에 저장되는 중간 결과물 줄이기 ✔️ 
<br>

### ◼ 중간 결과물을 줄이는법
![image](https://user-images.githubusercontent.com/41243762/145995316-cc987699-783d-482c-802e-5b15d4afe893.png)
- 필요 없는 연결 제거
   - locality-sensitive hashing Attention
- 메모리 저장 구조 변경
   - Reversible layers 
- 단계별 메모리 사용
   - chunks & removes
<br>

## 3. locality-sensitive hashing Attention (LSH Attention)
### ◼ 개념
![image](https://user-images.githubusercontent.com/41243762/145996157-bbf14453-8496-4df5-bfb8-337acabb222b.png)
- 기존 Attention은 L^2의 연산량과 그 결과물을 저장
- locality-sensitive hashing을 사용하여 유사한 데이터 사이에만 Attention을 선택적으로 연산하는 방법
<br>

### ◼ locality-sensitive hashing
<img src="https://user-images.githubusercontent.com/41243762/145996680-e5227bb0-7eeb-41d0-8dd8-742ef6d8f0fd.png" width="60%">
- 유사한 데이터를 같은 값으로 hash하는 기법
<br>

### ◼ LSH Attention 적용
1. 각 토큰의 (Key, Query), Value 생성
   - Key 와 Query는 동일한 Linear Layer에서 생성
      - LSH를 적용하기 위하여 key, Query를 같은 공간에 Projection<br><br>
2. Locality-Sensitive Hashing 적용
![image](https://user-images.githubusercontent.com/41243762/145996827-bd702899-9104-44e2-9478-3260e8f099c2.png)<br><br>
3. Index가 같은 것 끼리 Sorting<br><br>
![image](https://user-images.githubusercontent.com/41243762/145997860-1a448f4c-3b8e-47bf-b98f-4dc132b8c520.png)
4. Chunking Sequence
   - 동일한 Index를 갖고 있는 Token이 많아도 특정 길이 이상 Attention을 할 수 없으므로 문장길이에 Robust 하다. <br><br>
5. Attention 적용<br><br>

※ Multi Round LSH Attention
- 2~3번을 여러번 진행하여 다양한 결과를 얻음

<br>

## 4. Reversible layers
![image](https://user-images.githubusercontent.com/41243762/145998549-83f013ef-065c-4427-bef0-3dfd0d9c1b2a.png)

입력값 x를 복사하여 x1, x2로 구성, Reversible Network를 통과한 결과값 y1, y2은 평균하여 y로 변경
출력값(y1, y2)과 출력 Gradient (y1, y2) 로 입력값(x1, x2)과 입력 Gradient (x1, x2)을 추출
중간결과물 저장 없이 역전파를 계산할 수 있는 구조

![image](https://user-images.githubusercontent.com/41243762/145998710-84cf255d-1a0a-4810-b0ff-c40e92616877.png)

## 5. chunks & removes
![image](https://user-images.githubusercontent.com/41243762/145999530-d32cb5bb-f980-4650-a3b4-d1c153511c84.png)
<br>

## 6. Experiments
### ◼ shared-QK
![image](https://user-images.githubusercontent.com/41243762/145999678-7f7266bb-03d6-48e4-a983-0ad6829a1464.png)

### ◼ locality-sensitive hashing
![image](https://user-images.githubusercontent.com/41243762/145999805-6d2d6af2-5784-49e1-be26-66400800ecea.png)

---
### 참고
- paper : [Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf)
- [PINGPONG's blog](https://blog.pingpong.us/reformer-review/) | [Korea Univ DSBA Lab Youtube](https://www.youtube.com/watch?v=6ognBL6DEYM)
- [그림](https://drive.google.com/file/d/1Yrl0uecfkl-BkjlmJ6HZr6IzpzpIyPC_/view)
