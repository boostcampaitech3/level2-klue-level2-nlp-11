<h1 align="center">
Relation Extraction
</h1>

## Usage

### Install

```python
pip install -r requirements.txt
```

### Train

```python
python train.py
```

- 학습에 필요한 파라미터 설정은 `./config.yaml`에서 가능합니다.

### Inference

```python
python infernce.py
```

## Project Overview

### **KLUE Relation Extraction Competition**

![Untitled](https://user-images.githubusercontent.com/97524113/172302297-0d31b51e-2c11-40c9-9a6d-10cb543e9061.png)


- 위와 같이 Sentence, Subject entity, Object entity가 주어졌을 때 문장 안에서 Subject entity와 Object entity 간에 존재하는 관계를 예측하는 대회

### **Datasets**

- KLUE RE Dataset
- Train 32,470문장 / Test 7,765문장
- Label별 데이터 개수

![K-011](https://user-images.githubusercontent.com/97524113/172302321-6769d7ad-0c2d-4615-b61b-b72172d08b2b.jpg)


### **Model**

**R-BERT**

- Relation Classification을 위해 고안된 BERT 모델
    
    (Enriching Pre-trained Language Model with Entity Information for Relation Classification, Shanchan Wu & Yifan He, 2019)
    
    ![K-018](https://user-images.githubusercontent.com/97524113/172302349-59fc6387-9568-4e1f-b0fe-e806aa2cef94.jpg)

    
- ‘국립국어원 2021년 인공지능 언어능력평가’에서 1등을 차지한 Team BC의 동형이의어 구별을 위한 R-BERT 코드 참고 ([https://github.com/NIKL-Team-BC/NIKL-KLUE](https://github.com/NIKL-Team-BC/NIKL-KLUE))

**기존 R-BERT에서 추가된 것**

- R-BERT Classification Layer, R-BERT Input Sequence와 LSTM Input Sequence를 수정했습니다.
- R-BERT의 Input Sequence를 BERT가 pretrain할 때 사용했던 Input과 같이 구성했습니다.
    - 기존 Sentence에 “*sub entity 와 obj entity의 관계는?”* 이라는 문장을 매 입력마다 추가해 주어 Input Sequence를 BERT pretrain input과 유사하게 구성했습니다.
- Sequence의 매 Entity 마다 type_special_token, entity_token을 앞 뒤로 추가해 주었습니다.
    - “이순신은 조선 중기의 무신이다.” → “%PER이순신PER%은 #DAT조선 중기DAT#의 무신이다.”
- 단순하게 R-BERT를 통하여 출력된 Sequence를 Fc Layer를 통하여 분류하는 것보다는 Sequence Data의 특성을 이용하기 위하여 LSTM에 한 번 더 통과한 후 fc layer를 이용하여 분류를 진행했습니다.
- 길이가 비슷한 Sequence끼리 Batch를 구성하는 방식인 Sequence Bucketing을 사용하여 학습 속도를 개선했습니다.

### **Experiments**

**데이터**

- Data Augmentation
    - Back Translation
    - Random Masking
    - Token Deletion
    - **QA 모델(실제 사용)**
- Preprocess
    - 한자 → 한글
    - 특수 기호 제거
    - **Typed** **Entity Marker(실제 사용)**

**Model**

- R-BERT
    - 문장과 두 개의 Target Entity의 위치 정보를 전달하여 처리하도록 구현
    - F1-Score 70.6067, AUPRC 75.4665
- R-BERT with LSTM(full_sequence)
    - 마지막 FC Layer 앞에 LSTM을 추가
    - F1-Score 72.1511, AUPRC 75.7257
- R-BERT base with LSTM(entity_sequence)
    - LSTM을 추가하되, 전체 Sequence가 아닌 Target Entity만을 전달
    - F1-Score 71.9474  AUPRC 77.5083
- Ensemble
    - 유사한 성능을 보이지만 상관 관계는 낮은 모델 간 앙상블이 서로를 보완해줄 것이라 가정
    - 이를 바탕으로 모델 별 예측 확률 상관 관계를 파악하여 Hard Voting
    - F1-Score 75.41 AUPRC 80.85

### **Reference**

****[https://github.com/NIKL-Team-BC/NIKL-KLUE](https://github.com/NIKL-Team-BC/NIKL-KLUE)****

****[Enriching Pre-trained Language Model with Entity Information for Relation Classification(R-BERT)](https://arxiv.org/abs/1905.08284)****

****[An Improved Baseline for Sentence-level Relation Extraction(Typed Entity Marker)](https://arxiv.org/abs/2102.01373)****
