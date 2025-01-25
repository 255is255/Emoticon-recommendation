## 🌟 프로젝트 소개
비대면 소통 환경에서 감정 전달의 한계를 극복하기 위해, 한국어 및 영어 텍스트 기반 감정 분석과 이모티콘 추천 테스트를 목표하였습니다.  
간단하게 UI를 구성하고, 사용자의 텍스트에서 **5가지 감정** (공포불안, 놀람당황, 분노, 슬픔, 행복)을 탐지하여, 적합한 이모티콘을 제안합니다.

>[!NOTE]
> - 텍스트 기반 대화에서 감정을 실시간으로 탐지<br>
> - 한국어(KcBERT)와 영어(Logistic Regression)를 기반으로 감정 분석<br>
> - 유행어와 비속어가 포함된 텍스트에서도 정확한 감정 분석 가능<br>

<br>

## 🛠️ 기술 스택
- **언어 및 프레임워크**
  - 🐍 Python : 주요 프로그래밍 언어
  - 🖥️ PyTorch : KcBERT 모델 학습 및 Fine-Tuning
  - 📈 scikit-learn : Logistic Regression 및 Vectorizer 구현
- **데이터 처리**
  - 📚 Hugging Face : 토큰화 및 모델 관리
  - 📚 Konlpy : 한국어 데이터 전처리
- **개발 환경**
  - 💻 Local : RTX 4070 (Laptop 8GB)
  - ☁️ Google Colab T4

<br>

## 🗂️ 코드 구조
```plaintext
├── Korean
│   ├── data_filter.py : 데이터 전처리 및 불용어 제거
│   ├── KcBERT.py : KcBERT 기반 감정 분석 모델
│   ├── model_test_single.py : 단일 모델 테스트
│   └── model_test_ensemble.py : 앙상블 모델 테스트
├── English
│   ├── kaggle.json : Kaggle API 인증 파일
│   └── recommand_emoji_english.ipynb : Logistic Regression 기반 감정 분석
└── 통합 감정 분석 및 이모티콘 추천
    └── emotional_v2.py : 한국어/영어 통합 감정 분석 및 이모티콘 추천
````

<br>

## 🗂️ 코드 설명
### 한국어 감정 분석
- **data_filter.py** : 데이터 전처리(중복 제거, 불용어 처리 등)
- **KcBERT.py** : KcBERT 모델 학습 및 Fine-Tuning
- **model_test_ensemble.py** : 앙상블 기법 테스트

### 영어 감정 분석
- **recommand_emoji_english.ipynb** : Logistic Regression과 Vectorizer 기반 감정 분석 구현 및 모델 추출

### 통합 감정 분석 및 이모티콘 추천
- **emotional_v2.py** : 한국어/영어 통합 감정 분석 및 실시간 이모티콘 추천 시스템

<br>

## 🗂️ 모델 기능
### 한국어 감정 분석 (KcBERT)
- AIHub 데이터셋 기반 학습
- 불용어 제거 및 형태소 분석 적용
- 가변 가중치 기반 앙상블 기법으로 성능 개선
- 5가지 감정 분류(공포불안, 놀람당황, 분노, 슬픔, 행복)

### 영어 감정 분석 (Logistic Regression)
- Kaggle Emotions Dataset 기반 학습
- Bag of Words 및 TF-IDF를 활용한 벡터화
- 경량화된 고속 감정 분류 구현
- 5가지 감정 분류(공포불안, 놀람당황, 분노, 슬픔, 행복)


<br>

## 📈 결과물

### 한국어 감정 분석 성능
- 싱글모델(전체 데이터)

| ***감정*** | ***Precision*** | ***Recall*** | ***F1-Score*** | ***Support*** |
| --- | --- | --- | --- | --- |
| 공포불안 | 0.92 | 0.77 | 0.84 | 100 |
| 놀람당황 | 0.74 | 0.89 | 0.81 | 100 |
| 분노 | 0.81 | 0.92 | 0.86 | 100 |
| 슬픔 | 0.94 | 0.81 | 0.87 | 100 |
| 행복 | 0.97 | 0.93 | 0.95 | 100 |
| **Accuracy** |  |  | 0.86 | 500 |

- 싱글모델(불용어 사전제거 데이터)

| ***감정*** | ***Precision*** | ***Recall*** | ***F1-Score*** | ***Support*** |
| --- | --- | --- | --- | --- |
| 공포불안 | 1.00 | 0.71 | 0.83 | 100 |
| 놀람당황 | 0.75 | 0.91 | 0.82 | 100 |
| 분노 | 0.74 | 0.93 | 0.83 | 100 |
| 슬픔 | 0.90 | 0.70 | 0.79 | 100 |
| 행복 | 0.91 | 0.95 | 0.93 | 100 |
| **Accuracy** |  |  | 0.84 | 500 |

- 앙상블 모델(싱글모델 2개 결합)
  
| ***감정*** | ***Precision*** | ***Recall*** | ***F1-Score*** | ***Support*** |
| --- | --- | --- | --- | --- |
| 공포불안 | 0.99 | 0.78 | 0.87 | 100 |
| 놀람당황 | 0.74 | 0.93 | 0.83 | 100 |
| 분노 | 0.81 | 0.91 | 0.85 | 100 |
| 슬픔 | 0.97 | 0.83 | 0.89 | 100 |
| 행복 | 0.96 | 0.93 | 0.94 | 100 |
| **Accuracy** |  |  | 0.88 | 500 |

### 영어 감정 분석 성능

- 싱글모델(로지스틱 회귀)
  
| ***감정*** | ***Precision*** | ***Recall*** | ***F1-Score*** | ***Support*** |
| --- | --- | --- | --- | --- |
| 공포불안 | 0.83 | 0.83 | 0.83 | 9484 |
| 놀람당황 | 0.68 | 0.70 | 0.69 | 2960 |
| 분노 | 0.90 | 0.90 | 0.90 | 11625 |
| 슬픔 | 0.94 | 0.94 | 0.94 | 24100 |
| 행복 | 0.97 | 0.97 | 0.97 | 28282 |
| **Accuracy** |  |  | 0.92 | 76451 |

<br>

### 이모티콘 추천 예제

![image](https://github.com/user-attachments/assets/0424fc28-162e-472b-95d5-38a65b9de336)
![image](https://github.com/user-attachments/assets/9d0583d9-fd07-443d-826c-8868ec7f005f)

<br>



## 참조

### 데이터 출처
- [한국어 감정 정보가 포함된 단발성 대화 데이터셋(AI hub/아크릴)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=270)
- [한국어 감정 정보가 포함된 연속적 대화 데이터셋(AI hub/아크릴)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=271)
- [감성대화말뭉치(AI hub/미디어젠)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)
- [Emotion Dataset(Kaggle/BHAVIK JIKADARA)](https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset)

### 모델 출처
- [KcBERT](https://github.com/Beomi/KcBERT.git)

<br>

## 프로젝트 인원
- 최훈서 (Ajou University)
  - KcBERT 기반 한국어 감정분석
- 나성진 (Ajou University)
  - 로지스틱 회귀 기반 영어 감정 분석
