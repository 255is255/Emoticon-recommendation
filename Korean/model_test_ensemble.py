import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 두 모델의 경로
model_paths = {
    "stopwords_removed": "./MODEL_NAME_STOPWORDS_REMOVED",
    "original": "./MODEL_NAME_ORIGINAL"
}

# 모델 및 토크나이저 로드
models = {}
tokenizers = {}

for key, path in model_paths.items():
    models[key] = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True).to(device).eval()
    tokenizers[key] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# 감정 라벨 매핑
emotion_labels = {
    0: "공포불안",
    1: "놀람당황",
    2: "분노",
    3: "슬픔",
    4: "행복"
}

# 데이터 로드
file_path = r"./TEST_DATA.csv"
test_data = pd.read_csv(file_path)

# 입력 데이터 추출
test_sentences = test_data["Sentence"].tolist()
test_labels = test_data["label"].tolist()

# 입력 데이터 토큰화 및 GPU 전송
def tokenize_data(sentences, tokenizer):
    return tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# 앙상블 예측
def ensemble_predict(models, tokenizers, sentences, weights=None):
    # 가중치 초기화
    if weights is None:
        weights = {key: 1.0 for key in models.keys()}  # 모든 모델의 가중치를 동일하게 설정
    else:
        total_weight = sum(weights.values())
        weights = {key: weight / total_weight for key, weight in weights.items()}  # 가중치 정규화

    ensemble_probs = None

    for key, model in models.items():
        tokenizer = tokenizers[key]
        inputs = tokenize_data(sentences, tokenizer)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()

        # 가중치 적용
        weighted_probs = probs * weights[key]

        if ensemble_probs is None:
            ensemble_probs = weighted_probs
        else:
            ensemble_probs += weighted_probs

    # 최종 예측
    predictions = torch.argmax(ensemble_probs, dim=-1).tolist()
    return predictions, ensemble_probs

# 모델 예측 수행
weights = {
    "stopwords_removed": 0.65,
    "original": 0.35
}
predictions, ensemble_probs = ensemble_predict(models, tokenizers, test_sentences, weights=weights)

# 성능 평가
target_names = [emotion_labels[i] for i in range(len(emotion_labels))]
print("Classification Report:")
print(classification_report(test_labels, predictions, target_names=target_names))

# 결과 출력
print("\nPrediction Results:")
for sentence, pred, true, prob in zip(test_sentences, predictions, test_labels, ensemble_probs):
    predicted_emotion = emotion_labels[pred]
    true_emotion = emotion_labels[true]
    confidence = max(prob).item()
    print(f"문장: {sentence} -> 예측된 감정: {predicted_emotion} (Confidence: {confidence:.2f}), 실제 감정: {true_emotion}")
