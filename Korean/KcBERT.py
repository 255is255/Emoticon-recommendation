import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# OS Setting(GPU 설정, 모델 경로 설정)
device = 0 if torch.cuda.is_available() else -1
os.environ["HF_HOME"] = r"D:\Project\SNS\huggingface"
print("Hugging Face 캐시 경로:", os.getenv("HF_HOME"))

# 감정 라벨 매핑
emotion_labels = {
    "공포불안": 0,
    "놀람당황": 1,
    "분노": 2,
    "슬픔": 3,
    "행복": 4,
}
num_labels = len(emotion_labels)
file_path = r"D:\Project\SNS\new_dataset\ALL_data_except_medium.csv"


# 평가 메트릭 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 데이터 로드 및 처리 함수
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df = df[["Sentence", "Emotion"]].dropna()  # 'Sentence'와 'Emotion' 열만 사용
    df["label"] = df["Emotion"].map(emotion_labels)
    # 라벨을 정수형으로 강제 변환
    df["label"] = df["label"].astype(int)
    return df

# 토큰화 함수
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

all_data = load_and_process_data(file_path)
# 데이터셋 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_data["Sentence"].tolist(),
    all_data["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# Hugging Face Dataset 생성
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

# KcBERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    "beomi/KcBERT-base",
    trust_remote_code=True,
    cache_dir=os.getenv("HF_HOME")
)

# 데이터셋 토큰화
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 데이터셋 포맷 설정
train_dataset = train_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset = val_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])

# KcBERT 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    "beomi/KcBERT-base",
    num_labels=num_labels,
    trust_remote_code=True
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,  # GPU 용량에 따라 조절 가능
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 데이터 콜레이터
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 모델 학습
trainer.train()

# 모델 평가
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)
with open("5emotion_evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Evaluation Metrics saved to evaluation_metrics.json")

# 모델 저장
model.save_pretrained("./emotion_classification_model_20241213_ALLDATA__KCBERT")
tokenizer.save_vocabulary("./emotion_classification_model_20241213_ALLDATA_KCBERT")
tokenizer.save_pretrained("./emotion_classification_model_20241213_ALLDATA_KCBERT")
