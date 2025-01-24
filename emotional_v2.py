import tkinter as tk
from tkinter import ttk
from functools import partial
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import string

# 모델 및 토크나이저 로드
model_paths = {
    "stopwords_removed": "./emotion_classification_model_20241213_ALLDATA_except_medium_Delete_Stopwords_KCBERT",
    "original": "./emotion_classification_model_20241213_ALLDATA_except_medium_KCBERT"
}

models = {}
tokenizers = {}

for key, path in model_paths.items():
    models[key] = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True).to("cuda").eval()
    tokenizers[key] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# 로지스틱 회귀 모델 로드
classifier = joblib.load('./emotion_classifier.pkl')
vectorizer = joblib.load('./vectorizer.pkl')

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 감정 라벨 매핑
emotion_labels_korean = {
    0: "공포불안",
    1: "놀람당황",
    2: "분노",
    3: "슬픔",
    4: "행복"
}

emotion_to_emoji_english = {
    0: ["😢", "😭", "😞"],
    1: ["😊", "😁", "😄"],
    3: ["😡", "😠", "🤬"],
    4: ["😨", "😰", "😱"],
    5: ["😳", "😮", "😵"]
}

emoji_data_korean = {
    "공포불안": ["😨", "😰", "😱"],
    "놀람당황": ["😳", "😮", "😵"],
    "분노": ["😡", "😠", "🤬"],
    "슬픔": ["😢", "😭", "😞"],
    "행복": ["😊", "😁", "😄"]
}

weights = {
    "stopwords_removed": 0.65,
    "original": 0.35
}

# 텍스트 전처리 함수
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# 입력 데이터 토큰화 함수
def tokenize_data(sentence, tokenizer):
    return tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

# 한국어 모델 예측 함수
def predict_korean(models, tokenizers, sentence, weights=None):
    if weights is None:
        weights = {key: 1.0 for key in models.keys()}
    else:
        total_weight = sum(weights.values())
        weights = {key: weight / total_weight for key, weight in weights.items()}

    ensemble_probs = None

    for key, model in models.items():
        tokenizer = tokenizers[key]
        inputs = tokenize_data(sentence, tokenizer)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()

        weighted_probs = probs * weights[key]

        if ensemble_probs is None:
            ensemble_probs = weighted_probs
        else:
            ensemble_probs += weighted_probs

    predictions = torch.argmax(ensemble_probs, dim=-1).item()
    return predictions, ensemble_probs

# 영어 모델 예측 함수
def predict_english(text):
    cleaned_text = clean_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    emotion = classifier.predict(text_vec)[0]
    probabilities = classifier.predict_proba(text_vec)[0]
    confidence = max(probabilities)
    return emotion, confidence

# GUI 기반 채팅창
class ChatInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat 이모지")
        self.root.geometry("600x800")
        self.root.configure(bg="#D4E6F1")

        self.mode = tk.StringVar(value="Korean")

        # 모드 선택
        self.mode_frame = tk.Frame(root, bg="#D4E6F1")
        self.mode_frame.pack(pady=10)

        tk.Label(self.mode_frame, text="언어 모드 선택:", bg="#D4E6F1", font=("Arial", 14)).pack(side="left", padx=5)
        mode_selector = ttk.Combobox(self.mode_frame, textvariable=self.mode, values=["Korean", "English"],
                                      state="readonly", width=10)
        mode_selector.pack(side="left", padx=5)

        # 채팅창 스타일
        self.chat_frame = tk.Frame(root, bg="#D4E6F1")
        self.chat_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.chat_window = tk.Canvas(self.chat_frame, bg="#D4E6F1", highlightthickness=0)
        self.chat_window.pack(side="left", fill="both", expand=True)

        self.chat_scrollbar = ttk.Scrollbar(self.chat_frame, orient="vertical", command=self.chat_window.yview)
        self.chat_scrollbar.pack(side="right", fill="y")
        self.chat_window.config(yscrollcommand=self.chat_scrollbar.set)

        self.messages_frame = tk.Frame(self.chat_window, bg="#D4E6F1")
        self.chat_window.create_window((0, 0), window=self.messages_frame, anchor="nw")

        self.messages_frame.bind("<Configure>",
                                 lambda event: self.chat_window.configure(scrollregion=self.chat_window.bbox("all")))

        # 입력창 스타일
        self.entry_frame = tk.Frame(root, bg="#D4E6F1")
        self.entry_frame.pack(pady=10, padx=10, fill="x")

        self.entry = tk.Entry(self.entry_frame, font=("Arial", 16), bg="#ffffff", bd=2, relief="groove")
        self.entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        self.send_button = tk.Button(self.entry_frame, text="보내기", font=("Arial", 14), bg="#4CAF50", fg="#ffffff",
                                     command=self.send_message)
        self.send_button.pack(side="right", padx=5, pady=5)

        root.bind('<Return>', lambda event: self.send_message())

        self.is_user_turn = True

    def send_message(self):
        message = self.entry.get()
        if not message.strip():
            return

        sender = "나" if self.is_user_turn else "상대"
        self.display_message(sender, message, is_self=self.is_user_turn)

        if self.mode.get() == "Korean":
            emotion, probs = predict_korean(models, tokenizers, message)
            emotion_label = emotion_labels_korean[emotion]
            emojis = emoji_data_korean.get(emotion_label, [])
            threshold = probs[0, emotion].item()
            print(f"입력한 문장: {message}")
            print(f"분류된 감정: {emotion_label}")
            print(f"추천된 이모티콘: {emojis}")
        else:
            emotion, confidence = predict_english(message)
            emojis = emotion_to_emoji_english.get(emotion, ["\U0001F937"])
            threshold = confidence
            print(f"입력한 문장: {message}")
            print(f"분류된 감정: {emotion}")
            print(f"추천된 이모티콘: {emojis}")

        self.display_emotion_recommendation(emojis, threshold)
        self.is_user_turn = not self.is_user_turn
        self.entry.delete(0, tk.END)

    def display_message(self, sender, message, is_self=False):
        frame = tk.Frame(self.messages_frame, bg="#D4E6F1", pady=5)
        frame.pack(anchor="e" if is_self else "w", fill="x", padx=10, pady=5)

        bubble_frame = tk.Frame(frame, bg="#D4E6F1")
        bubble_frame.pack(anchor="e" if is_self else "w", padx=(100, 0) if is_self else (0, 400))

        bg_color = "#FFF176" if is_self else "#ffffff"
        bubble = tk.Label(bubble_frame, text=message, font=("Arial", 14), bg=bg_color, wraplength=350, padx=10, pady=5,
                          bd=2, relief="ridge")
        bubble.pack(anchor="e" if is_self else "w")

        sender_label = tk.Label(frame, text=sender, font=("Arial", 12, "bold"), bg="#D4E6F1", fg="#555555")
        sender_label.pack(anchor="e" if is_self else "w")

        self.chat_window.update_idletasks()
        self.chat_window.yview_moveto(1)

    def display_emotion_recommendation(self, emojis, threshold):
        if emojis:
            emoji_string = " ".join([f"{emoji}" for emoji in emojis])
            self.display_system_message(f"추천 이모티콘: {emoji_string}")

    def display_system_message(self, message):
        frame = tk.Frame(self.messages_frame, bg="#D4E6F1", pady=2)
        frame.pack(anchor="center", pady=2)

        system_message = tk.Label(frame, text=message, font=("Arial", 20), bg="#D4E6F1", fg="#777777", wraplength=350,
                                  justify="center")
        system_message.pack(anchor="center")

        self.chat_window.update_idletasks()
        self.chat_window.yview_moveto(1)

# 애플리케이션 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatInterface(root)
    root.mainloop()
