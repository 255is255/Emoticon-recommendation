import os
import pandas as pd
import re
from konlpy.tag import Okt
from collections import Counter

# 디렉터리 경로 설정
directory_path = r"D:\Project\SNS\new_dataset"

# 파일 탐색 및 중복 제거 처리
for filename in os.listdir(directory_path):
    if filename.endswith("ALL_data.csv"):
        file_path = os.path.join(directory_path, filename)
        try:
            # CSV 파일 읽기G
            df = pd.read_csv(file_path, encoding='utf-8')

            # 중복 제거 (문장이 있는 컬럼을 지정)
            if "Sentence" in df.columns:  # "문장" 컬럼 이름을 실제 파일에 맞게 수정
                df = df.drop_duplicates(subset=['Sentence'])

            # 중복 제거된 데이터 저장
            output_path = os.path.join(directory_path, f"deduplicated_{filename}")
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"중복 제거 완료: {output_path}")
        except Exception as e:
            print(f"오류 발생: {file_path}, {e}")

# Okt 객체 초기화
okt = Okt()

# 데이터 로드
dataset_path = r"D:\Project\SNS\new_dataset\ALL_data.csv"
data = pd.read_csv(dataset_path)

# 특수문자 제거 및 한글 문자만 유지
data['Sentence'] = data['Sentence'].apply(lambda x: re.sub(r"[^ㄱ-ㅎ가-힣ㅏ-ㅣ\s]", "", str(x)))

# 형태소 분석 후 단어 빈도 계산
all_tokens = [token for sentence in data['Sentence'] for token in okt.morphs(sentence)]
word_freq = Counter(all_tokens)

# 상위 n개의 단어 확인
print(word_freq.most_common(50))

#########형태소 분석에 따라 나타난 불용어 목록(명사, 감탄사 제외)
stopwords = [
    "이", "가", "을", "에", "는", "은", "로", "의", "를", "도", "한", "에서",
    "으로", "게", "해", "다", "요", "하고", "하는", "한테", "야", "좀", "뭐", "왜",
    "네", "그", "말", "만", "안", "못", "이야", "같아", "오늘", "우리", "정말", "해서",
    "내", "거", "너무", "친구", "나", "돈"
]


# 불용어 제거 함수 정의
def remove_stopwords(sentence, stopwords):
    tokens = okt.morphs(sentence)  # 형태소 분석
    filtered_tokens = [token for token in tokens if token not in stopwords]  # 불용어 제거
    return " ".join(filtered_tokens)  # 다시 문장으로 결합

# 'Sentence' 컬럼에서 불용어 제거
data['Processed_Sentence'] = data['Sentence'].apply(lambda x: remove_stopwords(x, stopwords))

# 결과 저장
output_path = (r"D:\Project\SNS\new_dataset\ALL_Data_Delete_Stopwords.csv")
data.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"불용어 제거 완료! 결과는 {output_path}에 저장되었습니다.")

