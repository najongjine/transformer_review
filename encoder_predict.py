import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ----------------------------------------------------
# 1. 모델 및 토크나이저 로드 (수정된 경로)
# ----------------------------------------------------
# 훈련 시 사용했던 모델 이름 (토크나이저 로드용)
MODEL_NAME = "beomi/kcbert-base"
# 🚨🚨🚨 수정된 부분: 이미지에서 확인된 가장 최신 체크포인트 폴더를 지정합니다. 🚨🚨🚨
FINETUNED_MODEL_PATH = "./kcbert_results/checkpoint-240"

print("✅ 훈련된 모델 로드 중...")

# 훈련된 모델과 토크나이저를 로드합니다.
# Trainer는 체크포인트 폴더 안에 config.json, pytorch_model.bin 등을 저장합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval() # 모델을 평가 모드로 설정

# ----------------------------------------------------
# 2. 예측 함수 정의
# ----------------------------------------------------
def predict_sentiment(text):
    """
    주어진 텍스트에 대한 감성(긍정/부정)을 예측합니다.
    """
    # 텍스트 토큰화 및 텐서 변환
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True
    )

    # 데이터를 모델과 동일한 장치(GPU 또는 CPU)로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 예측 수행
        outputs = model(**inputs)
        logits = outputs.logits

    # 로짓을 확률로 변환 (소프트맥스)
    probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # 가장 높은 확률을 가진 클래스(0 또는 1) 선택
    prediction = np.argmax(probabilities)

    # 결과 해석
    sentiment_map = {0: "부정 (Negative)", 1: "긍정 (Positive)"}

    print("-" * 30)
    print(f"입력 텍스트: {text}")
    print(f"예측 결과: {sentiment_map[prediction]}")
    print(f"긍정 확률: {probabilities[1]:.4f}")
    print(f"부정 확률: {probabilities[0]:.4f}")
    print("-" * 30)

    return prediction

# ----------------------------------------------------
# 3. 새로운 텍스트로 테스트
# ----------------------------------------------------
print("🚀 새로운 텍스트로 감성 예측 시작...")

# 긍정적인 예시
predict_sentiment("느금마가 좋아할듯")

# 부정적인 예시
predict_sentiment("느금마 만수무강.")

# 중립적인/모호한 예시
predict_sentiment("그냥 평범했고, 특별히 좋지도 나쁘지도 않았습니다.")

print("\n✅ 예측 완료!")