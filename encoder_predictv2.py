import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 저장된 모델과 토크나이저 경로 설정
MODEL_DIR = './model_save_minilm/'

# 2. 토크나이저 및 모델 로드
print("⏳ 모델을 불러오는 중입니다...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# GPU 사용 가능 여부 확인 및 적용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델을 평가(추론) 모드로 변경 (필수)
model.eval()
print(f"✅ 모델 로드 완료! (사용 장치: {device})\n")

# 3. 예측 함수 정의
def predict_sentiment(text):
    # 입력된 텍스트를 토큰화
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length', 
        max_length=128
    )
    
    # 데이터를 모델이 있는 장치(GPU/CPU)로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 기울기 계산 비활성화 
    with torch.no_grad():
        outputs = model(**inputs)

    # 모델의 출력값(로짓)을 확률값으로 변환
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    
    # 가장 높은 확률을 가진 클래스와 그 확률 추출
    pred_class = torch.argmax(probabilities, dim=-1).item()
    pred_prob = probabilities[0][pred_class].item() * 100 

    # 🚨 [수정된 부분] 결과 문자열 생성 (0: 부정, 1: 중립, 2: 긍정)
    if pred_class == 2:
        sentiment = "긍정 🟢"
    elif pred_class == 1:
        sentiment = "중립 🟡"
    else:
        sentiment = "부정 🔴"
    
    return sentiment, pred_prob

# --- 4. 실제 텍스트로 테스트해보기 ---

test_sentences = [
    "걍 쓸만함"
]

print("--- 🔍 리뷰 감성 분석 결과 ---")
for sentence in test_sentences:
    sentiment, confidence = predict_sentiment(sentence)
    print(f"리뷰: {sentence}")
    print(f"결과: {sentiment} (확신도: {confidence:.2f}%)\n")