import torch
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification
import time

# --- 1. 저장된 모델 및 토크나이저 로드 경로 설정 ---
# 학습 코드에서 모델과 토크나이저가 저장된 디렉토리입니다.
MODEL_DIR = './model_save/' 

# --- 2. Device 설정 ---
# 학습 때와 동일하게 GPU 사용 가능 여부를 확인합니다.
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('✅ GPU 사용 가능: {}'.format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print('⚠️ GPU를 찾을 수 없습니다. CPU를 사용합니다.')

# --- 3. 모델 및 토크나이저 로드 ---
try:
    # 1. 토크나이저 로드 (학습 시 사용한 BertTokenizerFast)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    print(f"✅ 토크나이저 로드 완료: {MODEL_DIR}")

    # 2. 모델 로드 (학습된 가중치를 포함하는 BertForSequenceClassification)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval() # 모델을 평가 모드로 설정
    print(f"✅ BERT 모델 로드 완료: {MODEL_DIR}")

except Exception as e:
    print(f"❌ 모델/토크나이저 로드 실패: {e}")
    print("   'model_save' 디렉토리에 config.json, model.safetensors 등이 존재하는지 확인하세요.")
    exit()

# --- 4. 추론(Inference) 함수 정의 ---

def predict_sentiment(text: str, max_length: int = 128):
    """
    주어진 텍스트에 대해 감성 분석(긍정/부정)을 수행합니다.
    
    :param text: 분류할 입력 텍스트
    :param max_length: 토큰화 시 최대 길이
    :return: (예측 라벨 (1: 긍정, 0: 부정), 긍정 확률, 부정 확률)
    """
    
    # 1. 입력 텍스트 토큰화
    # PyTorch 텐서 형태로 반환
    encoded_input = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt' 
    )

    # 2. 입력 데이터를 Device로 이동
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    token_type_ids = encoded_input['token_type_ids'].to(device)
    
    # 3. 모델 추론 실행
    with torch.no_grad(): # 평가 모드이므로 기울기 계산 비활성화
        outputs = model(
            input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
    
    # 4. 예측 결과 (Logits) 처리
    logits = outputs.logits # Logits: 분류 전의 원시 점수 텐서
    
    # Logits을 확률로 변환 (Softmax 사용)
    probabilities = torch.softmax(logits, dim=1)
    
    # 결과를 CPU로 이동하고 NumPy 배열로 변환
    prob_np = probabilities.cpu().numpy()[0]
    
    # 예측 라벨 (점수가 가장 높은 인덱스)
    predicted_label = np.argmax(prob_np) 
    
    # 긍정(Positive, 인덱스 1) 확률, 부정(Negative, 인덱스 0) 확률
    # 학습 시 라벨링: 긍정(1): 4~5점, 부정(0): 1~2점
    neg_prob = prob_np[0]
    pos_prob = prob_np[1]
    
    return predicted_label.item(), pos_prob.item(), neg_prob.item()

# --- 5. 테스트 실행 ---
if __name__ == "__main__":
    
    print("\n--- 🧠 감성 분류 테스트 ---")
    
    test_texts = [
        "느그집 누렁이도 거를듯.", # 부정
        "느금마 만수무강",
        "느금마",
        "Justine Beaver might like it" # 다국어 테스트
    ]
    
    for text in test_texts:
        t_start = time.time()
        label, pos_prob, neg_prob = predict_sentiment(text)
        t_end = time.time()

        sentiment = "긍정 (Positive)" if label == 1 else "부정 (Negative)"
        
        print(f"\n[입력]: {text}")
        print(f"  [결과]: {sentiment}")
        print(f"  [확률]: 긍정 {pos_prob:.4f} | 부정 {neg_prob:.4f}")
        print(f"  [시간]: {(t_end - t_start) * 1000:.2f} ms")