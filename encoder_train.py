import pandas as pd
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, SequentialSampler, RandomSampler
import numpy as np
import time
import datetime

# 1. 사용할 모델의 이름 정의
# BERT 모델 중 다국어 지원, 기본 사이즈, 대소문자 구분 버전을 사용합니다.
MODEL_NAME = "bert-base-multilingual-cased"

# 2. 다국어 토크나이저 로드
# Fast Tokenizer를 사용하여 속도를 개선합니다.
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
print(f"✅ 토크나이저 로드 완료: {MODEL_NAME}")

# 3. 데이터 로딩 및 라벨링 함수 (이전과 동일)
def load_and_preprocess_data(file_path):
    """
    파일을 읽고, 평점을 기반으로 긍정/부정 라벨을 부여하며, 3점 리뷰를 제외합니다.
    """
    data = []
    # 파일 읽기: 평점과 리뷰 내용이 탭(\t)으로 구분되어 있음
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', 1)
            if len(parts) == 2:
                try:
                    score = int(parts[0])
                    review = parts[1]

                    # 3점 제외
                    if score == 3:
                        continue

                    # 긍정(1): 4~5점, 부정(0): 1~2점
                    label = 1 if score >= 4 else 0
                    data.append([review, label])

                except ValueError:
                    continue

    df = pd.DataFrame(data, columns=['text', 'label'])
    print(f"✅ 총 {len(df)}개의 리뷰 데이터 준비 완료 (3점 제외).")
    return df

# 4. 토큰화 실행 함수
def tokenize_data(df, tokenizer, max_length=128):
    """
    DataFrame의 텍스트 데이터를 지정된 다국어 토크나이저를 사용하여 토큰화합니다.
    """
    # Hugging Face 토크나이저는 인코딩 과정에서 토큰화, 인덱스 변환, 패딩, 어텐션 마스크 생성을 모두 처리합니다.
    tokenized_data = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt' # PyTorch 텐서 형태로 반환
    )
    return tokenized_data, df['label'].tolist()


# --- 실행 ---
file_path = 'shopping.txt' # 파일 경로 지정 (실제 파일이 현재 경로에 있다고 가정)

# 1. 데이터 로딩 및 라벨링
df_data = load_and_preprocess_data(file_path)

# 2. 토큰화 실행
tokenized_inputs, labels = tokenize_data(df_data, tokenizer, max_length=128)

# 결과 확인 (첫 번째 리뷰)
print("\n--- 토큰화 결과 (첫 번째 리뷰) ---")
print(f"원문: {df_data.iloc[0]['text']}")
print(f"라벨: {df_data.iloc[0]['label']} (1:긍정, 0:부정)")
print("Input IDs (토큰 인덱스):", tokenized_inputs['input_ids'][0][:15]) # 앞 15개 출력
print("Attention Mask:", tokenized_inputs['attention_mask'][0][:15])
print("Token Type IDs:", tokenized_inputs['token_type_ids'][0][:15])

print(f"\n✅ 최종 준비된 데이터 개수: {len(labels)}")


# --- 3. 데이터셋 분할 및 데이터로더 생성 ---

# PyTorch 텐서 형태로 변환된 토큰화 결과와 라벨
input_ids = tokenized_inputs['input_ids']
attention_masks = tokenized_inputs['attention_mask']
token_type_ids = tokenized_inputs['token_type_ids']
labels_tensor = torch.tensor(labels)

# 1. TensorDataset 생성
# BERT 입력에 필요한 모든 텐서를 하나의 데이터셋으로 묶습니다.
dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels_tensor)

# 2. 훈련, 검증, 테스트 셋 크기 계산 (예: 80% / 10% / 10%)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# 3. random_split을 사용하여 데이터셋 분할
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"\n--- 데이터셋 분할 결과 ---")
print(f"총 데이터 수: {len(dataset)}")
print(f"훈련 셋 (Train Set) 크기: {len(train_dataset)}")
print(f"검증 셋 (Validation Set) 크기: {len(val_dataset)}")
print(f"테스트 셋 (Test Set) 크기: {len(test_dataset)}")

# 4. DataLoader 생성 (배치 학습 준비)
batch_size = 16 

# 훈련 데이터로더: 무작위 샘플링
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset), # 데이터를 무작위로 섞음
    batch_size=batch_size
)

# 검증 및 테스트 데이터로더: 순차적 샘플링
val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset), # 순서대로 샘플링
    batch_size=batch_size
)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset), # 순서대로 샘플링
    batch_size=batch_size
)

print(f"✅ DataLoader 생성 완료 (Batch Size: {batch_size}).")


# --- 4. BERT 모델 로드 및 설정 ---

# 1. Device 설정 (GPU 사용 가능 여부 확인)
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('✅ GPU 사용 가능: {}'.format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print('⚠️ GPU를 찾을 수 없습니다. CPU를 사용합니다.')

# 2. 모델 로드
# 분류 태스크를 위해 BertForSequenceClassification을 사용하며, 클래스 개수(2개: 긍정/부정)를 지정합니다.
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels = 2,    # 출력 클래스 개수 (긍정, 부정)
    output_attentions = False, # Attention 가중치 반환 안 함
    output_hidden_states = False, # 모든 hidden state 반환 안 함
)

# 모델을 설정된 Device로 이동
model.to(device)

print(f"✅ BERT 모델 로드 완료: {MODEL_NAME} (num_labels=2)")


# 3. 옵티마이저 및 학습률 스케줄러 설정
# BERT Fine-tuning에 일반적으로 사용되는 하이퍼파라미터
epochs = 4 # 학습 에폭 수 (권장: 2~4)
learning_rate = 2e-5 # BERT Fine-tuning에 적합한 작은 학습률 (권장: 1e-5 ~ 5e-5)
adam_epsilon = 1e-8 
warmup_steps = 0 

# 옵티마이저 설정 (AdamW: 가중치 감쇠(Weight Decay)가 개선된 Adam)
optimizer = AdamW(
    model.parameters(),
    lr = learning_rate,
    eps = adam_epsilon
)

# 학습 스케줄러 설정 (Linear Warmup and Decay)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = warmup_steps, 
    num_training_steps = total_steps
)

print(f"✅ 옵티마이저 및 스케줄러 설정 완료 (학습률: {learning_rate}, 에폭: {epochs})")
print("이제 학습 루프(Training Loop)를 추가하여 Fine-tuning을 진행하시면 됩니다.")



# --- 5. 학습 루프 (Fine-tuning) 및 평가 ---

# 정확도 계산 함수 정의
def flat_accuracy(preds, labels):
    """예측 결과와 실제 라벨을 비교하여 정확도를 계산합니다."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 시간 포맷팅 함수
def format_time(elapsed):
    """시간을 HH:MM:SS 형태로 포맷팅합니다."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# 학습 준비
training_stats = []
total_t0 = time.time()

# 모델을 훈련 모드로 설정
model.zero_grad()
model.train()

print("\n\n--- 🚀 BERT Fine-tuning 시작 ---")

for epoch_i in range(0, epochs):
    
    # ========================================
    #               훈련 (Training)
    # ========================================

    print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
    print('Training...')

    t0 = time.time()
    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        # 1. 배치 데이터 Device로 이동
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        # 2. 모델에 입력
        # forward() 실행 시, labels를 인자로 제공하면 loss를 계산해 반환함
        outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        loss = outputs.loss
        total_train_loss += loss.item()

        # 3. 역전파 및 가중치 업데이트
        loss.backward()
        
        # 클리핑(Clipping)을 통해 기울기가 너무 커지는 것을 방지
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 옵티마이저로 파라미터 업데이트
        optimizer.step()

        # 학습률 스케줄러 업데이트
        scheduler.step()
        
        # 기울기 초기화
        model.zero_grad()
        
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(f'  Batch {step:>5,} of {len(train_dataloader):>5,}. Loss: {loss.item():.2f}. Elapsed: {elapsed}.')

    avg_train_loss = total_train_loss / len(train_dataloader)           
    training_time = format_time(time.time() - t0)

    print(f'\n  평균 훈련 손실: {avg_train_loss:.2f}')
    print(f'  훈련 완료 시간: {training_time}')


    # ========================================
    #             검증 (Validation)
    # ========================================
    
    print('\nRunning Validation...')

    t0 = time.time()
    model.eval() # 모델을 평가 모드로 설정 (드롭아웃 등이 비활성화됨)

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in val_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        with torch.no_grad(): # 기울기 계산 비활성화 (메모리 절약)
            outputs = model(b_input_ids, 
                            token_type_ids=b_token_type_ids, 
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
        loss = outputs.loss
        logits = outputs.logits # 예측 결과

        total_eval_loss += loss.item()
        
        # 정확도 계산
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print(f'  정확도: {avg_val_accuracy:.4f}')

    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)
    
    print(f'  검증 손실: {avg_val_loss:.2f}')
    print(f'  검증 완료 시간: {validation_time}')

    # 에폭별 결과 저장
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print('\n\n--- ✅ Fine-tuning 완료 ---')
print(f'전체 학습 소요 시간: {format_time(time.time()-total_t0)}')


# --- 6. 모델 저장 ---

import os

# 모델 및 토크나이저를 저장할 디렉토리 경로 지정
output_dir = './model_save/'

# 디렉토리 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n✅ 모델 저장 디렉토리 생성: {output_dir}")

# 1. 모델 저장
print("Saving model to %s" % output_dir)
model_to_save = model.module if hasattr(model, 'module') else model  # 데이터 병렬화 처리
model_to_save.save_pretrained(output_dir)

# 2. 토크나이저 저장
tokenizer.save_pretrained(output_dir)

print("✅ 모델 및 토크나이저 저장 완료.")