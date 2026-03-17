import pandas as pd
from torch.optim import AdamW
# ✅ 수정 1: Bert 전용 클래스 대신 Auto 클래스로 변경
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, SequentialSampler, RandomSampler
import numpy as np
import time
import datetime

# 1. 사용할 모델의 이름 정의
# ✅ 수정 2: 모델 이름 변경
MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"

# 2. 다국어 토크나이저 로드
# ✅ 수정 3: AutoTokenizer 사용
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"✅ 토크나이저 로드 완료: {MODEL_NAME}")

# 3. 데이터 로딩 및 라벨링 함수
def load_and_preprocess_data(file_path):
    """
    파일을 읽고, 평점을 기반으로 긍정/부정 라벨을 부여하며, 3점 리뷰를 제외합니다.
    """
    data = []
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
    tokenized_data = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt' 
    )
    return tokenized_data, df['label'].tolist()


# --- 실행 ---
file_path = 'shopping.txt' 

# 1. 데이터 로딩 및 라벨링
df_data = load_and_preprocess_data(file_path)

# 2. 토큰화 실행
tokenized_inputs, labels = tokenize_data(df_data, tokenizer, max_length=128)

# 결과 확인 (첫 번째 리뷰)
print("\n--- 토큰화 결과 (첫 번째 리뷰) ---")
print(f"원문: {df_data.iloc[0]['text']}")
print(f"라벨: {df_data.iloc[0]['label']} (1:긍정, 0:부정)")
print("Input IDs:", tokenized_inputs['input_ids'][0][:15]) 
print("Attention Mask:", tokenized_inputs['attention_mask'][0][:15])

# ✅ 수정 4: token_type_ids가 없는 모델을 위한 예외 처리 출력
if 'token_type_ids' in tokenized_inputs:
    print("Token Type IDs:", tokenized_inputs['token_type_ids'][0][:15])
else:
    print("Token Type IDs: (이 모델은 token_type_ids를 사용하지 않음)")

print(f"\n✅ 최종 준비된 데이터 개수: {len(labels)}")


# --- 3. 데이터셋 분할 및 데이터로더 생성 ---

input_ids = tokenized_inputs['input_ids']
attention_masks = tokenized_inputs['attention_mask']

# ✅ 수정 5: token_type_ids가 없다면 0으로 채워진 더미 텐서 생성
if 'token_type_ids' in tokenized_inputs:
    token_type_ids = tokenized_inputs['token_type_ids']
else:
    token_type_ids = torch.zeros_like(input_ids)

labels_tensor = torch.tensor(labels)

# 1. TensorDataset 생성
dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels_tensor)

# 2. 훈련, 검증, 테스트 셋 크기 계산
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

# 4. DataLoader 생성 
batch_size = 16 

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=batch_size
)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=batch_size
)

print(f"✅ DataLoader 생성 완료 (Batch Size: {batch_size}).")


# --- 4. BERT 모델 로드 및 설정 ---

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('✅ GPU 사용 가능: {}'.format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print('⚠️ GPU를 찾을 수 없습니다. CPU를 사용합니다.')

# ✅ 수정 6: AutoModelForSequenceClassification 사용
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels = 2,    
    output_attentions = False, 
    output_hidden_states = False, 
)

model.to(device)
print(f"✅ 모델 로드 완료: {MODEL_NAME} (num_labels=2)")

epochs = 4 
learning_rate = 2e-5 
adam_epsilon = 1e-8 
warmup_steps = 0 

optimizer = AdamW(
    model.parameters(),
    lr = learning_rate,
    eps = adam_epsilon
)

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = warmup_steps, 
    num_training_steps = total_steps
)


# --- 5. 학습 루프 (Fine-tuning) 및 평가 ---

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

training_stats = []
total_t0 = time.time()

model.zero_grad()
model.train()

print("\n\n--- 🚀 Fine-tuning 시작 ---")

for epoch_i in range(0, epochs):
    
    print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
    print('Training...')

    t0 = time.time()
    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        # 모델에 입력
        outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(f'  Batch {step:>5,} of {len(train_dataloader):>5,}. Loss: {loss.item():.2f}. Elapsed: {elapsed}.')

    avg_train_loss = total_train_loss / len(train_dataloader)           
    training_time = format_time(time.time() - t0)

    print(f'\n  평균 훈련 손실: {avg_train_loss:.2f}')
    print(f'  훈련 완료 시간: {training_time}')
    
    print('\nRunning Validation...')

    t0 = time.time()
    model.eval() 

    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        with torch.no_grad(): 
            outputs = model(b_input_ids, 
                            token_type_ids=b_token_type_ids, 
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
        loss = outputs.loss
        logits = outputs.logits 

        total_eval_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print(f'  정확도: {avg_val_accuracy:.4f}')

    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)
    
    print(f'  검증 손실: {avg_val_loss:.2f}')
    print(f'  검증 완료 시간: {validation_time}')

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

output_dir = './model_save_minilm/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n✅ 모델 저장 디렉토리 생성: {output_dir}")

print("Saving model to %s" % output_dir)
model_to_save = model.module if hasattr(model, 'module') else model  
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ 모델 및 토크나이저 저장 완료.")