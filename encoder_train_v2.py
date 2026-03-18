import pandas as pd
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, SequentialSampler, RandomSampler
import numpy as np
import time
import datetime
import os

# 1. 사용할 모델의 이름 정의
MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"

# 2. 다국어 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"✅ 토크나이저 로드 완료: {MODEL_NAME}")

# 3. 데이터 로딩 및 라벨링 함수
def load_and_preprocess_data(file_path):
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
                    # 부정(0): 1~2점, 중립(1): 3점, 긍정(2): 4~5점
                    if score <= 2:
                        label = 0
                    elif score == 3:
                        label = 1
                    else:
                        label = 2
                    data.append([review, label])
                except ValueError:
                    continue

    df = pd.DataFrame(data, columns=['text', 'label'])
    print(f"✅ 총 {len(df)}개의 리뷰 데이터 준비 완료 (3점 포함).") 
    return df

# 4. 토큰화 실행 함수
def tokenize_data(df, tokenizer, max_length=128):
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
df_data = load_and_preprocess_data(file_path)
tokenized_inputs, labels = tokenize_data(df_data, tokenizer, max_length=128)

print(f"\n✅ 최종 준비된 데이터 개수: {len(labels)}")

# --- 3. 데이터셋 분할 및 데이터로더 생성 ---
input_ids = tokenized_inputs['input_ids']
attention_masks = tokenized_inputs['attention_mask']

if 'token_type_ids' in tokenized_inputs:
    token_type_ids = tokenized_inputs['token_type_ids']
else:
    token_type_ids = torch.zeros_like(input_ids)

labels_tensor = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

batch_size = 16 
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

print(f"✅ DataLoader 생성 완료 (Batch Size: {batch_size}).")

# --- 4. BERT 모델 로드 및 설정 ---
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('✅ GPU 사용 가능: {}'.format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print('⚠️ GPU를 찾을 수 없습니다. CPU를 사용합니다.')

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels = 3,  
    output_attentions = False, 
    output_hidden_states = False, 
)
model.to(device)

# 🚨 [수정사항] Early Stopping을 위해 최대 에포크를 넉넉하게 10으로 늘림
epochs = 10 
learning_rate = 2e-5 
adam_epsilon = 1e-8 
warmup_steps = 0 

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# --- 모델 저장 디렉토리 미리 생성 ---
output_dir = './model_save_minilm/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# 🚨 [핵심 추가] Early Stopping 관련 변수 초기화
patience = 2 # 검증 손실이 개선되지 않아도 참아줄 에포크 횟수 (2번 참음)
patience_counter = 0
best_val_loss = float('inf') # 가장 낮은 검증 손실을 기록하기 위해 무한대로 초기화

model.zero_grad()
print("\n\n--- 🚀 Fine-tuning 시작 (Early Stopping 적용) ---")

for epoch_i in range(0, epochs):
    print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].to(device)
        
        outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels)
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
    print(f'\n  평균 훈련 손실: {avg_train_loss:.2f}')
    
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
            outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels)
            
        loss = outputs.loss
        logits = outputs.logits 

        total_eval_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_loss = total_eval_loss / len(val_dataloader)
    
    print(f'  검증 정확도: {avg_val_accuracy:.4f}')
    print(f'  검증 손실: {avg_val_loss:.4f}')

    # 🚨 [핵심 추가] Early Stopping 로직 및 베스트 모델 저장
    if avg_val_loss < best_val_loss:
        print(f"  ✨ 검증 손실이 감소했습니다! ({best_val_loss:.4f} -> {avg_val_loss:.4f})")
        print("  💾 현재 시점의 최고 성능 모델을 저장합니다.")
        best_val_loss = avg_val_loss
        patience_counter = 0
        
        # 성능이 가장 좋을 때 덮어쓰기 방식으로 저장
        model_to_save = model.module if hasattr(model, 'module') else model  
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        patience_counter += 1
        print(f"  ⚠️ 검증 손실이 감소하지 않았습니다. (Patience: {patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print("\n🛑 Early Stopping 발동! 더 이상 성능이 개선되지 않아 학습을 조기 종료합니다.")
            break

print('\n\n--- ✅ Fine-tuning 종료 ---')
print(f'전체 학습 소요 시간: {format_time(time.time()-total_t0)}')
print(f"✅ 최고 성능의 모델이 '{output_dir}' 폴더에 안전하게 보관되어 있습니다.")