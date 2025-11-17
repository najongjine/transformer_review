import torch
import torch.nn as nn
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import math
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------
# 1. 모델 및 토크나이저 설정 (Hugging Face 표준 - WordPiece/SentencePiece 기술)
# ----------------------------------------------------
# Hugging Face 표준 Multilingual BERT 토크나이저를 "tiktokken"으로 사용합니다.
MODEL_NAME = "bert-base-multilingual-cased"
NUM_LABELS = 2
EMBEDDING_DIM = 768
N_HEAD = 12
N_LAYERS = 2
MAX_SEQ_LENGTH = 128

print(f"✅ 토크나이저 로드 (Hugging Face 표준 'tiktokken' 기술): {MODEL_NAME}")
# Multilingual BERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ----------------------------------------------------
# 2. 커스텀 트랜스포머 인코더 분류기 정의 (새 모델)
# ----------------------------------------------------

# 트랜스포머 모델에 필수적인 위치 인코딩 클래스
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class CustomTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, num_labels, max_len):
        super().__init__()

        self.d_model = d_model

        # 1. 임베딩 레이어 (토크나이저의 어휘 크기 사용)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=tokenizer.pad_token_id)

        # 2. 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # 3. 트랜스포머 인코더 스택 (순수 PyTorch Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_model * 4, dropout=0.1, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # 4. 분류 헤드
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # 1. 입력 형태 변환: (batch_size, seq_len) -> (seq_len, batch_size)
        input_ids = input_ids.transpose(0, 1)

        # 2. 임베딩 및 위치 인코딩 추가
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)

        # 3. 마스크 생성: attention_mask 0 위치에 True (패딩 무시)
        src_key_padding_mask = (attention_mask == 0)

        # 4. 트랜스포머 인코더 실행
        output = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )

        # 5. 분류: 첫 번째 토큰([CLS])의 출력을 문장 특징으로 사용
        cls_output = output[0]

        # 6. 최종 로짓 생성
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)

# 커스텀 모델 초기화
model = CustomTransformerClassifier(
    vocab_size=tokenizer.vocab_size,
    d_model=EMBEDDING_DIM,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    num_labels=NUM_LABELS,
    max_len=MAX_SEQ_LENGTH
)

print("✅ 커스텀 트랜스포머 인코더 모델 생성 완료.")

# ----------------------------------------------------
# 3. shopping.txt 데이터셋 생성 및 정제
# ----------------------------------------------------
texts = []
labels = []
file_path = "shopping.txt"

print(f"\n📂 파일 로드 및 정제 시작: {file_path}")

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            parts = line.split('\t', 1)

            if len(parts) == 2:
                try:
                    rating = int(parts[0])
                    text = parts[1]

                    if rating in [1, 2]:
                        label = 0 # 부정
                    elif rating in [4, 5]:
                        label = 1 # 긍정
                    else:
                        continue # 평점 3점 제외

                    texts.append(text)
                    labels.append(label)

                except ValueError:
                    continue
except FileNotFoundError:
    print(f"❌ 오류: 파일 '{file_path}'를 찾을 수 없습니다.")
    exit()

if not texts:
    print("❌ 오류: 유효한 데이터를 로드하지 못했습니다. 파일 내용 및 형식을 확인해주세요.")
    exit()

raw_dataset = Dataset.from_dict({'text': texts, 'label': labels})
train_test_split = raw_dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"로드된 전체 샘플 크기: {len(raw_dataset)} (평점 3점 제외)")
print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"평가 데이터셋 크기: {len(eval_dataset)}")


# ----------------------------------------------------
# 4. 데이터 전처리 (표준 토큰화)
# ----------------------------------------------------
def tokenize_function(examples):
    # 'tiktokken' 토크나이저 기술을 사용하여 정수 ID로 변환합니다.
    tokenized_inputs = tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])


# ----------------------------------------------------
# 5. 훈련 설정 및 Trainer 실행
# ----------------------------------------------------
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./custom_standard_transformer_results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

# Data Collator: 토크나이저에 맞춤
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("\n🚀 커스텀 트랜스포머 인코더 분류기 훈련 시작 (Hugging Face 표준 토크나이저 기반)...")
trainer.train()

print("\n✅ 훈련 완료! 이제 표준 기반 커스텀 모델이 작동할 것입니다.")