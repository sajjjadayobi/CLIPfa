from transformers import TrainingArguments, AutoTokenizer
import torch
from utils import optimal_workers

DATA_FILE = 'data.csv'
TEST_SIZE = 0.05
TEXT_MODEL = 'm3hrdadfi/roberta-zwnj-wnli-mean-tokens'
TOKENIZER = AutoTokenizer.from_pretrained(TEXT_MODEL)
IMAGE_MODEL = 'openai/clip-vit-base-patch32'
BATCH_SIZE = 32
IMAGE_SIZE = 224
MAX_LEN = 120
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


args = TrainingArguments(
    "clip-fa",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.001,
    warmup_steps=100,
    fp16=True,
    prediction_loss_only=True,
    dataloader_num_workers=optimal_workers(),
    gradient_accumulation_steps=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    report_to='tensorboard'
)
