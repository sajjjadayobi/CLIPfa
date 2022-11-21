import torch

DATA_FILE = 'dataset.csv'
TEST_SIZE = 0.05
TEXT_MODEL = 'm3hrdadfi/roberta-zwnj-wnli-mean-tokens'
IMAGE_MODEL = 'openai/clip-vit-base-patch32'
BATCH_SIZE = 256
IMAGE_SIZE = 224
MAX_LEN = 64  
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

if __name__ == '__main__':
    from transformers import TrainingArguments, AutoTokenizer, CLIPFeatureExtractor
    vision_preprocessor = CLIPFeatureExtractor.from_pretrained(IMAGE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    args = TrainingArguments(
        "clip-fa",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        learning_rate=3e-5,
        prediction_loss_only=True,
        weight_decay=0.003,
        warmup_steps=500,
        fp16=True,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=10,
        report_to='wandb'
    )
