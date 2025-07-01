from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np

# === Config ===
MODEL_NAME = "Qwen/Qwen-1_8B"
BATCH_SIZE = 2
EPOCHS = 2
MAX_LEN = 128
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./qwen_sst2_outputs"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, trust_remote_code=True)

# === Load and preprocess dataset ===
dataset = load_dataset("glue", "sst2")

def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=MAX_LEN)

encoded = dataset.map(preprocess, batched=True)
encoded = encoded.rename_column("label", "labels")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# === Evaluation metric ===
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# === Training setup ===
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=4,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{OUTPUT_DIR}/logs",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    compute_metrics=compute_metrics,
)

# === Run training and evaluation ===
trainer.train()
eval_results = trainer.evaluate()
print("✅ Validation Results:", eval_results)
