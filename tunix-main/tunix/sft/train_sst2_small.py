import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from evaluate import load as load_metric

# Step 1: Load SST-2 dataset
dataset = load_dataset("glue", "sst2")

# Step 2: Use a small model for local testing
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Step 3: Tokenize dataset
def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded = dataset.map(tokenize, batched=True)
encoded = encoded.remove_columns(["sentence", "idx"])
encoded.set_format("torch")

# Step 4: Data Collator
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 5: Metric
metric = load_metric("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# Step 6: TrainingArguments (CPU-friendly)
args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=2,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Step 7: Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"].shuffle(seed=42).select(range(1000)),  # smaller subset
    eval_dataset=encoded["validation"].select(range(200)),
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# Step 8: Training
start = time.time()
trainer.train()
end = time.time()
eval_metrics = trainer.evaluate()
eval_metrics["training_time_seconds"] = end - start

# Step 9: Plot
log_history = trainer.state.log_history
epochs = [log["epoch"] for log in log_history if "eval_accuracy" in log]
val_acc = [log["eval_accuracy"] for log in log_history if "eval_accuracy" in log]
train_loss = [log["loss"] for log in log_history if "loss" in log]

plt.figure()
plt.plot(epochs, val_acc, marker="o", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Val Accuracy")
plt.grid()
plt.legend()
plt.savefig("val_accuracy.png")

plt.figure()
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss")
plt.grid()
plt.legend()
plt.savefig("train_loss.png")

# Step 10: Save metrics
with open("metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)

print("✅ Done. Final metrics:")
print(json.dumps(eval_metrics, indent=2))
