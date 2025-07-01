import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from evaluate import load as load_metric

# Load SST-2 subset for faster training (adjust size if needed)
dataset = load_dataset("glue", "sst2")
train_data = dataset["train"].shuffle(seed=42).select(range(1000))  # 1000 train samples
val_data = dataset["validation"].select(range(200))                 # 200 val samples

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization function
def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

train_data = train_data.map(tokenize, batched=True).remove_columns(["sentence", "idx"])
val_data = val_data.map(tokenize, batched=True).remove_columns(["sentence", "idx"])
train_data.set_format("torch")
val_data.set_format("torch")

# Metric
metric = load_metric("glue", "sst2")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return metric.compute(predictions=preds, references=p.label_ids)

# Training arguments
args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,           # Train for 10 epochs
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to="none",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and evaluate
start = time.time()
trainer.train()
end = time.time()
metrics = trainer.evaluate()
metrics["training_time_sec"] = round(end - start, 2)

# Log parsing for plots
log_history = trainer.state.log_history

eval_logs = [log for log in log_history if "eval_accuracy" in log and "epoch" in log]
epochs = [log["epoch"] for log in eval_logs]
val_acc = [log["eval_accuracy"] for log in eval_logs]

train_loss_entries = [log for log in log_history if "loss" in log and "epoch" in log]
train_loss_epochs = sorted(set(log["epoch"] for log in train_loss_entries))
train_loss = []
for e in train_loss_epochs:
    losses = [log["loss"] for log in train_loss_entries if log["epoch"] == e]
    train_loss.append(sum(losses) / len(losses))

# Align lengths for plotting
min_len = min(len(epochs), len(val_acc), len(train_loss))
epochs = epochs[:min_len]
val_acc = val_acc[:min_len]
train_loss = train_loss[:min_len]

# Plot validation accuracy
plt.figure()
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.grid()
plt.legend()
plt.savefig("val_accuracy.png")

# Plot training loss
plt.figure()
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.grid()
plt.legend()
plt.savefig("train_loss.png")

# Save metrics to JSON
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Training complete!")
print(json.dumps(metrics, indent=2))
