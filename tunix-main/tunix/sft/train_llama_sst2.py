# train_llama_sst2.py

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

# ---- Step 1: Load SST-2 dataset ----
dataset = load_dataset("glue", "sst2")

# ---- Step 2: Load LLaMA tokenizer and model ----
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# If the tokenizer has no pad_token, assign it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Added pad token")

# Load model for classification with 2 labels (positive/negative sentiment)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# ---- Step 3: Tokenize the SST-2 sentences ----
def tokenize(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

encoded = dataset.map(tokenize, batched=True)
encoded = encoded.remove_columns(["sentence", "idx"])
encoded.set_format("torch")

# ---- Step 4: Data Collator ----
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---- Step 5: Metric (Accuracy from GLUE) ----
metric = load_metric("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# ---- Step 6: TrainingArguments ----
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=1,   # LLaMA requires batch size 1 unless fully padded
    per_device_eval_batch_size=1,
    learning_rate=3e-5,
    num_train_epochs=4,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# ---- Step 7: Trainer ----
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ---- Step 8: Train the model ----
start = time.time()
trainer.train()
end = time.time()
total_time = end - start

# ---- Step 9: Final Evaluation ----
eval_metrics = trainer.evaluate()
eval_metrics["training_time_seconds"] = total_time

# ---- Step 10: Log parsing for plot ----
log_history = trainer.state.log_history
epochs = [log["epoch"] for log in log_history if "eval_accuracy" in log]
val_acc = [log["eval_accuracy"] for log in log_history if "eval_accuracy" in log]
train_loss = [log["loss"] for log in log_history if "loss" in log]

# ---- Step 11: Plot Validation Accuracy ----
plt.figure()
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.grid()
plt.legend()
plt.savefig("val_accuracy.png")

# ---- Step 12: Plot Training Loss ----
plt.figure()
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.grid()
plt.legend()
plt.savefig("train_loss.png")

# ---- Step 13: Estimate FLOPs ----
flops_per_forward = sum(p.numel() for p in model.parameters())
eval_metrics["flops_per_batch"] = flops_per_forward * 2  # forward + backward

# ---- Step 14: Save metrics ----
with open("metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)

# ---- Step 15: Print result ----
print("✅ Final Evaluation Metrics:")
print(json.dumps(eval_metrics, indent=2))
