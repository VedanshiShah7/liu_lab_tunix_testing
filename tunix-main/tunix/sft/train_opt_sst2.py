from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from evaluate import load as load_metric
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

# Load SST-2 dataset
dataset = load_dataset("glue", "sst2")

# Use public OPT model
MODEL_NAME = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Preprocess
def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

encoded = dataset.map(preprocess, batched=True)
encoded = encoded.remove_columns(["sentence", "idx"])
encoded.set_format("torch")

# Collator & metrics
collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = load_metric("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# Training args
args = TrainingArguments(
    output_dir="./results_opt_sst2",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# Run training
start = time.time()
trainer.train()
end = time.time()
total_time = end - start

# Evaluate
metrics = trainer.evaluate()
metrics["training_time_seconds"] = total_time
metrics["flops_estimate"] = sum(p.numel() for p in model.parameters()) * 2

# Plot results
log = trainer.state.log_history
epochs = [l["epoch"] for l in log if "eval_accuracy" in l]
acc = [l["eval_accuracy"] for l in log if "eval_accuracy" in l]
loss = [l["loss"] for l in log if "loss" in l]

plt.plot(epochs, acc, label="Val Accuracy", marker="o")
plt.title("opt-350m on SST-2")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("val_accuracy_opt.png")
plt.show()

plt.plot(epochs, loss, label="Train Loss", marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("train_loss_opt.png")
plt.show()

print("✅ Final Metrics:\n", metrics)
