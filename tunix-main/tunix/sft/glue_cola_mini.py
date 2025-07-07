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

# Load GLUE cola dataset
dataset = load_dataset("glue", "cola")
train_data = dataset["train"].shuffle(seed=42).select(range(1000))
val_data = dataset["validation"].select(range(200))

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization
def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

train_data = train_data.map(tokenize, batched=True).remove_columns(["sentence", "idx"])
val_data = val_data.map(tokenize, batched=True).remove_columns(["sentence", "idx"])
train_data.set_format("torch")
val_data.set_format("torch")

# Metric
metric = load_metric("glue", "cola")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return metric.compute(predictions=preds, references=p.label_ids)

# Training arguments
args = TrainingArguments(
    output_dir="./results_cola",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=50,
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to="none",
    learning_rate=5e-5,
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
metrics["total_training_time_sec"] = round(end - start, 2)

# Parse logs
log_history = trainer.state.log_history
log_dict = {
    "epoch": [],
    "train_loss": [],
    "eval_loss": [],
    "eval_accuracy": [],
    "learning_rate": [],
    "epoch_time": []
}

prev_time = start
for log in log_history:
    if "epoch" in log:
        log_dict["epoch"].append(log["epoch"])
        log_dict["train_loss"].append(log.get("loss", np.nan))
        log_dict["eval_loss"].append(log.get("eval_loss", np.nan))
        log_dict["eval_accuracy"].append(log.get("eval_accuracy", np.nan))
        log_dict["learning_rate"].append(log.get("learning_rate", np.nan))
        
        current_time = log.get("train_runtime", None)
        log_dict["epoch_time"].append(time.time() - prev_time)
        prev_time = time.time()

# Save all logs
with open("training_logs_cola.json", "w") as f:
    json.dump(log_dict, f, indent=2)

# Plotting utility
def plot_metric(metric_list, name, ylabel):
    plt.figure()
    plt.plot(log_dict["epoch"], metric_list, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{name} vs. Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{name.lower().replace(' ', '_')}_cola.png")

# Plot everything
plot_metric(log_dict["train_loss"], "Training Loss", "Loss")
plot_metric(log_dict["eval_loss"], "Validation Loss", "Loss")
plot_metric(log_dict["eval_accuracy"], "Validation Accuracy", "Accuracy")
plot_metric(log_dict["learning_rate"], "Learning Rate", "LR")
plot_metric(log_dict["epoch_time"], "Epoch Time", "Seconds")

# Final metrics
with open("final_metrics_cola.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ All done! Plots + logs saved.")
