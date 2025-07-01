#!/usr/bin/env python
"""
all_fixed.py

Fine-tune a sequence classification model on GLUE SST-2 or PubMedQA,
with dynamic text column detection to avoid KeyError.
Requirements: transformers>=4.8.0, accelerate>=0.15, datasets, evaluate, torch, matplotlib
"""

import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from evaluate import load as load_metric

def main():
    parser = argparse.ArgumentParser(description="Train a classification model on GLUE SST-2 or PubMedQA.")
    parser.add_argument("--task", type=str, default="glue", choices=["glue", "pubmedqa"], help="Task to train on.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name.")
    parser.add_argument("--train_subset", type=int, default=1000, help="Number of training samples to use.")
    parser.add_argument("--eval_subset", type=int, default=200, help="Number of evaluation samples to use.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    # Load dataset
    if args.task == "glue":
        dataset = load_dataset("glue", "sst2")
        label_column = "label"
    else:
        dataset = load_dataset("pubmed_qa", "pqal")
        label_column = "label"

    # Inspect columns and pick text column
    cols = dataset["train"].column_names
    print(f"Columns in train split: {cols}")
    possible_text_cols = ["sentence", "question", "text", "passage"]
    text_column = next((c for c in possible_text_cols if c in cols), None)
    if text_column is None:
        raise ValueError(f"No suitable text column found. Available columns: {cols}")
    print(f"Using text column: {text_column}")

    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )

    # Tokenize function
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    # Apply tokenization
    encoded = dataset.map(tokenize_fn, batched=True)
    # Remove all original columns except the label
    encoded = encoded.remove_columns([c for c in cols if c not in [label_column]])
    encoded.set_format("torch")

    # Data collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Metric
    metric = load_metric("glue", "sst2") if args.task == "glue" else load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs=args.num_epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"].shuffle(seed=42).select(range(args.train_subset)),
        eval_dataset=encoded["validation"].select(range(args.eval_subset)),
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Train
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    # Evaluate
    eval_metrics = trainer.evaluate()
    eval_metrics["training_time_seconds"] = total_time
    with open("metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print("✅ Final metrics:", json.dumps(eval_metrics, indent=2))

    # Plot training curves
    log_history = trainer.state.log_history

    # Extract eval accuracy per epoch (average if multiple logs per epoch)
    eval_logs = [log for log in log_history if "eval_accuracy" in log and "epoch" in log]
    eval_epochs = sorted(set(log["epoch"] for log in eval_logs))
    val_acc = []
    for e in eval_epochs:
        vals = [log["eval_accuracy"] for log in eval_logs if log["epoch"] == e]
        val_acc.append(sum(vals) / len(vals))

    # Extract train loss per epoch (average if multiple logs per epoch)
    train_logs = [log for log in log_history if "loss" in log and "epoch" in log]
    train_epochs = sorted(set(log["epoch"] for log in train_logs))
    train_loss = []
    for e in train_epochs:
        vals = [log["loss"] for log in train_logs if log["epoch"] == e]
        train_loss.append(sum(vals) / len(vals))

    # Use the intersection of epochs to plot aligned data
    common_epochs = sorted(set(eval_epochs) & set(train_epochs))

    val_acc_aligned = [val_acc[eval_epochs.index(e)] for e in common_epochs]
    train_loss_aligned = [train_loss[train_epochs.index(e)] for e in common_epochs]

    # Debug prints to verify alignment
    print("Epochs:", common_epochs)
    print("Validation Accuracy:", val_acc_aligned)
    print("Training Loss:", train_loss_aligned)

    # Plot Validation Accuracy
    plt.figure()
    plt.plot(common_epochs, val_acc_aligned, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid()
    plt.legend()
    plt.savefig("val_accuracy.png")
    plt.close()

    # Plot Training Loss
    plt.figure()
    plt.plot(common_epochs, train_loss_aligned, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.legend()
    plt.savefig("train_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
