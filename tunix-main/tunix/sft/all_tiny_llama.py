#!/usr/bin/env python
"""
all_tiny_llama.py

Fine-tune a tiny LLaMA sequence classification model on GLUE SST-2 or PubMedQA
with dynamic text column detection.
Requirements: transformers>=4.31.0, accelerate>=0.15, datasets, evaluate, torch, matplotlib
"""

import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

from datasets import load_dataset
from transformers import (
    LlamaTokenizer,
    LlamaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from evaluate import load as load_metric


def main():
    parser = argparse.ArgumentParser(
        description="Train a tiny LLaMA classification model on GLUE SST-2 or PubMedQA."
    )
    parser.add_argument(
        "--task", type=str, default="glue", choices=["glue", "pubmedqa"],
        help="Task to train on."
    )
    parser.add_argument(
        "--model_name", type=str,
        default="hf-internal-testing/tiny-random-llama",
        help="Pretrained tiny LLaMA model name or path."
    )
    parser.add_argument(
        "--train_subset", type=int, default=1000,
        help="Number of training samples to use."
    )
    parser.add_argument(
        "--eval_subset", type=int, default=200,
        help="Number of evaluation samples to use."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5,
        help="Number of training epochs."
    )
    args = parser.parse_args()

    # Load dataset
    if args.task == "glue":
        dataset = load_dataset("glue", "sst2")
        label_column = "label"
        metric_name = ("glue", "sst2")
    else:
        dataset = load_dataset("pubmed_qa", "pqal")
        label_column = "label"
        metric_name = ("accuracy", )

    # Detect text column
    cols = dataset["train"].column_names
    possible_text_cols = ["sentence", "question", "text", "passage"]
    text_column = next((c for c in possible_text_cols if c in cols), None)
    if text_column is None:
        raise ValueError(f"No suitable text column found. Available: {cols}")
    print(f"Using text column: {text_column}")

    # Tokenizer & model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    model = LlamaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # Tokenization
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column], truncation=True, padding="max_length", max_length=128
        )

    encoded = dataset.map(tokenize_fn, batched=True)
    encoded = encoded.remove_columns([c for c in cols if c not in [label_column]])
    encoded.set_format("torch")

    # Data collator & metric
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = load_metric(*metric_name)

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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"].shuffle(seed=42).select(range(args.train_subset)),
        eval_dataset=encoded["validation"].select(range(args.eval_subset)),
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Training
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    # Evaluation
    eval_metrics = trainer.evaluate()
    eval_metrics["training_time_seconds"] = total_time
    with open("metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print("✅ Final metrics:", json.dumps(eval_metrics, indent=2))

    # Plotting
    log_history = trainer.state.log_history
    # Eval per epoch
    eval_logs = [log for log in log_history if "eval_accuracy" in log and "epoch" in log]
    epochs = sorted({log["epoch"] for log in eval_logs})
    val_acc = [
        sum(l["eval_accuracy"] for l in eval_logs if l["epoch"] == e) /
        len([l for l in eval_logs if l["epoch"] == e])
        for e in epochs
    ]
    # Train loss per epoch
    train_logs = [log for log in log_history if "loss" in log and "epoch" in log]
    train_epochs = sorted({log["epoch"] for log in train_logs})
    train_loss = [
        sum(l["loss"] for l in train_logs if l["epoch"] == e) /
        len([l for l in train_logs if l["epoch"] == e])
        for e in train_epochs
    ]
    # Align
    common = [e for e in epochs if e in train_epochs]
    acc_aligned = [val_acc[epochs.index(e)] for e in common]
    loss_aligned = [train_loss[train_epochs.index(e)] for e in common]

    # Accuracy plot
    plt.figure()
    plt.plot(common, acc_aligned, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Validation Accuracy"); plt.grid(); plt.legend()
    plt.savefig("val_accuracy.png"); plt.close()

    # Loss plot
    plt.figure()
    plt.plot(common, loss_aligned, marker="o", label="Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss"); plt.grid(); plt.legend()
    plt.savefig("train_loss.png"); plt.close()


if __name__ == "__main__":
    main()
