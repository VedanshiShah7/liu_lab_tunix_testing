#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from ptflops import get_model_complexity_info

def main():
    # Use BF16 on TPU and PJRT backend
    os.environ.setdefault("XLA_USE_BF16", "1")
    os.environ.setdefault("PJRT_DEVICE", "TPU")

    # 1) Load SST-2 + metric
    raw    = load_dataset("glue", "sst2")
    metric = load_metric("glue", "sst2")

    # 2) Tokenizer & model
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=2
                 )

    #  ⚡️ Optionally compile for fused kernels (PyTorch ≥2.0)
    # model = torch.compile(model, backend="inductor")

    # 3) FLOPs & params
    def dummy_input_constructor(input_res):
        bs, seq = input_res
        return {
            "input_ids":      torch.zeros((bs, seq), dtype=torch.long),
            "attention_mask": torch.ones((bs, seq), dtype=torch.long),
        }
    macs, params = get_model_complexity_info(
        model,
        (1, 128),
        as_strings=False,
        input_constructor=dummy_input_constructor,
        print_per_layer_stat=False,
        verbose=False,
    )
    flops = 2 * macs
    print(f"\nEstimated FLOPs (fwd+backward, batch=1, len=128): {flops:,}")
    print(f"Parameter count: {params:,}\n")

    # 4) Tokenize **and drop** the raw text column so collator only sees model inputs + label
    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=["sentence"],   # ← drop raw text!
    )

    # 5) Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6) TrainingArguments: 30 epochs, per-epoch log/eval/save
    args = TrainingArguments(
        output_dir="sst2_distilbert_long",
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=30,
        weight_decay=0.01,

        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # 7) Metrics fn
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset= tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9) Train!
    trainer.train()

    # 10) Final validation
    val_metrics = trainer.evaluate()
    print(f"\nFinal Validation Results: {val_metrics}\n")

    # 11) Test-set predictions
    test_ds = tokenized["test"]
    if "label" in test_ds.column_names:
        test_ds = test_ds.remove_columns("label")
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    os.makedirs("sst2_distilbert_long", exist_ok=True)
    with open("sst2_distilbert_long/sst2_test_preds.txt", "w") as f:
        for p in preds:
            f.write(f"{p}\n")
    print("✅ Test predictions saved to sst2_distilbert_long/sst2_test_preds.txt")

    # 12) Plot curves
    history    = trainer.state.log_history
    train_logs = [h for h in history if "loss" in h and "eval_loss" not in h]
    val_logs   = [h for h in history if "eval_loss" in h]

    df_train = pd.DataFrame({
        "epoch":      [h["epoch"] for h in train_logs],
        "train_loss": [h["loss"]  for h in train_logs],
    })
    df_val = pd.DataFrame({
        "epoch":        [h["epoch"]         for h in val_logs],
        "val_loss":     [h["eval_loss"]     for h in val_logs],
        "val_accuracy": [h["eval_accuracy"] for h in val_logs],
    })

    # Loss curve
    plt.figure()
    plt.plot(df_train["epoch"], df_train["train_loss"], marker="o", label="Train Loss")
    plt.plot(df_val["epoch"],   df_val["val_loss"],   marker="o", label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend(); plt.grid(True)
    plt.savefig("sst2_distilbert_long/loss_curve.png")
    plt.close()

    # Accuracy curve (highlight best-epoch)
    best_idx = df_val["val_accuracy"].idxmax()
    best_ep  = int(df_val.loc[best_idx, "epoch"])
    best_acc = df_val.loc[best_idx, "val_accuracy"]

    plt.figure()
    plt.plot(df_val["epoch"], df_val["val_accuracy"], marker="o", label="Val Accuracy")
    plt.scatter(best_ep, best_acc, color="red", s=100, label=f"Best (Epoch {best_ep})")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.legend(); plt.grid(True)
    plt.savefig("sst2_distilbert_long/accuracy_curve.png")
    plt.close()

    print("✅ Plots saved under sst2_distilbert_long/")

if __name__ == "__main__":
    main()
