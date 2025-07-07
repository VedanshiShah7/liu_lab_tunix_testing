#!/usr/bin/env python3
"""
run_sst2_tunix.py

Fine-tune DistilBERT on SST-2 with HuggingFace’s FlaxTrainer (JAX+TPU).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from evaluate import load as load_metric
from ptflops import get_model_complexity_info

from transformers import (
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
    FlaxTrainingArguments,
    FlaxTrainer,
    DataCollatorWithPadding,
)

def estimate_flops():
    """Estimate FLOPs via PyTorch + ptflops."""
    pt_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    def dummy_input(res):
        bs, sl = res
        return {
            "input_ids":      torch.zeros(bs, sl, dtype=torch.long),
            "attention_mask": torch.ones(bs, sl, dtype=torch.long),
        }
    macs, params = get_model_complexity_info(
        pt_model, (1,128),
        input_constructor=dummy_input,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"\nEstimated FLOPs (fwd+backward): {2*macs:,}")
    print(f"Parameter count: {params:,}\n")

def main():
    # 0) FLOPs
    estimate_flops()

    # 1) Load dataset & metric
    ds     = load_dataset("glue", "sst2")
    metric = load_metric("glue", "sst2")

    # 2) Tokenizer & model
    MODEL = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model     = FlaxAutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    # 3) Preprocess
    def tokenize_fn(ex):
        return tokenizer(
            ex["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    tokenized = ds.map(tokenize_fn, batched=True)
    train_ds = tokenized["train"]
    val_ds   = tokenized["validation"]
    test_ds  = tokenized["test"].remove_columns("label")

    # 4) Training args
    args = FlaxTrainingArguments(
        output_dir="tunix_sst2_ckpts",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # 5) Data collator & metrics
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 6) Trainer
    trainer = FlaxTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7) Train & evaluate
    trainer.train()
    val_metrics = trainer.evaluate()
    print("Validation metrics:", val_metrics)

    # 8) Test predictions
    test_out = trainer.predict(test_ds)
    preds    = np.argmax(test_out.predictions, axis=-1)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/sst2_test_preds.txt", "w") as f:
        for p in preds:
            f.write(f"{p}\n")
    print("Test predictions saved.")

    # 9) Plot history
    history   = trainer.state.log_history
    train_log = [h for h in history if "loss" in h and "eval_loss" not in h]
    val_log   = [h for h in history if "eval_loss" in h]

    df_tr = pd.DataFrame({
        "epoch":      [h["epoch"] for h in train_log],
        "train_loss": [h["loss"]  for h in train_log],
    })
    df_vl = pd.DataFrame({
        "epoch":        [h["epoch"]         for h in val_log],
        "val_loss":     [h["eval_loss"]     for h in val_log],
        "val_accuracy": [h["eval_accuracy"] for h in val_log],
    })

    plt.figure()
    plt.plot(df_tr.epoch, df_tr.train_loss, label="Train Loss")
    plt.plot(df_vl.epoch, df_vl.val_loss,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(f"{args.output_dir}/loss_curve.png"); plt.close()

    plt.figure()
    plt.plot(df_vl.epoch, df_vl.val_accuracy, label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(f"{args.output_dir}/accuracy_curve.png"); plt.close()

    print("Plots saved.")

if __name__ == "__main__":
    main()
