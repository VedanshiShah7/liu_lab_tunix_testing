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
    # Tell XLA/Accelerate to use the TPU
    os.environ.setdefault("XLA_USE_BF16", "1")      # use BF16 if available
    os.environ.setdefault("PJRT_DEVICE", "TPU")     # PJRT backend

    # --- 1. Load SST-2 + metric
    raw    = load_dataset("glue", "sst2")
    metric = load_metric("glue", "sst2")

    # --- 2. Tokenizer & model
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # --- 3. FLOPs & param count
    def dummy_input_constructor(input_res):
        bs, seq = input_res
        return {
            "input_ids":      torch.zeros((bs, seq), dtype=torch.long),
            "attention_mask": torch.ones((bs, seq), dtype=torch.long),
        }
    macs, params = get_model_complexity_info(
        model, (1,128), as_strings=False,
        input_constructor=dummy_input_constructor,
        print_per_layer_stat=False, verbose=False
    )
    flops = 2 * macs
    print(f"\nEstimated FLOPs (fwd+backward, batch=1, len=128): {flops:,}")
    print(f"Parameter count: {params:,}\n")

    # --- 4. Tokenize
    def tokenize_fn(batch):
        return tokenizer(batch["sentence"],
                         padding="max_length",
                         truncation=True,
                         max_length=128)
    tokenized = raw.map(tokenize_fn, batched=True)

    # --- 5. Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 6. Training args
    output_dir = "sst2_distilbert"
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # --- 7. Metric fn
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # --- 8. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset= tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 9. Train
    trainer.train()

    # --- 10. Eval on validation
    val_metrics = trainer.evaluate()
    print(f"\nValidation results: {val_metrics}\n")

    # --- 11. Predict on test (drop dummy labels)
    test_ds = tokenized["test"]
    if "label" in test_ds.column_names:
        test_ds = test_ds.remove_columns("label")
    test_out = trainer.predict(test_ds)
    preds    = np.argmax(test_out.predictions, axis=-1)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sst2_test_preds.txt"), "w") as f:
        for p in preds:
            f.write(f"{p}\n")
    print(f"✅ Test predictions saved to {output_dir}/sst2_test_preds.txt")

    # --- 12. Plot curves
    history    = trainer.state.log_history
    train_logs = [h for h in history if "loss" in h and "epoch" in h and "eval_loss" not in h]
    val_logs   = [h for h in history if "eval_loss" in h]

    df_train = pd.DataFrame({
        "epoch":      [h["epoch"] for h in train_logs],
        "train_loss": [h["loss"]  for h in train_logs],
    })
    df_val = pd.DataFrame({
        "epoch":        [h["epoch"]        for h in val_logs],
        "val_loss":     [h["eval_loss"]     for h in val_logs],
        "val_accuracy": [h["eval_accuracy"] for h in val_logs],
    })

    plt.figure(); plt.plot(df_train["epoch"], df_train["train_loss"], label="train loss")
    plt.plot(df_val["epoch"],   df_val["val_loss"],   label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curves"); plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png")); plt.close()

    plt.figure(); plt.plot(df_val["epoch"], df_val["val_accuracy"], label="val accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Validation Accuracy"); plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png")); plt.close()

    print(f"\n✅ Plots saved under {output_dir}/")
