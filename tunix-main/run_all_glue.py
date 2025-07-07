#!/usr/bin/env python3
import os
import numpy as np

# Try TPU; fallback to CPU
try:
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()
    print(f"Using TPU device: {DEVICE}")
except ImportError:
    import torch
    DEVICE = torch.device("cpu")
    print(f"torch_xla not found; using CPU device: {DEVICE}")

from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# List of GLUE tasks
TASKS = [
    "cola",
    "sst2",
    "mrpc",
    "sts-b",
    "qqp",
    "mnli",
    "qnli",
    "rte",
    "wnli",
    "ax"
]

# Define best-metric mapping per task
BEST_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "sts-b": "pearson",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "wnli": "accuracy",
    "ax": "accuracy",
}

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_ROOT = "glue_all"


def fine_tune_and_predict(task):
    raw = load_dataset("glue", task)
    metric = load_metric("glue", task)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_labels = 1 if task == "sts-b" else (3 if task == "mnli" else 2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # Move model to DEVICE
    model.to(DEVICE)

    splits = []
    if task == "mnli":
        splits = [("matched", raw["test_matched"]), ("mismatched", raw["test_mismatched"])]
    else:
        splits = [(task, raw["test"])]

    def tokenize_fn(batch):
        # Handle different text fields
        if task == "mnli":
            texts = batch["premise"]
            texts2 = batch["hypothesis"]
            return tokenizer(texts, texts2, padding="max_length", truncation=True, max_length=128)
        else:
            key = "sentence" if task not in ["qqp"] else None
            if task == "qqp":
                return tokenizer(batch["question1"], batch["question2"],
                                 padding="max_length", truncation=True, max_length=128)
            return tokenizer(batch[key], padding="max_length", truncation=True, max_length=128)

    tokenized = raw.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    out_dir = os.path.join(OUTPUT_ROOT, task)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC[task],
        no_cuda=(DEVICE.type == "cpu"),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task == "sts-b":
            return metric.compute(predictions=logits.flatten(), references=labels)
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train & evaluate
    trainer.train()

    # Predict for each split
    for name, ds in splits:
        ds = ds.remove_columns("label") if "label" in ds.column_names else ds
        preds = trainer.predict(ds).predictions
        if task != "sts-b":
            preds = np.argmax(preds, axis=-1)
        else:
            preds = preds.flatten().tolist()

        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, f"{task}{'' if name == task else f'-{name}'}_test_preds.txt")
        with open(fpath, "w") as f:
            for p in preds:
                f.write(f"{p}\n")
        print(f"Wrote {fpath}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_USE_BF16", "1")
    for t in TASKS:
        fine_tune_and_predict(t)
