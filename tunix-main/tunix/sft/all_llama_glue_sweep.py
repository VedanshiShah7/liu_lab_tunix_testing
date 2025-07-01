#!/usr/bin/env python
# all_llama_glue_sweep.py

import os
import time
import json
import argparse
from itertools import product

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    LlamaTokenizer,
    LlamaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric

# Optional: estimate FLOPs if fvcore is installed
try:
    from fvcore.nn import FlopCountAnalysis
    HAVE_FVCORE = True
except ImportError:
    HAVE_FVCORE = False


class TimeCallback(torch.nn.Module):
    def __init__(self):
        self.epoch_start = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_times.append(time.time() - self.epoch_start)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for LLaMA-2-7B on GLUE SST-2 / PubMedQA"
    )
    parser.add_argument(
        "--task", choices=["glue", "pubmedqa"], default="glue",
        help="Which task to run (default: glue/sst2)"
    )
    parser.add_argument(
        "--train_subset", type=int, default=1000,
        help="Number of training examples to use"
    )
    parser.add_argument(
        "--eval_subset", type=int, default=200,
        help="Number of evaluation examples to use"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Hugging Face model repo ID to test (e.g. meta-llama/Llama-2-7b-hf). "
             "If omitted, will compare both Llama-2-7b-hf and Llama-2-7b-chat-hf."
    )
    args = parser.parse_args()

    # 1) Data loading & split
    if args.task == "glue":
        raw = load_dataset("glue", "sst2")
        label_col = "label"
        metric = load_metric("glue", "sst2")
        text_cols = ["sentence"]
    else:
        raw = load_dataset("pubmed_qa", "pqal")
        label_col = "label"
        metric = load_metric("accuracy")
        text_cols = ["question", "passage", "text"]

    # custom 90/10 train/val split
    split = raw["train"].train_test_split(test_size=0.1, seed=42)
    train_raw = split["train"]
    val_raw = split["test"]

    # detect which column holds the text
    text_col = next(c for c in text_cols if c in train_raw.column_names)

    # 2) Model list
    DEFAULT_MODELS = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf"
    ]
    if args.model_name:
        MODEL_NAMES = [args.model_name]
    else:
        MODEL_NAMES = DEFAULT_MODELS

    # 3) Hyperparam grid
    BATCH_SIZES = [256, 512]
    LEARNING_RATES = [5e-5, 3e-5, 1e-5]
    WARMUP_RATIO = 0.1
    SCHEDULER = "linear"
    EARLY_STOP = 2

    all_results = []

    # 4) Loop over models × batch sizes × learning rates
    for model_name, bs, lr in product(MODEL_NAMES, BATCH_SIZES, LEARNING_RATES):
        run_id = f"{model_name.split('/')[-1]}_bs{bs}_lr{lr:.0e}"
        out_dir = os.path.join("results", run_id)
        os.makedirs(out_dir, exist_ok=True)

        # 4a) Load tokenizer & model (with auth from CLI/login)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = LlamaForSequenceClassification.from_pretrained(
            model_name, num_labels=2, use_auth_token=True
        )
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 4b) Optional FLOPs estimation
        if HAVE_FVCORE:
            sample = tokenizer(
                train_raw[0][text_col],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            )
            flop_an = FlopCountAnalysis(model, sample.to(model.device))
            flops_batch = flop_an.total()
        else:
            flops_batch = None

        # 4c) Tokenize once
        def tok_fn(batch):
            return tokenizer(
                batch[text_col],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        train_tok = (
            train_raw.select(range(args.train_subset))
            .map(tok_fn, batched=True)
            .remove_columns([c for c in train_raw.column_names if c != label_col])
            .with_format("torch")
        )
        val_tok = (
            val_raw.select(range(args.eval_subset))
            .map(tok_fn, batched=True)
            .remove_columns([c for c in val_raw.column_names if c != label_col])
            .with_format("torch")
        )

        collator = DataCollatorWithPadding(tokenizer)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return metric.compute(predictions=preds, references=labels)

        # 4d) TrainingArguments & Trainer
        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs // 2,
            learning_rate=lr,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type=SCHEDULER,
            save_total_limit=1,
            report_to="none",
        )

        time_cb = TimeCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP), time_cb],
        )

        # 4e) Train & time
        start_time = time.time()
        trainer.train()
        total_time = time.time() - start_time

        # 4f) Final evaluation & record metrics
        metrics = trainer.evaluate()
        metrics.update({
            "model": model_name,
            "batch_size": bs,
            "learning_rate": lr,
            "total_time_sec": total_time,
            "epoch_times_sec": time_cb.epoch_times,
            "flops_per_batch": flops_batch,
        })
        if flops_batch is not None:
            steps = len(trainer.get_train_dataloader()) * trainer.state.epoch
            metrics["estimated_total_flops"] = flops_batch * steps

        # save metrics.json
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        all_results.append(metrics)

    # 5) Print a leaderboard
    print("\n=== Leaderboard ===")
    header = ["model", "bs", "lr", "acc", "time(s)", "flops_batch", "est_flops"]
    print("\t".join(header))
    for r in sorted(all_results, key=lambda x: -x.get("eval_accuracy", 0.0)):
        print(
            f"{r['model'].split('/')[-1]}\t"
            f"{r['batch_size']}\t"
            f"{r['learning_rate']:.0e}\t"
            f"{r.get('eval_accuracy', 0):.4f}\t"
            f"{r['total_time_sec']:.1f}\t"
            f"{r['flops_per_batch']}\t"
            f"{r.get('estimated_total_flops','N/A')}"
        )


if __name__ == "__main__":
    main()
