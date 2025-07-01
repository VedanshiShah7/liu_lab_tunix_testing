import os
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
import optax
from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load and tokenize the SST-2 dataset
    raw = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    tokenized = raw.map(tokenize_fn, batched=True)

    # 2. Prepare JAX-friendly datasets
    train_ds = tokenized["train"].with_format("numpy")
    val_ds   = tokenized["validation"].with_format("numpy")
    # SST-2 test split contains dummy labels (-1), so drop before prediction
    test_ds  = tokenized["test"].remove_columns("label").with_format("numpy")

    # 3. Load a Flax model for sequence classification
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    # 4. Configure TUNiX training
    total_steps = len(train_ds) * 3  # epochs = 3
    config = TrainingConfig(
        max_steps=total_steps,
        eval_every_n_steps=len(train_ds),  # one evaluation per epoch
        checkpoint_root_directory="tunix_sst2_ckpts"
    )

    # 5. Define input function mapping dataset to model inputs
    def input_fn(batch):
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch.get("label")  # only present for train/val
        }

    # 6. Initialize the Trainer
    trainer = PeftTrainer(
        model=model,
        optimizer=optax.adamw(5e-5),
        config=config
    ).with_gen_model_input_fn(input_fn)

    # 7. Train and evaluate
    trainer.train(train_ds, val_ds)
    metrics = trainer.evaluate(val_ds)
    print("Validation metrics:", metrics)

    # 8. Predict on the test split
    preds = trainer.predict(test_ds).predictions.argmax(-1)
    os.makedirs(config.checkpoint_root_directory, exist_ok=True)
    out_path = os.path.join(config.checkpoint_root_directory, "sst2_test_preds.txt")
    with open(out_path, "w") as f:
        for p in preds:
            f.write(f"{p}\n")
    print(f"Test predictions saved to {out_path}")

    # 9. Extract training history and plot metrics
    history = getattr(trainer.state, 'log_history', [])
    train_logs = [h for h in history if "loss" in h and "eval_loss" not in h]
    val_logs   = [h for h in history if "eval_loss" in h]

    df_train = pd.DataFrame({
        "epoch": [h.get("epoch") for h in train_logs],
        "train_loss": [h.get("loss") for h in train_logs]
    })
    df_val = pd.DataFrame({
        "epoch": [h.get("epoch") for h in val_logs],
        "val_loss": [h.get("eval_loss") for h in val_logs],
        "val_accuracy": [h.get("eval_accuracy") for h in val_logs]
    })

    # Plot Loss Curve
    plt.figure()
    plt.plot(df_train["epoch"], df_train["train_loss"], label="Train Loss")
    plt.plot(df_val["epoch"],   df_val["val_loss"],    label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    loss_path = os.path.join(config.checkpoint_root_directory, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss curve saved to {loss_path}")

    # Plot Accuracy Curve  
    plt.figure()
    plt.plot(df_val["epoch"], df_val["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.legend()
    acc_path = os.path.join(config.checkpoint_root_directory, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Accuracy curve saved to {acc_path}")

if __name__ == "__main__":
    main()
