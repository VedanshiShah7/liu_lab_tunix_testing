import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tunix import data, models, trainers, config_utils, evaluation

GLUE_TASKS = [
    "cola", "sst2", "mrpc", "sts-b", "qqp", "mnli", "qnli", "rte", "wnli"
]

MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 3
TRAIN_SUBSET = 1000
EVAL_SUBSET = 200

OUTPUT_DIR = "sft"
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_records = []

for task in GLUE_TASKS:
    print(f"\n--- TUNiX fine-tuning for GLUE task: {task} ---")

    dataset = data.load_glue_dataset(task, train_subset=TRAIN_SUBSET, eval_subset=EVAL_SUBSET)
    model_cfg = config_utils.ModelConfig.from_pretrained(MODEL_NAME)
    model = models.build_model(model_cfg, num_labels=dataset.num_labels)

    train_cfg = config_utils.TrainConfig(
        num_epochs=NUM_EPOCHS,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy" if task != "sts-b" else "pearson",
    )

    trainer = trainers.TunixTrainer(
        model=model,
        dataset=dataset,
        train_config=train_cfg,
        tokenizer=model_cfg.tokenizer,
        compute_metrics=evaluation.glue_compute_metrics(task),
    )

    start = time.time()
    trainer.train()
    duration = time.time() - start

    metrics = trainer.evaluate()
    metrics["training_time_sec"] = duration

    # Save metrics JSON
    metrics_path = os.path.join(OUTPUT_DIR, f"{task}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions in GLUE submission format (usually TSV)
    preds = trainer.predict(dataset["validation"])
    preds_df = evaluation.glue_format_predictions(preds.predictions, dataset["validation"], task)
    preds_path = os.path.join(OUTPUT_DIR, f"{task}_predictions.tsv")
    preds_df.to_csv(preds_path, sep="\t", index=False)

    # Extract logs for plotting
    logs = trainer.state.log_history

    eval_logs = [log for log in logs if "eval_accuracy" in log and "epoch" in log]
    eval_epochs = sorted(set(log["epoch"] for log in eval_logs))
    val_acc = [np.mean([log["eval_accuracy"] for log in eval_logs if log["epoch"] == e]) for e in eval_epochs]

    train_logs = [log for log in logs if "loss" in log and "epoch" in log]
    train_epochs = sorted(set(log["epoch"] for log in train_logs))
    train_loss = [np.mean([log["loss"] for log in train_logs if log["epoch"] == e]) for e in train_epochs]

    common_epochs = sorted(set(eval_epochs) & set(train_epochs))
    val_acc_aligned = [val_acc[eval_epochs.index(e)] for e in common_epochs]
    train_loss_aligned = [train_loss[train_epochs.index(e)] for e in common_epochs]

    # Plot validation accuracy
    plt.figure()
    plt.plot(common_epochs, val_acc_aligned, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{task.upper()} Validation Accuracy")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{task}_val_accuracy.png"))
    plt.close()

    # Plot training loss
    plt.figure()
    plt.plot(common_epochs, train_loss_aligned, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{task.upper()} Training Loss")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{task}_train_loss.png"))
    plt.close()

    # Append results to summary records
    summary_records.append({
        "task": task,
        "accuracy": metrics.get("eval_accuracy", None),
        "pearson": metrics.get("eval_pearson", None),
        "training_time_sec": duration,
        "predictions_file": preds_path,
        "metrics_file": metrics_path,
        "val_acc_plot": os.path.join(OUTPUT_DIR, f"{task}_val_accuracy.png"),
        "train_loss_plot": os.path.join(OUTPUT_DIR, f"{task}_train_loss.png"),
    })

# Create summary DataFrame and save as markdown for easy viewing
summary_df = pd.DataFrame(summary_records)
summary_md_path = os.path.join(OUTPUT_DIR, "summary.md")

with open(summary_md_path, "w") as f:
    f.write(summary_df.to_markdown(index=False))

print(f"\nAll GLUE tasks complete. Summary saved to {summary_md_path}")
