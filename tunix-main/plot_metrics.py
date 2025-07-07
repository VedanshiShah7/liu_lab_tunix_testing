#!/usr/bin/env python3
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# 1) Paths
output_dir = "sst2_distilbert"
state_path = os.path.join(output_dir, "trainer_state.json")
loss_png   = os.path.join(output_dir, "loss_curve.png")
acc_png    = os.path.join(output_dir, "accuracy_curve.png")

# 2) Remove old plots if they exist
for f in (loss_png, acc_png):
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

# 3) Load training history
with open(state_path, "r") as f:
    state = json.load(f)
history = state["log_history"]

# 4) Split into train vs val logs
train_logs = [
    h for h in history
    if "loss" in h and "eval_loss" not in h and "epoch" in h
]
val_logs = [
    h for h in history
    if "eval_loss" in h and "epoch" in h
]

df_train = pd.DataFrame({
    "epoch":      [h["epoch"] for h in train_logs],
    "train_loss": [h["loss"]  for h in train_logs],
})
df_val = pd.DataFrame({
    "epoch":        [h["epoch"]          for h in val_logs],
    "val_loss":     [h["eval_loss"]      for h in val_logs],
    "val_accuracy": [h["eval_accuracy"]  for h in val_logs],
})

# 5) Plot Loss Curves with markers
plt.figure()
plt.plot(df_train["epoch"], df_train["train_loss"], marker="o", label="Train Loss")
plt.plot(df_val["epoch"],   df_val["val_loss"],   marker="s", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_png)
plt.close()

# 6) Plot Val Accuracy with markers + highlight best epoch
best = df_val.loc[df_val["val_accuracy"].idxmax()]
plt.figure()
plt.plot(df_val["epoch"], df_val["val_accuracy"], marker="o", label="Val Accuracy")
plt.scatter([best["epoch"]], [best["val_accuracy"]], 
            s=100, facecolors="none", edgecolors="r", linewidths=2,
            label=f"Best: Epoch {int(best['epoch'])} ({best['val_accuracy']:.3f})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(acc_png)
plt.close()

print(f"✅ Plots updated in `{output_dir}`:")
print(f"   • Loss curve → {loss_png}")
print(f"   • Accuracy curve → {acc_png}")
