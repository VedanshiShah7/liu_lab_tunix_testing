from datasets import load_from_disk
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
import torch

# Load eval data
eval_dataset = load_from_disk("data/sst2/validation")

# Load tokenizer and model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("checkpoints/llama_sst2_final")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Evaluation loop
model.eval()
preds, labels = [], []

for batch in torch.utils.data.DataLoader(eval_dataset, batch_size=16):
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
    logits = outputs.logits
    preds.extend(torch.argmax(logits, dim=1).tolist())
    labels.extend(batch["label"].tolist())

print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
