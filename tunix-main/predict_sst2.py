import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)

def main():
    model_dir = "sst2_distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)

    raw = load_dataset("glue", "sst2")
    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], padding="max_length",
                         truncation=True, max_length=128)
    tokenized = raw.map(tokenize_fn, batched=True)

    test_ds = tokenized["test"].remove_columns("label")

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    test_out = trainer.predict(test_ds)
    preds    = np.argmax(test_out.predictions, axis=-1)

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "sst2_test_preds.txt"), "w") as f:
        for p in preds:
            f.write(f"{p}\n")

    print(f"✅ Test predictions saved to {model_dir}/sst2_test_preds.txt")

if __name__ == "__main__":
    main()
