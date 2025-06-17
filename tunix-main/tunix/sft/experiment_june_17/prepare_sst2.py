from datasets import load_dataset
from transformers import AutoTokenizer

# Load SST-2
dataset = load_dataset("glue", "sst2")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

# Tokenize train and validation
encoded = dataset.map(preprocess, batched=True)
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

encoded["train"].save_to_disk("data/sst2/train")
encoded["validation"].save_to_disk("data/sst2/validation")
