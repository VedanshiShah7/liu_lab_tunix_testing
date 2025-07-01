# Qwen3 SFT on SST-2 using TUNiX internal structure (Updated for dynamic HF model config + evaluation)

import os
import sys
import tqdm
import json
import jax
import jax.numpy as jnp
from flax import nnx
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.flax import save_file

# Add repo root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tunix.models.qwen3.model import Qwen3, ModelConfig
from tunix.models.qwen3.params import create_model_from_safe_tensors
from tunix.sft.metrics_logger import log_metrics
from tunix.sft.progress_bar import ProgressBar
from tunix.sft.checkpoint_manager import save_checkpoint

# === Config ===
CHECKPOINT_DIR = "./"  # Directory where qwen1_8B_tunix.safetensors is stored
SAVE_PATH = "./outputs/qwen3-sft-sst2.safetensors"
RESULTS_PATH = "./outputs/qwen3-sft-sst2-results.json"
MODEL_NAME = "Qwen/Qwen-1_8B"
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LEN = 128

# === Load Hugging Face model config dynamically ===
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
hf_config = hf_model.config

model_config = ModelConfig(
    num_layers=hf_config.num_hidden_layers,
    vocab_size=hf_config.vocab_size,
    embed_dim=hf_config.hidden_size,
    hidden_dim=hf_config.intermediate_size,
    num_heads=hf_config.num_attention_heads,
    head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
    num_kv_heads=4,
    norm_eps=1e-6,
    rope_theta=1_000_000
)

# === Load model ===
model = create_model_from_safe_tensors(
    file_dir=CHECKPOINT_DIR,
    config=model_config,
)

# === Load & Tokenize Dataset ===
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def preprocess(example):
    tok = tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=MAX_LEN)
    tok["labels"] = example["label"]
    return tok

dataset = dataset.map(preprocess, batched=True)

# Convert to JAX arrays
def to_jax(data):
    return {
        "input_ids": jnp.array(data["input_ids"]),
        "attention_mask": jnp.array(data["attention_mask"]),
        "labels": jnp.array(data["labels"]),
    }

train_data = to_jax(dataset["train"])
val_data = to_jax(dataset["validation"])

# === Optimizer & Training Setup ===
import optax
from flax.training import train_state

class TrainState(train_state.TrainState):
    batch_stats: dict

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def compute_metrics(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return {"accuracy": jnp.mean(preds == labels)}

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits, _ = state.apply_fn({"params": params}, batch["input_ids"], batch["input_ids"], None, batch["attention_mask"])
        loss = cross_entropy_loss(logits, batch["labels"])
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)()
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch["labels"])
    return state, metrics

# === Initialize State ===
scheduler = optax.linear_schedule(init_value=LEARNING_RATE, end_value=0.0, transition_steps=len(train_data["input_ids"]) // BATCH_SIZE * EPOCHS)
optimizer = optax.adamw(scheduler)

rng = jax.random.PRNGKey(0)
state = TrainState.create(apply_fn=model.__call__, params=model.variables["params"], tx=optimizer)

# === Training Loop ===
progress = ProgressBar(total=len(train_data["input_ids"]) * EPOCHS // BATCH_SIZE)
for epoch in range(EPOCHS):
    for i in range(0, len(train_data["input_ids"]), BATCH_SIZE):
        batch = {k: v[i:i+BATCH_SIZE] for k, v in train_data.items()}
        state, metrics = train_step(state, batch)
        log_metrics(epoch=epoch, step=i, metrics=metrics)
        progress.update(1)

# === Evaluation ===
@jax.jit
def eval_step(state, batch):
    logits, _ = state.apply_fn({"params": state.params}, batch["input_ids"], batch["input_ids"], None, batch["attention_mask"])
    return compute_metrics(logits, batch["labels"])

print("\n🧪 Evaluating on validation set...")
eval_metrics = {"accuracy": 0.0}
num_batches = 0
for i in range(0, len(val_data["input_ids"]), BATCH_SIZE):
    batch = {k: v[i:i+BATCH_SIZE] for k, v in val_data.items()}
    metrics = eval_step(state, batch)
    eval_metrics["accuracy"] += metrics["accuracy"]
    num_batches += 1

eval_metrics["accuracy"] /= num_batches
print("✅ SST-2 Validation Accuracy:", float(eval_metrics["accuracy"]))

with open(RESULTS_PATH, "w") as f:
    json.dump(eval_metrics, f, indent=2)

# === Save Final Model ===
print("Saving model to:", SAVE_PATH)
save_checkpoint(SAVE_PATH, state.params)
print("✅ Training complete. Results saved.")
