# Qwen3 Supervised Fine-Tuning on SST-2 using TUNiX (JAX/Flax)

import os
import tqdm
import jax
import jax.numpy as jnp
from flax import nnx
from datasets import load_dataset
from transformers import AutoTokenizer
from safetensors.flax import save_file
from tunix.models.qwen3 import model as qwen_model

# === Step 1: Setup Model ===
model_config = qwen_model.ModelConfig.qwen3_1_7_b()  # or qwen3_30_b()

model = qwen_model.create_model_from_safe_tensors(
    file_dir="/path/to/safetensors",  # Change this
    config=model_config,
)

# === Step 2: Prepare SST-2 Dataset ===
print("Loading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True)

MAX_LEN = 128

print("Tokenizing...")
def preprocess(example):
    tokenized = tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    tokenized["labels"] = example["label"]
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True)
train_data = tokenized_dataset["train"]
val_data = tokenized_dataset["validation"]

# Convert to numpy arrays
def to_numpy(data):
    return {
        "input_ids": jnp.array(data["input_ids"]),
        "attention_mask": jnp.array(data["attention_mask"]),
        "labels": jnp.array(data["labels"]),
    }

train_np = to_numpy(train_data)
val_np = to_numpy(val_data)

# === Step 3: Training Setup ===
import optax
from flax.training import train_state

class TrainState(train_state.TrainState):
    batch_stats: dict

lr = 2e-5
batch_size = 4
epochs = 2

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def compute_metrics(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return {"accuracy": accuracy}

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits, _ = state.apply_fn({"params": params}, batch["input_ids"], batch["input_ids"], None, batch["attention_mask"])
        loss = cross_entropy_loss(logits, batch["labels"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch["labels"])
    return state, metrics

# === Step 4: Initialize Training ===
print("Initializing optimizer...")
scheduler = optax.linear_schedule(init_value=lr, end_value=0, transition_steps=len(train_np["input_ids"]) // batch_size * epochs)
optimizer = optax.adamw(scheduler)

rng = jax.random.PRNGKey(0)
params = model.variables["params"]
state = TrainState.create(apply_fn=model.__call__, params=params, tx=optimizer)

# === Step 5: Training Loop ===
print("Training...")
for epoch in range(epochs):
    pbar = tqdm.trange(0, len(train_np["input_ids"]), batch_size, desc=f"Epoch {epoch+1}")
    for i in pbar:
        batch = {k: v[i:i+batch_size] for k, v in train_np.items()}
        state, metrics = train_step(state, batch)
        pbar.set_postfix(metrics)

# === Step 6: Save Checkpoint ===
print("Saving checkpoint...")
save_file(state.params, "qwen3-sft-sst2.safetensors")
print("Done!")
