# run_sst2_sft.py

import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset
from transformers import AutoTokenizer

from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig, TrainingInput
from tunix.sft.checkpoint_manager import CheckpointManagerOptions
from tunix.sft.metrics_logger import MetricsLoggerOptions
from tunix.sft.profiler import ProfilerOptions

# 1. Hyperparams & Config
TRAIN_BATCH = 16
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

training_config = TrainingConfig(
    eval_every_n_steps=500,
    max_steps=None,  # let it run through entire dataset × epochs
    checkpoint_root_directory="./sft_checkpoints",
    checkpointing_options=CheckpointManagerOptions(),
    metrics_logging_options=MetricsLoggerOptions(log_dir="./logs", flush_every_n_steps=100),
    profiler_options=ProfilerOptions(log_dir="./prof", skip_first_n_steps=10, profiler_steps=50),
    max_inflight_computations=2,
)

# 2. Load & preprocess GLUE SST-2
raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(split):
    def _tok(batch):
        toks = tokenizer(batch["sentence"],
                         padding="max_length",
                         truncation=True,
                         max_length=128)
        return {
            "input_tokens": jnp.array(toks["input_ids"]),
            "input_mask":   jnp.array(toks["attention_mask"]),
            "label":        jnp.array(batch["label"]),
        }
    ds = raw[split].map(_tok, batched=True)
    ds.set_format(type="numpy", columns=["input_tokens","input_mask","label"])
    return ds

train_ds = preprocess("train")
val_ds   = preprocess("validation")
test_ds  = preprocess("test")

def ds_generator(ds):
    for itm in ds:
        ti = TrainingInput(input_tokens=itm["input_tokens"],
                           input_mask=itm["input_mask"])
        yield ti, itm["label"]

# 3. Instantiate model & optimizer
#    (replace `create_tunix_model` with however you build your Flax model)
from tunix.model import create_tunix_model
model = create_tunix_model(num_labels=2)
optimizer = optax.adamw(LR, weight_decay=WEIGHT_DECAY)

# 4. Set up trainer
trainer = PeftTrainer(model, optimizer, training_config)

# 4a. Tell it how to extract inputs + labels from our generator
def gen_model_input(batch_and_label):
    inp, lbl = batch_and_label
    return {
        "input_tokens": inp.input_tokens,
        "positions":    jnp.arange(inp.input_tokens.shape[1])[None, :],
        "attention_mask": inp.input_mask,
        "labels": lbl,
    }
trainer = trainer.with_gen_model_input_fn(gen_model_input)

# 4b. Use a classification loss fn (returns scalar loss only)
def cls_loss_fn(model, input_tokens, positions, attention_mask, labels):
    logits, _ = model(input_tokens, positions, None, attention_mask)
    # logits shape: (batch, seq_len, vocab) → for classification use first token:
    cls_logits = logits[:, 0, :]  # adjust if your model outputs differently
    loss = optax.softmax_cross_entropy_with_integer_labels(cls_logits, labels).mean()
    return loss
trainer = trainer.with_loss_fn(cls_loss_fn, has_aux=False)

# 5. Train!
trainer.train(
    train_ds=ds_generator(train_ds),
    eval_ds=ds_generator(val_ds),
)

# 6. Inference & accuracy on test split
total, correct = 0, 0
for inp, lbl in ds_generator(test_ds):
    logits, _ = model(
        inp.input_tokens,
        jnp.arange(inp.input_tokens.shape[1])[None, :],
        None,
        inp.input_mask,
    )
    cls_logits = logits[:, 0, :]
    preds = jnp.argmax(cls_logits, axis=-1)
    correct += int((preds == lbl).sum())
    total   += len(lbl)

print(f"\n▶ Test Accuracy: {correct/total:.4f}")
