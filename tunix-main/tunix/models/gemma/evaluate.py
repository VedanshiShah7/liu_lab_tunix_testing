#!/usr/bin/env python3
# evaluate.py

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import sacrebleu
from tqdm import tqdm

from data import create_datasets, GemmaTokenizer
from gemma import Transformer
from params import load_and_format_params


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Gemma model")
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to the Orbax checkpoint directory")
    p.add_argument("--version", type=str, default="2b",
                   choices=["2b","7b","2-2b","2-9b"],
                   help="Model version string (e.g. 2b, 7b, 2-2b, 2-9b)")
    p.add_argument("--dataset", type=str, default="mtnt/en-fr",
                   choices=["mtnt/en-fr","Helsinki-NLP/opus-100"],
                   help="Dataset to evaluate on")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Eval batch size")
    p.add_argument("--max_len", type=int, default=128,
                   help="Max target sequence length")
    p.add_argument("--instruct_tuned", action="store_true",
                   help="Whether to use the instruction-tuned template")
    p.add_argument("--bleu", action="store_true",
                   help="Compute BLEU over argmax decoding")
    return p.parse_args()


@partial(jax.jit, static_argnums=(0,))
def forward_apply(model, params, inputs, mask):
    logits, _ = model(inputs,                             # [B,L]
                      jnp.arange(inputs.shape[-1])[None, :],  # positions
                      cache=None,
                      attention_mask=mask)
    return logits  # [B,L,V]


def cross_entropy_and_accuracy(logits, labels, mask):
    # logits: [B,L,V], labels: [B,L], mask: [B,L] bool
    logp = jax.nn.log_softmax(logits, axis=-1)
    # Gather the log-probs at the gold positions
    nll = -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
    # Mask out the source-part: only evaluate on target tokens
    nll = nll * mask
    total_nll = nll.sum()
    total_tokens = mask.sum()
    # token accuracy
    pred = jnp.argmax(logits, axis=-1)
    correct = (pred == labels) * mask
    total_correct = correct.sum()
    return total_nll, total_tokens, total_correct


def main():
    args = parse_args()

    # 1) load params & build model
    print("Loading checkpoint...")
    raw_params = load_and_format_params(args.ckpt_path)
    model = Transformer.from_params(raw_params, version=args.version)
    params = jax.tree_leaves(model)  # flatten NN parameters for jit

    # 2) prepare tokenizer & eval loader
    tok = GemmaTokenizer()
    _, eval_loader = create_datasets(
        dataset_name=args.dataset,
        global_batch_size=args.batch_size,
        max_target_length=args.max_len,
        num_train_epochs=None,
        tokenizer=tok,
        instruct_tuned=args.instruct_tuned,
    )

    # 3) loop over eval
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    all_hyps = []
    all_refs = []

    print("Starting evaluation...")
    tic = time.perf_counter()
    for batch in tqdm(eval_loader, desc="Eval"):
        # batch: TrainingInput(input_tokens, input_mask)
        tokens = jnp.array(batch.input_tokens)       # [B, L_src+L_tgt]
        mask = jnp.array(batch.input_mask)           # bool mask separating tgt
        # split into labels & inputs
        # in our pipeline, input_tokens = [src; tgt], mask= [F...F; T...T]
        labels = jnp.where(mask, tokens, -1)         # keep only tgt for CE
        logits = forward_apply(model, params, tokens, mask[..., None, :])
        # 3.a) CE & accuracy
        nll, n_tok, n_corr = cross_entropy_and_accuracy(logits, labels, mask)
        total_nll   += float(nll)
        total_tokens+= int(n_tok)
        total_correct+= int(n_corr)
        # 3.b) BLEU references & hyps
        if args.bleu:
            # decode argmax only on target segment
            preds = np.argmax(np.array(logits), axis=-1)
            # for each sample, strip src tokens and decode
            for inp, pred_ids in zip(np.array(batch.input_tokens), preds):
                # find where mask starts
                tgt_ids = pred_ids[inp.shape[0] - args.max_len:]  # last max_len
                hyp = tok.DecodeIds(tgt_ids.tolist()).strip()
                ref = tok.DecodeIds(inp[inp != tok.pad_id()][-len(tgt_ids):].tolist()).strip()
                all_hyps.append(hyp)
                all_refs.append([ref])

    toc = time.perf_counter()
    elapsed = toc - tic

    # 4) report
    ppl = np.exp(total_nll / total_tokens)
    acc = total_correct / total_tokens * 100.0
    print(f"\nPerplexity: {ppl:.4f}")
    print(f"Token-accuracy: {acc:.2f}%")
    print(f"Throughput: {total_tokens/elapsed:.1f} tokens/sec  ({elapsed:.1f}s total)")

    if args.bleu:
        bleu = sacrebleu.corpus_bleu(all_hyps, list(zip(*all_refs)))
        print(f"BLEU: {bleu.score:.2f}")

if __name__ == "__main__":
    main()
