import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import os

model_name = "Qwen/Qwen-1_8B"
save_path = "./qwen1_8B_tunix.safetensors"

print(f"🔄 Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

print("💾 Extracting and transforming state_dict...")
state_dict = model.state_dict()
new_state = {}

for k, v in state_dict.items():
    # === Handle fused qkv projection ===
    if "c_attn.weight" in k:
        qkv = v.chunk(3, dim=-1)
        prefix = k.replace("c_attn.weight", "")
        new_state[prefix + "q_proj.weight"] = qkv[0].contiguous()
        new_state[prefix + "k_proj.weight"] = qkv[1].contiguous()
        new_state[prefix + "v_proj.weight"] = qkv[2].contiguous()

    elif "c_attn.bias" in k:
        qkv = v.chunk(3, dim=-1)
        prefix = k.replace("c_attn.bias", "")
        new_state[prefix + "q_proj.bias"] = qkv[0].contiguous()
        new_state[prefix + "k_proj.bias"] = qkv[1].contiguous()
        new_state[prefix + "v_proj.bias"] = qkv[2].contiguous()

    # === Handle attention output projection ===
    elif ".attn.c_proj.weight" in k:
        new_k = k.replace(".attn.c_proj.weight", ".attn.o_proj.weight")
        new_state[new_k] = v.contiguous()

    elif ".attn.c_proj.bias" in k:
        new_k = k.replace(".attn.c_proj.bias", ".attn.o_proj.bias")
        new_state[new_k] = v.contiguous()

    # === Handle embeddings ===
    elif "embed_tokens.weight" in k:
        new_state["transformer.embed.weight"] = v.contiguous()

    # === Handle final LM head ===
    elif "lm_head.weight" in k:
        new_state["lm_head.weight"] = v.contiguous()

    # === Copy everything else unchanged ===
    else:
        new_state[k] = v.contiguous()

print(f"💾 Saving to {save_path}...")
save_file(new_state, save_path)
print(f"✅ Saved: {save_path}")
