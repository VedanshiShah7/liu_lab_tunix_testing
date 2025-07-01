from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen-1_8B"

# Download tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# Save locally (optional)
tokenizer.save_pretrained("./qwen-1.8b")
model.save_pretrained("./qwen-1.8b")

from safetensors.torch import save_file

state_dict = model.state_dict()
save_file(state_dict, "./qwen-1.8b/pytorch_model.safetensors")
