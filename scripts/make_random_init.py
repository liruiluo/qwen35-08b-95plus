"""Re-initialize Qwen3.5-0.8B weights from scratch (seed 17).

Creates /root/models/Qwen3.5-0.8B-RandomInit: SAME architecture, tokenizer
and configs as the source model, but every weight drawn fresh from the
model class's own init distribution. Purpose: measure the true from-zero
starting point (expect ~0% success, no JSON, no reasoning) and test
whether the RL recipe can bootstrap from random weights.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

SRC = "/root/models/Qwen3.5-0.8B"
DST = "/root/models/Qwen3.5-0.8B-RandomInit"

torch.manual_seed(17)
cfg = AutoConfig.from_pretrained(SRC)
try:
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
model.save_pretrained(DST)
AutoTokenizer.from_pretrained(SRC).save_pretrained(DST)
n = sum(p.numel() for p in model.parameters())
print(f"saved {DST}: {n/1e9:.3f}B params, torch.manual_seed(17)")
