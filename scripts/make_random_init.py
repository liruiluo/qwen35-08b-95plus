#!/usr/bin/env python3
"""Re-initialize a model's weights from scratch (seed 17).

Creates <dst>: SAME architecture, tokenizer and configs as the source
model, but every weight drawn fresh from the model class's own init
distribution. Purpose: the true from-zero starting point (训练前模型).

Usage: python make_random_init.py <src> <dst>
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("src", help="source model dir (HF format)")
ap.add_argument("dst", help="output dir for the re-initialized model")
args = ap.parse_args()

torch.manual_seed(17)
cfg = AutoConfig.from_pretrained(args.src)
try:
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
model.save_pretrained(args.dst)
AutoTokenizer.from_pretrained(args.src).save_pretrained(args.dst)
n = sum(p.numel() for p in model.parameters())
print(f"saved {args.dst}: {n/1e9:.3f}B params, torch.manual_seed(17)")
