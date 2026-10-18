#!/usr/bin/env python3
"""Push a checkpoint's weights into the RUNNING sglang engine (which was
launched from the base model), so evaluations can run against arbitrary
weights without restarting the engine. Usage:

  python push_ckpt.py http://127.0.0.1:30001 /path/to/ckpt
"""
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "/root/train95")
from methods.sphere_grpo import push_weights  # noqa: E402


def main():
    url, path = sys.argv[1], sys.argv[2]
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16).cuda()
    m.eval()
    n = sum(1 for _ in m.state_dict())
    push_weights(url, m)
    print(f"PUSHED_OK {path} tensors={n}")


if __name__ == "__main__":
    main()
