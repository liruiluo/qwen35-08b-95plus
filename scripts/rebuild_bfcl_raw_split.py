"""Rebuild raw-format BFCL train/val split from items_bfcl_format.jsonl.

gen_train_data.py wrote train.jsonl/val.jsonl in SFT chat format (messages
+ tools), but the RL trainer (methods/sphere_grpo.py load_rows) needs the
raw rows: question / function / ground_truth. The items file preserves the
original shuffled order and the split was items[:660]=val, items[660:]=train,
so this rebuild reproduces the exact original split.
"""
import json
import os
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "bfcl/data/items_bfcl_format.jsonl"
n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 660
out_dir = os.path.dirname(src)

items = [json.loads(l) for l in open(src) if l.strip()]
val, train = items[:n_val], items[n_val:]
assert len(val) == n_val, (len(val), n_val)

with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
    for it in train:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
with open(os.path.join(out_dir, "val.jsonl"), "w") as f:
    for it in val:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"train={len(train)} val={len(val)} from {len(items)} items")
print("train row0 keys:", sorted(train[0].keys()))
