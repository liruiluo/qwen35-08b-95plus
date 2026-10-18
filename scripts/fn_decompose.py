"""Decompose SNIPS eval FNs: name-missing vs name-right-value-wrong.

Reads an eval jsonl with gold_slots + pred_slots (eval_snips.py dumps both).
This decides the extension strategy: value-wrong FNs convert with more RL
steps (the mapping is learnable); name-missing FNs need more exploration.
"""
import json
import sys
from collections import Counter

path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]
tp = name_right = 0
fn_missing = Counter()
fn_value = Counter()
fp_extra = Counter()
for r in rows:
    pred = [tuple(p) for p in r["pred_slots"]]
    gold = [tuple(g) for g in r["gold_slots"]]
    used = [False] * len(pred)
    for gname, gval in gold:
        hit = None
        for i, (pn, pv) in enumerate(pred):
            if not used[i] and pn == gname:
                hit = i
                break
        if hit is None:
            fn_missing[gname] += 1
        elif pred[hit][1] == gval:
            tp += 1
            used[hit] = True
        else:
            name_right += 1
            used[hit] = True
            fn_value[gname] += 1
    for i, (pn, pv) in enumerate(pred):
        if not used[i]:
            fp_extra[pn] += 1

print(f"TP={tp} name-right-value-wrong={name_right} "
      f"name-missing={sum(fn_missing.values())} "
      f"extra-FP={sum(fp_extra.values())}")
print("value-wrong FNs by slot:")
for k, v in fn_value.most_common(12):
    print(f"  {k}: {v}")
print("name-missing FNs by slot:")
for k, v in fn_missing.most_common(12):
    print(f"  {k}: {v}")
print("extra FPs by slot (no gold counterpart):")
for k, v in fp_extra.most_common(8):
    print(f"  {k}: {v}")
