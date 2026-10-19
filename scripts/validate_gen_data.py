"""Validate generated BFCL data: the canonical answer (first allowed values)
must pass the official ast_checker, otherwise the item is unearnable in RL.

Usage: python validate_gen_data.py <items_or_train.jsonl> [checker_path]
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(sys.argv[1]))
if len(sys.argv) > 2:
    sys.path.insert(0, sys.argv[2])
from ast_checker_official import ast_checker  # noqa: E402


def markup(name, params):
    lines = [f"<function={name}>"]
    for k, v in params.items():
        lines.append(f"<parameter={k}>")
        lines.append(json.dumps(v, ensure_ascii=False))
        lines.append("</parameter>")
    lines.append("</function>")
    return "\n".join(lines)


def undeep(x):
    """A concrete value that may nest dicts whose values are alt-lists."""
    if isinstance(x, dict):
        return {k: unval(v) for k, v in x.items()}
    if isinstance(x, list):
        return [undeep(e) for e in x]
    return x


def unval(allowed):
    """Allowed alternatives for one slot -> first non-empty, unwrapped."""
    vals = [v for v in allowed if v != ""]
    if not vals:
        return None
    return undeep(vals[0])


bad = Counter()
total = 0
first_bad = []
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    it = json.loads(line)
    total += 1
    fname = it["function"][0]["name"]
    gt = it["ground_truth"][0][fname]
    params = {}
    for k, allowed in gt.items():
        v = unval(allowed)
        if v is not None:
            params[k] = v
    resp = markup(fname.replace(".", "_"), params)
    try:
        import re
        calls = []
        for fm in re.finditer(r"<function=([\w.]+)>(.*?)</function>", resp, re.S):
            n, body = fm.group(1), fm.group(2)
            ps = {}
            for pm in re.finditer(r"<parameter=([\w.]+)>(.*?)</parameter>", body, re.S):
                try:
                    ps[pm.group(1)] = json.loads(pm.group(2).strip())
                except Exception:
                    ps[pm.group(1)] = pm.group(2).strip()
            calls.append({n: ps})
        v = ast_checker(it["function"], calls, it["ground_truth"])
        ok = v["valid"]
    except Exception as e:  # noqa: BLE001
        ok = False
        if len(first_bad) < 3:
            first_bad.append((it["id"], str(e)[:120]))
    if not ok:
        bad[it["id"].rsplit("_", 1)[0]] += 1
        if len(first_bad) < 6:
            first_bad.append((it["id"],
                              json.dumps(params, ensure_ascii=False)[:150]))

print(f"total={total} bad={sum(bad.values())} bad_by_family={dict(bad)}")
for fid, info in first_bad[:6]:
    print("  e.g.", fid, info)
