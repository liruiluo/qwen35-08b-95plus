"""Full failure catalog: for each failed eval item, emit per-param
(produced value, expected alternatives) so the convention patterns can be
tallied systematically for synthetic-data design."""
import json
import re
import sys

res_path, ans_path = sys.argv[1], sys.argv[2]


def parse_markup(text):
    calls = []
    if not text:
        return calls
    for fm in re.finditer(r"<function=([\w.]+)>(.*?)</function>", text, re.S):
        name, body = fm.group(1), fm.group(2)
        params = {}
        for pm in re.finditer(r"<parameter=([\w.]+)>(.*?)</parameter>", body, re.S):
            k, v = pm.group(1), pm.group(2).strip()
            try:
                params[k] = json.loads(v)
            except Exception:
                params[k] = v
        calls.append({name: params})
    return calls


ANS = {}
for line in open(ans_path):
    if line.strip():
        d = json.loads(line)
        ANS[d["id"]] = d["ground_truth"]

for line in open(res_path):
    if not line.strip():
        continue
    r = json.loads(line)
    if r["ok"]:
        continue
    gt_list = ANS[r["id"]]
    calls = parse_markup(r.get("raw") or "")
    got = {}
    for c in calls:
        for fname, params in c.items():
            got.update(params)
    # expected: single gold item (simple python = one call)
    exp = {}
    if gt_list and isinstance(gt_list[0], dict):
        for fname, params in gt_list[0].items():
            exp.update(params)
    for k, alts in exp.items():
        pv = got.get(k, "<OMITTED>")
        if isinstance(alts, list) and pv in alts:
            continue
        print(f"{r['id']}\t{k}\tprod={json.dumps(pv, ensure_ascii=False)[:60]}"
              f"\texp={json.dumps(alts, ensure_ascii=False)[:80]}")
    for k in got:
        if k not in exp:
            print(f"{r['id']}\t{k}\tprod={json.dumps(got[k], ensure_ascii=False)[:60]}\texp=<EXTRA>")
