"""Show official ground truth for specific eval ids to diagnose value errors."""
import json
import sys

ids = set(sys.argv[2:])
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    d = json.loads(line)
    if d["id"] in ids:
        print(d["id"], "=>", json.dumps(d["ground_truth"], ensure_ascii=False))
