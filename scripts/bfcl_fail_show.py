"""Show BFCL eval failures grouped by error_type with raw model output."""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
per_type = defaultdict(list)
for line in open(path):
    if not line.strip():
        continue
    r = json.loads(line)
    if not r["ok"]:
        per_type[r["error_type"]].append(r)

for et, rows in sorted(per_type.items(), key=lambda x: -len(x[1])):
    print(f"=== {et}: {len(rows)}")
    for r in rows[:4]:
        print("  id:", r["id"], "raw:", (r.get("raw") or "")[:220].replace("\n", " "))
    print()
