"""Show official question + param schema for specific eval ids."""
import json
import sys

want = set(sys.argv[2:])
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    d = json.loads(line)
    if d["id"] in want:
        print("===", d["id"])
        print("Q:", d["question"][0][0]["content"][:200])
        fd = d["function"][0]
        print("SCHEMA:", json.dumps(fd["parameters"], indent=1)[:800])
        print()
