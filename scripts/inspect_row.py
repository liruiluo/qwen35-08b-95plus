import json, sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for i, line in enumerate(open(path)):
    if i >= n:
        break
    d = json.loads(line)
    print('row', i, 'type', type(d).__name__)
    if isinstance(d, dict):
        for k, v in d.items():
            s = json.dumps(v, ensure_ascii=False)
            print(' ', k, '=>', s[:300])
