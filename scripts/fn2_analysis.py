import json, sys
sys.path.insert(0, '/root/train95')
from methods.sphere_grpo import parse_pred_snips

path = sys.argv[1] if len(sys.argv) > 1 else \
    '/root/train95/results/final-snips-all_hidden_adaptive.jsonl'
fn = {}; fp = {}; tp = 0
for line in open(path):
    d = json.loads(line)
    gold = d.get('gold')
    if isinstance(gold, str):
        gold = json.loads(gold)
    resp = d.get('response') or d.get('pred') or ''
    gslots = [(s['name'], s['value']) for s in gold['slots']]
    intent, pslots = parse_pred_snips(resp)
    rest = list(pslots)
    for gname, gval in gslots:
        hit = False
        for p in list(rest):
            if p[0] == gname:
                rest.remove(p)
                if p[1] == gval:
                    tp += 1
                else:
                    fn[gname] = fn.get(gname, 0) + 1
                    fp[gname] = fp.get(gname, 0) + 1
                hit = True
                break
        if not hit:
            fn[gname] = fn.get(gname, 0) + 1
    for pname, pval in rest:
        fp[pname] = fp.get(pname, 0) + 1

print('TP', tp, 'total FN', sum(fn.values()), 'total FP', sum(fp.values()))
print('--- FN by slot (true miss or value-wrong):')
for k, v in sorted(fn.items(), key=lambda x: -x[1])[:14]:
    print('  %s: %d' % (k, v))
print('--- FP by slot:')
for k, v in sorted(fp.items(), key=lambda x: -x[1])[:8]:
    print('  %s: %d' % (k, v))
