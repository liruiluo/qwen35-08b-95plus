#!/usr/bin/env python3
"""Evaluate a chat model on SNIPS NLU (validate split): intent accuracy +
span-exact slot micro-F1.

Protocol B (identical to the training rollout): the chat template is rendered
locally with the model tokenizer (thinking disabled) and sent to the sglang
engine's /generate endpoint as token ids. See common/bproto.py for why.

Usage:
  python eval_snips.py --engine-url http://127.0.0.1:30001 \
      --model /path/to/model --data data/validate.jsonl --out results.jsonl
"""
import argparse, json, os, queue, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.bproto import render_ids, generate  # noqa: E402


def parse_pred(content):
    if not content:
        return None, []
    try:
        s = content.strip()
        if "```" in s:
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
        a, b = s.find("{"), s.rfind("}")
        d = json.loads(s[a:b + 1])
        intent = d.get("intent")
        slots = []
        for sl in d.get("slots", []) or []:
            if isinstance(sl, dict) and "name" in sl and "value" in sl:
                slots.append((str(sl["name"]).strip(), str(sl["value"]).strip()))
        return intent, slots
    except Exception:
        return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-url", default="http://127.0.0.1:30001",
                    help="sglang engine root (its /generate is called)")
    ap.add_argument("--model", required=True,
                    help="local model path (tokenizer source; must match the "
                         "weights the engine serves)")
    ap.add_argument("--data", default="data/validate.jsonl")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    test = []
    for r in rows:
        system = r["messages"][0]["content"]
        text = r["messages"][1]["content"]
        gold = json.loads(r["messages"][2]["content"])
        test.append({"messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": text}],
                     "intent": gold["intent"],
                     "slots": [(s["name"], s["value"]) for s in gold["slots"]]})

    results = [None] * len(test)
    q = queue.Queue()
    for i in range(len(test)):
        q.put(i)

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            item = test[i]
            try:
                ids = render_ids(tok, item["messages"])
                content = generate(args.engine_url, ids)
            except Exception as e:  # noqa: BLE001
                results[i] = {"id": i, "pred_intent": None, "intent_ok": 0,
                              "tp": 0, "fp": 0, "fn": len(item["slots"]),
                              "worker_error": str(e)[:200]}
                q.task_done()
                continue
            intent, slots = parse_pred(content)
            i_ok = 1 if intent == item["intent"] else 0
            pred_all = list(slots)
            gold = list(item["slots"])
            tp = 0
            for g in gold:
                if g in slots:
                    tp += 1
                    slots.remove(g)
            results[i] = {"id": i, "text": item["messages"][1]["content"],
                          "gold_intent": item["intent"],
                          "pred_intent": intent, "intent_ok": i_ok,
                          "gold_slots": gold, "pred_slots": pred_all,
                          "tp": tp, "fp": len(slots),
                          "fn": len(gold) - tp}
            q.task_done()

    ths = [threading.Thread(target=worker) for _ in range(args.workers)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0

    n = len(results)
    TP = sum(r["tp"] for r in results)
    FP = sum(r["fp"] for r in results)
    FN = sum(r["fn"] for r in results)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    acc = sum(r["intent_ok"] for r in results) / n
    unparsable = sum(1 for r in results if r["pred_intent"] is None)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"model={args.model}")
    print(f"n={n} wall={wall:.1f}s")
    print(f"INTENT_ACC={acc*100:.2f}%")
    print(f"SLOT_P={P*100:.2f}% SLOT_R={R*100:.2f}% SLOT_F1={F1*100:.2f}%")
    print(f"TP={TP} FP={FP} FN={FN} unparsable={unparsable}")


if __name__ == "__main__":
    main()
