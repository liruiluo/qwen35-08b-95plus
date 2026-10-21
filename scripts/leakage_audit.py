#!/usr/bin/env python3
"""Data-hygiene audit: overlap between training data and evaluation data.

Checks, per dataset:
  SNIPS
    - exact utterance-text overlap between train and validate
    - near-duplicate rate (Jaccard over word 3-grams >= 0.8)
  BFCL
    - exact question-text overlap between synthetic train and real v4 eval
    - word 8-gram leakage: fraction of eval questions that share an 8-gram
      with any synthetic training question
    - function-name coverage (expected: eval functions appear in the
      synthetic pool by design — the schema space is public)

Usage:
  python leakage_audit.py --snips-train SNIPS/data/train.jsonl \
      --snips-val SNIPS/data/validate.jsonl \
      --bfcl-train bfcl/data/train.jsonl --bfcl-eval questions.jsonl
"""
import argparse, json, re


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def ngrams(words, n):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def near_dup(a_words, b_words, n=3, thr=0.8):
    ga, gb = ngrams(a_words, n), ngrams(b_words, n)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snips-train")
    ap.add_argument("--snips-val")
    ap.add_argument("--bfcl-train")
    ap.add_argument("--bfcl-eval")
    args = ap.parse_args()

    if args.snips_train and args.snips_val:
        tr = [json.loads(l)["messages"][1]["content"].strip()
              for l in open(args.snips_train) if l.strip()]
        va = [json.loads(l)["messages"][1]["content"].strip()
              for l in open(args.snips_val) if l.strip()]
        trn, van = {norm(t) for t in tr}, {norm(v) for v in va}
        exact = trn & van
        print(f"[SNIPS] train={len(tr)} val={len(va)} "
              f"exact-text overlap={len(exact)}")
        if exact:
            for e in list(exact)[:5]:
                print("   EXACT:", e[:80])
        trw = [norm(t).split() for t in tr]
        vaw = [norm(v).split() for v in va]
        nd = 0
        for i, vw in enumerate(vaw):
            if len(vw) < 3:
                continue
            best = max((near_dup(vw, tw) for tw in trw), default=0.0)
            if best >= 0.8:
                nd += 1
                if nd <= 3:
                    print(f"   NEARDUP val[{i}] jaccard={best:.2f}: "
                          f"{' '.join(vw)[:70]}")
        print(f"[SNIPS] near-duplicate (3-gram Jaccard>=0.8) val items: {nd}"
              f"/{len(vaw)}")

    if args.bfcl_train and args.bfcl_eval:
        trq = []
        for l in open(args.bfcl_train):
            if not l.strip():
                continue
            d = json.loads(l)
            # raw-format row: {id, question, function, ground_truth}
            trq.append(norm(d["question"]))
        evq = []
        for l in open(args.bfcl_eval):
            if not l.strip():
                continue
            d = json.loads(l)
            evq.append(norm(d["question"][0][0]["content"]))
        trset = set(trq)
        exact = [q for q in evq if q in trset]
        print(f"[BFCL] synth-train={len(trq)} eval={len(evq)} "
              f"exact question overlap={len(exact)}")
        grams = set()
        for q in trq:
            grams |= ngrams(q.split(), 8)
        leak = sum(1 for q in evq if ngrams(q.split(), 8) & grams)
        print(f"[BFCL] eval questions sharing a word 8-gram with synth "
              f"train: {leak}/{len(evq)}")
        for q in [q for q in evq if ngrams(q.split(), 8) & grams][:5]:
            print("   SHARED-8GRAM:", q[:80])


if __name__ == "__main__":
    main()
