#!/usr/bin/env python3
"""Build SNIPS NLU SFT data from the snipsco/nlu-benchmark repo.

Input:  <repo>/2017-06-custom-intent-engines/<Intent>/train_<Intent>_full.json
        (entity-annotated segments; some files need latin-1 fallback decoding)
Output: train.jsonl / validate.jsonl with {"messages": [...]} where the
        assistant target matches the eval protocol exactly:
        {"intent": "...", "slots": [{"name": "...", "value": "..."}]}
        Slot values are exact substrings of the utterance (span-exact eval).

The system prompt is identical to eval_snips.py so train/eval distributions
line up byte-for-byte on the instruction side.

Usage:
  python prepare_data.py --repo /path/to/nlu-benchmark --out-dir data
"""
import argparse, json, os, random


def load_json(f):
    raw = open(f, "rb").read()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("latin-1"))


def to_item(entry):
    """Convert one {'data': [...]} entry to (text, [(slot, value), ...])."""
    text = "".join(seg["text"] for seg in entry["data"])
    slots = []
    for seg in entry["data"]:
        if "entity" in seg:
            val = seg["text"]
            # trim whitespace the benchmark pads entity segments with
            lead = len(val) - len(val.lstrip())
            trail = len(val) - len(val.rstrip())
            val = val.strip()
            start = text.find(val)
            if start < 0 or (lead and not text[:text.find(val)].endswith(" ") and lead):
                # value not a clean substring; keep stripped form (rare)
                pass
            slots.append((seg["entity"], val))
    return text.strip(), slots


def build_split(repo, split, system):
    rows, intents, slots_all = [], [], set()
    base = os.path.join(repo, "2017-06-custom-intent-engines")
    for intent in sorted(os.listdir(base)):
        d = os.path.join(base, intent)
        if not os.path.isdir(d):
            continue
        if split == "train":
            fname = [f for f in os.listdir(d)
                     if f.startswith("train_") and f.endswith("_full.json")]
        else:
            fname = [f for f in os.listdir(d)
                     if f.startswith(f"{split}_") and f.endswith(".json")]
        if not fname:
            continue
        data = load_json(os.path.join(d, fname[0]))[intent]
        for entry in data:
            text, slots = to_item(entry)
            intents.append(intent)
            slots_all.update(s for s, _ in slots)
            assistant = json.dumps(
                {"intent": intent,
                 "slots": [{"name": n, "value": v} for n, v in slots]},
                ensure_ascii=False)
            rows.append({"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
                {"role": "assistant", "content": assistant}]})
    return rows, sorted(set(intents)), sorted(slots_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to nlu-benchmark clone")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    schema = json.load(open(os.path.join(args.repo, "2017-06-custom-intent-engines",
                                         "AddToPlaylist", "validate_AddToPlaylist.json")))  # probe only
    # schema is derived from the train_full entity vocabulary below
    train_rows, intents, slot_names = build_split(args.repo, "train", "")
    system = (
        "You are a Natural Language Understanding engine for a voice assistant. "
        "Given a user utterance, predict (1) the intent and (2) all slot spans in the utterance.\n"
        f"Possible intents (choose exactly one): {', '.join(intents)}.\n"
        f"Possible slot names: {', '.join(slot_names)}.\n"
        "A slot value must be the exact substring of the utterance that fills that slot. "
        "Only output slot spans that actually appear in the utterance; omit slots that are absent.\n"
        'Answer with strict JSON only, no other text: {"intent": "...", "slots": [{"name": "...", "value": "..."}]}'
    )
    for r in train_rows:
        r["messages"][0]["content"] = system

    val_rows, _, _ = build_split(args.repo, "validate", system)
    test_rows, _, _ = build_split(args.repo, "test", system)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.jsonl"), "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "validate.jsonl"), "w") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if test_rows:
        # The public snipsco/nlu-benchmark repo does NOT ship a test split;
        # write one only if a fork/clone provides test_<Intent>.json files.
        with open(os.path.join(args.out_dir, "test.jsonl"), "w") as f:
            for r in test_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"intents": intents, "slot_names": slot_names},
              open(os.path.join(args.out_dir, "schema.json"), "w"), indent=1)
    print(f"train={len(train_rows)} validate={len(val_rows)} test={len(test_rows)} "
          f"intents={len(intents)} slots={len(slot_names)}")


if __name__ == "__main__":
    main()
