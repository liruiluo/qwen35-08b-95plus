#!/usr/bin/env python3
"""Build SNIPS train/dev/test data from the JointBERT-standard split.

Source: monologg/JointBERT repo, data/snips/{train,dev,test}/{seq.in,seq.out,label}
        (the literature-standard 13,084/700/700 division of the SNIPS NLU
        corpus — the split JointBERT/Stack-Propagation-style papers report
        on, and the test-set split LLM2LLM-style reporting assumes).
        The official snipsco/nlu-benchmark repo ships NO test split, and its
        train file substantially overlaps the JointBERT test set, so
        test-set numbers must be trained on THIS split.

Input:  <src>/{train,dev,test}-{seq.in,seq.out,label}
Output: <out-dir>/{train,dev,test}.jsonl + schema.json in the
        {"messages": [...]} row schema, so the trainer and
        eval_snips.py consume it unchanged.

Slots are reconstructed from the BIO tags aligned to the whitespace tokens;
a slot value is the space-joined token span (an exact substring of the
whitespace-normalized utterance).

Usage:
  python prepare_jointbert.py --src ../jointbert-snips --out-dir data-jb
"""
import argparse, json, os


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def bio_to_slots(tokens, tags):
    """Group consecutive B-x (I-x)* runs into [{"name", "value"}]."""
    assert len(tokens) == len(tags), f"token/tag mismatch: {len(tokens)} vs {len(tags)}"
    slots, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            name = tags[i][2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{name}":
                j += 1
            slots.append({"name": name, "value": " ".join(tokens[i:j])})
            i = j
        else:
            i += 1
    return slots


def read_split(src, split, system):
    """One row per line: {"messages": [...]} schema."""
    utts = read_lines(os.path.join(src, f"{split}-seq.in"))
    tags = read_lines(os.path.join(src, f"{split}-seq.out"))
    labels = read_lines(os.path.join(src, f"{split}-label"))
    rows = []
    for raw, tag_line, intent in zip(utts, tags, labels):
        text = " ".join(raw.split())
        tokens, tlist = text.split(" "), tag_line.split()
        slots = bio_to_slots(tokens, tlist)
        assistant = json.dumps({"intent": intent, "slots": slots},
                               ensure_ascii=False)
        rows.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
            {"role": "assistant", "content": assistant}]})
    return rows


INTENT_DEFINITIONS = (
    "Intent definitions:\n"
    "- AddToPlaylist: add a track, album, or artist to an existing playlist "
    "(\"add X to Y\", \"put X onto/into Y\" where Y is a playlist).\n"
    "- BookRestaurant: reserve a table at a restaurant (party size, time, date, cuisine).\n"
    "- GetWeather: ask about weather conditions or forecast for a place or time, "
    "including \"will it be sunny / foggy / nice\" questions.\n"
    "- PlayMusic: play, put on, listen to, hear, or open music - a song, artist, album, "
    "playlist, EP, record, single, soundtrack, or genre - for listening now.\n"
    "- RateBook: give a star or point rating to a book (\"give X four stars\", "
    "\"rate X 2 of 6 points\").\n"
    "- SearchCreativeWork: find, search for, or look up a creative work - a movie, show, "
    "episode, game, trailer, or book - as content; looking for or getting a book, and "
    "watching, seeing, or purchasing a TV series or a typed work (\"the show X\", "
    "\"the tv series X\", \"the album Y\"), is SearchCreativeWork. Playing a non-music "
    "work (a movie, show, game, or trailer) is SearchCreativeWork; asking to play or "
    "hear music is PlayMusic instead. Wanting to see or watch a movie is "
    "SearchScreeningEvent, not SearchCreativeWork.\n"
    "- SearchScreeningEvent: the user wants to see or watch a movie - showtimes, "
    "theaters, what is playing near a place; \"I want to see X\", \"I'd like to watch X\", "
    "\"is X playing\", \"what movies are showing\", \"find the schedule for X\"."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with {split}-{seq.in,seq.out,label}")
    ap.add_argument("--out-dir", default="data-jb")
    ap.add_argument("--definitions", action="store_true",
                    help="append intent definitions to the system prompt")
    args = ap.parse_args()

    train_raw = read_lines(os.path.join(args.src, "train-seq.in"))
    train_tags = [t for line in read_lines(os.path.join(args.src, "train-seq.out"))
                  for t in line.split()]
    train_intents = sorted({l.strip() for l in read_lines(os.path.join(args.src, "train-label"))})
    slot_names = sorted({t[2:] for t in train_tags if t.startswith("B-")})

    system = (
        "You are a Natural Language Understanding engine for a voice assistant. "
        "Given a user utterance, predict (1) the intent and (2) all slot spans in the utterance.\n"
        f"Possible intents (choose exactly one): {', '.join(train_intents)}.\n"
    )
    if args.definitions:
        system += INTENT_DEFINITIONS + "\n"
    system += (
        f"Possible slot names: {', '.join(slot_names)}.\n"
        "A slot value must be the exact substring of the utterance that fills that slot. "
        "Only output slot spans that actually appear in the utterance; omit slots that are absent.\n"
        'Answer with strict JSON only, no other text: {"intent": "...", "slots": [{"name": "...", "value": "..."}]}'
    )

    os.makedirs(args.out_dir, exist_ok=True)
    counts = {}
    for split, fname in [("train", "train"), ("dev", "dev"), ("test", "test")]:
        rows = read_split(args.src, split, system)
        counts[split] = len(rows)
        with open(os.path.join(args.out_dir, f"{fname}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"intents": train_intents, "slot_names": slot_names},
              open(os.path.join(args.out_dir, "schema.json"), "w"), indent=1)
    print(f"train={counts['train']} dev={counts['dev']} test={counts['test']} "
          f"intents={len(train_intents)} slots={len(slot_names)}")


if __name__ == "__main__":
    main()
