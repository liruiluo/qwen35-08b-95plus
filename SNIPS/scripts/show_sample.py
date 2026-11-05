#!/usr/bin/env python3
"""Print SNIPS test samples as "utterance + answer" instead of raw JSON lines.

A raw line of SNIPS/dataset/test.jsonl is ~2.9k characters, of which ~2.5k is
the system prompt, so `head -n 1` on a projector shows a wall of text and hides
the part that matters (the user utterance and the gold answer). This script
prints one sample per block: utterance, intent, slots, and the gold JSON.

Usage (from the repo root):
  python SNIPS/scripts/show_sample.py                  # first test sample
  python SNIPS/scripts/show_sample.py -n 2             # first two
  python SNIPS/scripts/show_sample.py --data SNIPS/dataset/train.jsonl -i 100
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA = os.path.join(ROOT, "SNIPS", "dataset", "test.jsonl")


def render(row, idx, total, path, show_system):
    m = row["messages"]
    system, user, answer = m[0]["content"], m[1]["content"], m[2]["content"]
    gold = json.loads(answer)
    lines = [
        f"样本 {idx + 1} / {total}    （{os.path.relpath(path, ROOT)}）",
        "",
        f"用户指令   {user}",
        "",
        f"标准意图   {gold['intent']}",
    ]
    slots = gold.get("slots") or []
    if slots:
        width = max(len(s["name"]) for s in slots)
        lines.append("标准槽位   " + slots[0]["name"].ljust(width) + " = " + slots[0]["value"])
        for s in slots[1:]:
            lines.append("           " + s["name"].ljust(width) + " = " + s["value"])
    else:
        lines.append("标准槽位   （无）")
    lines += ["", f"标准答案   {answer}", ""]
    if show_system:
        lines.append(f"题面       system 消息 {len(system)} 字符：任务说明 + 7 类意图定义 "
                     f"+ 39 个槽位名（完整内容见 SNIPS/data-jbdef4/schema.json）")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DEFAULT_DATA, help="jsonl 文件（默认 test.jsonl）")
    ap.add_argument("-i", "--index", type=int, default=0, help="从第几条开始（0 起）")
    ap.add_argument("-n", "--count", type=int, default=1, help="看几条（默认 1）")
    ap.add_argument("--no-system", action="store_true", help="不打印题面说明行")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    start = max(0, args.index)
    for i in range(start, min(start + args.count, len(rows))):
        if i > start:
            print("\n" + "─" * 72 + "\n")
        print(render(rows[i], i, len(rows), args.data, not args.no_system))


if __name__ == "__main__":
    main()
