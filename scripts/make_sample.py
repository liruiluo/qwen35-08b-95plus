#!/usr/bin/env python3
"""把 SNIPS/dataset/test.jsonl 的前若干条展开成可读的 sample.json（给演示时点开看）。

原始文件是一行一条 JSONL，一行约 2.9k 字符（其中 2.5k 是 system 提示词），
直接点开在编辑器里是一堵墙。这个脚本输出一个格式化的小文件：
system 只保留开头并注明长度，每条样本展开成 user + answer。

用法（仓库根目录）：
  python scripts/make_sample.py            # 前 2 条 -> SNIPS/dataset/sample.json
  python scripts/make_sample.py -n 3
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "SNIPS", "dataset", "test.jsonl")
OUT = os.path.join(ROOT, "SNIPS", "dataset", "sample.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", "--count", type=int, default=2)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    sys_msg = rows[0]["messages"][0]["content"]
    samples = []
    for r in rows[: args.count]:
        samples.append({
            "user": r["messages"][1]["content"],
            "answer": json.loads(r["messages"][2]["content"]),
        })
    doc = {
        "说明": f"{os.path.relpath(args.data, ROOT)} 的前 {len(samples)} 条"
                "（原始文件一行一条 JSONL），按角色展开便于阅读",
        "system": " ".join(sys_msg.split())[:220] + f" …（共 {len(sys_msg)} 字符，"
                  "完整内容即文件里的第一条 system 消息）",
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[sample] {len(samples)} 条 -> {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
