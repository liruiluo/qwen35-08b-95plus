#!/usr/bin/env python3
"""Evaluate a chat model on BFCL v4 simple_python (400 items) with the
OFFICIAL AST-checker semantics (vendored in ast_checker_official.py).

Protocol: native tools API (function schemas passed as `tools`, the server
renders them into the prompt), BFCL persona system prompt, temp 0, thinking
disabled. The model answers with Qwen tool markup; we parse it and score with
the official checker. This is the strict number comparable to the official
leaderboard (modulo serving stack).

Usage:
  python eval_bfcl.py --base-url http://127.0.0.1:30000/v1 \
      --model /path/to/model \
      --questions BFCL_v4_simple_python.json \
      --answers BFCL_v4_simple_python.json.possible_answer \
      --out results.jsonl
"""
import argparse, ast as ast_mod, json, queue, re, threading, time, urllib.request

from ast_checker_official import ast_checker

SYSTEM = ("You are an expert in composing functions. You are given a question and a set of possible functions. "
          "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
          "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. "
          "You should only return the function calls in your response and nothing else.")


def to_oai_tool(fd):
    props = dict(fd["parameters"]["properties"])
    req = fd["parameters"].get("required", [])
    schema = {"type": "object", "properties": {}, "required": req}
    for k, v in props.items():
        schema["properties"][k] = {"type": v.get("type", "string"),
                                  "description": v.get("description", "")}
    return {"type": "function",
            "function": {"name": fd["name"].replace(".", "_"),
                         "description": fd.get("description", ""),
                         "parameters": schema}}


def call_api(base, model, user, tool, tries=3):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "tools": [tool], "tool_choice": "auto",
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            return r["choices"][0]["message"]["content"] or ""
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(2 * (i + 1))


def parse_markup(text):
    """Parse Qwen tool markup into [{name: {param: value}}] call dicts."""
    calls = []
    if not text:
        return calls
    for fm in re.finditer(r"<function=([\w.]+)>(.*?)</function>", text, re.S):
        name, body = fm.group(1), fm.group(2)
        params = {}
        for pm in re.finditer(r"<parameter=([\w.]+)>(.*?)</parameter>", body, re.S):
            k, v = pm.group(1), pm.group(2).strip()
            try:
                params[k] = json.loads(v)
            except Exception:
                try:
                    params[k] = ast_mod.literal_eval(v)
                except Exception:
                    params[k] = v
        calls.append({name: params})
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True,
                    help="official BFCL_v4_simple_python.json (jsonl)")
    ap.add_argument("--answers", required=True,
                    help="official possible_answer jsonl")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    QS = [json.loads(l) for l in open(args.questions) if l.strip()]
    ANS = {}
    for l in open(args.answers):
        if l.strip():
            d = json.loads(l)
            ANS[d["id"]] = d["ground_truth"]

    results = [None] * len(QS)
    q = queue.Queue()
    for i in range(len(QS)):
        q.put(i)

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            item = QS[i]
            user = item["question"][0][0]["content"]
            tool = to_oai_tool(item["function"][0])
            out = call_api(args.base_url, args.model, user, tool)
            calls = parse_markup(out)
            v = ast_checker(item["function"], calls, ANS[item["id"]])
            results[i] = {"id": item["id"],
                          "ok": 1 if v["valid"] else 0,
                          "error_type": v.get("error_type"),
                          "error": (v.get("error") or [""])[:1],
                          "n_calls": len(calls),
                          "raw": (out or "")[:600]}
            q.task_done()

    ths = [threading.Thread(target=worker) for _ in range(args.workers)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    n = len(results)
    ok = sum(r["ok"] for r in results)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"model={args.model}")
    print(f"n={n} wall={time.time()-t0:.1f}s")
    print(f"BFCL_SIMPLE_PYTHON_STRICT={ok/n*100:.2f}%  ({ok}/{n})")
    from collections import Counter
    et = Counter(r["error_type"] for r in results if not r["ok"])
    print("failure types:", dict(et))


if __name__ == "__main__":
    main()
