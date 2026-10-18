"""Shared B-protocol generation layer (train == eval, one source of truth).

Protocol B = the model is prompted through the SAME path both at training
rollout and at evaluation:

  1. the chat template is rendered LOCALLY with the model's own tokenizer
     (enable_thinking=False, add_generation_prompt=True, optional tools);
  2. the resulting token ids are sent to the sglang engine's /generate
     endpoint (`input_ids`, not `text`).

Why this matters (measured 2026-10-18, SNIPS validate n=700):
  - base Qwen3.5-0.8B:      A 82.71 intent / 19.61 slot-F1
                            B 82.71 intent / 19.38 slot-F1   (no difference)
  - SNIPS-supervised ckpt:  A  0.00 /  0.00 (700/700 unparsable)
                            B 98.86 / 96.68
A model whose behaviour was shaped under the local rendering COLLAPSES when
served through /v1/chat/completions (the server re-renders the prompt), so
any checkpoint produced by this repo MUST be evaluated under B. The base
model is protocol-insensitive; trained checkpoints are not. All numbers in
this repo are therefore B-protocol numbers, and this module is the single
implementation both evaluation scripts and the trainer's rollout path build
on.
"""
import json
import time
import urllib.request


def render_ids(tok, messages, tools=None):
    """Render messages (+ optional OpenAI-style tools) to input token ids."""
    kw = dict(tokenize=False, enable_thinking=False,
              add_generation_prompt=True)
    if tools:
        kw["tools"] = tools
    p = tok.apply_chat_template(messages, **kw)
    return tok(p, add_special_tokens=False)["input_ids"]


def generate(engine_url, input_ids, max_new_tokens=2048, temperature=0.0,
             tries=3, timeout=600):
    """Single-sample /generate against the sglang engine. Returns text."""
    body = {"input_ids": input_ids,
            "sampling_params": {"temperature": temperature, "top_p": 1.0,
                                 "max_new_tokens": max_new_tokens,
                                 "skip_special_tokens": True}}
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                engine_url.rstrip("/") + "/generate",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            t = r.get("text", "")
            return t if isinstance(t, str) else (t[0] if t else "")
        except Exception as e:  # noqa: BLE001
            last = e
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise last
