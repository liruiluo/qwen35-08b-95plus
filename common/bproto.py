"""Generation helpers: render the chat template locally and call the
sglang engine's /generate endpoint with token ids.

The chat template is rendered locally with the model's own tokenizer
(enable_thinking=False, add_generation_prompt=True, optional tools) and the
resulting token ids are sent as `input_ids` (not `text`). Training rollouts
and evaluation must use the SAME rendering: a checkpoint trained under this
local rendering breaks when the prompt is re-rendered server-side (e.g. via
/v1/chat/completions). SNIPS/eval_snips.py uses these functions;
methods/sphere_grpo.py renders prompts with identical parameters.
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
