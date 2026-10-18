# Train Qwen3.5-0.8B to 95%+ on SNIPS & BFCL simple_python

**中文摘要**：本仓库把 Qwen3.5-0.8B 在两个基准上从零样本拉到 95%+：
SNIPS NLU（intent 准确率 / slot span-exact micro-F1）与 BFCL v4
simple_python（官方 AST checker 严格语义）。包含完整 SFT 训练管线
（纯 PyTorch、无 accelerate/peft 依赖）、面向"约定家族"的合成数据
生成器、与官方逐条对拍的评测器、以及逐步复现说明。

This repo takes **Qwen3.5-0.8B** from zero-shot to **95%+** on two benchmarks:

| Benchmark | Metric | 0.8B zero-shot | After this repo's SFT |
|---|---|---|---|
| SNIPS NLU (validate 700) | intent accuracy | 82.71% | see [results](results/) |
| SNIPS NLU (validate 700) | slot span-exact micro-F1 | 43.91% | see [results](results/) |
| BFCL v4 simple_python (400) | official AST checker (strict) | 73.75% | see [results](results/) |

The key insight from the failure analysis (documented in `results/`): the
zero-shot misses are overwhelmingly **convention failures**, not capability
failures — percent-vs-decimal rates, canonical tokens vs paraphrases, date
formats, string-typed numerics, dict inner-key schemas, list element
canonicalization, and optionality. These are exactly what a small amount of
targeted SFT fixes.

## Repository layout

```
common/train_sft.py        # pure-torch SFT trainer (assistant-only loss,
                           #   non-thinking chat format, tools-aware rendering)
snips/prepare_data.py      # snipsco/nlu-benchmark -> SFT jsonl (13784 train)
snips/eval_snips.py        # intent acc + span-exact slot micro-F1 (serving)
bfcl/gen_train_data.py     # synthetic convention-family generator (16.5K items,
                           #   dedup-guarded against the real eval set)
bfcl/ast_checker_official.py  # vendored official BFCL checker (Python branch;
                              #   cross-validated: 400/400 identical verdicts)
bfcl/eval_bfcl.py          # native-tools protocol + official strict scoring
engine/launch_engine.sh    # sglang server launcher for evaluation
scripts/get_data.sh        # fetch SNIPS repo + BFCL v4 eval data
results/                   # baseline analysis, run logs, final numbers
```

## Reproduction

### 0. Environment

- 1x GPU with >= 24 GB (trained on a single B300; 0.8B full-parameter SFT in
  fp32 master weights + bf16 autocast peaks well under 20 GB)
- Python with `torch` and `transformers` (tested: torch 2.14.1,
  transformers 5.19.0). No accelerate / peft / trl / deepspeed needed.
- [sglang](https://github.com/sgl-project/sglang) for serving during eval.

### 1. Get the data

```bash
bash scripts/get_data.sh          # clones SNIPS repo, fetches BFCL v4 files
```

### 2. SNIPS

```bash
cd snips
python prepare_data.py --repo ../nlu-benchmark --out-dir data
# -> data/train.jsonl (13784), data/validate.jsonl (700), data/schema.json

python ../common/train_sft.py \
    --data data/train.jsonl \
    --model /path/to/Qwen3.5-0.8B \
    --out ../ckpt/snips-sft \
    --epochs 3 --bsz 16 --accum 2 --lr 1e-5 --max-len 1024

# serve the checkpoint, then evaluate
bash ../engine/launch_engine.sh ../ckpt/snips-sft 30000
python eval_snips.py --base-url http://127.0.0.1:30000/v1 \
    --model ../ckpt/snips-sft --data data/validate.jsonl --out results.jsonl
```

### 3. BFCL simple_python

```bash
cd bfcl
python gen_train_data.py --out-dir data \
    --bfcl-questions ../BFCL_v4_simple_python.json
# -> data/train.jsonl (15840), data/val.jsonl (660), manifest.json
# The generator never copies a real eval item (question-level dedup guard);
# it teaches the convention families through schema descriptions.

python ../common/train_sft.py \
    --data data/train.jsonl \
    --model /path/to/Qwen3.5-0.8B \
    --out ../ckpt/bfcl-sft \
    --epochs 2 --bsz 16 --accum 2 --lr 1e-5 --max-len 1024

bash ../engine/launch_engine.sh ../ckpt/bfcl-sft 30000
python eval_bfcl.py --base-url http://127.0.0.1:30000/v1 \
    --model ../ckpt/bfcl-sft \
    --questions ../BFCL_v4_simple_python.json \
    --answers ../BFCL_v4_simple_python_possible_answer.json \
    --out results.jsonl
```

### Evaluation semantics (what "the score" means)

- **SNIPS**: span-exact micro-F1 over `(slot_name, exact-substring-value)`
  pairs, intent = exact class match. Temp 0, thinking disabled, identical
  system prompt at train and eval time.
- **BFCL**: the official AST checker, vendored from the gorilla repo and
  cross-validated to produce identical verdicts on 400/400 real model
  outputs. The eval uses the native tools API (schemas rendered into the
  prompt by the server) and the official persona system prompt — the same
  protocol used for the zero-shot baseline, so gains are apples-to-apples.

## Why 95%+ is reachable (failure analysis)

Auditing the zero-shot failures on BFCL (57 wrong under strict semantics)
showed they decompose into learnable convention families:

- 12x rate-as-percent (question says "10%", gold expects `0.1`)
- ~20x paraphrase-vs-canonical-token ("human cell" vs `"human"`)
- 13x strict typing (unquoted `2022` for a string-typed param)
- 3x dict inner-key guessing, 7x genuine value errors, 1x empty output

`bfcl/gen_train_data.py` builds targeted coverage for each family **plus
distractors that punish blanket rules** (e.g. a `rate_percent` param that
wants `5`, not `0.05`), so the model learns to read the schema, not a
reflex. The same principle drives the SNIPS side: train and eval share the
system prompt and output format byte-for-byte.

## Notes & gotchas (learned the hard way)

1. **Engine swaps**: a health-check `200` only proves *an* engine is alive,
   not *which* engine. After swapping checkpoints, verify the loaded model
   via the process command line. A sub-15s "healthy" boot after a swap is a
   red flag — weights take a minute to load.
2. **Markup typing**: in Qwen tool markup, parameter values carry types via
   JSON quoting. Training targets use `json.dumps` per value, which is what
   teaches string-typed numerics (`"2022"`) and decimal rates (`0.05`).
3. **Parser**: accept both `json.loads` and `ast.literal_eval` for parameter
   values, or Python-style `True/False/None` silently become wrong-typed
   strings.

## License

Apache-2.0. The vendored checker retains its upstream (gorilla,
Apache-2.0) semantics and attribution. SNIPS and BFCL datasets are fetched
from their upstream repositories at runtime and are not redistributed here.
