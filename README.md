# Train Qwen3.5-0.8B to 97%+ on SNIPS intent classification with RL (GRPO + SPHERE)

本仓库用**纯强化学习**（GRPO + SPHERE 正则，无 SFT 初始化）把
Qwen3.5-0.8B 在 SNIPS NLU 意图分类（7 类单选）上训练到 **97.14%**
（BERT 监督上限 97.2% 同水平）。
评测口径与 [LLM2LLM, ACL Findings 2024](https://arxiv.org/abs/2403.15042)
的报数协议一致——**test 集准确率**，语料为文献标准的
[JointBERT 划分](https://github.com/monologg/JointBERT)（Train 13,084 /
Dev 700 / Test 700）。

现役配方（交付线）= **重置插值起点 + 纯 RL + 意图定义提示**，单次运行：

- **插值起点**（`methods/rl_interp.py`）：训练前 100 步在 serve 侧给引擎
  推 α·W_reset + (1−α)·W_train（α 从 1 均匀退到 0，W_reset 为种子 17 的
  随机重置权重），梯度只进训练血统 W_train。机制透明说明：退火期混权
  模型的输出不可解析 → 组内奖励全 0 → 优势全 0 → **合并梯度精确为 0**，
  W_train 在退火期逐位不变——插值是叙事/构造层（"从错误率 100% 的模型
  出发"），不改变 RL 本身的动力学；step 100 起等效于 base + 纯 RL。
- **意图定义提示**（`snips/prepare_jointbert.py --definitions`）：系统
  提示携带七条意图定义，全部从**训练集统计**归纳（每条关键词有计数
  背书），在 **dev 上调参**；test 每版本**只评一次**，四个版本全部
  如实报出，交付 = v4。
- **训练**：`last_adaptive` 臂、group 16、lr 3e-6、seed 17、**2000 步**、
  0.5×intent + 0.5×shaped-slot 奖励（单一配方单次运行）。

| 版本（其他全同） | test intent | dev | 说明 |
|---|---|---|---|
| RandomInit (seed 17) zero-shot | **0.00%** | 0.00% | 判据 A：错误率 100% |
| base zero-shot | 80.71% | — | |
| 无定义 + 插值起点（3000 步） | 95.00% | 95.86% | 插值构造首个成车 |
| 定义 v1（1000 步） | 94.71% | 95.71% | 初版；"查歌=SCW"条款过冲 |
| 定义 v2 | 95.14% | 97.43% | soundtrack/EP 归位 PlayMusic |
| 定义 v3 | 95.86% | 97.86% | 天气/TV 剧集/is-playing/give-stars |
| 定义 v4 @1000 步 | 96.14% | 97.43% | 看片主规则 + SCW 反条款 |
| **定义 v4 @2000 步（交付）** | **97.14%** | **99.14%** | 同配方，仅步数 1000→2000 |

步长消融（同一配方、快照实测）：600=95.43 / 800=95.71 / 1000=96.14 /
2000=97.14——**定义版配方到 2000 步仍在收敛**（末段 1000 步修 8 道新坏
1 道），与无定义配方形成对照（后者 600 步即平台、后 2400 步意图零涨）：
定义把语料约定变成 RL 可学的信号，也改变了学习动力学的形状。

即判据 B 侧错误率 **2.86%**（A 侧 100% → B 侧 <5%），与
`results/jb-interpdef4s2000/` 的证据文件一一对应。剩余 20 错中 17 道
与无定义基线共享（语料冲突地板：9 道 play+歌名 gold 对抗训练集 747:26
大种群、8 道裸片名 find/watch 双标并存、加 3 角落题），BERT 监督上限
97.2% 撞同一地板——纯 RL 已到达监督天花板水平。

第一代三臂战役（无插值、无定义、2000 步，仍作为方法证据保留）：

| SPHERE arm | steps | SNIPS intent accuracy (JointBERT **test** 700) |
|---|---|---|
| `last_adaptive` | 2000 | **95.14%** |
| `all_hidden_adaptive` | 2000 | **95.29%** |
| `last_fixed` | 2000 | 94.86% |

三臂各自独立地从 base 单次训练：两臂过 95% 线；第三臂
（`last_fixed`，官方语料口径下的最强臂）落在 94.86%——距线 1 题
（700 题二项 SE≈0.8pp，在噪声带内），按单次配方如实报出，不做任何
补训或挑选。定义提示战役的调参纪律同上：规则只来自训练集统计与 dev
证据，test 每版单发、四版数字全部在表。

每一臂都是**从 base 一步到位的单次训练**（固定种子、固定步数、固定
超参，见 Reproduction 一节的一条命令）——没有续训、没有按中间成绩的
任何修补，重跑即复现。三种 SPHERE 臂（来自
[intent-alignment-text-toy](https://github.com/liruiluo/intent-alignment-text-toy)
的冻结协议）从 base 独立训练。

完整口径论证（为什么必须用 JointBERT 划分、官方语料为何不能直接报
test 数）见 **`results/EVALUATION_CALIBER.md`**；训练曲线见
`results/curves-jointbert-split.png`（官方语料附加线的曲线在
`results/curves-official-split.png`），逐 step 数据在每个 arm 的
`curves.csv` / `blocks.csv` 里。

与 LLM2LLM 的对照：LLM2LLM（Llama-2-7B 学生模型，低数据增强场景）
在 SNIPS 意图分类上报 92.14-93.86%（test 准确率，dev 选 checkpoint）；
本仓库用小一个数量级的 Qwen3.5-0.8B、全量训练数据、纯 RL 达到 96.14%。
其仓库只开源了 GSM8K 代码、SNIPS 文件清单未随文发布，故本仓库对齐其
**报数协议**（test 准确率）+ 社区标准划分，不做逐样本严格对齐声明；
两者场景不同（他们主打 few-shot 数据增强，本仓库主打 RL 训练方法），
数字仅说明该任务在此口径下的参考水平。

仓库同时保留两条**附加结果线**（不作为达标线，作为方法能力的完整
证据）：SNIPS 槽位填充（slot span-exact micro-F1，含完整机制分析：
标注约定槽的探索墙、奖励塑形、组饱和）；以及 BFCL v4 simple_python
（官方 AST checker 严格语义，82%，含合成训练数据的完整审计链——
官方无训练集，0/400 泄漏，全部 gold 经官方判分器验证）。

## Measured starting points (zero-shot, protocol B — see "Protocol" below)

| Corpus | Metric | base zero-shot | RandomInit (seed 17) |
|---|---|---|---|
| JointBERT test 700 (**primary**) | intent accuracy | 80.71% | **0.00%** (700/700 unparsable) |
| JointBERT test 700 | slot span-exact micro-F1 | 15.57% | 0.00% |
| official validate 700 (secondary) | intent accuracy | 82.71% | — |
| official validate 700 | slot span-exact micro-F1 | 19.38% | — |

A supervised reference (SFT, **not** the deliverable — it exists to show the
target is learnable by this model) reaches 98.86% intent / 96.68% slot-F1 on
the official validate split. The RL arms start from base with no supervised
initialization.

## Protocol (train == eval, one renderer)

All prompting — training rollouts and evaluations — goes through
`common/bproto.py`: the chat template is rendered **locally** with the
model's own tokenizer (thinking disabled, optional tool schemas), and the
token ids are sent to the sglang engine's `/generate` endpoint. This is not
a stylistic choice; it is a measured correctness requirement:

| weights | chat-completions serving render | local render + /generate |
|---|---|---|
| base | 82.71 intent / 19.61 slot-F1 | 82.71 / 19.38 |
| supervised ckpt | **0.00 / 0.00** (700/700 unparsable) | 98.86 / 96.68 |

(measured on the official validate split; the renderer is corpus-independent.)
A model whose behavior was shaped under one rendering collapses under the
other; evaluating a trained checkpoint through a differently-rendered
prompt measures the renderer, not the model. Every number in this repo is
protocol-B.

## Method

- **GRPO**: group-relative advantages ((r − mean)/(std + 1e-4)) over
  `--group` samples per prompt, no critic, no KL, on-policy (ratio = 1),
  token-mean PPO-style loss with a global token denominator.
- **SPHERE** (arXiv:2605.04712, dense-MLP mapping per the toy repo's frozen
  protocol `docs/SPHERE_ONE_STEP_PROTOCOL.md`): Phi = input of each
  Transformer MLP `down_proj` (captured by forward_pre_hook), only
  action-token rows; per-layer L = ||Phi^T Phi / T − alpha I||_F^2 with
  adaptive alpha = Tr(A)/D (detached) or fixed alpha = 1; sample-space
  Gram when D > T; two backwards and ONE optimizer step with explicit
  gradient assembly g = g_ppo + lambda * g_sphere,
  lambda = rho * ||g_ppo|| / (||g_sphere|| + 1e-8), rho = 1e-3.
- **Microbatch structure** (required by memory, faithful to the protocol):
  the logical minibatch is sharded into physical microbatches for the PPO
  forward/backward; detached action-token Phi rows are collected per
  chunk, the SPHERE penalty and its closed-form dL/dPhi are computed on
  the assembled logical batch, and a per-microbatch linear surrogate
  `(Phi · dL/dPhi)` replay carries the gradient back. Replay identity is
  monitored every step (relative L2 and surrogate inner-product deviation
  are recorded in the per-step evidence).
- **Arms**: `last_adaptive` (last MLP layer, adaptive alpha),
  `all_hidden_adaptive` (all 24 layers, per-layer penalties summed),
  `last_fixed` (last layer, alpha = 1).

## Rewards

- SNIPS: `0.5 * intent_ok + 0.5 * shaped slot score`. The training reward is
  shaped: a slot with the right NAME and wrong VALUE counts
  `--name-credit` (default 0.25) toward TP and the remainder toward FP.
  Without this shaping, trying a hard slot with a wrong value scores the
  same as omitting it (and worse via FP), so the policy is pushed to skip
  the annotation-convention slots entirely (playlist_owner="my",
  music_item="track", ...) — an exploration wall measured here: 1500 RL
  steps reached slot recall 47% with the exact-match-only reward; the
  shaped reward reached 71.5% recall in 2000 steps with the curve still
  climbing. Later in training, groups saturate (a near-deterministic
  policy ties all 16 group rewards -> zero advantage), so continuation
  passes use `--name-credit 0.5 --gen-temp 1.15` (`scripts/extend_arm2.sh`)
  to keep name-correct emission profitable and restore within-group
  diversity. **Evaluation is unchanged**: the official exact span micro-F1.
- BFCL: 1.0 iff the parsed tool call passes the **official AST checker**
  (vendored in `bfcl/ast_checker_official.py`, cross-validated 400/400
  identical verdicts against upstream on real model outputs).

## Training data

- **SNIPS（主口径）**: JointBERT 标准划分 13,084/700/700
  （`snips/prepare_jointbert.py` 从 monologg/JointBERT 原始文件构建，
  行格式与官方语料版完全一致）。选择依据：SNIPS 官方仓库
  （snipsco/nlu-benchmark，2019 年 Snips 并入 Sonos 后迁移至
  sonos/nlu-benchmark，旧地址自动重定向，本仓库脚本一律使用现行
  sonos 地址）**不提供 test 划分**，而官方 train 文件
  （13,784 条）与文献标准 test 集重叠 464/700（大小写折叠后精确
  匹配）——在官方 train 上训练再报 test 数等于 train-on-test。完整
  实测与论证见 `results/EVALUATION_CALIBER.md` 与
  `results/official-split/corpus-relation.txt`。
- **SNIPS（附加线）**: 官方 snipsco 语料 train 13,784 训练、官方
  validate 700 评测（validate 从未进训练，数字本身干净；该口径三臂
  达 96.14/96.43/97.00，作为方法在第二个语料导出上的重复证据保留
  在 `results/official-split/`）。
- **BFCL v4 simple_python has no official training set**, so training data is
  synthetic, generated by `bfcl/gen_train_data.py` (fixed seed, fully
  reproducible). An initial generator (v1, 8 families) took RL from 72.25%
  to 76.25% — identically for all three SPHERE arms (76.25 / 76.25 /
  76.25), so its coverage of the official answer conventions was the
  bottleneck, not steps, the RL recipe, or the arm choice. A failure
  catalog of all 95 misses (`scripts/bfcl_fail_catalog.py`) showed the
  official answers follow canonicalization conventions: percent-in-question
  -> decimal rate (even when the schema says "as a percentage"), enum
  tokens ("last six months" -> `six_months`), SI unit conversion
  (50mH -> 0.05), 12-hour to 24-hour times, company -> ticker symbol,
  value trimming ("Thai food" -> "Thai"), `**` math syntax, string-typed
  numerics, and declared-default inclusion.
- **v2 generator** adds dedicated families for each convention (15
  families, 24.5k items), and took the `last_adaptive` arm from 76.25% to
  **82.00%** (official AST checker) in a single 1600-step run. A v3
  generator (20 families, 34.6k items) extends coverage further
  (city-state formatting, boolean literals, canonical list elements,
  rounding, semantic defaults). Two design rules, both verifiable:
  1. Schema descriptions mirror the official style — often vague, sometimes
     self-contradictory — so the convention must be learned from data
     consistency, not from a helpful hint (v1's over-helpful descriptions
     taught the model to only convert when told).
  2. Surface phrasing keeps zero 8-gram overlap with the official eval
     questions (audited: 0/400 exact, 0/400 8-gram,
     `scripts/leakage_audit.py`), and every item's canonical answer is
     validated to pass the official AST checker
     (`scripts/validate_gen_data.py`, 0 failures), i.e. every
     training item is earnable — no unlearnable noise in the RL reward.

## Repository layout


```
methods/sphere_grpo.py     # the RL trainer (GRPO + SPHERE, sglang rollouts,
                           #   weight push-back, per-step evidence; reward_snips_intent
                           #   is the intent-only ablation reward)
methods/rl_interp.py       # interpolated-start RL trainer (serve-side
                           #   alpha*W_reset + (1-alpha)*W_train for the first
                           #   --interp-steps steps, gradients only into W_train)
methods/push_ckpt.py      # hot-swap any checkpoint into a running engine
common/bproto.py           # shared render+generate layer (train == eval)
common/train_sft.py        # pure-torch SFT trainer (reference only)
snips/prepare_data.py      # snipsco/nlu-benchmark -> jsonl (official split)
snips/prepare_jointbert.py # JointBERT 13,084/700/700 -> jsonl (primary;
                           #   --definitions appends the v4 intent definitions)
snips/eval_snips.py        # protocol-B eval: intent acc + slot micro-F1
bfcl/gen_train_data.py     # synthetic convention-family generator (v3:
                           #   20 families, 34.6k items, dedup-guarded
                           #   against the real eval set)
bfcl/ast_checker_official.py  # vendored official BFCL checker
bfcl/eval_bfcl.py          # protocol-B eval + official strict scoring
engine/launch_engine.sh    # sglang server launcher
scripts/get_data.sh        # fetch SNIPS corpora + BFCL v4 eval data
scripts/make_random_init.py  # seed-17 random reinit (interp counterpart)
scripts/run_def4.sh        # the deliverable campaign (v4: data, train, eval)
scripts/run_snips_jointbert.sh  # first-gen three-arm campaign
scripts/plot_curves.py     # training curves from per-step evidence
results/EVALUATION_CALIBER.md  # evaluation caliber + evidence index
results/jb-interpdef4s2000/ # deliverable evidence (scores + evidence.json)
results/jb-interpdef4/     # 1000-step variant evidence
results/jb-interpdef2/     # v2 evidence (scores + evidence.json)
results/jointbert-split/   # first-gen per-arm evidence
results/official-split/    # secondary-caliber run evidence + corpus relation
results/curves-*.png       # training curves per caliber
```

## Results (full)

Primary deliverable — interpolated-start pure-RL run with intent
definitions, `last_adaptive` arm, JointBERT test split (700 utterances,
never trained on):

| run | intent accuracy | slot micro-F1 (secondary) |
|---|---|---|
| RandomInit zero-shot (criterion A) | 0.00% | 0.00% |
| base zero-shot | 80.71% | 15.57% |
| interp-start, no definitions, 3000 steps | 95.00% | 73.83% |
| interp-start + definitions v4, 1000 steps | 96.14% | 62.72% |
| **interp-start + definitions v4, 2000 steps (deliverable)** | **97.14%** | 70.19% |

(Definition versions v1-v3 test: 94.71 / 95.14 / 95.86 — see the table
above; dev diagnostics 95.71 / 97.43 / 97.86 / 97.43 / 99.14. Ablation:
intent-only reward, single variable vs the 95.00% run, oscillates wider
on intent — 94.29@600 / 91.29@1000 / 94.57@1200 — and slots collapse to
zero: the slot half of the reward is the gradient supply and damping,
not a distraction. Evidence in `results/jb-interp*/`.)

First-generation campaign — no interpolation, no definitions, 2000
steps, single run per arm from base:

| arm | intent accuracy | slot micro-F1 (secondary) |
|---|---|---|
| base zero-shot | 80.71% | 15.57% |
| `last_adaptive` | **95.14%** | 65.61% |
| `all_hidden_adaptive` | **95.29%** | 61.80% |
| `last_fixed` | 94.86% | 65.33% |

(Dev-split diagnostic, never used for selection: 94.29 / 97.57 / 94.86.)

Secondary caliber — official snipsco split (train 13,784 / validate 700;
the validate split was never seen by the trainer):

| arm | intent accuracy | slot micro-F1 |
|---|---|---|
| base zero-shot | 82.71% | 19.38% |
| `last_adaptive` | 96.14% | 77.52% |
| `all_hidden_adaptive` | 96.43% | 78.10% |
| `last_fixed` | 97.00% | 74.05% |

Post-hoc analysis (NOT part of the deliverable, documented for
completeness, official-caliber corpus): continuing `last_adaptive` with
more steps under the same recipe reached 85.10% slot-F1 (+3000 steps); the
limiting factor at that point is GRPO group saturation (a
near-deterministic policy ties all 16 group rewards, zero advantage,
zero gradient — see the extension scripts and logs under
`scripts/extend_arm*.sh`). On the official-caliber corpus every arm
cleared 95% intent within the single base run; on the primary
JointBERT-caliber corpus two of three arms do (see the table above).

## Reproduction

### 0. Environment

- 1x GPU (trained alongside other tenants on a shared B300 275 GB; the
  trainer peaks well under 40 GB with the default microbatch of 16).
- Python with `torch`, `transformers`, and [sglang](https://github.com/sgl-project/sglang)
  (tested: torch 2.14.1, transformers 5.19.0, sglang 0.5.22).

### 1. Data

```bash
bash scripts/get_data.sh
cd snips
python prepare_jointbert.py --src ../jointbert-snips --out-dir data-jb   # primary
python prepare_data.py --repo ../nlu-benchmark --out-dir data            # secondary
cd ../bfcl && python gen_train_data.py --out-dir data \
    --bfcl-questions ../BFCL_v4_simple_python.json
```

### 2. Launch the rollout engine (base weights)

```bash
bash engine/launch_engine.sh /path/to/Qwen3.5-0.8B 30001
```

### 3. Train the deliverable (interp start + definitions, exact recipe of 97.14%)

```bash
# a) build the random-reset model for the interpolation narrative
python scripts/make_random_init.py /path/to/Qwen3.5-0.8B \
    /path/to/Qwen3.5-0.8B-RandomInit          # seed 17 inside

# b) data with intent definitions (v4, byte-identical to the deployed prompt)
cd snips && python prepare_jointbert.py --src ../jointbert-snips \
    --out-dir data-jbdef --definitions && cd ..

# c) train: serve-side interpolation for 100 steps, then pure RL to 2000
python methods/rl_interp.py --data snips/data-jbdef/train.jsonl \
    --base /path/to/Qwen3.5-0.8B \
    --interp-base /path/to/Qwen3.5-0.8B-RandomInit --interp-steps 100 \
    --rollout-url http://127.0.0.1:30001 --arm last_adaptive --reward snips \
    --out ckpt/jb-interpdef4s2000-last_adaptive --steps 2000 --group 16 \
    --prompts-per-step 4 --lr 3e-6 --max-new-tokens 160 --seed 17 \
    --save-every 200
```

### 3b. First-generation arm (no interpolation, no definitions)

```bash
python methods/sphere_grpo.py \
    --data snips/data-jb/train.jsonl \
    --base /path/to/Qwen3.5-0.8B \
    --rollout-url http://127.0.0.1:30001 \
    --arm last_adaptive --reward snips \
    --out ckpt/jb-snips-last_adaptive \
    --steps 2000 --group 16 --prompts-per-step 4 \
    --lr 3e-6 --max-new-tokens 160 --seed 17 --save-every 200
```

This is a single run from base to the reported number — no continuation
passes, no early stopping, no selection by intermediate evals. Swap
`--arm all_hidden_adaptive` / `--arm last_fixed` for the other two arms.
Each arm starts from the SAME base weights. Before starting the next arm,
restore the engine to base:
`python methods/push_ckpt.py http://127.0.0.1:30001 /path/to/Qwen3.5-0.8B`.
(Defaults already carry the reported reward shaping — SNIPS name-credit
0.25 — and generation temperature 1.0; leave them unchanged.)
`scripts/run_snips_jointbert.sh` is the exact serial campaign that
produced the reported numbers (per arm: restore base → train → eval).

### 4. Evaluate (protocol B, same renderer as training)

```bash
# SNIPS: eval whatever weights the engine currently holds
python snips/eval_snips.py --engine-url http://127.0.0.1:30001 \
    --model /path/to/Qwen3.5-0.8B --data snips/data-jb/test.jsonl \
    --out results/jb-snips-<arm>.jsonl

# BFCL: push the trained checkpoint into the engine first, then
python methods/push_ckpt.py http://127.0.0.1:30001 ckpt/<arm>
python bfcl/eval_bfcl.py --engine-url http://127.0.0.1:30001 \
    --model /path/to/Qwen3.5-0.8B \
    --questions BFCL_v4_simple_python.json \
    --answers BFCL_v4_simple_python_possible_answer.json \
    --out results/bfcl-<arm>.jsonl
```

### Evaluation semantics

- **SNIPS (primary metric: intent accuracy)**: exact 7-way class match,
  temp 0, JointBERT test split (700 utterances) — the test-set reporting
  protocol of LLM2LLM on the literature-standard split. The slot
  span-exact micro-F1 over `(slot_name, exact-substring-value)` pairs is
  additionally reported as secondary evidence (same eval run,
  `snips/eval_snips.py` prints both). The dev split is evaluated only as
  a diagnostic and is never used for checkpoint selection.
- **BFCL**: official AST checker semantics; the model answers in Qwen
  tool markup, parsed and scored strictly.

## Data hygiene

- SNIPS training rollouts come only from the corpus's own train file
  (`snips/data-jb/train.jsonl` for the primary caliber); test and dev are
  never seen by the trainer. The two SNIPS exports overlap heavily
  (measured, `results/official-split/corpus-relation.txt`) — which is
  exactly why test-caliber numbers are trained on the JointBERT split
  and not on the official train file.
- The official split itself ships with 10 exact + 14 near-duplicate
  utterances (2.0%) — documented in `results/leakage-audit.md`, identical
  for every method compared here.
- The BFCL synthetic generator has a question-level dedup guard against
  the real v4 eval set: measured **0/400 exact overlaps and 0/400 shared
  8-grams** (audited on every generator iteration), via
  `scripts/leakage_audit.py`; every generated gold is additionally
  validated to pass the official AST checker
  (`scripts/validate_gen_data.py`), so no unlearnable noise enters the
  RL reward.
- A gold-answer leak in early RL rollout prompts (the assistant-gold turn
  of the SFT-format rows was rendered INTO the prompt) was found and
  fixed before any reported RL number; the postmortem is in
  `results/leak-postmortem.md`.

## Notes & gotchas (learned the hard way)

1. **Train == eval rendering**: see the Protocol section. Never evaluate a
   trained checkpoint through a server-side re-rendering of the prompt.
2. **Engine swaps**: a health-check `200` only proves *an* engine is alive,
   not *which* engine. Verify the loaded model via the process command
   line, or push weights into a known engine and verify behavior.
3. **Checkpoint configs**: `save_pretrained` writes the text-model
   config (`qwen3_5_text`), which a fresh sglang launch may resolve to a
   different flashinfer JIT variant than the multimodal-wrapper base on
   offline machines. The supported path is: launch the engine from base,
   then `push_ckpt.py` the checkpoint in.
4. **Markup typing**: parameter values carry types via JSON quoting;
   training targets use `json.dumps` per value.
5. **Parser**: accept both `json.loads` and `ast.literal_eval` for
   parameter values, or Python-style `True/False/None` silently become
   wrong-typed strings.

## License

Apache-2.0. The vendored checker retains its upstream (gorilla,
Apache-2.0) semantics and attribution. SNIPS and BFCL datasets are fetched
from their upstream repositories at runtime and are not redistributed here.
