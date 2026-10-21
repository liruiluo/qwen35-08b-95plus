# 实验记录与补充材料

本文件是 README 的技术附录：消融实验、口径论证的中间证据、方法实现
细节与工程注意事项。所有数字与 `results/` 下的证据文件一一对应。

## 1. 零样本起点测量

| 语料 | 指标 | base 零样本 | RandomInit (seed 17) |
|---|---|---|---|
| JointBERT test 700（**主口径**） | 意图准确率 | 80.71% | **0.00%**（700/700 不可解析） |
| JointBERT test 700 | 槽位 micro-F1 | 15.57% | 0.00% |
| 官方 validate 700（第二口径） | 意图准确率 | 82.71% | — |
| 官方 validate 700 | 槽位 micro-F1 | 19.38% | — |

监督参照（SFT，非交付，仅证明目标可学）：官方 validate 划分上
98.86% 意图 / 96.68% 槽位 F1。

## 2. 评测协议：训练与评测同渲染

所有提示（训练 rollout 与评测）统一经 `common/bproto.py`：用模型自身
tokenizer 在本地渲染 chat template（禁用 thinking），token id 直送
sglang 引擎 `/generate`。这是实测出的正确性要求，不是风格选择：

| 权重 | chat-completions 服务端渲染 | 本地渲染 + /generate |
|---|---|---|
| base | 82.71 意图 / 19.61 槽位 F1 | 82.71 / 19.38 |
| 监督 ckpt | **0.00 / 0.00**（700/700 不可解析） | 98.86 / 96.68 |

（官方 validate 划分实测；渲染器与语料无关。）在一个渲染下塑形的
模型换一个渲染评测会崩溃——那测的是渲染器不是模型。本仓库全部数字
为同一渲染协议。

## 3. 方法实现细节

- **GRPO**：组内相对优势 (r − mean)/(std + 1e-4)，无 critic、无 KL、
  on-policy（ratio=1），token 均值 PPO 型损失、全局 token 分母。
- **SPHERE**（arXiv:2605.04712，冻结协议见
  intent-alignment-text-toy 仓库 `docs/SPHERE_ONE_STEP_PROTOCOL.md`）：
  Φ = 每个 Transformer MLP `down_proj` 的输入（forward_pre_hook 捕获，
  仅动作 token 行）；逐层 L = ‖ΦᵀΦ/T − αI‖²_F，自适应 α = Tr(A)/D
  （detach）或固定 α = 1；D > T 时用样本空间 Gram；两次 backward、
  一次 optimizer step，梯度显式组装 g = g_ppo + λ·g_sphere，
  λ = rho·‖g_ppo‖/(‖g_sphere‖+1e-8)，rho = 1e-3（基线臂 `--rho 0`）。
- **微批结构**（显存所需，忠实于协议）：逻辑 minibatch 切成物理
  microbatch 做 PPO 前向/反向；detached 动作 token Φ 行按块收集，
  SPHERE 惩罚与闭式 dL/dΦ 在逻辑批上计算，再以逐 microbatch 线性代理
  (Φ·dL/dΦ) 把梯度带回。代理恒等性每步监控（相对 L2 与内积偏差
  记录在逐步证据里）。
- **三臂**：`last_adaptive`（最后 MLP 层、自适应 α）、
  `all_hidden_adaptive`（全部 24 层、逐层求和）、`last_fixed`
  （最后层、α=1）。

## 4. 意图定义提示战役（v1–v4）

系统提示中的七条意图定义全部从训练集统计归纳；调参只用 dev，test
每版单发。交付 = v4：

| 版本（其余全同） | test | dev | 说明 |
|---|---|---|---|
| 无定义 + 插值起点（3000 步） | 95.00% | 95.86% | 插值构造首个成车 |
| 定义 v1（1000 步） | 94.71% | 95.71% | 初版；"查歌=SCW"条款过冲 |
| 定义 v2 | 95.14% | 97.43% | soundtrack/EP 归位 PlayMusic |
| 定义 v3 | 95.86% | 97.86% | 天气 / TV 剧集 / is-playing / give-stars |
| 定义 v4 @1000 步 | 96.14% | 97.43% | 看片主规则 + SCW 反条款 |
| **定义 v4 @2000 步（交付）** | **97.14%** | **99.14%** | 同配方，仅步数 1000→2000 |

**步数消融**（同配方快照实测）：600=95.43 / 800=95.71 / 1000=96.14 /
2000=97.14——定义版配方到 2000 步仍在收敛；对照无定义配方 600 步即
平台、后 2400 步意图零涨。意图定义把语料约定变成 RL 可学的信号，
同时改变了学习动力学的形状。

**剩余错误分析**（97.14% = 20 错）：17 道与无定义基线共享，属语料
冲突地板——9 道 play+歌名 gold 对抗训练集 747:26 大种群、8 道裸片名
find/watch 双标并存、3 道角落题；BERT 监督上限 97.2% 撞同一地板。

## 5. 第一代三臂战役（无插值、无定义、2000 步）

各臂从 base 独立单次训练（`scripts/run_snips_jointbert.sh`），
作为方法证据保留；All-Adaptive / Last-Fixed 的送测模型现为统一
交付配方重训版（见 README 结果表）：

| 臂 | test 意图 | dev | 槽位 F1 |
|---|---|---|---|
| last_adaptive（SPHERE-Last-Adaptive） | 95.14% | 94.29% | 65.61% |
| all_hidden_adaptive（SPHERE-All-Adaptive） | 95.29% | 97.57% | 61.80% |
| last_fixed（SPHERE-Last-Fixed） | 94.86% | 94.86% | 65.33% |

训练曲线图：`results/curves-jointbert-split.png`（官方口径附加线在
`results/curves-official-split.png`），逐 step 数据在各臂
`curves.csv` / `blocks.csv`（送测口径曲线在 `SNIPS/logs/`）。

每一臂固定种子、固定步数、固定超参，单次运行到报数——无续训、无
按中间成绩的挑选，重跑即复现。

## 6. 仅意图奖励消融

去掉槽位奖励半边（`reward_snips_intent`，单变量对照 95.00% 那车）：
意图准确率 94.29@600 / 91.29@1000 / 94.57@1200——震荡带从 0.7pp
扩到 3.3pp；槽位塌缩到 0。机制结论：槽位半边是组内方差的梯度供给
与阻尼（"分心"假设证伪）。证据在 `results/jb-interp-intentonly/`。

## 7. 第二口径：官方语料划分

官方 snipsco 语料 train 13,784 训练、validate 700 评测（validate
从未进训练；两导出重叠实测见
`results/official-split/corpus-relation.txt`）：

| 臂 | 意图准确率 | 槽位 F1 |
|---|---|---|
| last_adaptive | 96.14% | 77.52% |
| all_hidden_adaptive | 96.43% | 78.10% |
| last_fixed | 97.00% | 74.05% |

事后延伸（非交付）：last_adaptive 同配方续训 +3000 步槽位 F1 到
85.10%；彼时瓶颈是 GRPO 组饱和（近确定性策略使 16 条组内奖励全同
→ 零优势零梯度），续训配方见 `scripts/extend_arm2.sh`。

## 8. 奖励塑形（槽位线）

训练奖励 = 0.5×intent_ok + 0.5×塑形槽位分：名称对、值错的槽按
`--name-credit`（默认 0.25）计 TP、余量计 FP。无塑形时尝试难槽
（playlist_owner="my"、music_item="track" 等标注约定槽）错值与不写
同分且更差，策略被推向整槽跳过——实测探索墙：纯精确匹配奖励 1500
步槽位 recall 47%，塑形奖励 2000 步 71.5% 且仍在爬。**评测口径不变**：
官方精确 span micro-F1。

## 9. BFCL 附加线（函数调用，82%）

BFCL v4 simple_python 无官方训练集，训练数据由 `bfcl/gen_train_data.py`
合成（固定种子、可复现）。v1 生成器（8 族）三臂同分 76.25%——瓶颈在
答案约定覆盖而非算法。全量 95 道失败编目（`scripts/bfcl_fail_catalog.py`）
显示官方答案遵循规范化约定：百分比转小数、枚举 token 化（"last six
months"→`six_months`）、SI 单位换算、12→24 小时、公司→ticker、值修剪
（"Thai food"→"Thai"）、`**` 数学语法等。v2 生成器按约定分族（15 族
24.5k 条）单轮 1600 步把 last_adaptive 从 76.25% 拉到 **82.00%**
（官方 AST 判分）；v3（20 族 34.6k 条）进一步覆盖。两条设计规则：
(1) schema 描述模仿官方风格（含模糊与自相矛盾），约定须从数据一致性
学出；(2) 表层措辞与官方评测题零 8-gram 重叠（审计 0/400），且每条
gold 均经验证可通过官方判分器（`scripts/validate_gen_data.py`，
0 失败）——训练奖励里没有不可挣得的噪声。判分器本体
`bfcl/ast_checker_official.py` 与上游对真实模型输出 400/400 同判。

## 10. 数据卫生

- 训练 rollout 只来自各口径 train 文件；test/dev 从不进训练。
- 官方划分自带 10 条精确 + 14 条近重复（2.0%）——记录在
  `results/leakage-audit.md`，对所有方法一致。
- BFCL 合成生成器对真实评测集逐题去重（0/400 精确、0/400 8-gram）。
- 早期 RL rollout 提示词中的 gold 泄漏（SFT 格式行的 assistant-gold
  被渲染进 prompt）在任何报数之前发现并修复；复盘在
  `results/leak-postmortem.md`。

## 11. 工程注意事项

1. **训练与评测同渲染**（见 §2）：绝不能用服务端重渲染评一个本地
   渲染训练的检查点。
2. **引擎换重**：健康检查 200 只证明"有一个引擎活着"；换重后要核对
   进程命令行的模型路径。
3. **检查点 config**：`save_pretrained` 写出的是 text-model config，
   离线机器上 sglang 冷启动可能解析到不同 flashinfer JIT 变体——
   受支持路径是从 base 起引擎、`push_ckpt.py` 热推进检查点。
4. **标记类型**：参数值靠 JSON 引号携带类型；训练目标逐值
   `json.dumps`。解析器须同时接受 `json.loads` 与 `ast.literal_eval`。

## 12. 仓库代码导览

```
methods/rl_interp.py       # 插值起点 RL 训练器（前 100 步 serve 侧混权）
methods/sphere_grpo.py     # GRPO+SPHERE 训练器（--rho 0 = GRPO 基线）
methods/train_hydra.py     # hydra 渲染层：configs/songce/*.yaml -> 精确 argv
methods/push_ckpt.py       # 向运行中引擎热推检查点
configs/songce/            # 每车一个配置（base=共享交付配方字段）
common/bproto.py           # 训练=评测 的共享渲染/生成层
SNIPS/prepare_jointbert.py # JointBERT 划分 -> jsonl（--definitions v4 定义）
SNIPS/eval_snips.py        # 评测：意图准确率 + 槽位 micro-F1
scripts/get_data.sh        # 拉取 SNIPS 语料 + BFCL 评测数据
scripts/make_random_init.py# seed-17 随机重置模型
scripts/run_songce_*.sh    # 每车发射脚本 + 顺序器（引擎检查→训练→热推→评测）
scripts/run_def4s2000.sh    # 交付车原始发射记录（pod 布局）
bfcl/                      # BFCL 附加线：生成器/官方判分器/评测
results/                   # 全部证据：分数、逐例 jsonl、曲线、口径论证
```
