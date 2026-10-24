# 评测口径与复现证据（Evaluation caliber）

本文件记录交付数字的评测口径（在哪个划分上、按什么协议报数）、
口径选择的依据、以及支持复现的全部证据文件索引。
最后更新：2026-10-21（三算法 + 基线统一配方重训收官；五模型证据齐）。

## 1. 任务范围

测试大纲 4.2.2.2 将本场景称为**指令认知识别**（SNIPS 数据集），
其评价指标为 7 类**意图分类**准确率。这与
[LLM2LLM (ACL Findings 2024)](https://arxiv.org/abs/2403.15042) 在 SNIPS
上的任务范围一致：LLM2LLM 同样只做意图分类，不做槽位填充。槽位
span-exact micro-F1 在本仓库作为附加证据线保留，不作为达标线。

## 2. 报数口径：标准三分划分上的 test 集

**主口径 = JointBERT 标准 13,084 / 700 / 700 划分上的 test 集（700 条）
意图准确率。**

依据链：

1. **LLM2LLM 报的是 test 准确率**。论文原文（arXiv 2403.15042）：
   > "To measure the efficacy of LLM2LLM, we fine-tune the student model
   > using samples of different sizes from each dataset and evaluate on
   > the validation sets"
   > "For all other datasets, we record the **test set performance** of
   > the checkpoint that performs best on the development set."

   即：dev 选 checkpoint，**test 上报最终数**。

2. **SNIPS 官方仓库没有 test 划分**。`snipsco/nlu-benchmark` 的
   `2017-06-custom-intent-engines/<Intent>/` 目录只提供
   `train_<Intent>.json`、`train_<Intent>_full.json`、
   `validate_<Intent>.json` 三种文件（逐目录核验过），全库不存在
   test 文件。

3. **文献标准 test 划分 = JointBERT 三分**。学术界（JointBERT、
   Stack-Propagation 等）通用的 SNIPS 划分为 Train 13,084 / Dev 700 /
   Test 700，文件直接内置于 [monologg/JointBERT](https://github.com/monologg/JointBERT)
   仓库 `data/snips/` 下（seq.in / seq.out / label 格式）。

## 3. 为什么必须换用 JointBERT 划分重训（而非直接在旧训练上评测）

两个语料导出**同源但不同份**，重叠关系全部实测
（`results/official-split/corpus-relation.txt`，大小写折叠后精确匹配）：

| 重叠 | 条数 | 意图标签一致率 |
|---|---|---|
| JointBERT train ∩ 官方 train (13,784) | 8,595 | 100% (8595/8595) |
| JointBERT test ∩ 官方 train (13,784) | **464 / 700** | 100% |
| JointBERT dev ∩ 官方 validate (700) | 352 / 700 | 100% |
| JointBERT test ∩ 官方 validate (700) | 1 / 700 | — |

（重叠处意图标签 100% 一致 → 两份导出来自同一标注源，差异只在
导出与文本规范化。）

**推论**：若用官方 train（13,784）训练、在 JointBERT test 上报数，
等于在 2/3 的考卷上训练过——train-on-test。因此主口径交付数字是
**在 JointBERT train（13,084）上重新单次训练、在 JointBERT test 上
评测**得到的；旧官方语料训练的三个算法模型（在官方 validate 上的
96.14 / 96.43 / 97.00）降级为附加证据线：那组数字本身干净
（validate 从未进训练），但不是文献可比的 test 口径。

## 4. 单次运行政策（复现性承诺）

每个交付数字 = **一条命令从 base 到最终数**：

- 固定随机种子（seed 17）、固定步数（2000）、固定超参（group 16、
  prompts-per-step 4、lr 3e-6（三算法；GRPO 基线为 1.5e-7）、
  max-new-tokens 160、save-every 200，
  其余取默认值：name-credit 0.25、gen-temp 1.0、microbatch 16）；
- 无续训/补训，无按中间成绩的任何修补，无按中间评测挑 checkpoint；
- 评测与训练同渲染协议（同一 chat 模板与生成参数，由
  common/bproto.py 统一实现）。

完整命令见 README「复现训练」一节；
`scripts/run_songce_*.sh` 是四个模型（三算法 + 基线）串行训练的原始脚本
（训练服务器布局）。

## 5. 证据文件索引

```
results/
  EVALUATION_CALIBER.md            ← 本文件（口径）
  sphere-last-adaptive/            ← SPHERE-Last-Adaptive（97.14%）单次运行证据
  sphere-all-adaptive/             ← SPHERE-All-Adaptive（96.00%）同构证据
  sphere-last-fixed/               ← SPHERE-Last-Fixed（96.43%）同构证据
  grpo-baseline/                   ← GRPO 基线（95.29%）同构证据
  randominit/                      ← 训练前模型（A）test 证据
  （前四者：evidence.json + final-{dev,test}.jsonl 逐例 + 分数文件；
    randominit：final-test.jsonl 逐例 + 分数文件）
  jointbert-split/                 ← 更早一代同划分运行记录（数据变体较旧，
                                     保留作附加证据线；交付数字全部来自上面
                                     五个模型目录）
    <arm>/evidence.json           运行配置 + 每 step 记录（g_ppo、
                                   g_sphere、lambda、mean_reward、
                                   action_tokens、replay 校验量）
    <arm>/trainer.log              训练器全程 stdout 日志
    <arm>/eval-{test,dev}.jsonl    test/dev 逐例输出（含预测 JSON）
    <arm>/eval-{test,dev}-score.txt 分数（逐字）
    base-test-score.txt            base 零样本 test 起点
  curves-jointbert-split.png       早期一代训练曲线图
  official-split/                  ← 附加证据线（官方 validate 口径）
    （各算法同构证据 + base 零样本 + 语料关系取证）
  curves-official-split.png
  leakage-audit.md                 ← 官方划分数据卫生审计记录
  leak-postmortem.md               ← 早期 gold 泄漏事后分析
```

## 6. 与 LLM2LLM 数字的关系（如何对照、如何不对照）

- LLM2LLM 的 SNIPS 意图分类报 92.14–93.86%（低数据场景：每类
  10–20 条种子样本 + LLM 数据增强，学生模型 Llama-2-7B）。
- 其仓库只开源了 GSM8K 代码，SNIPS 的确切文件清单未随文发布，
  因此**不做逐样本严格对齐声明**；本仓库采用其报数"协议"
  （test 集准确率）+ 社区标准划分（JointBERT）。
- 场景差异（LLM2LLM 为 few-shot 数据增强，本仓库为全量数据 +
  强化学习训练，模型还小一个数量级），数字只作参考水位，
  不作严格可比声明。

## 7. 数据卫生（两口径通用）

- 训练 rollout 提示只来自对应口径的 train 文件；test/dev 从不进训练。
- JointBERT 划分内部的固有近重复（其 train 与 test 间的少量相似
  句，社区划分原生属性，所有用该划分的论文相同）不在此重复审计；
  官方划分侧的 10 精确 + 14 近重复（2.0%）审计见
  `results/leakage-audit.md`。
- 早期 RL rollout 提示里的 gold 泄漏在报告任何 RL 数字前发现并
  修复，事后分析见 `results/leak-postmortem.md`。
