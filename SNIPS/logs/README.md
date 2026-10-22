# SNIPS/logs —— 训练曲线与评测汇总

每个模型一个子目录，含 `curves.png` 训练曲线图 +
`curves.csv` 逐 step 数据（生成脚本 `scripts/plot_songce_curves.py`）；
`curves-all.png` 为全部已训模型的对比总图；
`evaluate.py` 的汇总输出写在本目录 `evaluate-summary.txt`。

| 子目录 | 内容 | 说明 |
|---|---|---|
| `SPHERE-Last-Adaptive/` | `curves.png` + `curves.csv`（step, mean_reward, roll50）+ `training-log.jsonl`（逐步训练证据） | 交付模型，2000 步 |
| `SPHERE-All-Adaptive/` | `curves.png` + `curves.csv` + `training-log.jsonl` | 统一配方重训 |
| `SPHERE-Last-Fixed/` | `curves.png` + `curves.csv` + `training-log.jsonl` | 统一配方重训 |
| `GRPO-baseline/` | `curves.png` + `curves.csv` + `training-log.jsonl` | 基线模型 |

RandomInit（训练前模型）无训练曲线，不建子目录；零样本错误率 100%。

## 高效性判据（N<M）

大纲 4.2.2.4(3) 第 4 步要求"错误率达到 5% 所需训练迭代次数"的对比。
各模型训练时每 200 步保存检查点，逐检查点加载权重后跑一次 test 集评测
（`scripts/nm_sweep.sh`，数据 `nm-sweep.csv`），得到错误率-步数曲线。
错误率首次不超过各阈值的训练步数：

| 错误率阈值 | SPHERE-Last-Adaptive | SPHERE-All-Adaptive | SPHERE-Last-Fixed | GRPO 基线 |
|---|---|---|---|---|
| 5% | 200 | 200 | 200 | 200 |
| 4.5% | 600 | 600 | 600 | 未达到 |
| 4% | 600 | 800 | 800 | 未达到 |

大纲判据线（5%）上四个模型均在首个检查点（200 步）已达到，无法区分
N 与 M；高效性对比取更严的错误率线：4.5% 线上三种 SPHERE 算法均在
600 步达到（N=600），GRPO 基线在整个 2000 步训练中从未达到（全程
最低 4.57%），即 N < M；4% 线上同样全部达到而基线未达到。

## curves.csv 字段

- `step`：训练步（0 起）
- `mean_reward`：该步 4 prompt × 16 rollout 的平均训练奖励
  （0.5×intent_ok + 0.5×shaped-slot）
- `roll50`：最近 50 步 mean_reward 滚动均值（平滑用）
