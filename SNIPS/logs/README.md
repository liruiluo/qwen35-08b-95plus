# SNIPS/logs —— 训练曲线与评测汇总

每个模型一个子目录（`curves.png` 训练曲线图 + `curves.csv` 逐 step
数据；生成脚本 `scripts/plot_songce_curves.py`）；`curves-all.png`
为各模型对比总图；`evaluate.py` 的汇总输出写在本目录
`evaluate-summary.txt`。

| 子目录 | 内容 | 说明 |
|---|---|---|
| `SPHERE-Last-Adaptive/` | `curves.png` + `curves.csv`（step, mean_reward, roll50）+ `training-log.jsonl`（逐步训练证据） | 交付模型，2000 步 |
| `SPHERE-All-Adaptive/` | `curves.png` + `curves.csv` | 2000 步训练曲线（统一配方重训曲线待更新） |
| `SPHERE-Last-Fixed/` | `curves.png` + `curves.csv` | 2000 步训练曲线（统一配方重训曲线待更新） |
| `GRPO-baseline/` | （待训练） | 基线曲线 + 中间检查点评测，跑完补入 |
| `RandomInit/` | （无训练曲线） | 训练前模型未训练；零样本错误率 100% |

## 高效性判据（N<M）的取数方法

大纲 4.2.2.4(3) 第 4 步要求"错误率达到 5% 所需训练迭代次数"的对比。
取数方法：各模型训练均每 200 步存检查点（各模型目录的 `step{200..2000}`；
GRPO 基线训练命令见 `SNIPS/models/GRPO-baseline/README.md`），用
`methods/push_ckpt.py` + `SNIPS/eval_snips.py` 对每个检查点跑一次
test 集评测，得到错误率-步数曲线；N = 各 SPHERE 模型首次 ≤5% 的步数，
M = GRPO 基线首次 ≤5% 的步数。曲线作图脚本 `scripts/plot_songce_curves.py`。

## curves.csv 字段

- `step`：训练步（0 起）
- `mean_reward`：该步 4 prompt × 16 rollout 的平均训练奖励
  （0.5×intent_ok + 0.5×shaped-slot）
- `roll50`：最近 50 步 mean_reward 滚动均值（平滑用）
