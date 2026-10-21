# SNIPS/logs —— 训练曲线与评测汇总

每个模型一个子目录，已训练模型含 `curves.png` 训练曲线图 +
`curves.csv` 逐 step 数据（生成脚本 `scripts/plot_songce_curves.py`）；
`curves-all.png` 为全部已训模型的对比总图（三车落地后统一生成）；
`evaluate.py` 的汇总输出写在本目录 `evaluate-summary.txt`。

| 子目录 | 内容 | 说明 |
|---|---|---|
| `SPHERE-Last-Adaptive/` | `curves.png` + `curves.csv`（step, mean_reward, roll50）+ `training-log.jsonl`（逐步训练证据） | 交付模型，2000 步 |
| `SPHERE-All-Adaptive/` | （重训中，待补） | 统一配方重训完成后补入 |
| `SPHERE-Last-Fixed/` | （重训中，待补） | 统一配方重训完成后补入 |
| `GRPO-baseline/` | （待训练） | 基线曲线 + 中间检查点评测，跑完补入 |

RandomInit（训练前模型）无训练曲线，不建子目录；零样本错误率 100%。

## 高效性判据（N<M）的取数方法

大纲 4.2.2.4(3) 第 4 步要求"错误率达到 5% 所需训练迭代次数"的对比。
各模型训练时每 200 步保存检查点。对每个检查点用 `methods/push_ckpt.py`
加载权重后，以 `SNIPS/eval_snips.py` 跑一次 test 集评测，得到
错误率-步数曲线。N = 各 SPHERE 模型首次 ≤5% 的步数；M = GRPO 基线
首次 ≤5% 的步数。曲线图由 `scripts/plot_songce_curves.py` 生成。

## curves.csv 字段

- `step`：训练步（0 起）
- `mean_reward`：该步 4 prompt × 16 rollout 的平均训练奖励
  （0.5×intent_ok + 0.5×shaped-slot）
- `roll50`：最近 50 步 mean_reward 滚动均值（平滑用）
