# SNIPS/logs —— 训练曲线与评测汇总

每个模型一个子目录；`evaluate.py` 的汇总输出写在本目录
`evaluate-summary.txt`。

| 子目录 | 内容 | 说明 |
|---|---|---|
| `SPHERE-Last-Adaptive/` | `curves.csv`（step, mean_reward, roll50）+ `training-log.jsonl`（逐步全量证据：g_ppo/g_sphere/lambda/sphere_loss/interp_alpha/replay 校验） | 交付臂，2000 步；前 100 步为插值退火期（reward=0、梯度精确为 0，机制透明说明见根 README） |
| `SPHERE-All-Adaptive/` | `curves.csv` | 第一代战役臂，2000 步 |
| `SPHERE-Last-Fixed/` | `curves.csv` | 第一代战役臂，2000 步 |
| `GRPO-baseline/` | （待训练） | 基线曲线 + 中间检查点评测，跑完补入 |
| `RandomInit/` | （无训练曲线） | 训练前模型未训练；零样本错误率 100% |

## 高效性判据（N<M）的取数方法

大纲 4.2.2.4(3) 第 4 步要求"错误率达到 5% 所需训练迭代次数"的对比。
取数方法：各臂训练均每 200 步存检查点（pod `ckpt/*/step{200..2000}`，
GRPO 基线发射命令见 `SNIPS/models/GRPO-baseline/README.md`），用
`methods/push_ckpt.py` + `SNIPS/eval_snips.py` 对每个检查点跑一次
test 集评测，得到错误率-步数曲线；N = SPHERE 各臂首次 ≤5% 的步数，
M = GRPO 基线首次 ≤5% 的步数。曲线作图脚本 `scripts/plot_curves.py`。

## curves.csv 字段

- `step`：训练步（0 起）
- `mean_reward`：该步 4 prompt × 16 rollout 的平均训练奖励
  （0.5×intent_ok + 0.5×shaped-slot）
- `roll50`：最近 50 步 mean_reward 滚动均值（平滑用）
