# SPHERE-Last-Adaptive（交付臂）

- 训练配方：纯 RL（GRPO + SPHERE 谱塑性正则，last_adaptive 臂）+
  意图定义提示 v4 + 2000 步（group 16、lr 3e-6、seed 17，单次运行）。
- 结果：**test 意图准确率 97.14%（错误率 2.86%，判据 B≤5% 合格）**，
  dev 99.14%；槽位 span-exact micro-F1 70.19%（附加指标）。
- 证据：`results/jb-interpdef4s2000/`（evidence.json + 逐例 jsonl + 分数文件）；
  训练曲线见 `SNIPS/logs/SPHERE-Last-Adaptive/`。
- 权重：pod `ckpt/jb-interpdef4s2000-last_adaptive`（sha256 见 MANIFEST.md，
  含每 200 步中间检查点，供高效性判据 N 的错误率-步数扫描）。
- 精确复现命令：仓库根 README「复现训练」一节（hydra 配置见
  `configs/songce/`，交付臂等价直接命令在 `scripts/run_def4s2000.sh`）。
