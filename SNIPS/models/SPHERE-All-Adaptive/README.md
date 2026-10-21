# SPHERE-All-Adaptive

- 训练配方：第一代三臂战役（从 base 单次训练、无插值、无意图定义，
  sphere_grpo.py + all_hidden_adaptive 臂 + 2000 步 + seed 17）。
- 结果：test 意图准确率 **95.29%（错误率 4.71% ≤5% 合格）**；dev 97.57%。
- 证据：`results/jointbert-split/all_hidden_adaptive/`；曲线见
  `SNIPS/logs/SPHERE-All-Adaptive/curves.csv`。
- 权重：pod `ckpt/jb-snips-all_hidden_adaptive-s2000`（sha256 见 MANIFEST.md）。
