# SPHERE-Last-Fixed

- 训练配方：第一代三臂战役（从 base 单次训练、无插值、无意图定义，
  sphere_grpo.py + last_fixed 臂 + 2000 步 + seed 17）。
- 结果：test 意图准确率 **94.86%（错误率 5.14%，超判据线 0.14pp = 1 题，
  700 题二项 SE≈0.8pp，在噪声带内）**；dev 94.86%。
- 证据：`results/jointbert-split/last_fixed/`；曲线见
  `SNIPS/logs/SPHERE-Last-Fixed/curves.csv`。
- 权重：pod `ckpt/jb-snips-last_fixed-s2000`（sha256 见 MANIFEST.md）。
- 备注：官方语料口径下同臂达 97.00%（`results/official-split/`）。如需
  全臂统一到交付配方（插值起点+定义 v4），按 SPHERE-Last-Adaptive 的
  复现命令换 `--arm last_fixed` 重训即可（约 3-4 小时/臂，见 logs/README.md）。
