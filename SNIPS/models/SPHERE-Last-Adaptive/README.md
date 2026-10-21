# SPHERE-Last-Adaptive（交付）

- 训练配方：纯 RL（GRPO + SPHERE 谱塑性正则）+ 意图定义提示 +
  2000 步（group 16、lr 3e-6、seed 17）。
- 结果：**test 意图准确率 97.14%，错误率 2.86%（判据 B≤5% 合格）**。
- 证据：`results/jb-interpdef4s2000/`（evidence.json + 逐例 jsonl + 分数文件）；
  训练曲线见 `SNIPS/logs/SPHERE-Last-Adaptive/`。
- 权重：本目录（HF 格式，sha256 见 MANIFEST.md）；高效性判据 N 所需的
  每 200 步错误率-步数扫描见 `SNIPS/logs/README.md`。
- 复现命令：仓库根 README「复现训练」一节（本模型配置
  `configs/songce/last_adaptive.yaml`）。
