# SPHERE-Last-Adaptive

- 训练配方：GRPO 强化学习叠加 SPHERE 谱塑性正则，并用意图定义提示。
- 结果：test 意图准确率 **97.14%**，错误率 **2.86%**，低于 5% 判据线，
  测试通过。
- 训练曲线见 `SNIPS/logs/SPHERE-Last-Adaptive/`。
- 权重在本目录，HF 格式。高效性判据所需的每 200 步错误率-步数扫描
  见 `SNIPS/logs/README.md`。
- 复现命令见仓库根 README「复现训练」一节，本模型配置为
  `configs/songce/last_adaptive.yaml`。
