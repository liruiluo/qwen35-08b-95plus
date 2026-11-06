# FIXA

- 算法：谱正则只作用于**最后一层**，系数**固定为 1**。
- 训练配方：GRPO 强化学习叠加 SPHERE 谱塑性正则，并用意图定义提示，
  配置为 `configs/songce/last_fixed.yaml`。
- 结果：test 错误率 **3.57%**，意图准确率 96.43%，满足大纲判据。
- 权重与曲线：`MANIFEST.md` 与 `SNIPS/logs/FIXA/`。
