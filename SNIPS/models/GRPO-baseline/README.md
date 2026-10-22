# GRPO 基线模型

大纲 4.2.2.2 的基线算法：组相对策略优化（GRPO，DeepSeekMath,
arXiv:2402.03300）。与 SPHERE-Last-Adaptive 的差异仅两处：关闭
SPHERE 谱正则项与基线学习率；数据、步数、种子等其余配置完全一致。

- test 错误率 **5.57%**（意图准确率 94.43%），2000 步内未达到 5% 判据线
  （错误率-步数曲线见 `SNIPS/logs/README.md`）。

用途（大纲 4.2.2.4(3) 第 4/5 步）：
1. 作为第 5 个模型进 `evaluate.py` 报测试集错误率；
2. 高效性判据 N<M：错误率达到 5% 所需训练步数 M 由其训练曲线/中间检查点给出。

训练命令（配置：`configs/songce/grpo_baseline.yaml`）：
```bash
python methods/train_hydra.py -cn grpo_baseline   # GRPO 基线训练
```
