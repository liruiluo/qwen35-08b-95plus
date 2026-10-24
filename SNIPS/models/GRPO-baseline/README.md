# GRPO 基线模型

大纲 4.2.2.2 的基线算法为组相对策略优化 GRPO（DeepSeekMath，
arXiv:2402.03300）。

- test 错误率 **4.71%**，意图准确率 95.29%。5% 判据线 400 步达到，
  4.8% 线 1400 步达到，2000 步内未达到 4.5% 线。错误率-步数曲线见
  `SNIPS/logs/README.md`。

用途对应大纲 4.2.2.4(3) 的第 4、5 步：
1. 作为第 5 个模型进 `evaluate.py` 报测试集错误率。
2. 高效性对比：错误率达到 5% 所需训练步数由其训练曲线与中间检查点给出。

训练命令如下，配置为 `configs/songce/grpo_baseline.yaml`：
```bash
python methods/train_hydra.py -cn grpo_baseline
```
