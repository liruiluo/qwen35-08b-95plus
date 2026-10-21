# GRPO 基线模型（待训练）

大纲 4.2.2.2 的基线算法：组相对策略优化（GRPO，DeepSeekMath,
arXiv:2402.03300）。与 SPHERE-Last-Adaptive 的差异仅两处：关闭
SPHERE 谱正则项（纯 GRPO）与基线学习率；数据、步数、种子等其余
配置完全一致。

用途（大纲 4.2.2.4(3) 第 4/5 步）：
1. 作为第 5 个模型进 `evaluate.py` 报测试集错误率；
2. 高效性判据 N<M：错误率达到 5% 所需训练步数 M 由其训练曲线/中间检查点给出。

发射命令（训练服务器，引擎启动后；配置唯一来源 = `configs/songce/grpo_baseline.yaml`）：
```bash
python methods/train_hydra.py -cn grpo_baseline   # 渲染为训练器的精确 argv 并执行
bash scripts/run_songce_grpobase.sh               # 完整流程：训练 → 权重加载 → 评测
```
基线与交付模型的差异仅两个字段：`rho: 0.0`（关闭谱正则）、`lr: 1.5e-6`（基线学习率）。
