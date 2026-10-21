# GRPO 基线模型（待训练）

大纲 4.2.2.2 的基线算法：组相对策略优化（GRPO，DeepSeekMath, arXiv:2402.03300），
即在交付配方上**关闭 SPHERE 谱正则项**（`--rho 0`，梯度组合
g = g_ppo + rho·‖g_ppo‖/‖g_sphere‖·g_sphere 中的 rho 置 0），其余与
SPHERE-Last-Adaptive 完全一致（插值起点 + 定义 v4 数据 + 2000 步 + seed 17）。

用途（大纲 4.2.2.4(3) 第 4/5 步）：
1. 作为第 5 个模型进 `evaluate.py` 报测试集错误率；
2. 高效性判据 N<M：错误率达到 5% 所需训练步数 M 由其训练曲线/中间检查点给出。

发射命令（pod 侧，引擎起在 base 后）：
```bash
python methods/rl_interp.py --data snips/data-jbdef4/train.jsonl \
  --base /root/models/Qwen3.5-0.8B \
  --interp-base /root/models/Qwen3.5-0.8B-RandomInit --interp-steps 100 \
  --rollout-url http://127.0.0.1:30001 --arm last_adaptive --reward snips --rho 0 \
  --out ckpt/jb-grpobase-s2000-last_adaptive --steps 2000 --group 16 \
  --prompts-per-step 4 --lr 3e-6 --max-new-tokens 160 --seed 17 --save-every 200
```
