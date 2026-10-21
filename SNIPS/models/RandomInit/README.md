# 训练前模型（判据 A）

Qwen3.5-0.8B 的随机重置权重（`scripts/make_random_init.py`，torch seed 17，
对全部参数独立重采样，结构/分词器不变）。按大纲 4.2.2.5 的口径，本模型在
SNIPS 测试集上的错误率为 **100%**（意图准确率 0.00%，700/700 输出不可解析，
逐例存档见仓库根 `results/jb-interp*/` 证据链）。

这是交付线训练（插值起点 RL）的构造起点：训练前 100 步 serve 侧给引擎推
α·W_RandomInit + (1−α)·W_train（α: 1→0 线性退火），故"训练前模型"与
训练起点严格同一权重。复现：`python scripts/make_random_init.py <base> <out>`。
