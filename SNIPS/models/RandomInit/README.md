# 训练前模型

Qwen3.5-0.8B 的随机重置权重，torch seed 17，对全部参数独立重采样，
结构与分词器不变。按大纲 4.2.2.5 的口径，本模型在 SNIPS 测试集上的
错误率为 **100%**，700 条测试样本输出全部不可解析。复现命令为
`python scripts/make_random_init.py <base> <out>`。
