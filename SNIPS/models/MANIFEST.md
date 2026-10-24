# SNIPS/models —— 五个测试对象（测试大纲 4.2.2.2 / 4.2.2.4(3)）

| 目录 | 大纲角色 | 状态 | test 错误率（实测） |
|---|---|---|---|
| `RandomInit/` | 训练前模型 | ✅ 权重随材料交付 | **100%**（700/700 不可解析，意图准确率 0.00%） |
| `GRPO-baseline/` | 基线算法模型（GRPO） | ✅ 权重随材料交付 | **4.71%**（意图准确率 95.29%） |
| `SPHERE-Last-Adaptive/` | 项目算法 1 | ✅ 权重随材料交付 | **2.86%**（意图准确率 97.14%） |
| `SPHERE-All-Adaptive/` | 项目算法 2 | ✅ 权重随材料交付 | **4.00%**（意图准确率 96.00%） |
| `SPHERE-Last-Fixed/` | 项目算法 3 | ✅ 权重随材料交付 | **3.57%**（意图准确率 96.43%） |

## 命名映射（大纲名称 ↔ 代码名）

| 大纲名称 | 训练器 `--arm` |
|---|---|
| SPHERE-Last-Adaptive | `last_adaptive` |
| SPHERE-All-Adaptive | `all_hidden_adaptive` |
| SPHERE-Last-Fixed | `last_fixed` |
| GRPO 基线 | （`--arm` 任意 + `--rho 0`） |

## 权重获取

权重为 HF 格式目录，含 `model.safetensors`、config 与 tokenizer，
随交付材料提供。放置到本目录各子目录，只需顶层文件，`step*/`
中间检查点不必拷贝。

RandomInit 由 `scripts/make_random_init.py` 以 seed 17 从
Qwen3.5-0.8B 随机重置生成，可离线复现。
