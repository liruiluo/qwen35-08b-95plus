# SNIPS/models —— 五个测试对象（测试大纲 4.2.2.2 / 4.2.2.4(3)）

| 目录 | 大纲角色 | 状态 | test 错误率（实测） |
|---|---|---|---|
| `RandomInit/` | 训练前模型（判据 A） | ✅ 权重随材料交付 | **100%**（700/700 不可解析，意图准确率 0.00%） |
| `GRPO-baseline/` | 基线算法模型（GRPO，rho=0） | ⏳ **待训练** | — |
| `SPHERE-Last-Adaptive/` | 项目算法 1（交付） | ✅ 权重随材料交付 | **2.86%**（意图准确率 97.14%） |
| `SPHERE-All-Adaptive/` | 项目算法 2 | 统一配方重训中† | — |
| `SPHERE-Last-Fixed/` | 项目算法 3 | 统一配方重训中† | — |

## 命名映射（大纲名称 ↔ 代码名）

| 大纲名称 | 训练器 `--arm` |
|---|---|
| SPHERE-Last-Adaptive | `last_adaptive` |
| SPHERE-All-Adaptive | `all_hidden_adaptive` |
| SPHERE-Last-Fixed | `last_fixed` |
| GRPO 基线 | （`--arm` 任意 + `--rho 0`） |

## 权重获取

权重为 HF 格式目录（`model.safetensors` + config + tokenizer），随
交付材料提供；按下方 sha256 清单核对后放置到本目录各子目录（只需
顶层文件，`step*/` 中间检查点不必拷贝）：

```bash
sha256sum SNIPS/models/<模型名>/model.safetensors   # macOS: shasum -a 256 <同路径>
```

## sha256 清单

| 模型 | model.safetensors sha256 |
|---|---|
| RandomInit | `aa430a35da2d2855f8059cb45a7f28c2753f94ccb6ff363ea053e76922b31130` |
| SPHERE-Last-Adaptive | `d24f3ff584704c8f5ba8042163eb69efdf92cdff76b0be2157bb621357e3c006` |
| SPHERE-All-Adaptive | （重训完成后更新） | — |
| SPHERE-Last-Fixed | （重训完成后更新） | — |
| GRPO-baseline | （待训练，见 `GRPO-baseline/README.md`） | — |

RandomInit 由 `scripts/make_random_init.py`（seed 17）从
Qwen3.5-0.8B 随机重置生成，可离线复现。
