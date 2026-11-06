# SNIPS/models —— 五个测试对象（测试大纲 4.2.2.2 / 4.2.2.4(3)）

| 目录 | 大纲角色 | 状态 | test 错误率（实测） |
|---|---|---|---|
| `RandomInit/` | 训练前模型 | ✅ 权重随材料交付 | **100%**（700/700 不可解析，意图准确率 0.00%） |
| `GRPO-baseline/` | 基线算法模型（GRPO） | ✅ 权重随材料交付 | **4.71%**（意图准确率 95.29%） |
| `ATLAS/` | 项目算法 1 | ✅ 权重随材料交付 | **2.86%**（意图准确率 97.14%） |
| `SPHERE/` | 项目算法 2 | ✅ 权重随材料交付 | **4.00%**（意图准确率 96.00%） |
| `FIXA/` | 项目算法 3 | ✅ 权重随材料交付 | **3.57%**（意图准确率 96.43%） |

## 命名映射（大纲名称 ↔ 代码名）

| 大纲名称 | 训练器 `--arm` |
|---|---|
| ATLAS | `last_adaptive` |
| SPHERE | `all_hidden_adaptive` |
| FIXA | `last_fixed` |
| GRPO 基线 | （`--arm` 任意 + `--rho 0`） |

## 三种算法各是什么

三者的差别只在**谱正则的作用范围**和**系数调度**，其余训练配方完全相同
（GRPO、2000 步、每步 4 prompt × 16 rollout、lr 3e-6、seed 17、意图定义提示）：

| 算法 | 谱正则作用范围 | 系数 α | 训练配置 |
|---|---|---|---|
| **ATLAS** | 最后一层（layer 23） | 自适应，α = Tr(A)/D（detached） | `last_adaptive` |
| **SPHERE** | 全部 24 层 | 自适应，同上 | `all_hidden_adaptive` |
| **FIXA** | 最后一层 | 固定 α = 1 | `last_fixed` |

## 权重获取

权重为 HF 格式目录，含 `model.safetensors`、config 与 tokenizer，
随交付材料提供。从共享盘拷贝到本目录各子目录的命令见仓库根
README「评测」一节。

各模型权重在组共享盘上，根目录在训练 pod 挂载为
`/home/ai-jingyan-train/luolirui.1/qwen95-delivery/`，在跳板机 jd
挂载为 `/mnt/amg-nfs/luolirui.1/qwen95-delivery/`：

| 模型 | 共享盘路径 |
|---|---|
| RandomInit | `models/Qwen3.5-0.8B-RandomInit` |
| GRPO-baseline | `ckpt/jb-grpobase6-s2000` |
| ATLAS | `ckpt/jb-interpdef4s2000-last_adaptive` |
| SPHERE | `ckpt/jb-interpdef4s2000-all_hidden_adaptive` |
| FIXA | `ckpt/jb-interpdef4s2000-last_fixed` |

RandomInit 由 `scripts/make_random_init.py` 以 seed 17 从
Qwen3.5-0.8B 随机重置生成，可离线复现。
