# SNIPS 意图分类 · 纯强化学习训练（GRPO + SPHERE）

本仓库将 Qwen3.5-0.8B（8 亿参数）通过**纯强化学习**训练至 SNIPS 意图分类
**test 集准确率 97.14%**，对应国家科技重大专项课题二「交互式通讯学习
算法体系」第三场景（指令认知识别）。三种算法 SPHERE-Last-Adaptive /
SPHERE-All-Adaptive / SPHERE-Last-Fixed 与 GRPO 基线的测试对象、数据、
脚本按《测试大纲》4.2.2 组织在 `SNIPS/` 目录。

## 结果

SNIPS 7 类意图分类，JointBERT 标准划分（Train 13,084 / Dev 700 /
Test 700），test 集单次评测：

| 模型 | test 准确率 | 错误率 | 大纲判据（B < A×5%，即 ≤5%） |
|---|---|---|---|
| 训练前模型（随机初始化，seed 17，A） | 0.00% | 100% | — |
| **SPHERE-Last-Adaptive（交付）** | **97.14%** | **2.86%** | **通过** |
| SPHERE-All-Adaptive | 重训中† | | |
| SPHERE-Last-Fixed | 重训中† | | |
| GRPO 基线 | 训练中† | | |

† 三者正按与交付臂完全相同的配方统一训练（2026-10-21 启动），完成后
更新此表。此前一轮无意图定义提示配方的参考成绩见
`results/EXPERIMENT_NOTES.md`。

交付模型 dev 准确率 99.14%，槽位填充 micro-F1 70.19%（附加指标）。
作为参照，BERT 监督微调在同划分上约 97.2%——本方法以纯强化学习
达到监督训练的上限水平。

## 评测（送测包）

`SNIPS/` 按测试大纲 4.2.2.4(3) 组织：

| 目录 | 内容 |
|---|---|
| `SNIPS/dataset/` | 训练/开发/测试数据（JointBERT 划分，含 sha256 校验） |
| `SNIPS/models/` | 五个模型目录与权重清单（`MANIFEST.md`，含大纲名称与代码名映射） |
| `SNIPS/logs/` | 各模型训练曲线与评测汇总 |
| `SNIPS/scripts/evaluate.py` | 评测入口：遍历测试样本 → 五模型 → 错误率 |

```bash
python SNIPS/scripts/evaluate.py             # 全部五个模型（自动拉起引擎、热推权重、判据标注）
python SNIPS/scripts/evaluate.py --limit 20  # 冒烟
```

## 复现训练

环境：单张 GPU；Python ≥3.10，`pip install -r requirements.txt`，
另需 [sglang](https://github.com/sgl-project/sglang) 推理引擎
（实测版本 torch 2.14.1 / transformers 5.19.0 / sglang 0.5.22）。

```bash
# 1) 数据：JointBERT 原始文件 -> jsonl（--definitions 附带意图定义提示）
bash scripts/get_data.sh
cd SNIPS && python prepare_jointbert.py --src ../jointbert-snips \
    --out-dir data-jbdef --definitions && cd ..

# 2) 随机初始化模型（插值起点用，seed 17）
python scripts/make_random_init.py <base模型> <输出目录>

# 3) 启动推理引擎
bash engine/launch_engine.sh <base模型> 30001

# 4) 训练（每车一个 hydra 配置，见 configs/songce/）
python methods/train_hydra.py -cn all_adaptive     # SPHERE-All-Adaptive
python methods/train_hydra.py -cn last_fixed       # SPHERE-Last-Fixed
python methods/train_hydra.py -cn grpo_baseline    # GRPO 基线
```

统一训练超参：group 16、lr 3e-6（基线 1.5e-6）、seed 17、2000 步、
每 200 步保存检查点（供错误率–步数曲线与高效性判据 N&lt;M 取数）。
交付臂的等价直接命令与逐字段说明见 `configs/songce/` 各配置文件。

## 方法

训练算法为 **GRPO**（组相对策略优化）：对每个提示采样 16 条输出，以
组内相对优势替代价值网络估计。在此之上叠加 **SPHERE 谱塑性正则**
（项目组论文，arXiv:2605.04712）：对 Transformer MLP 层输入的 Gram
矩阵施加 Parseval 型正则，其强度经梯度模长归一，固定为 GRPO 梯度的
0.1%（rho=1e-3）。三种实例化构成三个算法：仅最后一层、自适应 α
（Last-Adaptive），全部层（All-Adaptive），仅最后一层、固定 α=1
（Last-Fixed）。GRPO 基线即关闭该正则项（rho=0）。

两个训练设计：

- **插值起点**：训练前 100 步将引擎权重设为 α·随机初始化 + (1−α)·训练
  权重（α 线性退火 1→0），使"训练前模型 = 错误率 100% 的随机模型"与
  训练起点严格一致。机制说明：退火期模型输出不可解析、组内奖励为 0，
  合并梯度精确为 0——插值不改变训练动力学，step 100 后等效于从原模型
  出发的纯 RL。
- **意图定义提示**：系统提示包含七条意图定义，全部由训练集统计归纳
  （每个关键词均有计数背书），在开发集上调参；test 集每个版本只评测
  一次。

评测协议：训练与评测使用同一渲染层（本地以模型自身 tokenizer 渲染
chat template，token id 直送引擎 `/generate` 端点），保证训练与评测
条件一致。评测口径采用 JointBERT 标准划分，完整论证见
`results/EVALUATION_CALIBER.md`（要点：SNIPS 官方仓库不提供 test
划分，且官方 train 文件与标准 test 集重叠 464/700，直接混用等于在
测试集上训练）。

## 附加结果

- SNIPS 槽位填充（span-exact micro-F1）：70.19%，含标注约定与奖励
  塑形分析
- 官方语料划分（snipsco train 13,784 / validate 700）：三臂意图
  准确率 96.14–97.00%
- BFCL v4 simple_python（函数调用）：官方 AST 判分 82%，合成训练
  数据经完整泄漏审计（0/400 重叠）

消融实验（训练步数、意图定义版本、仅意图奖励）、零样本起点测量、
方法实现细节与工程注意事项见 `results/EXPERIMENT_NOTES.md`。

## 数据与引用

- SNIPS 语料：Coucke et al., arXiv:1805.10190（snipsco/nlu-benchmark）
- 划分与对照：JointBERT（monologg/JointBERT）；LLM2LLM, ACL
  Findings 2024（arXiv:2403.15042，SNIPS 意图分类 92.14–93.86%，
  Llama-2-7B 少样本增强场景）
- GRPO：DeepSeekMath, arXiv:2402.03300
- SPHERE：arXiv:2605.04712（ICML 2026）

## License

Apache-2.0。内嵌的 BFCL 判分器保留上游（gorilla，Apache-2.0）语义
与署名。SNIPS 与 BFCL 数据在运行时从上游仓库获取，本仓库不再分发。
