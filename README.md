# SNIPS 意图分类 · 纯强化学习训练（GRPO + SPHERE）

本仓库将 Qwen3.5-0.8B 通过**纯强化学习**训练至 SNIPS 意图分类
**test 集准确率 97.14%**，对应国家科技重大专项课题二「交互式通讯学习
算法体系」第三场景（指令认知识别）。三种算法 SPHERE-Last-Adaptive /
SPHERE-All-Adaptive / SPHERE-Last-Fixed 与 GRPO 基线按《测试大纲》
4.2.2 组织在 `SNIPS/` 目录，评测步骤见下文「评测」。

## 结果

SNIPS 7 类意图分类，JointBERT 标准划分（Train 13,084 / Dev 700 /
Test 700），test 集单次评测：

| 模型 | test 准确率 | 错误率 | 大纲判据（B < A×5%，即 ≤5%） |
|---|---|---|---|
| 训练前模型（随机初始化，seed 17，A） | 0.00% | 100% | — |
| **SPHERE-Last-Adaptive（交付）** | **97.14%** | **2.86%** | **通过** |
| SPHERE-All-Adaptive | 统一配方训练中† | | |
| SPHERE-Last-Fixed | 统一配方训练中† | | |
| GRPO 基线 | 训练中† | | |

† 与交付模型完全相同的配方（2026-10-21 启动），完成后更新此表。

## 评测

`SNIPS/` 按测试大纲 4.2.2.4(3) 组织：`SNIPS/dataset/` 测试数据（含
sha256 校验）、`SNIPS/models/` 五个模型目录与权重清单
（`SNIPS/models/MANIFEST.md`）、`SNIPS/logs/` 各模型训练曲线、
`SNIPS/scripts/evaluate.py` 评测入口。

环境：单张 GPU；Python ≥3.10；`pip install -r requirements.txt`。
运行前将五个模型权重放入 `SNIPS/models/` 各子目录（HF 格式，
sha256 清单见 `SNIPS/models/MANIFEST.md`）。

```bash
python SNIPS/scripts/evaluate.py             # 遍历测试样本 → 五模型 → 错误率
python SNIPS/scripts/evaluate.py --limit 20  # 冒烟
```

脚本自动拉起推理引擎、逐模型加载权重，输出各模型错误率与判据标注
（训练前模型预期 100%，项目算法模型 ≤5% 为合格）。

## 复现训练

```bash
# 1) 数据：JointBERT 原始文件 -> jsonl（--definitions 附带意图定义提示）
bash scripts/get_data.sh
cd SNIPS && python prepare_jointbert.py --src ../jointbert-snips \
    --out-dir data-jbdef --definitions && cd ..

# 2) 模型：从 ModelScope 或 HuggingFace 下载 Qwen3.5-0.8B 作为
#    <base模型>，并生成训练前模型（随机初始化，seed 17）
python scripts/make_random_init.py <base模型> <随机初始化目录>

# 3) 启动推理引擎
bash engine/launch_engine.sh <base模型> 30001

# 4) 训练（每车一个 hydra 配置，见 configs/songce/）
python methods/train_hydra.py -cn last_adaptive   # SPHERE-Last-Adaptive（交付）
python methods/train_hydra.py -cn all_adaptive    # SPHERE-All-Adaptive
python methods/train_hydra.py -cn last_fixed      # SPHERE-Last-Fixed
python methods/train_hydra.py -cn grpo_baseline   # GRPO 基线
```

统一训练超参：group 16、lr 3e-6（基线 1.5e-6）、seed 17、2000 步、
每 200 步保存检查点。`configs/songce/base.yaml` 中的模型与数据路径
为训练服务器路径，本机复现时按环境修改。

## 方法

训练算法为 **GRPO**（组相对策略优化）：对每个提示采样 16 条输出，以
组内相对优势替代价值网络估计。在此之上叠加 **SPHERE 谱塑性正则**
（arXiv:2605.04712）：对 Transformer MLP 层输入的 Gram 矩阵施加
Parseval 型正则，强度固定为 GRPO 梯度的 0.1%（rho=1e-3）。三种实例化
构成三个算法：仅最后一层、自适应 α（Last-Adaptive），全部层
（All-Adaptive），仅最后一层、固定 α=1（Last-Fixed）；GRPO 基线即
关闭该正则项（rho=0）。

训练设计：系统提示含七条**意图定义**，全部由训练集统计归纳，
调参只用开发集，test 集仅用于最终评测。

评测协议：训练与评测使用同一渲染层（本地渲染 chat template，token
id 直送引擎），评测采用 JointBERT 标准划分（SNIPS 官方 train
与标准 test 集重叠 464/700，不能混用；完整论证见
`results/EVALUATION_CALIBER.md`）。

## 数据与引用

- SNIPS 语料：Coucke et al., arXiv:1805.10190
- 划分：JointBERT（monologg/JointBERT）
- GRPO：DeepSeekMath, arXiv:2402.03300
- SPHERE：arXiv:2605.04712（ICML 2026）

## License

Apache-2.0。
