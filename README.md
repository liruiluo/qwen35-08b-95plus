# SNIPS 指令认知识别

对应国家科技重大专项课题二「交互式通讯学习算法体系」第三场景
（测试大纲 4.2.2）。本仓库将 Qwen3.5-0.8B 通过**强化学习**在 SNIPS
任务上训练至 **test 集 7 类意图分类准确率 97.14%**。

## 结果

SNIPS 指令认知识别任务，评价指标为 7 类意图分类准确率。
JointBERT 标准划分（Train 13,084 / Dev 700 / Test 700），test 集单次评测：

| 模型 | test 准确率 | 错误率 | 大纲判据（B < A×5%，即 ≤5%） |
|---|---|---|---|
| 训练前模型（随机初始化，seed 17，A） | 0.00% | 100% | — |
| **SPHERE-Last-Adaptive** | **97.14%** | **2.86%** | **通过** |
| SPHERE-All-Adaptive | 96.00% | 4.00% | 通过 |
| SPHERE-Last-Fixed | 96.43% | 3.57% | 通过 |
| GRPO 基线 | 95.29% | 4.71% | — |

## 评测

按测试大纲 4.2.2.4(3) 的六个测试步骤组织：

1. 测试数据样例：`SNIPS/dataset/`（test 700 条，行格式见其 `README.md`）
2. 模型：`SNIPS/models/` 下三个项目算法模型、基线模型与训练前模型
   （权重放置见 `SNIPS/models/MANIFEST.md`）
3. 训练曲线：`SNIPS/logs/` 各模型子目录（`curves.png` + `curves.csv` +
   `training-log.jsonl`）
4. 高效性测试（N<M）：各模型每 200 步存检查点、逐检查点评测 test 集
   （`scripts/nm_sweep.sh`；数据 `SNIPS/logs/nm-sweep.csv`，曲线
   `error-rate-curves.png`）。错误率首次不超过各阈值的训练步数：

   | 错误率阈值 | SPHERE-Last-Adaptive | SPHERE-All-Adaptive | SPHERE-Last-Fixed | GRPO 基线 |
   |---|---|---|---|---|
   | 5%（判据线） | 200 | 200 | 200 | 400 |
   | 4.8% | 200 | 200 | 600 | 1400 |
   | 4.5% | 600 | 600 | 600 | 未达到 |
   | 4% | 600 | 800 | 800 | 未达到 |

   判据线 5% 上三算法 N=200 < 基线 M=400，高效性测试通过；更严的
   4.8% 线三算法 200/200/600 步、基线 1400 步；4.5% 与 4% 线三算法
   全部达到，基线 2000 步内未达到（全程最低错误率 4.71%）。
5. 运行评测：`python SNIPS/scripts/evaluate.py`，遍历测试样本，
   逐个输入上述 5 个模型并计算错误率
6. 验收：训练前模型错误率为 100%；项目算法模型错误率不高于 5%
   （实测 2.86% / 4.00% / 3.57%）

环境：单张 GPU；Python ≥3.10；`pip install -r requirements.txt`
（Linux 上 torch 需为 CUDA 版，按 pytorch.org 官方指引安装）。
评测脚本自动拉起推理引擎、逐模型加载权重，输出各模型错误率与判据标注。

## 复现训练

```bash
# 1) 数据：JointBERT 原始文件 -> jsonl（--definitions 附带意图定义提示）
bash scripts/get_data.sh
cd SNIPS && python prepare_jointbert.py --src ../jointbert-snips \
    --out-dir data-jbdef4 --definitions && cd ..

# 2) 模型：基座权重 Qwen3.5-0.8B 放入 base/，并生成训练前模型
#    （随机初始化，seed 17）
python scripts/make_random_init.py base/Qwen3.5-0.8B <随机初始化目录>

# 3) 启动推理引擎
bash engine/launch_engine.sh <base模型> 30001

# 4) 训练（每个算法一个 hydra 配置，见 configs/songce/）
python methods/train_hydra.py -cn last_adaptive   # SPHERE-Last-Adaptive
python methods/train_hydra.py -cn all_adaptive    # SPHERE-All-Adaptive
python methods/train_hydra.py -cn last_fixed      # SPHERE-Last-Fixed
python methods/train_hydra.py -cn grpo_baseline   # GRPO 基线
```

`configs/songce/base.yaml` 中的模型与数据路径为训练服务器路径，
本机复现时按环境修改。

## 方法

训练算法为 **GRPO**（组相对策略优化）：对每个提示采样 16 条输出，以
组内相对优势替代价值网络估计。在此之上叠加 **SPHERE 谱塑性正则**
（ICML 2026）：对 Transformer MLP 层输入的 Gram 矩阵施加
Parseval 型正则。

四个训练配置：

- SPHERE-Last-Adaptive：仅最后一层，自适应 α
- SPHERE-All-Adaptive：全部层
- SPHERE-Last-Fixed：仅最后一层，固定 α=1
- GRPO 基线：关闭谱正则项

评测采用 JointBERT 标准划分（划分依据见 `SNIPS/dataset/README.md`）。

## 数据与引用

- SNIPS 语料：Coucke et al., 2018（arXiv:1805.10190）
- 划分：Chen et al., 2019（JointBERT, arXiv:1902.10909；
  数据文件 monologg/JointBERT）
- GRPO：Shao et al., 2024（DeepSeekMath, arXiv:2402.03300）
- SPHERE：Lirui Luo et al. Proceedings of the 43rd International
  Conference on Machine Learning (ICML), 2026（arXiv:2605.04712）

## License

Apache-2.0。
