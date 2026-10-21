# SNIPS/dataset —— 测试数据

SNIPS NLU 基准（Coucke et al., arXiv:1805.10190，
https://github.com/snipsco/nlu-benchmark）的 **JointBERT 标准划分**
（Train 13,084 / Dev 700 / Test 700），由 `SNIPS/prepare_jointbert.py
--definitions` 从 monologg/JointBERT 原始文件构建。

| 文件 | 条数 | 用途 |
|---|---|---|
| `train.jsonl` | 13,084 | 训练语料（测试大纲：训练数据） |
| `dev.jsonl` | 700 | 开发集（调参诊断用，**从不用于 checkpoint 挑选**） |
| `test.jsonl` | 700 | **测试样本**：`SNIPS/scripts/evaluate.py` 的遍历对象 |

行格式（chat messages，第三条消息为标准答案）：

```json
{"messages": [
  {"role": "system", "content": "<任务说明 + 7 类意图定义（训练集统计归纳）>"},
  {"role": "user",   "content": "play something from the album erase by britney spears"},
  {"role": "assistant", "content": "{\"intent\": \"PlayMusic\", \"slots\": [...]}"}
]}
```

## 划分说明（为什么必须用 JointBERT 划分）

SNIPS 官方仓库不提供 test 划分，且官方 train 文件（13,784 条）与文献标准
test 集重叠 464/700——在官方 train 上训练再报 test 数等于 train-on-test。
完整实测与论证见仓库根 `results/EVALUATION_CALIBER.md` 与
`results/official-split/corpus-relation.txt`。官方划分的附加结果
（train 13,784 / validate 700）保留在 `results/official-split/`，
不作为主结果。

数据 sha256：

```
47d5f20f460bab8056290409826c0f9208a6e75d624e158da3ab2b10c6b7db52  train.jsonl
95a40d7de55cca3b1d536046b72add37fdf18cd2d46d34c65a15f68a4ce547ab  dev.jsonl
498ca42bd3069ff2838fd1679d104803c2a04db2a25fb0e87aef15583ab4efbb  test.jsonl
```
