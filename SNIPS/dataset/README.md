# SNIPS/dataset —— 测试数据

SNIPS NLU 基准（Coucke et al., arXiv:1805.10190，
https://github.com/snipsco/nlu-benchmark）的 **JointBERT 标准划分**，
Train 13,084 / Dev 700 / Test 700，由 `SNIPS/prepare_jointbert.py
--definitions` 从 monologg/JointBERT 原始文件构建。

| 文件 | 条数 | 用途 |
|---|---|---|
| `train.jsonl` | 13,084 | 训练语料 |
| `dev.jsonl` | 700 | 开发集 |
| `test.jsonl` | 700 | **测试样本**：`SNIPS/scripts/evaluate.py` 的遍历对象 |

行格式为 chat messages，第三条消息为标准答案：

```json
{"messages": [
  {"role": "system", "content": "<任务说明 + 7 类意图定义>"},
  {"role": "user",   "content": "play something from the album erase by britney spears"},
  {"role": "assistant", "content": "{\"intent\": \"PlayMusic\", \"slots\": [...]}"}
]}
```
