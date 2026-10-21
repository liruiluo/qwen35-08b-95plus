# SNIPS/models —— 五个测试对象（测试大纲 4.2.2.2 / 4.2.2.4(3)）

| 目录 | 大纲角色 | 状态 | test 错误率（实测） |
|---|---|---|---|
| `RandomInit/` | 训练前模型（判据 A） | ✅ 权重在 pod | **100%**（700/700 不可解析，意图准确率 0.00%） |
| `GRPO-baseline/` | 基线算法模型（GRPO，rho=0） | ⏳ **待训练** | — |
| `SPHERE-Last-Adaptive/` | 项目算法 1（交付） | ✅ 权重在 pod | **2.86%**（意图准确率 97.14%） |
| `SPHERE-All-Adaptive/` | 项目算法 2 | ✅ 权重在 pod | **4.71%**（95.29%） |
| `SPHERE-Last-Fixed/` | 项目算法 3 | ✅ 权重在 pod | **5.14%**（94.86%，超线 0.14pp = 1 题） |

## 命名映射（大纲名称 ↔ 代码名）

| 大纲名称 | 训练器 `--arm` | 机制 |
|---|---|---|
| SPHERE-Last-Adaptive | `last_adaptive` | 最后一个 MLP 层、自适应 α |
| SPHERE-All-Adaptive | `all_hidden_adaptive` | 全部 24 层、逐层求和 |
| SPHERE-Last-Fixed | `last_fixed` | 最后一个 MLP 层、固定 α=1 |
| GRPO 基线 | （任意臂 + `--rho 0`） | 关闭 SPHERE 谱正则项，纯 GRPO |

## 权重获取（不入 git；sha256 逐一核对）

权重为 HF 格式目录（`model.safetensors` + config + tokenizer）。
canonical 存放点：训练 pod `/root/train95/ckpt/` 与 `/root/models/`。
按 sha256 校验后放置到本目录各子目录（只需顶层文件，`step*/` 中间检查点不必拷贝）：

```bash
# 例（经跳板 jd 拉 SPHERE-Last-Adaptive）：
ssh jd "ssh -p 22228 -i /root/.ssh/id_ed25519_devbox_tailnet_20260627 root@6.235.104.34 \
  'tar czf - -C /root/train95/ckpt/jb-interpdef4s2000-last_adaptive --exclude=step* .'" \
  | tar xzf - -C SNIPS/models/SPHERE-Last-Adaptive/
shasum -a 256 SNIPS/models/SPHERE-Last-Adaptive/model.safetensors
```

## sha256 清单（pod 实测，2026-10-21）

| 模型 | pod 路径 | model.safetensors sha256 |
|---|---|---|
| RandomInit | `/root/models/Qwen3.5-0.8B-RandomInit` | `aa430a35da2d2855f8059cb45a7f28c2753f94ccb6ff363ea053e76922b31130` |
| SPHERE-Last-Adaptive | `/root/train95/ckpt/jb-interpdef4s2000-last_adaptive` | `d24f3ff584704c8f5ba8042163eb69efdf92cdff76b0be2157bb621357e3c006` |
| SPHERE-All-Adaptive | `/root/train95/ckpt/jb-snips-all_hidden_adaptive-s2000` | `02ed260c6d9afa7cf69fcb4c818a1589bfe147668f47f4d8f672d44430d0c635` |
| SPHERE-Last-Fixed | `/root/train95/ckpt/jb-snips-last_fixed-s2000` | `32f9782efd4ba7cc3776deb8ef9af0325ee895f135f1529e9678c0d9934b36e2` |
| GRPO-baseline | （待训练，见 `GRPO-baseline/README.md`） | — |

RandomInit 由 `scripts/make_random_init.py`（seed 17）从
Qwen3.5-0.8B-instruct 随机重置生成，可离线复现。
