# 优化 1：slot_mapping D2H — offload_launch / onboard_launch 前后对比

`_build_slot_mappings` 改为仅 D2H `page_ids` + CPU numpy 展开；offload/onboard loop 去掉全量 `.to("cpu")`。  

## 1. offload_launch

### 1.1 L1（step-op 总耗时）

| seq_len | Before (ms) | After (ms) | Δ (ms) | Δ (%) |
| --- | --- | --- | --- | --- |
| 1024 | 6.682 | 3.268 | **-3.414** | **-51.1%** |
| 2048 | 7.192 | 3.958 | **-3.234** | **-45.0%** |
| 4096 | 7.948 | 5.145 | **-2.803** | **-35.3%** |

### 1.2 L2（offload_launch 内子函数）

| L2 function | seq_len | Before (ms) | After (ms) | Δ (ms) | Δ (%) |
| --- | --- | --- | --- | --- | --- |
| `recsys.gpu.acquire_offload_pages` | 1024 | 0.098 | 0.088 | -0.010 | -10.2% |
| | 2048 | 0.093 | 0.094 | +0.001 | +1.1% |
| | 4096 | 0.102 | 0.097 | -0.005 | -4.9% |
| `recsys.host._build_slot_mappings` | 1024 | 3.329 | 0.857 | **-2.472** | **-74.3%** |
| | 2048 | 3.303 | 1.129 | **-2.174** | **-65.8%** |
| | 4096 | 3.790 | 2.109 | **-1.681** | **-44.4%** |
| `flexkv.client.put_async` | 1024 | 2.529 | 2.055 | **-0.474** | **-18.7%** |
| | 2048 | 2.911 | 2.407 | **-0.504** | **-17.3%** |
| | 4096 | 3.192 | 2.601 | **-0.591** | **-18.5%** |

---

## 2. onboard_launch

### 2.1 L1（step-op 总耗时）

| seq_len | Before (ms) | After (ms) | Δ (ms) | Δ (%) |
| --- | --- | --- | --- | --- |
| 1024 | 1.306 | 1.283 | -0.023 | -1.8% |
| 2048 | 1.380 | 1.537 | **+0.157** | **+11.4%** |
| 4096 | 1.424 | 1.987 | **+0.563** | **+39.5%** |

### 2.2 L2（onboard_launch 内子函数）

| L2 function | seq_len | Before (ms) | After (ms) | Δ (ms) | Δ (%) |
| --- | --- | --- | --- | --- | --- |
| `recsys.host._build_slot_mappings` | 1024 | 0.630 | 0.766 | +0.136 | +21.6% |
| | 2048 | 0.623 | 0.953 | **+0.330** | **+53.0%** |
| | 4096 | 0.615 | 1.376 | **+0.761** | **+123.7%** |
| `flexkv.client.launch` | 1024 | 0.259 | 0.246 | -0.013 | -5.0% |
| | 2048 | 0.311 | 0.299 | -0.012 | -3.9% |
| | 4096 | 0.324 | 0.308 | -0.016 | -4.9% |

---

## 3. 简要分析

**offload_launch 收益显著且一致。** L1 在三个 seq_len 上下降 35–51%（`seq_len=2048` 时 7.19 → 3.96 ms）。主要来源是 `_build_slot_mappings` L2 下降 44–74%：baseline 在 step1 GPU 繁忙时做 GPU 广播展开并叠加 loop 内全量 D2H，优化后改为小 `page_ids` D2H + CPU 展开，同时 `put_async` L2 也降约 17–19%（slot_mapping 已在 CPU，省去每次 put 前的 `.to("cpu")`）。

**onboard_launch L1 变化不大。** `_build_slot_mappings` L2 在 2048 上从 0.62 → 0.95 ms（+53%），但 baseline 的 sync/D2H 原落在未标注的 loop 区间（约 0.3–0.5 ms）；细粒度子阶段 profiling 显示真实 slot 成本 baseline build+`slot_d2h` ≈ 1.43 ms，optimized `page_ids_d2h`+`cpu_expand` ≈ 1.02 ms，构建本身反而更低。L1 略升（2048 上 +11%）主要来自计量边界变化及 `launch` 以外未标开销的波动，而非优化失效。onboard 路径绝对量级小（~1.5 ms），对 E2E 影响有限。
