# 优化 2：as_batch（在 ex1 slot_mapping v2 之上）

对比基线：**ex1 v2**（bulk D2H + CPU 广播 slot expand），`RECSYS_FLEXKV_AS_BATCH=0`。
优化组：**v2 + as_batch**，`RECSYS_FLEXKV_AS_BATCH=1`（onboard/offload 多 seq 时 `launch(as_batch=True)`；offload 用 `put_match` + batch launch）。

> Profiler：8L/4H/256D, page_size=32, bs=8；`RUNS=3` mean。onboard @1024 为独立 3-run（仅 len1024_bs8）；2048/4096 与 offload 为 full breakdown 3-run。

## 1. offload_launch L1

| seq_len | v2 only (ms) | v2+as_batch (ms) | Δ (%) |
| --- | --- | --- | --- |
| 1024 | 3.141 | 2.609 | -16.9% |
| 2048 | 3.232 | 3.173 | -1.8% |
| 4096 | 4.176 | 4.311 | +3.2% |

## 2. onboard_launch L1

| seq_len | v2 only (ms) | v2+as_batch (ms) | Δ (%) |
| --- | --- | --- | --- |
| 1024 | 1.323 | 0.801 | -39.5% |
| 2048 | 1.439 | 1.058 | -26.5% |
| 4096 | 1.446 | 0.777 | -46.3% |

## 3. L2 均值（跨 seq_len）

| L2 | v2 only (ms) | v2+as_batch (ms) | Δ (%) |
| --- | --- | --- | --- |
| `step1.flexkv._build_slot_mappings` | 0.598 | 0.587 | -1.7% |
| `step3.flexkv._build_slot_mappings` | 0.314 | 0.245 | -22.2% |
| `step3.flexkv.client.launch` | 0.480 | 0.339 | -29.5% |

offload batch 路径上 `put_async` L2 消失（改走 `put_match` + batch `launch`）；onboard/offload 的 `client.launch` L2 下降与 L1 趋势一致。

## 4. 相对 ex1 v2 的叠加（2048）

| 阶段 | offload_launch | onboard_launch |
| --- | --- | --- |
| ex1 Before | 7.192 ms | 1.380 ms |
| ex1 v2 | 3.958 ms | 1.537 ms |
| **ex2 v2+as_batch** | **3.173 ms** | **1.058 ms** |

相对 ex1 v2，as_batch 在 2048 上再降 offload **2.2%**、onboard **31.2%**；相对 Before 累计 offload **-55.9%**、onboard **-23.3%**。

## 5. Offload stress（64L/8H/256D）

**计时口径**：`total_burst_once_wait` = 从第 1 次 `offload_launch` 到全部 `offload_try_wait` 排空（launch → wait 结束）。

**吞吐公式**（bf16，K+V）：

- 每 token KV = `64L × 8H × 256D × 2 × 2B` = **512 KiB**
- 每 seq 数据量 = `seq_len × 512 KiB`（seq=1024 → **0.5 GiB/seq**）
- **有效 KV 吞吐** = `launch_count × batch_size × seq_len × 512 KiB / wait_time`
- **seq 吞吐** = `launch_count × batch_size / wait_time`

### 5.1 回归：bs=1, seq=1024

`batch_size=1` 不触发 as_batch；用于验证单 seq 路径无回归。

| launch_count | v2 wait (ms) | v2 seq/s | v2 GiB/s | as_batch wait (ms) | as_batch seq/s | as_batch GiB/s | Δ GiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 562.8 | 88.8 | 44.4 | 564.4 | 88.6 | 44.3 | -0.3% |
| 100 | 1126.4 | 88.8 | 44.4 | 1110.6 | 90.0 | 45.0 | +1.4% |
| 150 | 1628.2 | 92.1 | 46.1 | 1647.5 | 91.0 | 45.5 | -1.2% |

单 seq 路径吞吐 **~44–46 GiB/s（~89–92 seq/s）**，两组 ±1.5% 内，无回归。

### 5.2 吞吐：bs=4, seq=512

每次 launch 含 4 seq（**1.0 GiB/launch**），offload 走 put_match + batch launch，as_batch 生效。

| launch_count | v2 wait (ms) | v2 seq/s | v2 GiB/s | as_batch wait (ms) | as_batch seq/s | as_batch GiB/s | Δ GiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 148.5 | 134.7 | 33.7 | 135.8 | 147.3 | 36.8 | **+9.2%** |
| 10 | 240.3 | 166.5 | 41.6 | 222.7 | 179.6 | 44.9 | **+7.9%** |
| 15 | 347.8 | 172.5 | 43.1 | 332.3 | 180.6 | 45.1 | **+4.6%** |
| 20 | 470.9 | 169.9 | 42.5 | 441.1 | 181.4 | 45.3 | **+6.6%** |

as_batch 将有效 KV 吞吐从 **~34–43 GiB/s** 提升到 **~37–45 GiB/s**（**+5% ~ +9%**），seq 吞吐从 **~135–173 seq/s** 提升到 **~147–181 seq/s**。

两组 stress 均 `launch_succeeded = launch_count`，无 PUT 失败。

## 6. Conclusion

1. **onboard_launch**：1024/2048/4096 上 3-run mean 降 **26%–46%**；主要收益来自 batch `launch` 减少 submit 次数（L2 `client.launch` **-30%**）。
2. **offload_launch**：1024 降 **17%**；2048/4096 L1 在 ±3% 内，L2 已切至 batch 路径。
3. **Stress**：bs=1 吞吐 **~44 GiB/s** 无回归（±1.5%）；bs=4 有效 KV 吞吐 **+5% ~ +9%**（33–43 → 37–45 GiB/s）。
4. **代码**：`lyl-flexkv-optimization` 分支 `flex_kvcache_manager.py`；验证期保留 `RECSYS_FLEXKV_AS_BATCH`（默认开）。
