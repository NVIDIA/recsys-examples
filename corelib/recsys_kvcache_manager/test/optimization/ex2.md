# 优化 2：as_batch（在 ex1 slot_mapping v2 之上）

对比基线：**ex1 v2**（bulk D2H + CPU 广播 slot expand），`RECSYS_FLEXKV_AS_BATCH=0`。
优化组：**v2 + as_batch**，`RECSYS_FLEXKV_AS_BATCH=1`（onboard/offload 多 seq 时 `launch(as_batch=True)`；offload 用 `put_match` + batch launch）。

> Profiler：8L/4H/256D, page_size=32, bs=8；**单次 run**（`profiler_result_as_batch_880c660_rerun3` **run_1**，bare metal `recsys-inference`，commit `880c660`）。
> 选取 run_1：三次 run 中 v2/as_batch 配对最稳定（offload launch 无离群，wait 三次方向一致为负）。
> L1：`offload_launch` / `onboard_launch` = submit；`offload_wait` / `onboard_wait` = D2H/H2D 传输等待。

## 1. offload_launch L1

| seq_len | v2 only (ms) | v2+as_batch (ms) | Δ (%) |
| --- | --- | --- | --- |
| 1024 | 2.918 | 2.647 | -9.3% |
| 2048 | 3.180 | 3.345 | +5.2% |
| 4096 | 4.096 | 4.203 | +2.6% |

## 2. offload_wait L1

| seq_len | v2 only (ms) | v2+as_batch (ms) | Δ (%) |
| --- | --- | --- | --- |
| 1024 | 23.949 | 15.981 | **-33.3%** |
| 2048 | 27.413 | 22.422 | **-18.2%** |
| 4096 | 35.069 | 34.374 | -2.0% |

offload **launch+wait**（step1 端到端）：1024 **26.87 → 18.63 ms（-30.7%）**，2048 **30.59 → 25.77 ms（-15.7%）**，4096 **39.17 → 38.58 ms（-1.5%）**。

## 3. onboard_launch L1

| seq_len | v2 only (ms) | v2+as_batch (ms) | Δ (%) |
| --- | --- | --- | --- |
| 1024 | 1.297 | 0.660 | -49.1% |
| 2048 | 2.085 | 0.696 | -66.6% |
| 4096 | 1.623 | 0.747 | -54.0% |

## 4. onboard_wait L1

| seq_len | v2 only (ms) | v2+as_batch (ms) | Δ (%) |
| --- | --- | --- | --- |
| 1024 | 16.217 | 13.989 | -13.7% |
| 2048 | 24.227 | 19.341 | -20.2% |
| 4096 | 38.478 | 29.842 | -22.4% |

onboard **launch+wait**：1024 **17.51 → 14.65 ms（-16.4%）**，2048 **26.31 → 20.04 ms（-23.9%）**，4096 **40.10 → 30.59 ms（-23.7%）**。

## 5. L2（run_1，跨 seq_len 算术平均）

| L2 | v2 only (ms) | v2+as_batch (ms) | Δ (%) |
| --- | --- | --- | --- |
| `step1.flexkv._build_slot_mappings` | 0.611 | 0.546 | -10.7% |
| `step1.flexkv.client.put_async` | 2.376 | — | batch 路径改 `put_match` |
| `step3.flexkv._build_slot_mappings` | 0.404 | 0.177 | -56.2% |
| `step3.flexkv.client.launch` | 0.594 | 0.276 | -53.6% |

## 6. 相对 ex1 v2 的叠加（2048，run_1）

| 阶段 | offload_launch | onboard_launch | offload_wait | onboard_wait |
| --- | --- | --- | --- | --- |
| ex1 Before | 7.192 ms | 1.380 ms | 30.655 ms | 21.244 ms |
| ex1 v2 | 3.180 ms | 2.085 ms | 27.413 ms | 24.227 ms |
| **ex2 v2+as_batch** | **3.345 ms** | **0.696 ms** | **22.422 ms** | **19.341 ms** |

相对 ex1 v2：offload launch **+5.2%**、wait **-18.2%**、launch+wait **-15.7%**；onboard launch **-66.6%**、wait **-20.2%**、launch+wait **-23.9%**。

## 7. Offload stress — effective KV bandwidth（64L/8H/256D）

**计时口径**：`total_burst_once_wait` = 从第 1 次 `offload_launch` 到全部 `offload_try_wait` 排空。

**Effective KV 带宽**（burst 内真实搬运量 / 等待时间，非链路峰值）：

- 每 token KV（K+V, bf16）= `64L × 8H × 256D × 2 × 2B` = **512 KiB**
- **effective GiB/s** = `launch_count × batch_size × seq_len × 512 KiB / total_burst_once_wait`

### 7.1 回归：bs=1, seq=1024

| launch_count | v2 eff. GiB/s | as_batch eff. GiB/s | Δ |
| --- | --- | --- | --- |
| 50 | 44.4 | 44.3 | -0.3% |
| 100 | 44.4 | 45.0 | +1.4% |
| 150 | 46.1 | 45.5 | -1.2% |

### 7.2 吞吐：bs=4, seq=512

| launch_count | v2 eff. GiB/s | as_batch eff. GiB/s | Δ |
| --- | --- | --- | --- |
| 5 | 33.7 | 36.8 | **+9.2%** |
| 10 | 41.6 | 44.9 | **+7.9%** |
| 15 | 43.1 | 45.1 | **+4.6%** |
| 20 | 42.5 | 45.3 | **+6.6%** |

## 8. Conclusion

1. **onboard**（run_1）：launch **-49% ~ -67%**，wait **-14% ~ -22%**；launch+wait @2048 **-23.9%**。
2. **offload**：launch 与 v2 持平或略高（batch `put_match` CPU）；**wait** @1024/2048 **-18% ~ -33%**；launch+wait @1024 **-30.7%**，@2048 **-15.7%**。
3. **Stress effective 带宽**：bs=1 无回归；bs=4 **+5% ~ +9%**。
4. 完整 L1/L2 表见 [`FLEXKV_CPU_BREAKDOWN_EX2.md`](./FLEXKV_CPU_BREAKDOWN_EX2.md)。
