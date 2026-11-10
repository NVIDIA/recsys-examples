# 第 2 组：CPU 更快（c1、c3、c4、c5）

[English](group2-cpu-wins.md) | 中文

本组有四个用例：c1 过滤清洗、c3 每用户 TopN、c4 keep-last 去重、c5 Interval Join（含融合变体 join-filter 与 join-agg）。调优后（两边都开 `pipeline.object-reuse=true`），四个用例都是 CPU 吞吐更高。成本不在计算，而在序列化、JVM 状态读写和行列转换，GPU 替换不了这些。环境、数据集、负载档位、统计方法和一致性检查见 [benchmarks/README.zh-CN.md](README.zh-CN.md#method) 的方法部分。

> 状态：数字沿用此前合并文档里的数字。它们来自一份内部测量报告的改写版，该改写版**尚未经过独立复核**。"未核实"表示没有找到直接证据。

阅读说明：

- 吞吐单位为条/秒，延迟单位为 ms。"默认"指 Flink 默认配置，"调优"指两边同开 object-reuse（c3 另加 `table.exec.rank.topn-cache-size=100000`）。每个吞吐都是 3 轮的中位数。百分比 = 较快吞吐 / 较慢吞吐 - 1。
- **计划。** 每个用例列出各个实测模式的真实计划。EXPLAIN 输出三个阶段。**本组所有用例中，Optimized Physical Plan 在 CPU 与 GPU 上相同**（我们比对了保存下来的计划文本；唯一例外是 c4 的 `cpu-mb`，多了一个 mini-batch 节点）。**替换发生在 Optimized Execution Plan 这一层**：CPU 节点名被换成 `Gpu*` 节点名。此前的文字写"EXPLAIN 显示 CPU 与 GPU 计划没有差异"，这只对物理计划成立。下面的计划行摘自保存的计划文件，只裁掉了很长的 `fields=[...]` 列表，用 `…` 标出。
- **Gpu\* 节点。** 节点名和括号里的参数由 Mega-Flink 定义。关于它们的说法以 Mega-Flink 算子覆盖文档为依据；文档没有说明的，我们写"节点名由 Mega-Flink 定义"。所有 GPU 计划里，`mode=gpu-native` 是 Mega-Flink 定义的标签，`engine=batch|persistent|adaptive` 与覆盖文档里各算子的引擎设置一致（如 `join.engine`、`topn.engine`、`deduplicate.engine`）。`batchRows=4096, maxWaitMs=5` 是批大小与最长等待，与测试环境的 GPU 算子设置相同（批大小 4,096 行、最长等待 5 ms）；覆盖文档把 `batch.rows` / `batch.max-wait-ms` 描述为用户需要接受其等待的设置。`FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)` 是仍用 Flink CPU 实现的 Calc。覆盖文档列出 `calc.engine=auto|cpu|persistent|batch`，并说 `auto` 在 latency-safe 策略下使用 Flink CPU；本次运行用的是哪种策略：未核实。
- **GPU 确实在跑的证据。** Mega-Flink 算子带有运行时计数（批数、H2D 时间、计算时间、kernel 数、persistent 命令数）。我们按用例引用。这是运行时证据，不是 EXPLAIN 标签。

目录：

- [c1 过滤清洗](#c1-filter)
- [c3 每用户观看最久的前 10 个视频（TopN）](#c3-topn)
- [c4 keep-last 去重](#c4-dedup)
- [c5 Interval Join 及其融合](#c5-join)：[join](#c5-join-only)、[join-filter](#c5-join-filter)、[join-agg](#c5-join-agg)
- [本组小结](#summary)

<a id="c1-filter"></a>
## c1 过滤清洗

### 做什么

推荐样本清洗与基础特征：过滤无效或随机曝光记录，保留播放超过 3 秒或有点击的行，并计算观看比例、时长分桶、互动次数。实测模式：`cpu`、`gpu-batch`。

### SQL

```sql
SELECT seq, user_id, video_id, time_ms,
       CAST(play_time_ms AS DOUBLE) / CAST(duration_ms + 1 AS DOUBLE) AS watch_ratio,
       CASE WHEN duration_ms < 15000 THEN 0 WHEN duration_ms < 60000 THEN 1 ELSE 2 END AS duration_bucket,
       is_like + is_follow + is_comment + is_forward AS interact_cnt,
       is_click
FROM events
WHERE duration_ms > 0 AND play_time_ms >= 0 AND is_rand = 0
  AND (play_time_ms > 3000 OR is_click = 1)
```

### 真实执行计划

CPU，Optimized Execution Plan：

```text
Calc(select=[seq, user_id, video_id, time_ms, (CAST(play_time_ms AS DOUBLE) / CAST((duration_ms + 1) AS DOUBLE)) AS watch_ratio, CASE((duration_ms < 15000), 0, (duration_ms < 60000), 1, 2) AS duration_bucket, (((is_like + is_follow) + is_comment) + is_forward) AS interact_cnt, is_click], where=[((duration_ms > 0) AND (play_time_ms >= 0) AND (is_rand = 0) AND ((play_time_ms > 3000) OR (is_click = 1)))])
+- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

GPU batch，Optimized Execution Plan：

```text
GpuNativeStreamExecCalc(mode=gpu-native, engine=batch, batchRows=4096, maxWaitMs=5) Calc(select=[seq, user_id, video_id, time_ms, (CAST(play_time_ms AS DOUBLE) / CAST((duration_ms + 1) AS DOUBLE)) AS watch_ratio, CASE((duration_ms < 15000), 0, (duration_ms < 60000), 1, 2) AS duration_bucket, (((is_like + is_follow) + is_comment) + is_forward) AS interact_cnt, is_click], where=[((duration_ms > 0) AND (play_time_ms >= 0) AND (is_rand = 0) AND ((play_time_ms > 3000) OR (is_click = 1)))])
+- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

两边的 Optimized Physical Plan 相同（`TableSourceScan` 之上一个 `Calc`）。替换发生在 Optimized Execution Plan：`Calc` 变成 `GpuNativeStreamExecCalc`，原来的 Calc 表达式保留在参数之后。

- `GpuNativeStreamExecCalc`：由 Mega-Flink 执行的过滤加投影节点。节点名由 Mega-Flink 定义。`engine=batch` 是攒批形态；`batchRows` 与 `maxWaitMs` 见上文。覆盖文档把固定形状的过滤投影 Calc 列为实验性 GPU 算子，默认回退 CPU。
- source 扫描不变。

### 各模式的计划差异

只测了两个模式。GPU 的作业图与 CPU 的作业图顶点链相同（source、Calc、约束检查、sink 在一条链里），只有 Calc 的节点名不同，没有多出 exchange。

### GPU 确实在跑的证据

batch 模式下，运行时计数里的批数、H2D 时间和计算时间在所有正式轮次都不为 0。默认配置打满档（927 批）的实测计时见下面的时间构成，并有 nsys 核对。

### 结果

**默认配置**

| 档 | CPU 吞吐 | CPU p50 / p99 | GPU 吞吐 | GPU p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| 低 | 20,000 | <0.01 / <0.01 | 19,999 | 2.74 / 6.16 | MATCH |
| 中 | 100,000 | <0.01 / 0.21 | 99,989 | 2.96 / 7.60 | MATCH |
| 打满 | 831,997 | <0.01 / <0.01 | 1,025,360 | 1.86 / 3.38 | MATCH |

**调优（两边同开 object-reuse）**

| 档 | CPU 吞吐 | CPU p50 / p99 | GPU 吞吐 | GPU p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| 低 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 中 | 100,000 | <0.01 / 0.01 | 99,978 | 2.80 / 6.23 | MATCH |
| 打满 | 2,472,421 | <0.01 / <0.01 | 2,059,561 | 0.92 / 1.77 | MATCH |

### 时间构成

GPU 一侧的实测计时（默认，打满，927 批；Mega-Flink 的 CUDA event 计时，为实测）：行→列 254 ms、H2D 20 ms、kernel 25 ms、D2H 14 ms、列→行 186 ms。这五段合计 499 ms；kernel 约占 5%，行列转换约占 88%。nsys 核对：kernel 13.2 ms、H2D 15.9 ms、D2H 8.1 ms。

### 结论

默认配置下 GPU +23%，调优后 CPU +20%。**配置很关键。** 默认配置（`pipeline.object-reuse=false`）下 GPU 在 c1 更快（1.03M 对 0.83M，GPU +23%）；两边同开 object-reuse 后结果翻转为 CPU 更快。object-reuse 对 CPU 的帮助大于对 GPU（c1：约 x2.95 对 x2.0）。GPU kernel 约占 GPU 侧实测时间的 5%，行列转换约占 88%。

<a id="c3-topn"></a>
## c3 每用户观看最久的前 10 个视频（TopN）

### 做什么

用户行为序列上的 Top-K 特征，如"最近观看最久的 10 个视频"。append-only 的 `ROW_NUMBER` TopN。GPU 把 `Rank` 换成 GPU TopN，有三种形态：batch（攒批）、persistent（常驻）、adaptive（自适应）。实测模式：`cpu`、`gpu-batch`、`gpu-persistent`、`gpu-adaptive`。

### SQL

```sql
SELECT seq, user_id, video_id, play_time_ms FROM (
  SELECT seq, user_id, video_id, play_time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_time_ms DESC, seq ASC) AS rn
  FROM events) WHERE rn <= 10
```

### 真实执行计划

CPU，Optimized Execution Plan：

```text
Rank(strategy=[AppendFastStrategy], rankType=[ROW_NUMBER], rankRange=[rankStart=1, rankEnd=10], partitionBy=[user_id], orderBy=[play_time_ms DESC, seq ASC], select=[seq, user_id, video_id, play_time_ms])
+- Exchange(distribution=[hash[user_id]])
   +- Calc(select=[seq, user_id, video_id, play_time_ms])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

GPU batch，Optimized Execution Plan：

```text
GpuNativeStreamExecRank(mode=gpu-native, engine=batch, rankType=ROW_NUMBER, topN=10, batchRows=4096, maxWaitMs=5) Rank(strategy=[AppendFastStrategy], rankType=[ROW_NUMBER], rankRange=[rankStart=1, rankEnd=10], partitionBy=[user_id], orderBy=[play_time_ms DESC, seq ASC], select=[seq, user_id, video_id, play_time_ms])
+- Exchange(distribution=[hash[user_id]])
   +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, user_id, video_id, play_time_ms])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

四个模式的 Optimized Physical Plan 相同（`Exchange(hash[user_id])` 之上的 `Rank(AppendFastStrategy)`，其下是 `Calc`）。替换发生在 Optimized Execution Plan。

- `GpuNativeStreamExecRank`：替换 `Rank`。`rankType=ROW_NUMBER` 与 `topN=10` 重复了其后 Flink Rank 的排名类型和范围。覆盖文档把 Rank 列为实验性混合执行算子；`topn.engine=adaptive` 按压力选 CPU 或 batch，persistent 仅适用于 ROW_NUMBER。
- `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)`：exchange 之下的 Calc 仍在 CPU（见文首 Gpu\* 说明）。
- `Exchange(hash[user_id])` 与 source 扫描不变。

### 各模式的计划差异

三个 GPU 计划形状相同，只有 Rank 行里的 `engine` 值不同。另外两个模式的 Rank 行：

```text
GpuNativeStreamExecRank(mode=gpu-native, engine=persistent, rankType=ROW_NUMBER, topN=10, batchRows=4096, maxWaitMs=5) Rank(strategy=[AppendFastStrategy], rankType=[ROW_NUMBER], rankRange=[rankStart=1, rankEnd=10], partitionBy=[user_id], orderBy=[play_time_ms DESC, seq ASC], select=[seq, user_id, video_id, play_time_ms])
```

```text
GpuNativeStreamExecRank(mode=gpu-native, engine=adaptive, rankType=ROW_NUMBER, topN=10, batchRows=4096, maxWaitMs=5) Rank(strategy=[AppendFastStrategy], rankType=[ROW_NUMBER], rankRange=[rankStart=1, rankEnd=10], partitionBy=[user_id], orderBy=[play_time_ms DESC, seq ASC], select=[seq, user_id, video_id, play_time_ms])
```

- `engine=batch`：攒批的 GPU Top-N。
- `engine=persistent`：常驻形态。覆盖文档说明 persistent 仅适用于 ROW_NUMBER。
- `engine=adaptive`：算子内同时保留 CPU 路径和 GPU 路径，按压力选择。实测中低/中档 0 行走 GPU（延迟与 CPU 相同，不是 GPU 的结果）；打满档约 36% 的行走 CPU 路径。

四个模式的作业图相同：一条链是 source 与 Calc，另一条链是 Rank、约束检查与 sink。

### GPU 确实在跑的证据

batch 模式下，批数、H2D 时间和计算时间在所有正式轮次都不为 0。persistent 的 persistent 命令数等于全部输入。adaptive 的计数在低/中档全部为 0、打满档多数轮次不为 0，与低/中档 0 行走 GPU 一致。nsys，默认配置，独占：910 次 `select_top_n_by_group` kernel 合计 108 ms，加 H2D 5 ms、D2H 2 ms。

### 结果

**默认配置**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 低 | CPU | 19,995 | 14.70 / 33.03 | MATCH |
| 低 | GPU batch | 19,995 | 16.81 / 33.23 | MATCH |
| 低 | GPU persistent | 19,995 | 14.73 / 33.14 | MATCH |
| 低 | GPU adaptive | 19,995 | 14.99 / 33.04 | MATCH |
| 中 | CPU | 99,977 | 3.58 / 6.69 | MATCH |
| 中 | GPU batch | 99,976 | 8.59 / 16.11 | MATCH |
| 中 | GPU persistent | 99,976 | 4.14 / 8.92 | MATCH |
| 中 | GPU adaptive | 99,979 | 4.15 / 7.40 | MATCH |
| 打满 | CPU | 1,024,400 | 4.69 / 20.25 | MATCH |
| 打满 | GPU batch | 930,760 | 5.61 / 25.01 | MATCH |
| 打满 | GPU persistent | 489,815 | 11.20 / 33.67 | MATCH |
| 打满 | GPU adaptive | 690,440 | 10.28 / 32.89 | MATCH |

GPU batch 打满档轮间波动大：8 轮复测 0.90–1.16M，呈双峰（5 轮约 0.93M，3 轮 1.13–1.16M）；CPU 8 轮 1.02–1.12M，中位数 1.06M。因此不能说 GPU 稳定胜出或稳定落后。

**调优（两边开 object-reuse，另加 topn-cache-size=100000）**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 中 | CPU | 99,978 | 3.53 / 6.70 | MATCH |
| 中 | GPU batch | 99,972 | 8.64 / 15.12 | MATCH |
| 中 | GPU persistent | 99,977 | 4.16 / 13.48 | MATCH |
| 中 | GPU adaptive | 99,978 | 4.05 / 10.77 | MATCH |
| 打满 | CPU | 1,363,629 | 3.59 / 22.79 | MATCH |
| 打满 | GPU batch | 1,231,075 | 5.22 / 29.97 | MATCH |
| 打满 | GPU persistent | 430,154 | 12.87 / 38.05 | MATCH |
| 打满 | GPU adaptive | 718,364 | 9.48 / 31.29 | MATCH |

低档调优：未测。

### 时间构成

默认配置，独占，nsys 与参数扫描：910 次 `select_top_n_by_group` kernel 合计 108 ms，加 H2D 5 ms、D2H 2 ms，约占约 3.2 s 作业时间的 3.6%，GPU 设备 95% 以上空闲。并行度 1 下，kernel 约 0.12 s，整个作业（3.3M 行）约 3.2 s（按正式中位数）。

### 结论

CPU 不输：调优后 CPU +11%（对 GPU batch），并行度 2/4 下 CPU +38% / +21%（见 [benchmarks/README.zh-CN.md](README.zh-CN.md) 第 5 节）。persistent 仅在低/中档延迟优于 batch，但不优于 CPU，且打满吞吐很低；adaptive 在低/中档 0 行走 GPU，打满档约 36% 的行走 CPU 路径。

**为什么 GPU 没赢**（默认配置，独占，nsys 与参数扫描）：批大小（1,024 到 65,536）与最长等待（1/5/20 ms）扫描下，打满吞吐都在 0.87-1.11M，无稳定趋势；只有 source + 交换 + sink、不做任何计算的上限探针，CPU 0.856M、GPU 0.851M，说明瓶颈在 JVM 流水线。GPU batch 在打满档出现两簇结果，其机制我们不下结论。

<a id="c4-dedup"></a>
## c4 keep-last 去重

### 做什么

样本去重、取最新状态。按 (user_id, video_id) 保留时间最新的一条。GPU 把去重节点换成 GPU Deduplicate（batch / adaptive）。另测开 mini-batch 的 CPU（`cpu-mb`）。实测模式：`cpu`、`cpu-mb`、`gpu-batch`、`gpu-adaptive`。

### SQL

```sql
SELECT seq, user_id, video_id, play_time_ms, time_ms FROM (
  SELECT seq, user_id, video_id, play_time_ms, time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY ts DESC) AS rn
  FROM events) WHERE rn = 1
```

### 真实执行计划

CPU，Optimized Execution Plan：

```text
Calc(select=[seq, user_id, video_id, play_time_ms, time_ms])
+- Deduplicate(keep=[LastRow], key=[user_id, video_id], order=[ROWTIME], outputInsertOnly=[false])
   +- Exchange(distribution=[hash[user_id, video_id]])
      +- Calc(select=[seq, user_id, video_id, time_ms, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
         +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

GPU batch，Optimized Execution Plan：

```text
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, user_id, video_id, play_time_ms, time_ms])
+- GpuNativeStreamExecDeduplicate(mode=gpu-native, engine=batch, batchRows=4096, maxWaitMs=5) Deduplicate(keep=[LastRow], key=[user_id, video_id], order=[ROWTIME], outputInsertOnly=[false])
   +- Exchange(distribution=[hash[user_id, video_id]])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, user_id, video_id, time_ms, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
         +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

对此前描述的更正：c4 的计划此前被写成 `Rank(AppendFast, keep-last)`。保存下来的计划显示，Flink 把这条 SQL 变成 `Deduplicate(keep=[LastRow], ...)` 节点，而不是 Rank。

`cpu`、`gpu-batch` 与 `gpu-adaptive` 的 Optimized Physical Plan 相同。替换发生在 Optimized Execution Plan。

- `GpuNativeStreamExecDeduplicate`：替换 `Deduplicate`。覆盖文档把 keep-last Deduplicate 列为实验性混合执行算子，用 `deduplicate.engine=adaptive` 或明确的 batch 设置开启。
- 去重节点上下的 Calc 显示为 `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)`（见文首 Gpu\* 说明）。

### 各模式的计划差异

`gpu-adaptive` 与 `gpu-batch` 只有 engine 值不同：

```text
+- GpuNativeStreamExecDeduplicate(mode=gpu-native, engine=adaptive, batchRows=4096, maxWaitMs=5) Deduplicate(keep=[LastRow], key=[user_id, video_id], order=[ROWTIME], outputInsertOnly=[false])
```

`cpu-mb`（开 mini-batch 的 CPU）是另一份 CPU 计划：它的 Optimized Physical Plan 与 Optimized Execution Plan 都多了一个 `MiniBatchAssigner` 节点。与 `cpu` 不同的行：

```text
         +- MiniBatchAssigner(interval=[5ms], mode=[RowTime])
            +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

作业图：CPU 作业图里，带 `Deduplicate` 的顶点后面接着一个 Calc。两个 GPU 作业图里，Deduplicate 顶点直接接约束检查，没有单独列出 Calc 顶点，尽管上面的执行计划里仍有这个 Calc。原因：未核实。

### GPU 确实在跑的证据

batch 模式下，批数、H2D 时间和计算时间在所有正式轮次都不为 0。GPU 实测计时（默认，打满）：983 批上 kernel 13 ms、H2D 8 ms、D2H 8 ms。adaptive 的计数在低/中档全部为 0、打满档不为 0，与低/中档 0 行走 GPU 一致。

### 结果

**默认配置**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 低 | CPU | 19,995 | 11.49 / 24.98 | MATCH |
| 低 | CPU mini-batch | 19,995 | 22.50 / 43.65 | MATCH |
| 低 | GPU batch | 19,995 | 14.54 / 25.23 | MATCH |
| 低 | GPU adaptive | 19,995 | 11.52 / 24.95 | MATCH |
| 中 | CPU | 99,962 | 2.88 / 5.06 | MATCH |
| 中 | CPU mini-batch | 99,950 | 14.93 / 30.94 | MATCH |
| 中 | GPU batch | 99,967 | 5.99 / 16.08 | MATCH |
| 中 | GPU adaptive | 99,967 | 2.86 / 5.06 | MATCH |
| 打满 | CPU | 685,882 | 5.23 / 19.63 | MATCH |
| 打满 | CPU mini-batch | 580,305 | 7.74 / 26.90 | MATCH |
| 打满 | GPU batch | 660,054 | 7.22 / 27.85 | MATCH |
| 打满 | GPU adaptive | 675,786 | 6.98 / 28.34 | MATCH |

**调优**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 中 | CPU | 99,970 | 2.74 / 5.06 | MATCH |
| 中 | GPU batch | 99,964 | 5.93 / 13.95 | MATCH |
| 中 | GPU adaptive | 99,970 | 2.86 / 7.19 | MATCH |
| 打满 | CPU | 1,033,538 | 3.30 / 17.82 | MATCH |
| 打满 | GPU batch | 850,292 | 5.96 / 21.96 | MATCH |
| 打满 | GPU adaptive | 852,773 | 6.03 / 22.28 | MATCH |

低档调优：未测。

### 时间构成

去重的主要成本是序列化与 JVM 状态读写，不在 GPU 能替换的部分。JFR 估算（采样，非精确）：CPU 状态约 14%，GPU 模式状态约 17%。GPU 实测 983 批上 kernel 13 ms、H2D 8 ms、D2H 8 ms。

### 结论

默认配置下 CPU 略好（0.686M 对 0.660M），调优后 CPU +22%。adaptive 在低/中档 0 行走 GPU；打满档约 4.5% 的行走 CPU 路径。

<a id="c5-join"></a>
## c5 曝光 × 点击 Interval Join，及后接 Filter / 聚合的融合

曝光-点击归因、训练样本拼接。曝光流 `imp` 与点击流 `clk`（`is_click=1`）取自同一份日志，join 键为 `user_id*10^8+video_id`（单 BIGINT），条件为点击时间落在曝光后 30 分钟内。三个变体：

- `c5-join`：只做 join。
- `c5-join-filter`：join 后加 `c.play_time_ms >= i.duration_ms`。Flink 将其下推为 join 的残差条件；GPU 融合模式会把残差条件放到设备端执行，因此"融合 vs 不融合"之差 = 融合 + 残差下放。
- `c5-join-agg`：join 后按 user_id 聚合（开 mini-batch 5 ms / 4096，两边相同）。

共用的两个视图（三个变体都用）：

```sql
CREATE TEMPORARY VIEW imp AS
  SELECT seq, user_id * 100000000 + video_id AS uv, user_id, duration_ms, ts FROM events;
CREATE TEMPORARY VIEW clk AS
  SELECT seq, user_id * 100000000 + video_id AS uv, play_time_ms, ts FROM events WHERE is_click = 1;
```

GPU 融合模式下，join 后的 Calc / LocalAggregate 与 join 在同一个 GPU 阶段内完成，少一次列→行与行→列。`c5-join` 的 GPU batch 默认即为融合（Join→投影）。

**融合在计划里长什么样。** 对 `join-filter` 和 `join-agg`，融合与不融合两次运行的 Optimized Execution Plan 文本**完全相同**。差别只出现在作业图：融合时，join 之后的算子（Calc，join-agg 还有 LocalGroupAggregate）不再作为 `IntervalJoin` 之后的独立顶点列出；不融合时它们接在 join 后面的同一条链里。下面各节分别列出。

**c5 里的 Gpu\* 节点。**

- `GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=..., bounds=[-1800000,0], ...)`：替换 `IntervalJoin`。覆盖文档把 Interval Join 列为带 batch 与 persistent 引擎的实验性 GPU 算子，适用于 append-only、event-time、单个 BIGINT 等值 key、一个有界时间条件的 INNER join。`bounds` 与 IntervalJoin 行里的 `leftLowerBound` / `leftUpperBound` 是同样的两个值（30 分钟，以毫秒表示）；其余参数见文首 Gpu\* 说明。按文档，不满足这个形状的 join 回退 CPU。
- `FlinkCpuStreamExecCalc(...)`：Calc 节点仍在 Flink CPU。
- 覆盖文档描述了 Interval Join 与紧随其后的 Calc（数值条件、直接字段或安全 cast 的投影）以及后接的两阶段 Local Aggregate 的深融合。

<a id="c5-join-only"></a>
### c5-join

#### 做什么

只做 join：把每条曝光与同一用户同一视频 30 分钟内的点击配对。实测模式：`cpu`、`gpu-batch`（默认融合）、`gpu-persistent`（默认融合）。表里还有"GPU 不融合"一列；该次运行没有保存计划文件（未核实）。

#### SQL

```sql
-- 视图 imp 与 clk 见上文
SELECT i.seq, c.seq, i.uv, c.play_time_ms, i.duration_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE;
```

#### 真实执行计划

CPU，Optimized Execution Plan：

```text
Calc(select=[seq, seq0, uv, play_time_ms, duration_ms])
+- IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
   :- Exchange(distribution=[hash[uv]])
   :  +- Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, duration_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
   :     +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])(reuse_id=[1])
   +- Exchange(distribution=[hash[uv]])
      +- Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
         +- Reused(reference_id=[1])
```

GPU batch，Optimized Execution Plan：

```text
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, seq0, uv, play_time_ms, duration_ms])
+- GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=batch, bounds=[-1800000,0], batchRows=4096, maxWaitMs=5) IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
   :- Exchange(distribution=[hash[uv]])
   :  +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, duration_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
   :     +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])(reuse_id=[1])
   +- Exchange(distribution=[hash[uv]])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
         +- Reused(reference_id=[1])
```

`cpu`、`gpu-batch` 与 `gpu-persistent` 的 Optimized Physical Plan 相同。替换发生在 Optimized Execution Plan：`IntervalJoin` 变成 `GpuNativeStreamExecIntervalJoin`，Calc 节点变成 `FlinkCpuStreamExecCalc`。

#### 各模式的计划差异

`gpu-persistent` 与 `gpu-batch` 只有 engine 值不同：

```text
+- GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=persistent, bounds=[-1800000,0], batchRows=4096, maxWaitMs=5) IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
```

作业图：CPU 作业图里，`IntervalJoin` 后面接着一个 Calc。`gpu-batch` 与 `gpu-persistent` 的作业图里，该顶点只有 `IntervalJoin`，直接接约束检查：投影已融合进 join（这是默认的融合形态）。

CPU 作业图（省略 source 顶点）：

```text
- IntervalJoin[13] -> Calc[14] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

GPU batch 作业图（省略 source 顶点）：

```text
- IntervalJoin[13] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

#### GPU 确实在跑的证据

batch 模式下，join 的计数（kernel 数、H2D 时间、计算时间）在所有正式轮次都不为 0。persistent 的命令计数只在 breakdown 轮有记录，正式轮为 0，原因未核实。

#### 结果

**默认**（同一作业内显式 fuse / nofuse，3 轮中位数；吞吐 / p50 / p99）

| 用例 | 档 | CPU 吞吐 / p50 / p99 | GPU 融合 | GPU 不融合 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| join | 低 | 19,994 / 29.91 / 79.26 | 19,993 / 29.88 / 80.09 | 19,993 / 29.98 / 79.91 | MATCH |
| join | 中 | 99,928 / 10.57 / 24.84 | 99,921 / 14.08 / 27.56 | 99,933 / 14.23 / 26.58 | MATCH |
| join | 打满 | 259,430 / 20.62 / 55.11 | 232,091 / 24.97 / 56.24 | 221,821 / 26.10 / 56.35 | MATCH |

GPU persistent 形态（默认融合）：join 打满 166,494（p99 199.94 ms）、join-filter 打满 171,651（p99 197.54 ms），各档尾延迟约 200 ms，均明显差于 CPU 与 batch。

**调优**（两边同开 object-reuse；低档：未测）

| 用例 | 档 | CPU 吞吐 / p50 / p99 | GPU 融合 | GPU 不融合 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| join | 中 | 99,928 / 10.38 / 37.56 | 99,931 / 13.89 / 26.79 | 99,938 / 13.84 / 25.97 | MATCH |
| join | 打满 | 291,797 / 19.63 / 60.01 | 253,233 / 24.07 / 54.08 | 251,270 / 24.24 / 52.71 | MATCH |

#### 时间构成

两边最大的成本都是 JVM 状态读写（join 两侧的缓冲，JFR 估算，采样：CPU join 约 52%、GPU join 约 43%），GPU 没有替换它。本数据上 join 放大倍数约为 1，融合能省的搬运量有限。

#### 结论

吞吐上 CPU 更快：调优后 join +15%（对 GPU 融合）。调优后融合没有吞吐收益（join +0.8%，噪声内）；join 延迟无差别。persistent 的尾延迟明显差于 CPU 与 batch。

<a id="c5-join-filter"></a>
### c5-join-filter

#### 做什么

join 之后再按配对结果过滤：`c.play_time_ms >= i.duration_ms`。实测模式：`cpu`、`gpu-batch-fuse`、`gpu-batch-nofuse`；persistent（默认融合）见结果表下的说明。

#### SQL

```sql
-- 视图 imp 与 clk 见上文
SELECT i.seq, c.seq, i.uv, c.play_time_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE
WHERE c.play_time_ms >= i.duration_ms;
```

#### 真实执行计划

CPU，Optimized Execution Plan：

```text
Calc(select=[seq, seq0, uv, play_time_ms])
+- IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)) AND (play_time_ms >= duration_ms))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
   :- Exchange(distribution=[hash[uv]])
   :  +- Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, duration_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
   :     +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])(reuse_id=[1])
   +- Exchange(distribution=[hash[uv]])
      +- Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
         +- Reused(reference_id=[1])
```

GPU batch 融合（`gpu-batch-fuse`），Optimized Execution Plan：

```text
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, seq0, uv, play_time_ms])
+- GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=batch, bounds=[-1800000,0], batchRows=4096, maxWaitMs=5) IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)) AND (play_time_ms >= duration_ms))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
   :- Exchange(distribution=[hash[uv]])
   :  +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, duration_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
   :     +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])(reuse_id=[1])
   +- Exchange(distribution=[hash[uv]])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
         +- Reused(reference_id=[1])
```

`cpu`、`gpu-batch-fuse` 与 `gpu-batch-nofuse` 的 Optimized Physical Plan 相同。过滤条件已被下推到 join：IntervalJoin 行的 `where=[...]` 里包含 `play_time_ms >= duration_ms`，所以 join 之后的 Calc 只是投影。替换发生在 Optimized Execution Plan，与 c5-join 相同。

#### 各模式的计划差异（fuse / nofuse）

`gpu-batch-nofuse` 的 Optimized Execution Plan 与 `gpu-batch-fuse` 完全相同（我们比对了两份文本）。差别在作业图（省略 source 顶点）：

融合：

```text
- IntervalJoin[13] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

不融合：

```text
- IntervalJoin[13] -> Calc[14] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

CPU：

```text
- IntervalJoin[13] -> Calc[14] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

融合时，`IntervalJoin` 后面不再有单独的 Calc 顶点；不融合时仍有，与 CPU 相同。按覆盖文档，残差条件和投影随后在 join 阶段内求值。这里"融合 vs 不融合"还包含残差条件下放到设备端。persistent 没有保存计划文件（未核实）。

#### GPU 确实在跑的证据

batch 模式（融合与不融合）下，join 的计数在所有正式轮次都不为 0。persistent 与 c5-join 一样，正式轮的命令计数不能作为证据（c5 persistent 正式轮为 0，原因未核实）。

#### 结果

**默认**（同一作业内显式 fuse / nofuse，3 轮中位数；吞吐 / p50 / p99）

| 用例 | 档 | CPU 吞吐 / p50 / p99 | GPU 融合 | GPU 不融合 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| join-filter | 低 | 19,993 / 29.64 / 79.08 | 19,990 / 30.00 / 80.25 | 19,989 / 29.97 / 80.03 | MATCH |
| join-filter | 中 | 99,928 / 10.36 / 26.64 | 99,953 / 13.95 / 26.62 | 99,904 / 13.95 / 25.87 | MATCH |
| join-filter | 打满 | 272,760 / 20.17 / 56.22 | 254,153 / 22.89 / 54.92 | 250,751 / 23.30 / 54.38 | MATCH |

join-filter 的"不融合 + 设备端残差"（打满）为 251,131，与"不融合"251k 基本相同，说明残差下放单独几乎没有收益。

GPU persistent 形态（默认融合）：join 打满 166,494（p99 199.94 ms）、join-filter 打满 171,651（p99 197.54 ms），各档尾延迟约 200 ms，均明显差于 CPU 与 batch。

**调优**（两边同开 object-reuse；低档：未测）

| 用例 | 档 | CPU 吞吐 / p50 / p99 | GPU 融合 | GPU 不融合 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| join-filter | 中 | 99,927 / 10.09 / 30.54 | 99,903 / 13.82 / 25.54 | 99,921 / 13.71 / 25.68 | MATCH |
| join-filter | 打满 | 307,921 / 18.80 / 63.22 | 262,297 / 23.49 / 51.55 | 259,502 / 23.67 / 52.45 | MATCH |

#### 时间构成

与 c5-join 相同：两边最大的成本都是 JVM 状态读写（join 两侧的缓冲，JFR 估算，采样：CPU join 约 52%、GPU join 约 43%），GPU 没有替换它。本数据上 join 放大倍数约为 1。

#### 结论

吞吐上 CPU 更快：调优后 join-filter +17%（对 GPU 融合）。调优后融合没有吞吐收益（join-filter +1.1%，噪声内）；延迟无差别。

<a id="c5-join-agg"></a>
### c5-join-agg

#### 做什么

join 之后按 user_id 做 COUNT / SUM（mini-batch 5 ms / 4096，两边相同）。实测模式：`cpu`、`gpu-batch-fuse`、`gpu-batch-nofuse`。

#### SQL

```sql
-- 视图 imp 与 clk 见上文
SELECT i.user_id, COUNT(*) AS clicks, SUM(c.play_time_ms) AS play_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE
GROUP BY i.user_id;
```

#### 真实执行计划

CPU，Optimized Execution Plan：

```text
GlobalGroupAggregate(groupBy=[user_id], select=[user_id, COUNT(count1$0) AS clicks, SUM(sum$1) AS play_ms])
+- Exchange(distribution=[hash[user_id]])
   +- LocalGroupAggregate(groupBy=[user_id], select=[user_id, COUNT(*) AS count1$0, SUM(play_time_ms) AS sum$1])
      +- Calc(select=[user_id, play_time_ms])
         +- IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=2, rightTimeIndex=2], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[uv, user_id, ts, uv0, play_time_ms, ts0])
            :- Exchange(distribution=[hash[uv]])
            :  +- Calc(select=[((user_id * 100000000) + video_id) AS uv, user_id, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            :     +- MiniBatchAssigner(interval=[5ms], mode=[RowTime])(reuse_id=[1])
            :        +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
            +- Exchange(distribution=[hash[uv]])
               +- Calc(select=[((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
                  +- Reused(reference_id=[1])
```

GPU batch 融合（`gpu-batch-fuse`），Optimized Execution Plan：

```text
GpuAssistedStreamExecGlobalGroupAggregate(local=project-owned-cuda, merge=project-owned-cpu) GlobalGroupAggregate(groupBy=[user_id], select=[user_id, COUNT(count1$0) AS clicks, SUM(sum$1) AS play_ms])
+- Exchange(distribution=[hash[user_id]])
   +- GpuNativeStreamExecLocalGroupAggregate(mode=gpu-native, engine=batch, batchRows=4096, maxWaitMs=5, persistentMaxGroups=65536) LocalGroupAggregate(groupBy=[user_id], select=[user_id, COUNT(*) AS count1$0, SUM(play_time_ms) AS sum$1])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, play_time_ms])
         +- GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=batch, bounds=[-1800000,0], batchRows=4096, maxWaitMs=5) IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=2, rightTimeIndex=2], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[uv, user_id, ts, uv0, play_time_ms, ts0])
            :- Exchange(distribution=[hash[uv]])
            :  +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[((user_id * 100000000) + video_id) AS uv, user_id, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            :     +- MiniBatchAssigner(interval=[5ms], mode=[RowTime])(reuse_id=[1])
            :        +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
            +- Exchange(distribution=[hash[uv]])
               +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
                  +- Reused(reference_id=[1])
```

`cpu`、`gpu-batch-fuse` 与 `gpu-batch-nofuse` 的 Optimized Physical Plan 相同（两阶段聚合：`LocalGroupAggregate`、`Exchange(hash[user_id])`、`GlobalGroupAggregate`）。替换发生在 Optimized Execution Plan。这里有三个节点被换掉：`IntervalJoin`、`LocalGroupAggregate` 与 `GlobalGroupAggregate`。

- `GpuNativeStreamExecLocalGroupAggregate(..., persistentMaxGroups=...)`：替换 `LocalGroupAggregate`（两阶段聚合的 Local 阶段）。`persistentMaxGroups` 由 Mega-Flink 定义。覆盖文档把 Local + Global 聚合列为实验性 GPU 算子，计划里还需要 Flink mini-batch。
- `GpuAssistedStreamExecGlobalGroupAggregate(local=project-owned-cuda, merge=project-owned-cpu)`：替换 `GlobalGroupAggregate`。参数名由 Mega-Flink 定义。覆盖文档把 Global 阶段描述为 JVM 上的 merge 加 Flink keyed state，所以状态与 merge 留在 JVM。

#### 各模式的计划差异（fuse / nofuse）

与 join-filter 相同，`gpu-batch-nofuse` 的 Optimized Execution Plan 与 `gpu-batch-fuse` 完全相同。差别在作业图（省略 source 顶点）：

融合：

```text
- GlobalGroupAggregate[22] -> ConstraintEnforcer[23] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
- IntervalJoin[18]  [parallelism=1]
```

不融合：

```text
- GlobalGroupAggregate[22] -> ConstraintEnforcer[23] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
- IntervalJoin[18] -> Calc[19] -> LocalGroupAggregate[20]  [parallelism=1]
```

CPU：

```text
- GlobalGroupAggregate[22] -> ConstraintEnforcer[23] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
- IntervalJoin[18] -> Calc[19] -> LocalGroupAggregate[20]  [parallelism=1]
```

融合时，`IntervalJoin` 自成一个顶点，其后的 Calc 与 `LocalGroupAggregate` 没有列出。不融合时（以及 CPU 上）它们接在 join 后面的同一条链里。按覆盖文档，深融合在与 join 相同的一次 CUDA range probe 里更新 COUNT/SUM，只向 exchange 输出压缩后的 partial。Global 阶段在所有模式里都是独立顶点。

#### GPU 确实在跑的证据

batch 模式（融合与不融合）下，join 的计数在所有正式轮次都不为 0。

#### 结果

**默认**（同一作业内显式 fuse / nofuse，3 轮中位数；吞吐 / p50 / p99）

| 用例 | 档 | CPU 吞吐 / p50 / p99 | GPU 融合 | GPU 不融合 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| join-agg | 低 | 19,861 / 107.58 / 195.47 | 19,958 / 92.68 / 180.70 | 19,859 / 96.89 / 183.59 | MATCH |
| join-agg | 中 | 99,252 / 55.91 / 102.08 | 99,774 / 37.97 / 76.68 | 99,397 / 43.59 / 86.78 | MATCH |
| join-agg | 打满 | 268,556 / 60.56 / 121.23 | 233,707 / 44.49 / 83.84 | 232,152 / 50.97 / 98.09 | MATCH |

**调优**（两边同开 object-reuse；低档：未测）

| 用例 | 档 | CPU 吞吐 / p50 / p99 | GPU 融合 | GPU 不融合 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| join-agg | 中 | 99,141 / 55.20 / 102.33 | 99,798 / 37.57 / 76.84 | 99,377 / 42.76 / 86.52 | MATCH |
| join-agg | 打满 | 293,385 / 58.96 / 127.08 | 236,056 / 45.31 / 84.37 | 253,569 / 48.46 / 98.14 | MATCH |

join-agg 打满档 CPU 三轮为 293k、257k、300k，波动较大。

#### 时间构成

两边最大的成本都是 JVM 状态读写（join 两侧的缓冲，JFR 估算，采样：CPU join 约 52%、GPU join 约 43%），GPU 没有替换它。本数据上 join 放大倍数约为 1，融合能省的搬运量有限。

#### 结论

吞吐上 CPU 更快：调优后 join-agg +24%（对 GPU 融合）。接聚合时 GPU 延迟更低（p50 约低 23%）。融合：调优后吞吐无收益（join-agg 融合反而低约 7%），只有 join-agg 的 p50 延迟低约 6.5%（打满）/ 12%（中档）。

<a id="summary"></a>
## 本组小结

| 用例 | CPU | GPU | 较快 |
| --- | --- | --- | --- |
| c1 过滤清洗 | 2,472,421 | 2,059,561 | CPU +20% |
| c3 TopN | 1,363,629 | 1,231,075（batch） | CPU +11% |
| c4 去重 | 1,033,538 | 850,292（batch） | CPU +22% |
| c5 join | 291,797 | 253,233（融合） | CPU +15% |
| c5 join-filter | 307,921 | 262,297（融合） | CPU +17% |
| c5 join-agg | 293,385 | 236,056（融合） | CPU +24%（GPU p50 延迟约低 23%） |

调优后打满吞吐（条/秒），单卡，并行度 1。这四个用例里，GPU 接手的计算量比序列化、JVM 状态读写和行列转换小，而这些 GPU 替换不了。persistent 与 adaptive 形态没有得到"选 persistent"的支持（adaptive 在低/中档 0 行走 GPU）。适用的限制：单卡、并行度主要为 1；c4、c5 在并行度 2/4 未测；调优后的低档未测；改写后的报告未经独立复核。
