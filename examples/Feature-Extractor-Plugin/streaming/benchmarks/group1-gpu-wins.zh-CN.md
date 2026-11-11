# 第 1 组：GPU 赢（c2 窗口聚合、c6 维表 Lookup Join）

返回[总览](README.zh-CN.md)。这两个用例在打满档，默认与调优两套表里都是 GPU 更快。测试环境、档位、统计方法以及"默认""调优"的含义见 [README 第 2、3 节](README.zh-CN.md#env)。吞吐单位为条/秒，延迟单位为 ms；"对拍"列比较最终表的 SHA-256，按用例与档位。

- [c2 窗口聚合（HOP 与 TUMBLE）](#c2-window)
- [c6 维表 Lookup Join](#c6-lookup)

## 怎么读计划

Flink 对一条查询会打印多份计划，这里有两份有用。

- **优化后的物理计划（Optimized Physical Plan）**：Flink 优化器的输出，CPU 与 GPU 相同。
- **优化后的执行计划（Optimized Execution Plan）**：选定 ExecNode 之后的计划。GPU 模式下，被 Mega-Flink 替换的节点在原 Flink 节点文本前多一个 `<节点名>(<参数>)` 前缀；留在 CPU 的节点显示为 `FlinkCpuStreamExecCalc(...)`。CPU 与 GPU 的差别就在这一份。

下文的计划文本取自测试作业导出的 EXPLAIN，只把数据源扫描里很长的 `fields=[...]` 列表截掉，写成 `…`。节点名和参数由 Mega-Flink 定义；我们只解释其算子覆盖文档里有依据的部分，没有依据的写"文档未定义"。

两个用例里都有 `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)`。Mega-Flink 文档里有 Calc 引擎设置（`auto`、`cpu`、`persistent`、`batch`），并说明 `auto` 为了低延迟让独立的 Calc 走 CPU。所以这两个参数读作"实际引擎：cpu""配置值：auto"（这是我们对名字的理解）；GPU 模式下这些 Calc 仍在 Flink CPU 上运行。

<a id="c2-window"></a>
## c2 每用户最近 30 分钟点击数的窗口聚合（HOP 与 TUMBLE）

### 做什么

实时统计特征，如"用户最近 N 分钟的曝光 / 点击 / 播放时长"。事件时间窗口，按 user_id 聚合。HOP 窗口长 30 分钟、步长 5 分钟（每条记录落入多个窗口）；TUMBLE 为 5 分钟滚动窗口。CPU 默认把窗口聚合拆为 Local + Global 两阶段；另测一阶段（`1p`，直接 WindowAggregate）。GPU 两阶段对应实验性的 `LocalWindowAggregate` 路径，一阶段对应 Mega-Flink 较成熟的路径。

### SQL

```sql
-- HOP
SELECT user_id, window_start, window_end, COUNT(*) AS expo_cnt,
       SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms
FROM TABLE(HOP(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE, INTERVAL '30' MINUTE))
GROUP BY user_id, window_start, window_end;

-- TUMBLE：把 HOP(...) 换成
TABLE(TUMBLE(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE))
```

### 真实执行计划

优化后的物理计划（CPU 与 GPU 相同；默认的两阶段，已截掉数据源字段）：

```
Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GlobalWindowAggregate(groupBy=[user_id], window=[HOP(slice_end=[$slice_end], size=[30 min], slide=[5 min])], select=[user_id, COUNT(count1$0) AS expo_cnt, SUM(sum$1) AS click_cnt, SUM(sum$2) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- LocalWindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS count1$0, SUM(is_click) AS sum$1, SUM(play_time_ms) AS sum$2, slice_end('w$) AS $slice_end])
         +- Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

优化后的执行计划，CPU（两阶段）：

```
Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GlobalWindowAggregate(groupBy=[user_id], window=[HOP(slice_end=[$slice_end], size=[30 min], slide=[5 min])], select=[user_id, COUNT(count1$0) AS expo_cnt, SUM(sum$1) AS click_cnt, SUM(sum$2) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- LocalWindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS count1$0, SUM(is_click) AS sum$1, SUM(play_time_ms) AS sum$2, slice_end('w$) AS $slice_end])
         +- Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

优化后的执行计划，GPU（两阶段）：

```
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GpuAssistedStreamExecGlobalWindowAggregate(local=project-owned-cuda, merge=project-owned-cpu, direct-watermark-output=true) GlobalWindowAggregate(groupBy=[user_id], window=[HOP(slice_end=[$slice_end], size=[30 min], slide=[5 min])], select=[user_id, COUNT(count1$0) AS expo_cnt, SUM(sum$1) AS click_cnt, SUM(sum$2) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- GpuNativeStreamExecLocalWindowAggregate(mode=gpu-native, batchRows=4096, maxWaitMs=5) LocalWindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS count1$0, SUM(is_click) AS sum$1, SUM(play_time_ms) AS sum$2, slice_end('w$) AS $slice_end])
         +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

替换发生在执行计划这一层，不在物理计划。GPU 模式下各节点的含义：

| 节点与参数 | 说明 |
| --- | --- |
| `GpuNativeStreamExecLocalWindowAggregate(mode=gpu-native, batchRows=4096, maxWaitMs=5)` | 替换 Local 窗口聚合。Mega-Flink 文档说明：支持 UTC 事件时间的 TUMBLE / HOP / CUMULATE、仅 INSERT 输入，在 GPU 上直接生成 Flink 的局部 accumulator 与 `$slice_end`，多个 GPU 小批在主机上按 key 与 slice 合并后再交给 Exchange。这是实验性路径（需打开实验算子开关）。`batchRows=4096`、`maxWaitMs=5` 即攒批大小与最长等待（见总览第 2 节）。`mode=gpu-native` 的含义文档未定义。 |
| `GpuAssistedStreamExecGlobalWindowAggregate(local=project-owned-cuda, merge=project-owned-cpu, direct-watermark-output=true)` | 替换 Global 窗口聚合。文档说明：Global 阶段用项目专用的 JVM 合并（不再发起第二次 GPU kernel），未关闭的窗口仍写入 Flink keyed state 并注册事件时间 timer，watermark 已关闭的窗口直接合并并只写一次最终行。三个参数值读作：Local 阶段是项目自己的 CUDA 代码，合并是项目自己的 CPU 代码，watermark 关闭的窗口直接输出。具体取值字符串文档未定义。同为实验性。 |
| `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)` | 两个 Calc 仍在 Flink CPU 上（见上）。 |
| `Exchange(distribution=[hash[user_id]])`、`TableSourceScan` | Flink 原节点，没有变化。 |

一阶段（`1p`）的优化后执行计划；优化后的物理计划两边同样相同：

CPU（`cpu-1p`）：

```
Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- WindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS expo_cnt, SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
         +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

GPU（`gpu-1p`）：

```
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GpuNativeStreamExecWindowAggregate(mode=gpu-native, batchRows=4096, maxWaitMs=5) WindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS expo_cnt, SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
         +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

`GpuNativeStreamExecWindowAggregate(mode=gpu-native, batchRows=4096, maxWaitMs=5)` 用一个算子替换整个一阶段 `WindowAggregate`。Mega-Flink 文档把 TUMBLE / HOP / CUMULATE 窗口聚合列为默认 GPU 路径、无需额外开关，并说明 watermark 到达窗口末尾时输出一条最终 INSERT 并清理该窗口状态。本文早先版本也称它为 Mega-Flink 较成熟的路径。`gpu-1p` 的运行配置另加一阶段设置 `table.optimizer.agg-phase-strategy=ONE_PHASE`。

TUMBLE 的节点结构与 HOP 相同，只是窗口参数不同。TUMBLE（两阶段）的 GPU 行：

```
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GpuAssistedStreamExecGlobalWindowAggregate(local=project-owned-cuda, merge=project-owned-cpu, direct-watermark-output=true) GlobalWindowAggregate(groupBy=[user_id], window=[TUMBLE(slice_end=[$slice_end], size=[5 min])], select=[user_id, COUNT(count1$0) AS expo_cnt, SUM(sum$1) AS click_cnt, SUM(sum$2) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
```

### 各模式的计划差异

| 模式 | Local 阶段 | Global / 窗口阶段 | 出处 |
| --- | --- | --- | --- |
| `cpu`（两阶段） | `LocalWindowAggregate` | `GlobalWindowAggregate` | 上面的 CPU 计划 |
| `cpu-1p` | 无 | `WindowAggregate` | CPU 一阶段计划 |
| `gpu`（两阶段） | `GpuNativeStreamExecLocalWindowAggregate` | `GpuAssistedStreamExecGlobalWindowAggregate` | 上面的 GPU 计划 |
| `gpu-1p` | 无 | `GpuNativeStreamExecWindowAggregate` | GPU 一阶段计划 |

c2 没有测 persistent 与 adaptive。JobGraph（算子链）CPU 与 GPU 形状相同：两个顶点，第一个到 Local 窗口聚合为止（两阶段），第二个从 Global 开始；JobGraph 里的算子名不带 Gpu 前缀。

CPU 两阶段的 JobGraph：

```
JobGraph vertices (operator chains):
- GlobalWindowAggregate[11] -> Calc[12] -> ConstraintEnforcer[13] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
- Source: Sequence Source -> kuairand-replay -> kuairand-watermarks -> *anonymous_datastream_source$1*[7] -> Calc[8] -> LocalWindowAggregate[9]  [parallelism=1]
```

### GPU 确实在跑的证据

窗口算子有运行时指标，所以每一轮 GPU 都能核对。已发布运行的所有 GPU 轮次里，批数、H2D、kernel、D2H 计时都不为 0。这些数据来自运行指标（Mega-Flink 的 CUDA event 计时；H2D 与 D2H 含排队，见总览第 3 节），不在本文早先已复核的数字之内。3 轮的各轮范围：

| 模式 | 档 | 批数 | H2D (ms) | Kernel (ms) | D2H (ms) |
| --- | --- | --- | --- | --- | --- |
| `gpu` HOP | 打满 | 842-951 | 12.9-13.8 | 12.2-13.8 | 8.0-9.8 |
| `gpu-1p` HOP | 打满 | 889-914 | 52.5-53.4 | 14.7 | 24.0-24.5 |
| `gpu` HOP | 中 | 6,177-6,194 | | | |
| `gpu-1p` HOP | 中 | 5,579-5,667 | | | |
| `gpu` HOP | 低 | 8,482-8,486 | | | |
| `gpu-1p` HOP | 低 | 3,688-3,702 | | | |
| `gpu` TUMBLE | 打满 / 中 / 低 | 835-857 / 6,153-6,180 / 8,485-8,489 | | | |

限速档的批数更多；原因（批由 5 ms 最长等待截断而不是攒满）是我们的理解，未核实。

### 结果

**HOP，默认配置**（"1p" = 一阶段）

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 低 | CPU 两阶段 | 19,951 | 64.08 / 123.64 | MATCH |
| 低 | CPU 1p | 19,951 | 30.69 / 67.84 | MATCH |
| 低 | GPU 两阶段 | 19,971 | 54.76 / 105.96 | MATCH |
| 低 | GPU 1p | 19,973 | 26.36 / 52.98 | MATCH |
| 中 | CPU 两阶段 | 99,752 | 68.30 / 132.81 | MATCH |
| 中 | CPU 1p | 99,747 | 25.19 / 64.30 | MATCH |
| 中 | GPU 两阶段 | 99,821 | 38.24 / 69.66 | MATCH |
| 中 | GPU 1p | 99,870 | 17.28 / 40.35 | MATCH |
| 打满 | CPU 两阶段 | 744,721 | 53.80 / 117.34 | MATCH |
| 打满 | CPU 1p | 628,892 | 26.11 / 78.29 | MATCH |
| 打满 | GPU 两阶段 | 898,493 | 37.33 / 83.32 | MATCH |
| 打满 | GPU 1p | 892,390 | 16.62 / 57.67 | MATCH |

**HOP，调优**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 中 | CPU 两阶段 | 99,824 | 62.89 / 124.43 | MATCH |
| 中 | CPU 1p | 99,828 | 22.46 / 65.07 | MATCH |
| 中 | GPU 两阶段 | 99,924 | 37.16 / 64.68 | MATCH |
| 中 | GPU 1p | 99,920 | 16.06 / 34.12 | MATCH |
| 打满 | CPU 两阶段 | 1,475,619 | 48.34 / 97.40 | MATCH |
| 打满 | CPU 1p | 1,079,916 | 25.01 / 86.26 | MATCH |
| 打满 | GPU 两阶段 | 1,861,833 | 36.44 / 99.76 | MATCH |
| 打满 | GPU 1p | 1,184,784 | 16.02 / 51.10 | MATCH |

低档调优：未测。

**TUMBLE，默认配置**

| 档 | CPU 吞吐 | CPU p50 / p99 | GPU 吞吐 | GPU p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| 低 | 19,992 | 73.37 / 119.64 | 19,993 | 52.67 / 105.76 | MATCH |
| 中 | 99,961 | 59.86 / 122.12 | 99,967 | 34.96 / 77.36 | MATCH |
| 打满 | 724,616 | 43.42 / 105.17 | 965,671 | 32.37 / 94.38 | MATCH |

**TUMBLE，调优**

| 档 | CPU 吞吐 | CPU p50 / p99 | GPU 吞吐 | GPU p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| 中 | 99,972 | 61.94 / 121.82 | 99,976 | 34.66 / 63.30 | MATCH |
| 打满 | 1,613,516 | 37.31 / 107.71 | 2,020,259 | 33.91 / 86.85 | MATCH |

### 时间构成

- JFR 采样占比是估算；证据表里的 Mega-Flink CUDA event 计时是实测（总览第 3 节）。breakdown 作业只跑 1 轮且开了 profiling，不用于比速度。
- 状态读写、timer、changelog 合并与 Global 聚合仍在 JVM，GPU 没有替换（总览第 6 节）。GPU 一侧的 breakdown 里状态读写（stateRead / merge）并未消失，"GPU 批量处理省掉状态读写"没有实测支持。
- c2 各段的采样百分比本文不列出：此处未核实。

### 结论

GPU 吞吐与延迟都更好。调优后 HOP 两阶段 GPU +26%，TUMBLE +25%；一阶段对一阶段 GPU +10%（CPU 一阶段本身比两阶段慢，所以"最优对最优"为 +26%）。默认配置下的"GPU +42%（对 CPU 一阶段）"是非最优对比，不建议引用。CPU 一阶段在默认配置下轮间波动较大（0.59–0.69M）。GPU 一侧的 breakdown 里状态读写（stateRead / merge）并未消失，"GPU 批量处理省掉状态读写"没有实测支持。

<a id="c6-lookup"></a>
## c6 查特征维表 Lookup Join

### 做什么

实时拼接物品侧特征：事件流按 video_id 查维表取 author_id。仅测 INNER。CPU 用全量内存 HashMap（相当于 Flink 全缓存的较好情形），GPU 把维表装入显存哈希表。注意：每批少于 1,024 行时 GPU 路径回落到 JVM 查表，所以低/中档的 `gpu` 模式多半没有真正上 GPU；另测 `gpu-min1`（强制每批上 GPU）作对照。 维表为 1,039,412 行（低档 413,436 行），不是全表 3,204 万行。维表声明了 `PRIMARY KEY (video_id) NOT ENFORCED` 并使用测试用的自定义 connector；CPU 与 GPU 读同一份数据。

### SQL

```sql
SELECT e.seq, e.video_id, d.author_id
FROM events AS e
JOIN video_dim FOR SYSTEM_TIME AS OF e.proc_time AS d
  ON e.video_id = d.video_id
```

### 真实执行计划

优化后的物理计划（CPU 与 GPU 相同）：

```
Calc(select=[seq, video_id, author_id])
+- LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], lookup=[video_id=video_id], select=[seq, video_id, video_id0, author_id])
   +- Calc(select=[seq, video_id])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

优化后的执行计划，CPU：

```
Calc(select=[seq, video_id, author_id])
+- LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], lookup=[video_id=video_id], select=[seq, video_id, video_id0, author_id])
   +- Calc(select=[seq, video_id])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

优化后的执行计划，GPU（`gpu`）：

```
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, video_id, author_id])
+- GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1024) LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], lookup=[video_id=video_id], select=[seq, video_id, video_id0, author_id])
   +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, video_id])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

替换同样只发生在执行计划。GPU 模式下各节点的含义：

| 节点与参数 | 说明 |
| --- | --- |
| `GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1024)` | 替换 `LookupJoin`。Mega-Flink 文档把"有界只读快照上的 Lookup Join（INNER / LEFT）"列为按负载分档的实验性 CPU/GPU 路径，用专用的快照提供者：唯一非 NULL 的 BIGINT 键、两列 BIGINT 维表、仅 INSERT 输入、无 async / retry / 残差条件、并打开实验算子开关；低/中负载的默认选择是 CPU。`batchRows=4096` 是攒批大小。`gpuMinRows=1024`：少于 1,024 行的批走 JVM 查表而不上 GPU（见总览第 2 节）；文档未定义这个参数名。 |
| `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)` | Calc 仍在 Flink CPU 上（见上）。 |
| `TableSourceScan` | Flink 原节点，没有变化。 |

### 各模式的计划差异

| 模式 | 查表节点 |
| --- | --- |
| `cpu` | `LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], ...)`（全量内存 HashMap） |
| `gpu` | `GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1024)` |
| `gpu-min1`（对照） | `GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1)`：强制每批都上 GPU |

`gpu-min1` 那一行，取自 `gpu-min1` 运行的计划：

```
+- GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1) LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], lookup=[video_id=video_id], select=[seq, video_id, video_id0, author_id])
```

c6 没有测 persistent 与 adaptive。JobGraph 两边都是一条算子链。CPU 上是 `Calc -> LookupJoin -> Calc`；GPU 上末尾的 Calc 没有作为独立算子出现（上面的执行计划里仍列出它）：

CPU：

```
JobGraph vertices (operator chains):
- Source: Sequence Source -> kuairand-replay -> kuairand-watermarks -> *anonymous_datastream_source$1*[5] -> Calc[6] -> LookupJoin[7] -> Calc[8] -> ConstraintEnforcer[9] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

GPU：

```
JobGraph vertices (operator chains):
- Source: Sequence Source -> kuairand-replay -> kuairand-watermarks -> *anonymous_datastream_source$1*[5] -> Calc[6] -> LookupJoin[7] -> ConstraintEnforcer[9] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

GPU 运行的规划期状态报告了融合投影，与缺少末尾 Calc 相符（这是我们的理解；文档提到该查表有融合谓词 / 投影）：

```
LookupJoinPlannerShimStatus = gpu-replaced:gpu-native:bounded-readonly-snapshot=kuairand-video-author-1039412-…:fused=directProjectionFields=3, d2hBytesPerProbe(inner/left)=10/10
```

### GPU 确实在跑的证据

上面的规划期状态在三档的 `gpu` 与 `gpu-min1` 上都是 `gpu-replaced`（低档快照为 413,436 行）。运行期证据：用 nsys 统计的 CUDA kernel 实例数（默认配置，各 1 次）：

| 档 | `gpu` kernel 实例 | `gpu-min1` kernel 实例 |
| --- | --- | --- |
| 低 | 3（几乎没上 GPU） | 8,484 |
| 中 | 22（几乎没上 GPU） | 6,151 |
| 打满 | 878 | 848 |

因此低/中档 `gpu` 的约 3 ms 延迟不是 GPU 带来的；只有打满档的 GPU 结果有运行时证据。c6 的正式计时轮次没有运行时计时指标（Lookup Join 无此类指标），证据来自规划期 `shim_status`（三档均为 gpu-replaced）和 nsys。

**低/中档的 `gpu` 大多数批走 JVM 查表**（每批少于 1,024 行），所以它在那里约 3 ms 的延迟不是 GPU 带来的；`gpu-min1` 才是每批都真正用 GPU 的模式。

### 结果

**默认配置**（同一作业内三种模式轮换）

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 低 | CPU | 20,000 | <0.01 / <0.01 | MATCH |
| 低 | GPU（实际走 JVM） | 19,998 | 2.70 / 6.10 | MATCH |
| 低 | GPU min1 | 19,998 | 2.71 / 6.14 | MATCH |
| 中 | CPU | 100,000 | <0.01 / 14.76 | MATCH |
| 中 | GPU（实际走 JVM） | 99,990 | 3.18 / 9.48 | MATCH |
| 中 | GPU min1 | 99,991 | 2.96 / 8.97 | MATCH |
| 打满 | CPU | 601,930 | <0.01 / <0.01 | MATCH |
| 打满 | GPU | 838,326 | 2.33 / 4.70 | MATCH |
| 打满 | GPU min1 | 834,406 | 2.33 / 4.76 | MATCH |

**调优**（低档：未测）

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 中 | CPU | 100,000 | <0.01 / 19.20 | MATCH |
| 中 | GPU | 99,990 | 2.93 / 6.28 | MATCH |
| 中 | GPU min1 | 99,994 | 2.78 / 6.20 | MATCH |
| 打满 | CPU | 1,299,187 | <0.01 / <0.01 | MATCH |
| 打满 | GPU | 2,284,322 | 0.88 / 1.75 | MATCH |
| 打满 | GPU min1 | 2,237,176 | 0.89 / 1.76 | MATCH |

### 时间构成

- Lookup Join 没有运行时 GPU 计时，所以 c6 的 H2D / kernel / D2H 没有拆分。上面的 nsys kernel 实例数是这个用例仅有的运行期 GPU 数据。
- **JFR 占比不采用**：c6 的 JFR 归类把 Mega-Flink 包名下的类一律计为 "GPU 算子 JVM 侧"，而 CPU 查表用的维表类恰好在同一个包下，导致 CPU 作业里也出现约 24% 的 "GPU 相关"，确认是误归类，因此 c6 的 JFR 占比不作结论，其他用到同一个包的用例（含驱动类本身）的 "GPU 相关" 占比也应复查。

### 结论

打满档 GPU 更快：默认 +39%（838k 对 602k），调优后 +76%；低/中档受限速，吞吐相同，GPU 因攒批多约 3 ms 延迟。只验证了单 BIGINT 键加少量 BIGINT 列这一最简情形。

**LEFT lookup**：在本构建下不能上 GPU，会回落到 CPU，所以没有性能对比。规划期状态：维表有主键时报输出维表列与维表 schema 不匹配（LEFT 把维表列变为可空，而主键列非空，类型不相等）；去掉主键则报 lookup key 未证明包含维表主键。
