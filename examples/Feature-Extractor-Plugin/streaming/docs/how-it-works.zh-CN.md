<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 流式部分：工作原理与安全性

[English](how-it-works.md) | 中文

本页描述 [Mega-Flink](https://github.com/cb521/Mega-Flink) **现在**的行为（标记 **已有**）。第 7 节说明本目录里的算子库带来了什么、没带什么。Mega-Flink 文档没写的地方，本页写"文档未说明"，不推测。

## 1. 一页概览

```
 SQL / Table API -> Flink 规划器 -> 优化后的 StreamExec* 物理节点
   规划期，逐节点：表达式是否全支持？latency-safe？实验开关是否打开？
      否 -> 回到原 Flink CPU 节点          explain：GpuNativeFallback*(reason=...)
      是 -> GPU 算子                       explain：GpuNativeStreamExec*(mode=gpu-native, ...)

 运行期，逐批：
   RowData -> 拷贝所需列 -> pinned 缓冲 -> H2D -> CUDA kernel -> D2H -> pinned 缓冲
           -> JVM 重组成行 -> 合入 Flink 状态 / 输出
   成批条件：batchRows | maxWaitMs | watermark | checkpoint barrier | 输入结束

 总账：Flink keyed state + timer（进 checkpoint）        GPU 显存：可重建的缓存
```

三条规则贯穿设计：

- 准入按物理节点判断，全有或全无。节点没有被完整支持，就留在 Flink CPU。
- Flink 状态和 timer 始终是总账。checkpoint 与恢复不依赖 GPU 显存。
- 每条 GPU 路径都和 CPU 结果对拍；没赢 CPU 的路径默认关闭。

## 2. 一个批次从进入到输出

1. **拷进去。** 算子（如 `CudfTumblingWindowOperator`、`CudfAggregateOperator`）只把用到的列从每行 `RowData` 拷进 native 侧创建的 pinned 缓冲（kernel 封装类如 `DenseGroupbyKernel`）。
2. **H2D、kernel、D2H。** 封装类经 JNI 提交。native 代码在 `dense_groupby_jni.cu` 这类 `.cu` 文件里。显存缓冲、pinned 缓冲、stream、event 在 `open()` 一次分配并复用，每批不做 `cudaMalloc`/`cudaFree`。
3. **重组行。** 结果在 JVM 里解码成 `BinaryRowData`（`PackedBinaryRowBatch` 把多行放进一个 `byte[]`），合入 Flink 状态，再输出。
4. **计时。** 每批拆成 `row_to_native`、`h2d`、`compute`、`d2h`、`native_to_row` 五段。

没有 Arrow 或列式 Exchange。除已列出的深融合外，每个节点之后数据都回到 CPU 行格式，GPU 不跨 Exchange。

**什么时候成批。** 下面任一条先发生就执行（架构文档"攒批边界"）：

- 缓冲行数达到 `cudf.flink.batch.rows`（默认 4096），或第一行进入后等待达到 `cudf.flink.batch.max-wait-ms`（默认 5）；
- 收到 watermark（双输入 Join 任一侧的 watermark）；
- checkpoint barrier 即将拍快照（`prepareSnapshotPreBarrier`）；
- 有界输入结束（`endInput`/`finish`）。

窗口算子在窗口与 watermark 边界内攒批，不会早于 watermark 允许的时间输出。逐条出结果的算子在默认 `cudf.flink.batch.policy=latency-safe` 下不攒批（第 4 节）。固定形状的 `Calc` 有逐条的 `persistent` 路径：1,024 个 slot 的 mapped ring，不攒批。

## 3. 状态、checkpoint、watermark、故障

### 3.1 状态

Flink 状态是 checkpoint 的总账。GPU 显存是可丢弃、可重建的缓存。

| 算子 | Flink 状态里有什么 | GPU 上有什么 |
|---|---|---|
| Group Aggregate | `ValueState` 累加值，`cudfAggregateAverageStateV1` | batch：本批 partial。persistent：缓存，用 Flink 状态重新播种 |
| Window Aggregate | `ValueState`（`cudfTumblingWindowStateV2`）+ event-time timer | 本批 partial |
| Inner Join | 左右两侧 `ListState`（`gpuNativeJoinLeftRows`、`gpuNativeJoinRightRows`） | persistent：hash 缓存，恢复后按 key 懒加载 |
| Interval / Temporal Join | `MapState` 为总账 + timer | 可选热缓存（Temporal 缓存默认关） |

`GpuDeviceBatch` 句柄从不进 checkpoint（岛设计文档）。

**例外：有界 Lookup Join。** 维表只在显存里（唯一 BIGINT key、只读，行数不超过 `cudf.flink.lookup-join.dimension.max-rows`，默认 1,048,576）。Flink 在算子状态 `cudfLookupJoinDimensionSnapshotV1` 里只存 identity 和内容的 SHA-256。恢复时每个 subtask 从 connector 重新加载快照；identity 或摘要不一致就拒绝启动。重建是双缓冲、原子切换，失败不会暴露半张表。

**状态名与 savepoint。** 文档写明：引擎（CPU、persistent、batch）切换时，operator UID 和 state serializer 不变。融合岛保留 anchor 算子原来的 UID、state descriptor 名和 serializer。从旧插件版本的 savepoint 恢复还在待办清单上，目前只有一个很窄的 Join→Calc→Aggregate savepoint 门禁。CPU 规划作业的 savepoint 能否恢复到 GPU 算子：文档未说明。

### 3.2 barrier、watermark、timer

- **barrier。** 算子先 flush 待处理行，把结果合入 Flink 状态，排空 ring，然后才放行 barrier。窗口算子还会先把缓存的 partial 写入状态并注册 timer，所以恢复不依赖那份临时缓存。无状态 `Calc` 的缓冲不进 checkpoint。
- **watermark。** `processWatermark` 先 flush，再转发 watermark。watermark 已经越过窗口末尾时，窗口结果直接输出，不写状态。
- **timer。** 所有 timer 由 Flink 的 timer service 管理（`registerEventTimeTimer`、`onEventTime`）。插件另外加一个 `max-wait-ms` 的 processing-time timer，以及一个 1 ms timer 用来取回 `Calc` ring 的尾部结果。迟到行会计数（`cudfWindowLateRowsDropped`）。

### 3.3 GPU 出错：两种不同的情况

- **结果还没对外可见时：** 无状态岛会保留主机批，直到设备输出发布，可以整批在 CPU 重放。聚合遇到 kernel 处理不了的值，会在 JVM 里归约这一批（`CpuAggregateFallback`），再合入同一份 Flink 状态。
- **输出已可见，或状态、timer 已变化后：** 论文写明 CUDA 错误会让任务失败，Flink 再从已完成的 checkpoint 恢复。这样避免出现一批里一半 GPU、一半 CPU。真实的 CUDA OOM 和非法 launch 各注入过一次，任务恢复后最终表一致。GPU reset、ECC、illegal address 故障没有测。
- 超出显存预算会在 `cudaMalloc` 之前失败。persistent 缓存满：`adaptive` 释放缓存并转 batch；固定 `persistent` 直接失败。

## 4. 准入与回退

| 层（`cudf.flink.*`） | 默认 | 作用 |
|---|---|---|
| `mode` | `cpu-row` | 不设成 `gpu-native` 就绝不用 GPU。这是总开关 |
| `fallback.enabled` | `true` | 不支持的节点回到 Flink 原节点；`false`（压测用）直接抛错 |
| `batch.policy` | `latency-safe` | 逐条出结果、却只有 batch GPU 路径的节点留在 CPU；`throughput` 表示接受等待；Flink 已开 mini-batch 视为已接受 |
| `experimental.operators.enabled` | `false` | 没过默认性能/稳定性准入的路径需要它 |
| `experimental.generic-calc.enabled` | `false` | 通用 Calc 解释器（还需要 `throughput`） |
| `cross-operator-fusion.enabled`、`operator-islands.enabled` | `true`、`false` | 融合子路径如 `calc.local-aggregate.native.enabled` 默认 `false` |

- **回退到哪。** 同一个 `StreamExec*` 类转而调用 Flink 原实现，explain 显示 `GpuNativeFallbackCalc(reason=...)`。融合证明失败时，整段回到独立算子，没有半融合。
- **已编译计划。** `mode`、batch、fallback、显存等设置会写进 compiled plan。恢复的计划即使新进程默认是 CPU，也保留 `gpu-native`。已编译的计划如何改回 CPU：文档未说明。安装会补丁 planner loader；`rollback` 精确还原原字节（校验 SHA-256）。
- **运行期 controller。** 它不改变选了哪个算子，只在该算子已有的路径之间切换。`engine=adaptive` 按压力采样（EWMA），进入和退出用不同阈值，需要连续确认和最短驻留，切换前先排空旧路径。聚合的默认值举例：采样 100 ms，batch 进入/退出 200k/150k 行每秒；`aggregate.adaptive.persistent-enabled=false`，所以只在 CPU 与 batch 之间切。Calc 的 `auto` 和 Over 的 `auto` 在规划期静态选择。
- **小批留在 JVM。** Lookup Join：低于 `cudf.flink.lookup-join.gpu.min-batch-rows`（默认 1,024）的不满批走精确的 JVM 查找（`cudfLookupJoinCpuBatches`）。Window Join：每个 key/窗口候选对不超过 64 时走 JVM 小任务。Temporal Sort：每组不足 256 行留在 JVM。全局的 `gpuMinRows` 选项：文档未说明。

## 5. "结果与 CPU 一致"靠什么

- **语义校验，失败即关闭。** 节点必须被完整支持才会迁移。检查涵盖行类型（多数有状态路径只收 INSERT）、key 与类型、NULL 与 NaN 规则、时间戳精度、watermark 规则。
- **独立对拍。** 每个压测 workload 都有自己的 CPU 期望值函数，避免 CPU 与 GPU 同时算错。校验用精确条数加与顺序无关的摘要、最终动态表，或精确 changelog。恢复测试在一次完整 checkpoint 之后注入一次故障。GPU 算子没有真正执行的运行，整轮作废（explain 和运行时状态都要显示）。
- **SHA-256。** 文档里 SHA-256 用在 Lookup 快照、安装文件、快照尾部校验。长稳核账协议明确说哈希不能证明集合相等。最终表对拍用的摘要算法：文档未说明。
- **已知差异与限制。** 原生 Flink 2.2.1 用原始 `>`/`<` 比较 NaN，会产生非法 retract，所以含 NaN 的 Rank/TopN 只能和独立 reference 对拍。原生 Flink 拒绝流式 RANK/DENSE_RANK，所以它的 CPU 档是插件自己的 reference。不支持、留在 CPU 的：UDF、DECIMAL 列、DATE/TIME/BINARY、嵌套类型（UNNEST 原型除外）、非零 TTL 的 retract、MIN/MAX multiset retract、多个 DISTINCT、processing-time 窗口。很多路径仍是实验性的；项目自评在 P0 清单完成前只是研究插件。

## 6. 可观测性

- **Explain。** `GpuNativeStreamExecCalc(mode=gpu-native, engine=batch, batchRows=4096, maxWaitMs=5)` 或 `GpuNativeFallback*(reason=...)`。文字计划仍可能显示已融合的 AVG `Calc`；以 `calc_planner_status=gpu-fused-window-avg-finalize:gpu-native` 为准。
- **Flink metrics，按算子。** `cudfBatches`、`cudfInputRows`、`cudfOutputRows`、`cudfRowToColumnNanos`、`cudfComputeNanos`、`cudfColumnToRowNanos`；`cudfAggregatePersistentCommands`、`gpuJoinPersistentCommands`、`cudfOverPersistentCommands`；`*AdaptiveMode`、`*AdaptiveSwitches`。`h2d`、`d2h` 记在 `RuntimeMetrics`；是否作为 Flink metric 暴露：文档未说明。
- **回退计数。** `gpuJoinJavaFallbacks`、`cudfFusedCalcNativeFallbackBatches`、`cudfOverExactJvmFallbackBatches`、`gpuJoinPersistentCapacityFallbacks`、`cudfTopNAdaptivePersistentCapacityFallbacks`。
- **显存。** `cudfGpuMemoryReservedBytes`、`cudfGpuTaskManagerMemoryLimitBytes`、`cudfGpuDeviceMemoryFreeBytes`/`UsedBytes`（遥测失败时为 `-1`）。还没有逐算子的 CUDA 分配归因。
- **看板。** Prometheus 规则和 Grafana 看板只是静态候选：没部署，没在 GPU 上跑过。native/JNI 错误计数仍是待办。
- **Tools。** `qualification` 把作业分为 `high`（GPU 候选家族占计算至少 60%）、`medium`（25% 到 60%）、`low`、`unknown`，并标出有风险的形状。`profiling` 报告 busy/反压、倾斜、`pending_work_is_growing`、`checkpoint_health` 和 CPU 回退标记。只读 HTTP GET，不需要 GPU，不需要插件。它们看的是计划形状和 Flink 指标，看不到 kernel 时间。

## 7. 库与 Mega-Flink：带来了什么，没带来什么

[`../library`](../library) 里的算子库（设计见 [DESIGN](DESIGN.zh-CN.md)，验证范围见 [LIMITS](LIMITS.zh-CN.md)）带来了第 2 到 3 节里窗口聚合和有界维表查询的算子。第 1 节和第 4 到 6 节描述的是 Mega-Flink 的 planner 一侧，不能原样套到库上。

| 主题 | 在库里 |
|---|---|
| 算子、攒批、barrier 与 watermark 时刷出 | **已带来。**窗口算子（一阶段、local、global）和维表查询算子就是 Mega-Flink 的类，去掉了 planner 融合分支。批大小和等待时间是 builder 选项（`batchRows`、`maxWaitMs`；窗口默认 4096 行和 5 ms，维表查询默认 1,048,576 行）。经 JVM 的运行只在四个用例上与 CPU 输出对拍过（见 LIMITS） |
| 状态 | **已带来。**Flink 状态仍是总账，恢复模型相同。库上没有测过恢复、savepoint 和还原 |
| 准入 | **改成构建期校验。**planner 的证明变成 `GpuWindows.build()` 和 `GpuJoins.build()` 里的形状校验，取自 Mega-Flink 自己的准入代码。校验不过会抛 `GpuUnsupportedShapeException`，运行中途不回退。append-only 输入在构建时无法证明，所以算子遇到第一条非 INSERT 行就抛异常 |
| 两阶段聚合 | **以开关形式带来。**`.twoPhase()` 会构建 local 合并和 global 合并，调用方不用自己写。它在 Mega-Flink 里是实验功能 |
| Plan 类、key 选择器、快照提供者 | **已重做。**Calcite 类型换成普通的可序列化类（`AggregateCallSpec`、`FieldRowDataKeySelector`、`LookupDimension`） |
| 设备获取 | **已重做。**是 builder 选项 `gpuResource`，默认用 Flink 外部资源 `gpu` |
| 系统属性 | **没有重做。**缓存大小和 native 库路径仍作为 `cudf.flink.*` 系统属性读取 |
| 安装器、planner 补丁、`mode=cpu-row` 总开关、`latency-safe` 与实验开关 | **没带。**不覆盖规划器，作业是显式调用库 |
| 自适应控制器、常驻路径、深融合、Calc 解释器 | **没带。**库只跑攒批模式 |
| 影子对比 | **没带。**作业运行时没有任何东西把 GPU 结果与 CPU 结果对比。对比是离线做的：示例作业用 CPU（Flink SQL）或 GPU（库）运行，并打印最终表的 SHA-256 |
| 状态名 | 沿用 Mega-Flink，因此与 Flink 原生算子的 savepoint 不兼容。CPU SQL 作业改用库要从头启动 |
| 指标、看板 | 算子的指标代码（例如迟到行计数 `cudfWindowLateRowsDropped`）随算子带来了；库作业里是否出现、怎么出现，没有检查。Prometheus 规则和 Grafana 面板没带 |
| Table 桥接 | **尚未开始。**不进第一版；转换开销未验证 |
