<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# 已知局限（流式部分）

[English](LIMITS.md) | 中文

每条都标了来源：**README** = 流式 README，**REPORT** = 测试报告（[benchmarks/README.zh-CN.md](../benchmarks/README.zh-CN.md)），**DESIGN** = [DESIGN.zh-CN.md](DESIGN.zh-CN.md)，或本仓库里的某个文件。本清单不保证完整。

这里有两种证据。README 和 REPORT 里的吞吐数字来自 Mega-Flink 构建，不是 `library/` 里的库。对库，我们只核对了结果与 CPU 结果一致（见下一节）。再往后各节的局限，除非来源写"library"，说的都是我们测过的 Mega-Flink 构建。

## 库验证到了什么程度

| 项目 | 来源 |
|---|---|
| Java 单元测试：63 个通过，不需要 GPU（在 `library/` 下 `mvn -q test`）。它们构建作业图并序列化算子工厂；调用 native kernel 的测试不在其中。 | `library/src/test` |
| native 自测（`library/native/selftest.sh`，不经过 JVM）：在一张 L20 上，只编 sm_89 的构建和编 "89 90a" 的构建都得到 `SELFTEST_PASS`，各 13 项检查。 | `library/native/selftest/` |
| 用示例作业经 JVM 端到端运行：1 张 NVIDIA L20（sm_89），CUDA 12.4，并行度 1，内嵌本地 mini cluster，KuaiRand-27K 数据，预热 300,000 行、测量 3,000,000 行。四个用例里，CPU 运行（Flink SQL）与 GPU 运行（本库）的排序后最终表 SHA-256 全部一致：c2 HOP 一阶段与两阶段（718,449 行，`60fccd33`）、c2 TUMBLE 5 分钟（283,544 行，`6dcf4c11`）、c6 INNER 查 `author_id`（3,300,000 行，`fe6b958a`）。每一对在各自独立的 JVM 里运行。这些摘要也与早先 Mega-Flink 运行的 c2-hop、c2-tumble、c6 最终表摘要相同。 | `examples/README.md` |
| 示例作业的吞吐这里不报：两个引擎喂的行对象不同（SQL 用 `Row`，库用 `RowData`），而且每个数字只跑了一次。 | `examples/README.md` |

库没有验证过的：L20 以外的 GPU、多卡、并行度大于 1、集群、Kafka 等真实数据源、其他数据集、2.2.1 以外的 Flink 版本、RocksDB 状态、checkpoint 与恢复，以及按 `gpu` 外部资源查找设备（示例用 `GpuResourceMode.LEGACY_DEVICE` 直接选 0 号卡）。

## 算子形状

| 局限 | 来源 |
|---|---|
| 窗口聚合至少要一个 key 列。Mega-Flink 自己的检查并不要求，但无 key 的路径没有测过，所以 builder 拒绝。 | `GpuWindows` Javadoc |
| DISTINCT 聚合一律拒绝：一阶段算子没有 distinct 状态，两阶段 local 的预检查也拒绝。只提供 COUNT、SUM、AVG、MIN、MAX。 | `GpuWindows`、`Agg` |
| 只支持事件时间的 TUMBLE 和 HOP。不提供 CUMULATE 和 SESSION 窗口。 | `GpuWindows` |
| 输入必须是 append-only。构建作业图时无法证明；算子遇到第一条撤回或更新行、或时间列为 NULL 时会抛异常。 | `GpuWindows`、`GpuJoins` Javadoc |
| 输出列的类型和是否可空是按 Flink SQL 规则推导的（SUM、AVG、MIN、MAX 与输入列类型相同，COUNT 为 BIGINT NOT NULL，`window_*` 为 TIMESTAMP(p) NOT NULL），没有与 planner 的输出对过。 | `GpuWindows` |
| 两阶段窗口聚合在 Mega-Flink 里是实验功能（那边要打开实验算子开关）。库里因为测过，就以 `.twoPhase()` 提供。 | `GpuWindows.Builder.twoPhase` |
| 维表查询只有 INNER，只有一个 BIGINT 探测 key，维表是两个 BIGINT 列且 key 唯一。其他 key 类型和更宽的行不支持。不提供 LEFT 维表查询。 | `GpuJoins`、`LookupDimension` |
| 维表有界：默认最多 1,048,576 行（`maxDimensionRows`），超过则作业失败。Mega-Flink 构建里 GPU 查找表的上限是 4,194,304 行。主要运行装了 1,039,412 行，有一次补到 400 万行。完整的 3,200 万行表没有测。 | `GpuJoins`、README 2.4、REPORT |
| `gpuMinBatchRows` 决定一个查表批是否运行 GPU kernel：小于它的批由 JVM 查表回答。库的默认值是 1,024 行（与基准运行一致；`batchRows` 默认 1,048,576）；示例作业 `LookupAuthorJob` 也设为 `--batch-rows 4096 --gpu-min-batch-rows 1024`，与基准运行一致。用算子计数器在一张 L20 上测得（并行度 1、3,300,000 行、打满回放）：取 1,024 时，809 批运行了 GPU kernel，2 批由 JVM 回答；取 1（每批都上 GPU）时，812 批运行 kernel，0 批走 JVM；nsys 数到的 `lookup_kernel` 启动次数分别是 809 和 812。两种设置下最终表 SHA-256 都与 CPU 运行一致。打满回放会攒满 4,096 行的批，所以这对低、中流量不说明什么：那里的批由 5 ms 的等待时间触发、不足 1,024 行，多数走 JVM（见基准报告 c6 一节）。用库的默认值时，按算子代码，不足 1,048,576 行的批都走 JVM，这个回放几乎全部走 JVM（据代码推断，没有实测）。 | `GpuJoins`、REPORT |
| 探测行不重新分区：每个并行子任务都会加载完整维表。 | `GpuJoins` Javadoc |
| LEFT/SEMI/ANTI Join、单独的 Calc、COUNT DISTINCT、SESSION 窗口、Cross Join 不在库里。去重、TopN、过滤、Interval Join 仍只在 Mega-Flink。 | README 3.1 |

## 原生层

| 局限 | 来源 |
|---|---|
| 设备准入放宽了。Mega-Flink 只放行名字含 "H20" 的 Hopper 卡。现在 `build.sh` 按编进去的架构定义 `FEP_ADMITTED_DEVICE_MAJORS_MASK`，库会放行任何计算能力主版本编进去了的 GPU（`--archs 89` 对应主版本 8，`90a` 对应 9），而不只是我们测过的那几种卡名。 | `library/native/gpu_arch_compat.h`、`build.sh` |
| 90a（Hopper）代码只编译过，从未在 Hopper GPU 上运行。 | `library/native/build.sh`、自测 |
| `.cu` 文件是整份拷贝，所以动态库导出 58 个 JNI 函数，Java 类只声明了其中 30 个。其余 28 个是死代码：18 个属于库里没有的 Java 类（`GpuDenseLongAggregateKernel`、`PersistentAggregateKernel`、`GpuDeviceBatch`、`LimitKernel`），10 个是 `DenseGroupbyKernel` 已不再声明的 Calc 与 device-batch 入口。 | `library/native/*.cu`、`DenseGroupbyKernel` |
| JNI 参数和返回类型只做了按名字匹配和自测检查，自测只覆盖一部分方法（设备绑定、维表查询、带 COUNT/SUM/MIN/MAX 的分组聚合）。其他路径，如 STRING 通道和两阶段 HOP 的窗口缓冲，只被上面的端到端运行顺带触及，没有做类型层面的检查。 | `library/native/selftest/` |
| 自测要手动运行；它需要 GPU 和 JDK 的 `jni.h`。 | `library/native/selftest.sh` |

## 版本、硬件、数据

| 局限 | 来源 |
|---|---|
| 只验证了 Flink 2.2.1。2.0、2.1 能否编译未验证。不以 Flink 1.x 为目标。库用到的部分 Flink 类标注为 `@Internal`，可能变化。 | README、DESIGN 4.4 |
| 只有一张 NVIDIA L20，单卡。其他 GPU 和多卡没测。 | README |
| 并行度主要为 1。并行度 2 和 4 只在部分 Mega-Flink 用例上测过（窗口聚合、TopN、维表查询）。大于 4 没测。去重和 Interval Join 在并行度 2/4 下没测。库只在并行度 1 下运行过。 | README 2.4、REPORT |
| 只有单机，没有跨机测试，没有集群。库是在本地 mini cluster 里运行的。 | README |
| 没有 Kafka 测试。输入是 KuaiRand 日志的回放。真实数据源格式没测。 | README 1、REPORT |
| 只有 KuaiRand 数据集。其他数据集和更重的表达式没测。 | README 2.4 |
| 只测了堆内状态，没测 RocksDB 状态。 | README |
| 不限速打满，预热 30 万条后测 300 万条，3 轮取中位数。低档和中档是限速运行，其吞吐只说明两边都跟得上。 | README、REPORT |

## 行为

| 局限 | 来源 |
|---|---|
| Mega-Flink 构建里默认模式是 `cpu-row`（原始 Flink 节点），只有显式设成 `gpu-native` 才用 GPU。库没有这个开关：作业调用 `GpuWindows` 或 `GpuJoins` 就用 GPU 算子。 | related-work.zh-CN.md |
| 对拍的是最终结果表（SHA-256，全部一致）。CPU 和 GPU 的中间 changelog 行数不同，因为 GPU 在一个批内合并了更新。 | REPORT |
| 常驻（resident kernel）模式在我们的数据上没有收益。TopN 打满时它只有 0.43M 行/秒，CPU 是 1.36M；Interval Join 在每个速率档都更差。库只有攒批模式。 | README 2.5 |
| 自适应模式没有收益。低档和中档有 0 行进了 GPU。我们没找到它胜过 CPU 与攒批中较好者的用例。库没有带这一模式。 | README 2.5 |
| Interval Join 后接 filter 或聚合的融合没有吞吐收益（join +0.8%、join+filter +1.1%、join+聚合 -7%），只有 join+聚合的延迟中位数更低。其他算子组合未验证。 | README 2.6 |
| GPU 的收益取决于 CPU 基线是否调优。`pipeline.object-reuse` 关闭时，过滤的结论会翻过来。其他 CPU 选项没有探索。 | README 2.3 |
| 时间占比来自更早一次用 Flink 默认配置做的 JFR 采样，是估算值。 | README 1 |
| `benchmarks/` 里 c1、c3、c4、c5 的 GPU 侧需要 Mega-Flink 构建。 | benchmarks/REPRODUCE.zh-CN.md |

## 工具

| 局限 | 来源 |
|---|---|
| qualification 和 profiling 按 SQL 物理节点名识别算子，纯 DataStream 作业识别不了。 | README 3.1、DESIGN 7 |
| 工具限定 Flink 2.2.1，其他 2.x 版本未验证。 | DESIGN 7 |
| 只在合成样例上测过，没有在真实生产作业或真实 History Server 归档上跑过。审查优先级阈值（60% 和 25%）是经验值，没有用实测的 GPU 收益验证过。 | tools/README.md |

## 状态与结果

| 局限 | 来源 |
|---|---|
| 算子状态名与 Flink 自带算子不同。SQL 作业的 savepoint 无法恢复到使用本库的作业。 | README 3.2、DESIGN 8 |
| 基准报告在评审后改写过，改写后尚未独立复核。 | README |
| Table 桥接的转换开销未验证。 | DESIGN 8 |
