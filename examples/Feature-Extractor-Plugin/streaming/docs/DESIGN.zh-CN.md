<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# 流式部分库设计

[English](DESIGN.md) | 中文

状态说明：**已在本目录** = 代码已放进来（`library/`、`library/native/`、`tools/`、`examples/`、`benchmarks/`）。**仅在 Mega-Flink** = Mega-Flink 里有，没有抽取。**尚未开始**。代码验证到什么程度见 [LIMITS.zh-CN.md](LIMITS.zh-CN.md)。标"未验证"的内容没有测过。

## 1. 目标与非目标

目标：

- 做一个小的 Flink GPU 算子库，作业显式调用，只用在 GPU 值得的步骤上。
- 结果与 Flink 自己的 CPU 输出对拍。
- 算子从研究插件 [Mega-Flink](https://github.com/cb521/Mega-Flink) 中抽取。

非目标：

- 不覆盖 Flink planner 的类，不自动改写 SQL 计划。
- 不自动证明某段 SQL 可以安全迁到 GPU，由调用者选择步骤。
- 不收 Mega-Flink 自己测试里也没跑赢 CPU 的算子（见 4.3）。

## 2. 第一版范围

**窗口聚合 + 有界 INNER 维表查询**（第 9 节决策 3）。其余算子只有重新测量确实赢才加。与流式 README 第 3 节的状态表一致。

| 组件 | 状态 |
|---|---|
| 窗口聚合（HOP、TUMBLE，一阶段或两阶段） | **已在本目录**：`library/`，类 `GpuWindows` |
| INNER 维表查询（有界维表） | **已在本目录**：`library/`，类 `GpuJoins` |
| 这两个算子的原生层（CUDA、JNI） | **已在本目录**：`library/native/` |
| Builder API 与形状校验 | **已在本目录**：在 `build()` 时检查 |
| 去重、TopN、Interval Join、带融合 filter/聚合的 INNER Join、过滤 | **仅在 Mega-Flink**。以后只有重新测量确实赢才加 |
| Table/SQL 桥接 | **尚未开始**。不进第一版（已定），只交 DataStream API |
| Tools（qualification、profiling） | **已在本目录**：`tools/` |
| 示例作业、基准驱动与脚本 | **已在本目录**：`examples/`、`benchmarks/`（c1、c3、c4、c5 的 GPU 侧需要 Mega-Flink 构建） |

## 3. 公开 API

DataStream API 加 builder。作业把 `DataStream<RowData>` 和描述行结构的 Flink `RowType` 交给 builder，拿回 `DataStream<RowData>`。包名 `com.nvidia.fep.flink`。

```java
// 窗口聚合：先是 key 列，再是聚合列，最后是 window_start 和 window_end
DataStream<RowData> stats =
    GpuWindows.hop(events, eventType, Duration.ofMinutes(30), Duration.ofMinutes(5))
        .timeField("ts")
        .keyBy("user_id")
        .aggregate(Agg.count().as("cnt"), Agg.sum("play_ms").as("play_ms_sum"))
        .twoPhase()            // 可选；另有 .offset(...)、.windowProperties(...)
        .build();

// 用 BIGINT 探测列对有界维表快照做 INNER 查询
LookupDimension videos = LookupDimension.ofLongPairs("videos-v7", videoIds, authorIds);
DataStream<RowData> enriched =
    GpuJoins.lookupInner(events, eventType, "video_id", videos)
        .dimensionColumns("dim_video_id", "author_id")
        .build();
```

滚动窗口用 `GpuWindows.tumble(stream, rowType, size)`。其他 builder 选项：`batchRows`、`maxWaitMs`、`gpuResource`；查询 join 另有 `gpuMinBatchRows`、`maxDimensionRows`。`GpuWindows` 和 `GpuJoins` 的类 Javadoc 列出了全部检查项。

- builder 在 `build()` 时检查声明的形状：列和聚合类型、窗口参数、至少一个 key 列、支持的 join 类型、批大小设置。检查失败会抛 `GpuUnsupportedShapeException`（参数本身无效，例如列名不存在，则抛 `IllegalArgumentException`）。运行中途不会静默回退。
- **append-only 在构建时无法证明**，因为输入是 `DataStream`。算子遇到非 INSERT 行会抛异常。时间列为 NULL 的行同样被拒绝。
- 现在是 Mega-Flink 的 planner 来证明这些条件。库里由 builder 来做，用的是从 planner 的 `tryCompile` 代码里拆出来的同一批准入检查，所以漏检可能导致结果错误（风险 1）。
- 算子是普通的 Flink 单输入流算子。攒批、barrier 与 watermark 时的刷出都从 Mega-Flink 带过来，变的只是构造算子的代码。
- 两阶段聚合（无状态 GPU 合并，再按 key 合并）是一个 builder 开关 `.twoPhase()`。它在 Mega-Flink 里标为实验功能；库里提供它，是因为我们的测量覆盖了它。
- **Table 桥接**（`GpuTables`）：尚未开始，不进第一版（已定）。转换开销未验证（风险 3）。

考虑过但没选的方案：

- Table API 的 process table function（Flink 2.1+）。它在分区内逐行调用，够不到"在 checkpoint barrier 前攒批"这类算子钩子。在 2.2.1 上未验证。
- 带自定义 planner 扩展的 SQL hint。Flink 没有替换物理算子的公开钩子，这样仍要用 planner 内部类，库会绑死一个 Flink 版本。以后可以做成可选模块。

## 4. 从 Mega-Flink 抽取了什么

### 4.1 Java

- `library/src/main` 共 51 个类、约 15,000 行（从 Mega-Flink 搬来 43 个类，按需裁剪，另有 8 个新写的门面类），测试 12 个类、约 2,100 行。Mega-Flink 的 runtime 包有几百个类、约十万行。
- 保留：
  - 第一版的算子（一阶段窗口、两阶段的 local 与 global 窗口、有界维表查询）；
  - 行列打包、显存管理、资源生命周期；
  - native kernel 的 Java 封装（`DenseGroupbyKernel`、`LookupJoinKernel`、`CudaDeviceBinding`）；
  - 可序列化的计划描述（只含数据和校验）；
  - 指标辅助类。
- 已改写：
  - 计划类去掉了 Calcite 和 planner 入口，用 `AggregateCallSpec` 替代 Calcite 的 `AggregateCall`；
  - planner 生成的 key 选择器换成 `FieldRowDataKeySelector`，planner 的快照提供者约定换成 `LookupDimension`；
  - Calc 与 Expand 融合路径从算子和 `DenseGroupbyKernel` 里删掉了。
- 没改：读 JVM 系统属性的地方（如 `cudf.flink.window.partial-cache.entries`、`cudf.flink.lookup-join.library`）仍是系统属性，没有变成 builder 参数。GPU 设备获取是 builder 选项（`gpuResource`，默认用 Flink 外部资源 `gpu`）。

### 4.2 原生层（CUDA、JNI）

- 4 个 `.cu` 文件加 2 个头文件，共 7,815 行：设备绑定、分组聚合（含它所链接的 device-batch 运行时）和维表查询。整文件拷贝，没有裁剪。`build.sh` 从中编出四个动态库：`libcudf_flink_device.so`、`libcudf_flink_device_batch.so`、`libcudf_flink_groupby.so`、`libcudf_flink_lookup_join.so`。
- 没有带入：通用 Calc 解释器、expand、fused filter/project、hash join、嵌套循环 join、TopN、去重、over、row number、temporal sort 以及基准可执行文件。
- 只需要 CUDA Toolkit。这些文件不用 CUB。
- JNI 符号名带 Java 包名，已按前缀替换改名（`Java_io_github_cb521_cudfflink_` 改为 `Java_com_nvidia_fep_flink_`）。
- 代码里有一处改动：`gpu_arch_compat.h` 里的设备准入（见 [LIMITS.zh-CN.md](LIMITS.zh-CN.md)）。
- 因为整文件拷贝，动态库里导出了一些没有任何 Java 类声明的 JNI 符号（死代码，见 LIMITS）。

### 4.3 删掉的

- Flink 内部的覆盖层、Flink 2.2.1 安装器和桥接类。
- 基准代码和参考实现（这里的基准驱动是新写的）。
- 算子岛（实验性，Mega-Flink 自己的记录里没有性能准入）。
- 在 Mega-Flink 测试里没赢过 CPU 的算子：单独的 Calc、LEFT/SEMI/ANTI Join、Cross Join、COUNT DISTINCT、SESSION 窗口、Over 聚合（第一版）。
- Java 侧的 Calc 解释器。Mega-Flink 自己的结论是默认关闭，端到端也没赢。
- 自适应控制器和常驻路径。库只跑攒批模式。
- Java 代码里 CUMULATE、SESSION、DISTINCT、FILTER、SUM0 和 LEFT 维表查询仍在内部存在，但 builder 不提供。

### 4.4 Flink 依赖

- `flink-streaming-java` 和 `flink-table-runtime`，都是 `provided`。没有 planner，没有 Calcite。测试另用 `flink-runtime` 和 JUnit。
- 构建并测试过的版本是 Flink 2.2.1，Java release 为 11。2.0、2.1 未验证（决策 4）。

## 5. 目录结构

```text
streaming/
  README.md, README.zh-CN.md   方法与结果
  docs/                        本文、LIMITS.md、how-it-works.md、瓶颈分析、相关工作、图
  library/                     Maven 模块 fep-flink-gpu（包 com.nvidia.fep.flink）
    pom.xml                    Flink 依赖为 provided
    src/main/java/             门面（GpuWindows、GpuJoins 等）与运行时类，同一个包
    src/test/java/             CPU 单元测试（不需要 GPU）
    NOTICE
    native/                    CUDA 与 JNI 源码、build.sh、selftest.sh、selftest/
  examples/                    两个 KuaiRand 示例作业（WindowFeaturesJob、LookupAuthorJob）
  tools/                       qualification 与 profiling（fep-flink-tools）
  benchmarks/                  结果与方法；driver/、scripts/、cases/、prepare_stream_data.py、REPRODUCE.md
```

抽取来的类大量使用包级私有访问，所以留在同一个包里，公开 API 只是一层薄门面。

## 6. 许可与 NOTICE

- Apache-2.0。每个新写或抽取的源文件（Java 和 CUDA）都带 SPDX 头：`SPDX-License-Identifier: Apache-2.0`。
- Mega-Flink 是 Apache-2.0，它的 NOTICE 把版权方写作 "Mega-Flink contributors"。这里的版权行是 NVIDIA（已定）。
- NOTICE 文字（在 `library/NOTICE`、`library/native/NOTICE`、`tools/NOTICE`）："Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0)"，外加 Apache Software Foundation 的声明行，作为保守做法保留。
- Mega-Flink 只把它的 Flink 覆盖层标为修改过的 Apache Flink 代码，该层没有抽取。我们在保留的源码里搜了 "adapted from"、"copied from"、"based on flink"，没有命中。没有逐行比对，所以"不含 Flink 衍生代码"未验证。
- CUDA Toolkit 是构建要求，不随库分发。工具不使用 XGBoost。

## 7. 工具

`fep-flink-tools`：Python 3，只用标准库，不需要 GPU，不需要 Flink jar。只读：读 Flink REST API、历史归档或已保存的快照。版本 0.1.0。

- `qualification`：把作业的 SQL 物理节点对应到库里的算子，并报告这些节点占多少计算时间。
- `profiling`：找热点、反压和倾斜。
- 算子族文件列出四个库已提供的节点名（窗口聚合及其 local/global 拆分、维表查询），对应 `GpuWindows.tumble` / `GpuWindows.hop` 和 `GpuJoins.lookupInner`。其余都是 `not-provided` 或 `cpu-boundary`。
- Flink 版本限定 2.2.1；其他 2.x 未验证。自测（`tests/self_test.py`）在合成样例上通过。
- **局限**：它们按 SQL 物理节点名识别算子。纯 DataStream 作业里算子名由用户起，识别不了。它们回答"现有 SQL 作业的哪一段可以迁到库里"，不回答"哪一步 DataStream 代码"。没有在真实生产作业上跑过。
- `compare`、`calibrate`、`train`（XGBoost，需要成对测量）没有带入（已定）。

## 8. 风险

1. **语义证明从 planner 移到 builder。**漏检可能导致结果错误。缓解：builder 复用 Mega-Flink 的准入检查，每个支持的形状都与 Flink CPU 输出对拍。append-only 输入在运行时强制，不在构建时。
2. **`@Internal` 的 Flink 类**（key 选择器、slice assigner、`InternalTypeInfo`、clock service）在小版本之间可能变。若要支持 2.2.x 之外的版本，需要做 2.x 的编译矩阵。
3. **Table 桥接的转换开销未验证。**如果不是零开销，会吃掉一部分收益。
4. **死代码。**native 文件是整份拷贝，有些导出的 JNI 函数没有 Java 调用方。裁掉它们是可能的第二步。
5. **状态不兼容。**算子状态名沿用 Mega-Flink。SQL 作业的 savepoint 无法恢复到使用本库的作业；迁到库里的作业从空状态开始。
6. **默认值要重新核对。**我们的测量没有看到常驻或自适应模式的收益（见 LIMITS）。维表查询的 `gpuMinBatchRows` 默认值是 1,024，与基准运行一致；它的 `batchRows` 默认值（1,048,576，从 Mega-Flink 带来）则不是。

## 9. 决策

| # | 决策 | 结果 | 状态 |
|---|---|---|---|
| 1 | Java 包名 | 改为 `com.nvidia.fep.flink`；按前缀替换，JNI 符号遵循同一规则 | **已定** |
| 2 | 版权行与 NOTICE 写法 | NVIDIA；NOTICE 按第 6 节的保守写法 | **已定** |
| 3 | 第一版范围 | 窗口聚合和有界 INNER 维表查询；其余可选，重测证明有收益才收 | **已定** |
| 4 | Flink 版本 | 只做 2.2.x | **已定** |
| 5 | 第一版是否含 Table 桥接 | 不含；只交 DataStream API | **已定** |
| 6 | Tools 里的 XGBoost 子命令 | 不带入 | **已定** |
| 7 | native 里的 Calc 解释器 | 第一版裁掉 | **已定** |
