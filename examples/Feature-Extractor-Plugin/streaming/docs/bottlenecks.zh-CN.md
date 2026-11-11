<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Flink 作业的性能瓶颈一般有哪些

[English](bottlenecks.md) | 中文

这是 [README](../README.zh-CN.md) 第 1 节背后的细节。
每类瓶颈列三项：时间花在哪、公开证据、我们的实测。每个公开数字只代表一家公司的一个场景。我们原样引用，不能当成通用占比（标"厂商"的，是厂商对自家产品的自测）。我们的数字来自本仓库的实测（见 benchmarks；条件见文首）。"未测"就是没测，我们不作任何说法。

## 序列化 / 算子间记录复制

- **时间花在哪**：Kryo/POJO 序列化、记录在算子间的深拷贝、网络交换前后的编解码。一般怎么解决：用 POJO/Tuple/Row 代替 Kryo、开 object reuse、算子链、列裁剪。
- **公开数字**：[Flink 官方博客](https://flink.apache.org/2020/04/15/flink-serialization-tuning-vol.-1-choosing-your-serializer-if-you-can/)（微基准）：POJO 退回 Kryo，性能降 75%。[Iron Vector](https://docs.google.com/document/d/1ledkVUcWSlx5l4yLaK9nmdIoOPMpUd_1GbjBezbNMPs)（厂商，单条 Kafka Avro→SQL→blackhole 管道）：只开 object reuse，原生 Flink 吞吐 +35%～40%。
- **我们的实测**（c1 简单过滤）：object-reuse 关→开，CPU 从 0.84M 升到 2.47M 行/秒，GPU 从 1.03M 升到 2.06M 行/秒。序列化/交换在两边各占约 29% 采样时间。方向一致，但我们在简单过滤上提升约 3 倍，比公开的端到端数字大得多。两者都不要外推。

## 状态读写

- **时间花在哪**：RocksDB 下是每次访问的序列化、seek、compaction、磁盘 IO；堆内状态下是哈希查找和对象访问。一般怎么解决：缓存热 key、KV 分离、换状态后端、把 Join 改成查表、TTL、MiniBatch。
- **公开数字**（几乎都是 RocksDB 大状态）：[Zalando](https://engineering.zalando.com/posts/2026/07/migrating-ad-event-processing-to-flink.html)：Interval Join 有 15%～30% 的 CPU 花在 seek 上。[淘宝](https://www.alibabacloud.com/blog/the-delta-join-in-apache-flink-architectural-decoupling-for-hyper-scale-stream-processing_602645)：Delta Join 去掉 50TB 的 Join 状态，CPU 和内存省 80% 以上。
- **我们的实测**（堆内状态、JFR 估算）：状态占采样时间的比例，c5 Interval Join 为 52%（CPU）、43%（GPU）；Join+聚合 32% / 40%；c2 窗口聚合 14%（CPU）；c4 去重 14% / 17%。GPU 版里状态仍在 JVM，所以这几个用例 GPU 收益很小或没有。方向一致；成本构成不同（公开案例是 RocksDB，我们是堆内）。

## Source / Sink 与格式解析

- **时间花在哪**：JSON/Avro/PB 反序列化、连接器线程、外部写入。一般怎么解决：二进制格式、裁剪无用列/字段、连接器并行度、sink 批量写、调 consumer 拉取参数。
- **公开数字**：[字节/抖音](https://developer.aliyun.com/article/796471)：Kafka 源的 PB 数据裁剪后，大部分任务 CPU 节省约 30%（是 CPU 节省，不是占总时间的比例）。[Iron Vector](https://docs.google.com/document/d/1ledkVUcWSlx5l4yLaK9nmdIoOPMpUd_1GbjBezbNMPs)（厂商，Avro 源解码）：274→528 MiB/s（+93%），但要先调大 Kafka 拉取参数；第一次跑反而慢 23%。
- **我们的实测**：本地回放 int64 列文件，没有文本解析。source 线程在 c1 占 15%（CPU）/18%（GPU），在 c3 的 GPU 侧占 10%。Kafka 和真实格式未测。部分一致：我们的 source 比真实的 Kafka 加 JSON/PB 解码轻得多，所以会低估这一类。

## 行列转换（含 GPU 上多出的 H2D/D2H）

- **时间花在哪**：把 RowData 转成向量化/GPU 引擎要的列格式，再转回来；GPU 上还多了 H2D、D2H 拷贝。一般怎么解决：端到端列式、少出入设备、只转用到的列、批量化。
- **公开数字**（没找到 Flink 上 GPU 的公开数字）：[蚂蚁 Flex](https://segmentfault.com/a/1190000046642882)（CPU 向量化）：所用字段全部转换，开销将近 11%；只转表达式用到的字段，降到 6.68%。[Gluten-Flink issue](https://github.com/apache/gluten/issues/12426)：每批多两次行列转换，无数字。
- **我们的实测**（c1 过滤、Flink 默认配置那轮、GPU 算子自带计时）：计时的 5 段 GPU 步骤合计 499 ms：行→列 254、H2D 20、kernel 25、D2H 14、列→行 186。行列转换占这 5 段的 88%，kernel 占 5%。没算在内的：JVM 侧 processElement 1,725 ms 和 collect 1,096 ms。nsys 核对：kernel 和拷贝时间与计时相差约 10% 以内。c5 Interval Join 的 GPU 侧：计时的 1,107 ms 里，列→行占 843 ms。蚂蚁的 11% 是 CPU 引擎，我们没有核实它为何更低，所以不能拿它佐证我们的数字。这是上 GPU 方案特有的成本，不是 Flink 的通用瓶颈。

## 数据倾斜

- **时间花在哪**：少数 subtask 满载，其余空闲，上游被反压。一般怎么解决：两阶段聚合、打散 key、按负载分发。
- **公开数字**：[Rippling](https://www.rippling.com/blog/eliminating-data-skew-in-flink)（CDC 管道）：60% 的负载落在 2 个 subtask 上，修复后吞吐 +38%，总耗时从 17 小时降到 12.4 小时。**我们的实测**：未测（单并行度看不出倾斜）。

## Checkpoint 与恢复

- **时间花在哪**：状态上传、对齐等待、元数据序列化、大状态恢复。一般怎么解决：增量 checkpoint、changelog、非对齐 checkpoint 加 buffer debloating、存算分离（ForSt）。
- **公开数字**：[Netflix](https://www.slideshare.net/StevenWu5/scaling-flink-in-cloud-94274600)（200 节点测试集群）：21TB 的 savepoint 要 27 分钟；修掉 S3 HEAD 请求问题后，checkpoint 时长降 10 倍。[阿里 VLDB'25](https://www.vldb.org/pvldb/vol18/p4846-mei.pdf)（对比 Flink 1.20）：checkpoint 时长最多缩短 94%，恢复最多快 49 倍。公开证据主要讲稳定性、尾延迟和恢复时间，对稳态吞吐的影响讲得少。**我们的实测**：未测。

## 网络与反压

- **时间花在哪**：跨机器传输、序列化、缓冲区排队。一般怎么解决：算子链或 multiple-input 去掉无用 shuffle、task 放置、credit-based 流控、buffer debloating。
- **公开数字**：[Pinterest](https://www.slideshare.net/slideshow/tuning-flink-clusters-for-stability-and-efficiency/266280602)：优化 task 放置后，跨主机网络流量降 70% 以上，作业成本降约 75%（成本不全来自网络）。**我们的实测**：单机、单并行度（部分用例补测并行度 2、4）；交换成本包含在"序列化/交换"里（各用例占采样时间 12%～34%）。没有跨机网络测试。部分一致。

## GC

- **时间花在哪**：堆内状态大时出现 Full GC，停顿引发心跳超时和反压。一般怎么解决：大状态改 RocksDB 或堆外、加内存或并行度、换垃圾回收器。
- **公开数字**：只有阈值和定性说法。例如[腾讯云](https://www.tencentcloud.com/document/product/1000/56810)把"5 分钟内 Full GC 超过 30%"当严重问题（告警阈值，不是测得的占比）。没找到可信的量化案例。**我们的实测**：没有单独测（16GB 堆，G1）。

## JVM 流水线上限（逐条记录路径）

- **时间花在哪**：即使没有任何计算，单并行度下，一个 source 线程、交换和一个 sink 线程就会限制吞吐。一般怎么解决：提高并行度、算子链、减少逐条处理。
- **公开数字**：没找到直接的数字。厂商的原生向量化引擎在 Nexmark 上报告了很大的提升（阿里 [Flash](https://developer.aliyun.com/article/1634363) 五倍以上；[VERA-X](https://ververica.com/blog/vera-x-introducing-the-first-native-vectorized-apache-flink-engine) 对 Flink 1.19 为 5～10 倍），但这些提升包含很多逐条路径之外的因素。
- **我们的实测**（c0-pass 上限探针：只有 source+交换+sink，输出 3.3M 行，Flink 默认配置）：CPU 0.856M 行/秒，GPU 0.851M 行/秒。c3 的 GPU Top-N：kernel 约占作业时间 3.6%，GPU 设备 95% 以上时间空闲，吞吐受 JVM 侧限制。这是我们自己的发现。

## 我们实测的各用例时间构成

占采样时间的百分比，JFR 估算，取自 object-reuse 关闭的那一轮。"GPU 相关"指 GPU 算子代码和拷贝。c6（维表查询）不列：它的 GPU 占比被分类器误归类。

| 用例 | CPU 侧 | GPU 侧 |
|---|---|---|
| c1 过滤 | 算子 29、序列化/交换 29、行转换 20、source 15 | 序列化/交换 29、算子 21、source 18、GPU 相关 13、行转换 9 |
| c2 窗口聚合（GPU 为一阶段） | 算子 35、序列化/交换 28、状态 14 | GPU 相关 38、序列化/交换 29、算子 12 |
| c3 每用户 Top-N | 算子 45、序列化 34、状态 3 | 序列化 30、算子 24、GPU 相关 23、source 10 |
| c4 保留最新去重 | 序列化 34、算子 22、状态 14、sink 10 | 序列化 34、算子 20、状态 17、GPU 相关 7 |
| c5 Interval Join | 状态 52、算子 27、序列化 12 | 状态 43、算子 26、GPU 相关 15 |
| c5 Join+聚合 | 状态 32、算子 31、序列化 28 | （融合）状态 40、算子 24、GPU 相关 21 |
