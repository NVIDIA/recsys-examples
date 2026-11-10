# 流式特征计算基准：Flink CPU 与 GPU（Mega-Flink）

六组性能用例（c1–c6）在回放的推荐日志上比较 Flink CPU 与 Flink 加 Mega-Flink GPU 算子：GPU 赢两组（窗口聚合、维表查询），两边都调优后 CPU 赢其余四组。本文只放共用的环境与跨用例章节；每个用例有单独页面，含执行计划、SQL 与结果。读者入口见上层 README 的"第五部分"。

> 状态说明：本文数字来自内部实测报告的最新改写版。该版本在补测和改写之后**尚未经过独立复核**；改写前的版本曾经独立复核，复核意见已逐条处理。标"未核实"的内容表示没有找到直接证据。

## 目录

- [1. 总览](#overview)
- [2. 测试环境与数据集](#env)
- [3. 测量方法](#method)
- [4. 用例按结论分组](#cases)：[第 1 组：GPU 赢](group1-gpu-wins.zh-CN.md)（c2、c6），[第 2 组：CPU 赢](group2-cpu-wins.zh-CN.md)（c1、c3、c4、c5）
- [5. 多并行度与大维表](#parallelism)
- [6. 时间花在哪里（breakdown 与 nsys）](#breakdown)
- [7. 已知局限](#limits)
- [8. 复现](#repro)

<a id="overview"></a>
## 1. 总览

打满档吞吐（条/秒），单卡、并行度 1、独占运行，两边都开 `pipeline.object-reuse`（"调优后"）：

| 用例 | CPU | GPU | 谁更快 |
| --- | --- | --- | --- |
| c1 过滤清洗 | 2,472,421 | 2,059,561 | CPU +20% |
| c2 HOP 两阶段 | 1,475,619 | 1,861,833 | GPU +26% |
| c2 TUMBLE | 1,613,516 | 2,020,259 | GPU +25% |
| c3 TopN | 1,363,629 | 1,231,075（batch） | CPU +11% |
| c4 去重 | 1,033,538 | 850,292（batch） | CPU +22% |
| c5 join | 291,797 | 253,233（融合） | CPU +15% |
| c5 join-filter | 307,921 | 262,297（融合） | CPU +17% |
| c5 join-agg | 293,385 | 236,056（融合） | CPU +24%（GPU p50 延迟低约 23%） |
| c6 Lookup（INNER） | 1,299,187 | 2,284,322 | GPU +76% |

百分比 = 较快一方吞吐 / 较慢一方吞吐 − 1。每个数字都是 3 轮中位数。

**配置口径很重要。** 默认配置（`pipeline.object-reuse=false`）下，c1 是 GPU 更快（1.03M 对 0.83M，GPU +23%）；两边同开 object-reuse 后翻转为 CPU 更快。object-reuse 对 CPU 的提升大于对 GPU（c1：CPU 约 ×2.95，GPU 约 ×2.0）。因此选型前应先把 CPU 侧调到位。每个用例页同时给出默认配置与调优后的结果。

<a id="env"></a>
## 2. 测试环境与数据集

**硬件与软件**

- GPU：NVIDIA L20 单卡（驱动对应 CUDA 12.4，原生代码按 sm_89 编译）。
- Flink 2.2.1；Mega-Flink 为与之配套的同一次构建。JVM 堆 16 GB、G1，默认 heap 状态后端，watermark 延迟 20 ms，其余为默认（含默认 buffer-timeout）。
- 单并行度为主，本地盘读数据。宿主机 CPU 型号与核数：未核实，发布前补充。
- GPU 算子参数：攒批 4,096 行、最长等待 5 ms；c6 维表显存哈希表上限 4,194,304 行；c6 每批少于 1,024 行时回落到 JVM 查表。

**数据集**

- [KuaiRand](https://kuairand.com/)（快手公开的推荐数据集，含随机曝光视频），使用 KuaiRand-27K 的用户行为日志，共 92,919,789 行，按 `time_ms` 排序，转成 int64 列文件，进程内按时间顺序回放。引用格式与许可请以其公开页面为准（页面链接发布前请再核对）。
- 视频表共 32,038,725 行。c6 的维表只装入回放切片里出现过的视频（中/高档 1,039,412 行，低档 413,436 行；只取 video_id 与 author_id 两个 BIGINT 列），CPU 的 HashMap 与 GPU 显存哈希表使用同一份数据。

<a id="method"></a>
## 3. 测量方法

**三档负载**

| 档 | 输入速率 | 记录数 |
| --- | --- | --- |
| 低 | 20,000 条/秒 | 60 万 |
| 中 | 100,000 条/秒 | 300 万 |
| 打满 | 不限速 | 300 万 |

每档先以同样速率预热 30 万条再计时。低、中档受限速，两边吞吐相同（都跑满），用来比延迟；打满档用来比吞吐。

**轮数与统计**：每个"用例 × 档 × 模式"跑 3 轮，各模式轮换顺序；吞吐报 3 轮中位数；p50 / p95 / p99 是各轮各自的分位数再取中位数（不是吞吐中位数那一轮的值）。

**延迟定义**：从该条记录的计划发送时刻到 sink 收到的时间。定速档按计划时刻算，避免协调遗漏。打满档源头被反压，延迟是积压下的值，只能做相对比较。表中 "<0.01" 或 "0.00" 表示小于 0.01 ms（c1、c6 的 source 到 sink 在同一个算子链内）。c2 有少量窗口输出没有触发引用（打满/中档 24,354 行，低档 19,205 行），不计入延迟。

**正确性对拍**：每个用例的 CPU 与各 GPU 模式的最终表计算 SHA-256，逐"用例 × 档"比较，所有已发布结果均为 MATCH（0 次不一致）。c5 的 join 对数另用独立 Java 程序核对（正式轮 1,284,869 对）。注意：对拍对象是**最终表**，CPU 与 GPU 的中间 changelog 行数并不相同（GPU 批内合并会少发或多发中间更新），例如 c3 打满 CPU 965,196 行对 GPU batch 876,508 行，c4 为 3,361,901 对 3,302,645，c5-join-agg（默认配置）CPU 765,548 对融合 825,672。所以延迟样本数也略有不同。

**独占运行**：每个作业独占整个节点（一次只跑一个作业），1 张 GPU。另有一轮"并发初筛"（4 个作业同节点同时跑），仅用于筛选，不用于下表；同型号节点间差异小（多数 ±5% 内），并发初筛的吞吐多数比独占偏高 3–6%，个别用例（c3 GPU batch）有离群值，因此下表只用独占结果。

**CPU 基线**：Flink 自带的最优配置，两边相同的并行度与 mini-batch 设置；c2 同时测了默认两阶段（Local+Global）与一阶段（`1p`），c4 测了开 mini-batch 的 CPU（`cpu-mb`），取各自最优对照。"调优后"指两边都加 `pipeline.object-reuse=true`（c3 另加 `table.exec.rank.topn-cache-size=100000`，因为打满档约有 20 万个不同用户）。除此之外未对 CPU 或 GPU 做逐用例的更多调参。

**耗时分解（breakdown）的两类数据，口径不同**：

- JFR 采样（CPU 侧各段占比）是**采样估算**：按线程栈归类，带采样开销，只用于看大致分布。
- Mega-Flink 算子自带的 CUDA event 计时（行→列、H2D、kernel、D2H、列→行）是**实测**。其中 H2D/D2H 含排队与同步，比 nsys 看到的拷贝时间高 1.5–2.5 倍。
- 另用 nsys 对少数用例做了独立核对：kernel 总时间与 Mega-Flink 计时的 kernel 段一致（相差约 10%）。
- breakdown 作业每组只跑 1 轮且开了 profiling，吞吐略低，不用于比速度。

<a id="cases"></a>
## 4. 用例按结论分组

六个用例按第 1 节"调优后、打满档吞吐"的结果分组。每个用例页依次包含：场景、完整 SQL、真实执行计划（CPU 与 GPU）、插件替换了什么、各模式的计划差异、GPU 确实在跑的证据、结果表（默认与调优）、时间构成和结论。只在默认配置下 GPU 更快的用例（c1）放在第 2 组，因为以调优后的结果为准。

**第 1 组：GPU 赢**（[group1-gpu-wins.zh-CN.md](group1-gpu-wins.zh-CN.md)）

| 用例 | 一句话结论 | 详情 |
| --- | --- | --- |
| c2 窗口聚合（HOP 与 TUMBLE） | GPU 吞吐与延迟都更好：调优后 HOP 两阶段 GPU +26%，TUMBLE +25%。 | [c2](group1-gpu-wins.zh-CN.md#c2-window) |
| c6 维表 Lookup Join（INNER） | 打满档调优后 GPU +76%（默认 +39%）；低/中档吞吐相同，且大多数批走 JVM 查表。 | [c6](group1-gpu-wins.zh-CN.md#c6-lookup) |

**第 2 组：CPU 赢**（[group2-cpu-wins.zh-CN.md](group2-cpu-wins.zh-CN.md)）

| 用例 | 一句话结论 | 详情 |
| --- | --- | --- |
| c1 过滤清洗 | 默认配置下 GPU +23%，调优后 CPU +20%。 | [c1](group2-cpu-wins.zh-CN.md#c1-filter) |
| c3 TopN | 调优后 CPU 比 GPU batch 快 11%。 | [c3](group2-cpu-wins.zh-CN.md#c3-topn) |
| c4 keep-last 去重 | 调优后 CPU +22%。 | [c4](group2-cpu-wins.zh-CN.md#c4-dedup) |
| c5 Interval Join（join、join-filter、join-agg） | 调优后 CPU +15%、+17%、+24%（对 GPU 融合）；接聚合时 GPU p50 延迟低约 23%。 | [c5](group2-cpu-wins.zh-CN.md#c5-join) |

**CPU 与 GPU 的计划差在哪。** 对每一对导出的计划，抽象语法树与优化后的物理计划（Optimized Physical Plan）在 CPU 与 GPU 上相同（c2 一阶段要与 CPU 一阶段计划比较）。差别在优化后的执行计划（Optimized Execution Plan）：GPU 模式下部分节点带 Mega-Flink 的名字（如 `GpuBoundedLookupJoin`、`GpuNativeStreamExecLocalWindowAggregate`），留在 CPU 的节点显示为 `FlinkCpuStreamExecCalc(...)`。真实计划行见各用例页。

<a id="parallelism"></a>
## 5. 多并行度与大维表

做法：同一份回放，单个轻量 Long 序列源经 `k mod p` 分发给 p 个回放 map（每个 subtask 按时间序回放），后续算子并行度 = p，CPU 与 GPU 并行度相同，p 个 subtask 共用 1 张 L20。仅测 c2、c3、c6。默认配置下的并行度 2/4 对拍 18 组（用例 × 档 × 并行度）全部 MATCH；调优后并行度 2/4 的对拍状态：未核实。

**调优后，打满档吞吐（条/秒）**（以下并行度 2/4 的数字取自内部报告，本草稿未逐项回源核对）

| 用例 | 并行度 2（CPU / GPU） | 并行度 4（CPU / GPU） | 谁更快 |
| --- | --- | --- | --- |
| c2 HOP 两阶段 | 2,015k / 2,532k | 2,907k / 3,365k | GPU +26% / +16% |
| c2 HOP 一阶段 | 1,846k / 2,017k | 3,028k / 3,340k | GPU +9% / +10% |
| c3 TopN（GPU batch） | 2,517k / 1,824k | 3,827k / 3,154k | CPU +38% / +21% |
| c6 Lookup | 1,592k / 2,991k | 3,168k / 3,809k | GPU +88% / +20% |

c3 GPU adaptive：并行度 2 为 980k，并行度 4 为 1,545k。

**c6 大维表**（调优，并行度 1，维表垫到 400 万行，GPU lookup 上限 4,194,304 行）：打满 CPU 1,292k，GPU 2,360k（+83%），gpu-min1 2,370k；中档两边跑满，GPU p50 3.1 ms，CPU <0.01 ms。

**延迟注意**：并行度大于 1 时，回放与算子之间多一次网络交换（Flink 默认 buffer-timeout 100 ms，低流量时缓冲块要等超时才发），所以低/中档 p50 比并行度 1 高很多（例如 c3 低档 83–126 ms 对并行度 1 的 14.7 ms）。这是测试驱动引入的、CPU 与 GPU 共有的加成，并行度大于 1 的低/中档延迟只用于两边相对比较。

<a id="breakdown"></a>
## 6. 时间花在哪里（breakdown 与 nsys）

- 单并行度下，Flink 自身的序列化 / 网络交换、source 回放、状态读写、DataStream 与 Table 的行转换占大头；GPU kernel 本身只占很小一块。例如 c3 GPU batch 打满：kernel 约 0.12 s，整个作业约 3.3M 行约 3.2 s（按正式中位数）。
- GPU 一侧新增的成本是行→列、H2D / D2H、列→行，以及 JVM 侧的批打包与解包；状态读写、timer、changelog 合并、Global 聚合仍在 JVM，GPU 没有替换。
- 采样估算与实测的口径区别见第 3 节。JFR 只采样线程名含 "(1/1)" 的任务线程，persistent 形态若有独立常驻线程则未被统计，其 GPU 占比仅供参考。
- CPU 与 GPU 的抽象语法树和优化后的物理计划（Calc / Rank / WindowAggregate / IntervalJoin / LookupJoin 的结构）完全相同；GPU 节点名出现在 EXPLAIN 的优化后执行计划里（如 `GpuBoundedLookupJoin`、`GpuNativeStreamExecLocalWindowAggregate`，CPU 的 Calc 显示为 `FlinkCpuStreamExecCalc`）。替换发生在 ExecNode 翻译为 Transformation 的阶段。本文早先版本写过"EXPLAIN 看不出区别"，这只对物理计划成立。
- 各用例的时间构成见各用例页。

<a id="limits"></a>
## 7. 已知局限

1. **单卡、并行度主要为 1。** 并行度 2、4 只测了 c2、c3、c6；c4、c5 在并行度 2/4 未测；多卡、更大并行度未测。
2. **c6 全量维表未测。** 全表 3,204 万行超过 GPU lookup 上限（4,194,304 行），只测了回放切片内的 1,039,412 行和垫到 400 万行的情形。
3. **报告改写后尚未独立复核。** 补测（object-reuse 调优、并行度、TUMBLE、nsys）之后的改写版没有经过独立复核。
4. **LEFT lookup 本构建不能上 GPU**，没有 LEFT 的性能对比。
5. **c6 的 JFR 占比因误归类不采用。** 凡用到同一包的用例，"GPU 相关"占比都需复查。
6. **JFR 是采样估算，**不是精确耗时；Mega-Flink 计时与 nsys 才是实测，其中 H2D / D2H 含排队。
7. **调优档位不全。** 调优后的低档（20k/s）未测；c4、c5、c6 的调优只测了中、打满两档。调优仅限 `pipeline.object-reuse`（及 c3 的 topn-cache），未做更多逐用例调参，也没有测试 mini-batch 等其他 CPU 侧选项（c4 的 `cpu-mb` 与 c5-join-agg 的 mini-batch 除外）。
8. **默认配置不是最优配置。** 默认配置下的结论（如 c1 GPU 赢）在调优后会变化，引用时请注明口径。
9. **对拍对象是最终表**，中间 changelog 在 CPU 与 GPU 之间不同（见第 3 节）。
10. **GPU persistent 与 adaptive：** 本次未得到"选 persistent"的支持；adaptive 在低/中档 0 行走 GPU。c5 persistent 只在 breakdown 轮记录了命令计数，正式轮为 0，原因未核实。
11. **未做：** 其他 GPU 型号（如 H20）、多卡、更大规模的维表、其他数据集；只验证了一种数据集与一种 GPU。
12. **c3 GPU batch 打满档有双峰波动**（见 [group2-cpu-wins.zh-CN.md](group2-cpu-wins.zh-CN.md#c3-topn) 的 c3 一节），机制没有更细的证据，不作断言。

<a id="repro"></a>
## 8. 复现

转换程序、测试驱动和运行脚本都在本目录（`prepare_stream_data.py`、`driver/`、`scripts/`），路径取自环境变量；分步说明见 [REPRODUCE.zh-CN.md](REPRODUCE.zh-CN.md)。c2、c6 的 GPU 侧也可以用库和 `../examples` 里的示例作业运行；c1、c3、c4、c5 的 GPU 侧仍需要 Mega-Flink 构建。

需要的东西：

- 1 张 NVIDIA L20（或同代 sm_89 的 GPU），驱动支持 CUDA 12.4；独占节点；宿主机内存足够 16 GB JVM 堆加数据文件。
- Flink 2.2.1 与与之配套构建的 Mega-Flink（GPU 算子与 shim）。获取方式与构建说明：未核实，待补。
- KuaiRand-27K 数据集（从其公开页面获取），以及把日志转成按 `time_ms` 排序的 int64 列文件的转换程序（`prepare_stream_data.py`；`driver/` 里另有 Java 参考转换器）。
- 测试驱动：负责进程内回放、SQL 定义、sink 收 changelog、延迟统计与 SHA-256 对拍（`driver/`）。
- 运行脚本：按"用例 × 模式 × 档位 × 轮数"轮换执行，汇总中位数并做对拍（`scripts/`）。
- 可选：JFR 与 nsys，用于 breakdown 与 kernel 核对。

模式名（沿用内部命名，仅供对照结果表）：`cpu`、`cpu-mb`、`cpu-1p`、`gpu`、`gpu-1p`、`gpu-batch`、`gpu-persistent`、`gpu-adaptive`、`*-fuse` / `*-nofuse`、`gpu-min1`；调优模式在后面加 `-tuned`。

注意事项：

- 运行期间不要修改正在被读取的 shell 脚本；内部曾因此出现个别轮次被误判为失败（数据本身有效，已计入）。
- 定速档的延迟按计划发送时刻计算；打满档延迟只作相对比较。
- 并发运行会改变吞吐（初筛里并发数字多数偏高 3–6%，个别用例有离群值），对比请在独占节点上进行。
