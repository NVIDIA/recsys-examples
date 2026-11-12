<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 复现流式基准

本页说明如何用本目录的驱动和脚本重跑六个流式用例（c1-c6，见 [README.zh-CN.md](README.zh-CN.md)）：编译驱动、准备 KuaiRand 数据、跑用例、核对各模式结果一致、读结果。

- **CPU 基线只需要开源 Flink 2.2.1。** 不涉及任何插件。
- **GPU 侧不能单靠本目录复现。** c2（窗口聚合）和 c6（维表查询）的 GPU 侧可以用 Feature-Extractor 库通过 `../examples` 里的示例作业运行（`WindowFeaturesJob`、`LookupAuthorJob`，见 `../examples/README.md`），并以同样方式与 CPU 运行对拍最终表。但本目录的基准脚本只通过 Mega-Flink 构建驱动 GPU 模式，已发表的 c2、c6 GPU 数字也来自它。c1、c3、c4、c5 的 GPU 数字同样来自 Mega-Flink 构建。**这四个用例的 GPU 侧需要 Mega-Flink 构建（[https://github.com/cb521/Mega-Flink](https://github.com/cb521/Mega-Flink)）；本目录提供驱动与脚本，库里没有对应的算子。**

下文路径一律用环境变量：`$BENCH` 是本目录（仓库中的 `examples/Feature-Extractor-Plugin/streaming/benchmarks`），`$FLINK_HOME` 是解压后的 Flink 发行版，`$FEP_DATA_DIR` 是准备好的数据，`$OUT` 是结果目录。

> **各步骤的验证状态。** 已验证：第 3 步（驱动用开源 Flink 2.2.1 二进制发行版和 JDK 11 编译通过）；第 4 步的转换部分（在一个小的合成 CSV 上，转换结果与独立的 Java 参考转换器逐字节一致；`data-sha256.txt` 里的 17 个参考校验和来自早先由该 Java 转换器处理完整数据得到的文件）；第 5 步（在一张 L20 上做的冒烟测试，见下）；脚本的 Mega-Flink 模式，用一份本地 Mega-Flink 构建跑过 c1、c2-hop、c3、c4、c5-join、c6（最终表 SHA-256 与 CPU 一致）。**没有运行的：第 4 步的下载**（下载地址和校验和来自数据集页面，用 `HEAD` 请求确认文件存在，9,892,191,178 字节）、**Python 转换器在完整 9,292 万行文件上的运行**（所以你的数据请用 `data-sha256.txt` 做 `sha256sum -c`）、**第 6 步在开源 Flink 上的完整矩阵**、第 9 步的 breakdown 与 nsys 脚本。已发表的数字来自 Mega-Flink 构建的普通 CPU 模式，不是开源 Flink；冒烟切片上两者最终表一致，但完整矩阵没有互相对比。

## 0. 需要什么

| | 测试环境 |
| --- | --- |
| GPU（只有 Mega-Flink 运行需要） | 1 张 NVIDIA L20（Ada，sm_89），驱动对应 CUDA 12.4。CPU 基线不需要 GPU |
| CPU / 内存 | 已发表的运行用独占节点、16 GB JVM 堆；主机 CPU 型号和核数：未核实。回放源、Flink 任务和 sink 都在同一个 JVM 里，请留出多个空闲核 |
| 磁盘 | 压缩包约 10 GB（第 4 步后可删），CSV 约 7 GB，转换后数据约 11 GB；预期用本地盘 |
| 操作系统 | Linux（用过 Ubuntu 22.04） |
| JDK | OpenJDK 11。没试过 JDK 17 及以上；需要用 `FEP_JVM_ARGS` 传 Flink 的 `bin/config.sh` 里列出的 `--add-opens` 参数 |
| Flink | 2.2.1 二进制发行版，Scala 2.12（`flink-2.2.1-bin-scala_2.12.tgz`） |
| Python | 3.8 及以上，需要 `numpy`；`pandas` 可选（转换快数倍） |
| CUDA | 只有 Mega-Flink 需要；用 `CUDA_LIB_DIR` 把运行时库加进 `LD_LIBRARY_PATH` |
| Mega-Flink | 只有 GPU 模式需要，见第 7 步 |

第 4 步内存：完整日志约 25 GB。驱动在堆上保存 `预热 + 测量` 行、14 列 int64（已发表规模约几百 MB），外加捕获的 sink 输出行。

## 1. 获取代码

```bash
export WORK=<一个空目录>
cd $WORK
git clone -b feature/feature-extractor-plugin https://github.com/cb521/recsys-examples.git
export BENCH=$WORK/recsys-examples/examples/Feature-Extractor-Plugin/streaming/benchmarks
```

## 2. 获取 Flink

```bash
export FLINK_HOME=$WORK/flink-2.2.1
cd $WORK
curl -fLO https://archive.apache.org/dist/flink/flink-2.2.1/flink-2.2.1-bin-scala_2.12.tgz
tar xzf flink-2.2.1-bin-scala_2.12.tgz          # 生成 $WORK/flink-2.2.1
export JAVA_HOME=<JDK 11 目录>                  # 不设则用 PATH 里的 java
```

不需要启动 Flink 集群。驱动在自己的 JVM 里跑 Flink 的 `MiniCluster`，只需要 jar：`build.sh` 和 `run-one.sh` 取发行版 `lib/*.jar`（去掉 planner loader），再加 `opt/flink-table-planner_2.12-2.2.1.jar`。

## 3. 编译驱动

```bash
export FEP_BUILD_DIR=$WORK/build                # 可选，默认 $BENCH/driver/build
$BENCH/driver/build.sh
```

预期输出：`built $FEP_BUILD_DIR/classes`。它用 `javac` 编译 `driver/src/main/java/fep/*.java`（PATH 上只有 JRE 启动器时改用 JDK 的编译器模块）。`src/main` 里没有任何对 Mega-Flink 的引用。

## 4. 准备数据

驱动读取 KuaiRand-27K 曝光日志（`part2` 文件，92,919,789 行）的 int64 列文件（按 `time_ms` 排序）和视频表（video_id、author_id）。[KuaiRand](https://kuairand.com)（Gao 等，CIKM 2022，CC BY-SA 4.0）是公开数据集；引用和许可请以数据集页面为准。

```bash
export FEP_DATA_DIR=<本地快盘，约 30 GB 空闲>/kuairand-stream
pip install numpy pandas                       # pandas 可选
python3 $BENCH/prepare_stream_data.py --out $FEP_DATA_DIR
```

脚本从 Zenodo 下载 `KuaiRand-27K.tar.gz`（约 9.9 GB，md5 `3e3c799a24e2d23a4d2c757fbf9adf59`，可断点续传），校验 md5，解出日志和视频 CSV 并转换。已有 CSV 时用 `--src <KuaiRand-27K/data 目录>`。预期输出末尾：

```
== log: 92919789 rows
== video: 32038725 rows
```

目录结构：`$FEP_DATA_DIR/log/<列名>.i64`（14 列）、`$FEP_DATA_DIR/video/{video_id,author_id,video_duration}.i64`，各有一个 `meta.txt`。

**核对数据**（确认转换结果与已发表数字所用的一致）：

```bash
cd $FEP_DATA_DIR && sha256sum -c $BENCH/data-sha256.txt     # 17 行，全部 OK
```

`driver/src/main/java/fep/KuaiRandConvert.java` 是规则相同、相互独立的 Java 转换器（参考数据就是它做的）：`java -cp $FEP_BUILD_DIR/classes fep.KuaiRandConvert log <日志 csv> <输出>/log`，视频表同理把 `log` 换成 `video`。

快速小数据：`--limit-rows 500000` 只读每个 CSV 的前 50 万行（是另一份更小的数据，校验和不在 `data-sha256.txt` 里）。没有 pandas 时用纯 Python 解析，每 1,000 万行预计要几分钟。

## 5. 冒烟测试（几分钟）

每个用例在 40 万行切片上各跑一次打满档（10 万预热 + 30 万测量），CPU 基线用开源 Flink：

```bash
export HEAP=16g WARMUP=100000
for c in c1-filter c2-hop c2-tumble c3-topn c4-dedup c5-join c5-join-filter c5-join-agg c6-lookup; do
  $BENCH/scripts/bench.sh $c cpu 0 300000 $OUT/smoke --dump-final true
done
```

每次运行打印一行（驱动完整输出在 `<运行目录>/stdout.log`，末尾是 `rc=0`）：

```
RESULT case=c1-filter label=cpu rate=0 thr=<条/秒> p50=<ms> p95=<ms> p99=<ms> out=199254 final=199254 sha=2f29e3f499b32bbf
```

`thr` 和延迟取决于你的机器；**`out`、`final` 和 `sha` 与机器无关**。用第 4 步的数据，应得到下表（最终表的 SHA-256，前 16 位十六进制；完整摘要在 `result.json`）：

| 用例 | 最终表行数 | sha256（前 16 位） |
| --- | ---: | --- |
| c1-filter | 199,254 | 2f29e3f499b32bbf |
| c2-hop | 85,727 | 528ec5b56b375126 |
| c2-tumble | 34,625 | ef46e2ce28a2ff15 |
| c3-topn | 58,028 | 21d7d0915cc5edfa |
| c4-dedup | 392,746 | 8593325de68ad677 |
| c5-join | 157,446 | d9600de61e056926 |
| c5-join-filter | 65,810 | 20e9ede74afdef3a |
| c5-join-agg | 6,246 | eff1aaeab5eb2b48 |
| c6-lookup | 400,000 | 07800d9c2c117f9f |

这些值用 `data-sha256.txt` 对应的数据得到，并在开源 Flink、Mega-Flink 的 CPU 模式、以及（c1、c2-hop、c3、c4、c5-join、c6 的）GPU 模式下完全一致。如果你的不同，先查 `sha256sum -c`。c3、c4、c5-join-agg 的 changelog 行数 `out` 在多次运行之间、CPU 与 GPU 之间可能不同（最终表不会）；例如 c5-join-agg 两次 CPU 运行分别打印 94,666 和 95,176，最终表 SHA 相同。这次冒烟的吞吐不是基准，不发表（运行时作业只看到 2 个 CPU 核）。

核对实际执行的 SQL 与 `cases/` 里的一致：

```bash
python3 $BENCH/scripts/check_sql.py $OUT/smoke              # 预期：checked N runs, 0 differ
```

## 6. 跑 CPU 基线矩阵

`scripts/matrix.sh` 按 档位 x 用例 x 轮次 运行，每轮轮换模式顺序，第 1 轮保存最终表：

```bash
export HEAP=16g
# 打满档，调优（pipeline.object-reuse=true），3 轮：README 里的"调优 CPU"一列
TIERS=high $BENCH/scripts/matrix.sh $OUT/run 3 \
    "c1-filter:cpu-tuned" "c2-hop:cpu-tuned,cpu-1p-tuned" "c2-tumble:cpu-tuned" "c3-topn:cpu-tuned" \
    "c4-dedup:cpu-tuned,cpu-mb-tuned" "c5-join:cpu-tuned" "c5-join-filter:cpu-tuned" "c5-join-agg:cpu-tuned" \
    "c6-lookup:cpu-tuned"
# 默认配置、三个档位：模式名去掉 -tuned，不设 TIERS
```

| 档位 | 输入速率 | 测量条数 | 预热 |
| --- | --- | --- | --- |
| 低 | 20,000 条/秒 | 600,000 | 300,000（同速率） |
| 中 | 100,000 条/秒 | 3,000,000 | 300,000 |
| 高（打满） | 不限速 | 3,000,000 | 300,000 |

限速档按定义要花墙钟时间：以 20,000 条/秒回放（600,000 + 300,000）行要 45 秒，以 100,000 条/秒回放（3,000,000 + 300,000）行要 33 秒，每次运行、每一轮都如此，外加启动。整个矩阵没有计时。单次运行超过 `RUN_TIMEOUT` 秒（默认 900）会被杀掉。每次运行单独起一个 JVM，输出到 `$OUT/run/<用例>/<模式>/r<速率>-n<条数>-t<轮次>/`。

CPU 基线的模式（也见 `scripts/bench.sh` 开头注释）：

- `cpu`：Flink 默认计划。`cpu-1p`：一阶段聚合（c2）。`cpu-mb`：mini-batch 5 ms / 4,096 行（c4；`c5-join-agg` 两边都固定开 mini-batch，因为两阶段聚合需要它）。
- `-tuned`：`pipeline.object-reuse=true` 加 `table.exec.rank.topn-cache-size=100000`（只对 c3 有意义）。README 的方法是每个用例取双方各自最好的 CPU 模式；每个用例实际测了哪些模式见用例页。比较前先把 CPU 侧调好。
- `--parallelism p`（用 `XARGS="--parallelism 2"` 传入）用 p 个子任务回放；`预热 + 测量` 必须能被 p 整除。c6 可用 `-Dfep.dim.target=4000000`（通过 `JAVA_OPTS`）把维表补到指定行数（补的是不会被查到的视频）。

## 7. GPU 侧（Mega-Flink）

需要一份适配 Flink 2.2.1 的 Mega-Flink 构建，来自 [https://github.com/cb521/Mega-Flink](https://github.com/cb521/Mega-Flink)（下面的 7a 节用一条命令按固定提交构建，并生成 env 文件；手工方式如下）。然后：

```bash
export MEGA_FLINK_CP=<类路径：Flink 2.2.1 jar + Mega-Flink runtime + 它的 Flink 2.2.1 shim + 它的 benchmark harness 类>
export MEGA_FLINK_NATIVE_DIR=<Mega-Flink native 库目录>
export CUDA_LIB_DIR=<CUDA 运行时库目录，例如 /usr/local/cuda/lib64>
$BENCH/driver/build.sh                        # 同时把 driver/src/megaflink 编译到 $FEP_BUILD_DIR/classes-megaflink
```

设了 `MEGA_FLINK_CP` 后，脚本用它代替 `$FLINK_HOME`，给驱动传 `-Dfep.megaflink=true`，GPU 模式才可用。类路径上必须有 harness 类 `io.github.cb521.cudfflink.benchmark.BenchmarkGpuResource`；驱动用反射调用它来设置基准用的 GPU 参数。此时 c6 的维表由 `FepVideoDimGpuFactory` 提供，它多了 GPU 查表所需的快照接口；CPU 查表函数是同一个。

```bash
# README 里的 GPU 模式：gpu-batch（c1、c3、c4、c5），gpu / gpu-1p（c2、c6），-fuse / -nofuse（c5），-min1（c6）
TIERS=high $BENCH/scripts/matrix.sh $OUT/run 3 "c1-filter:cpu-tuned,gpu-batch-tuned" "c2-hop:cpu-tuned,gpu-tuned" "c6-lookup:cpu-tuned,gpu-tuned"
```

涉及 Mega-Flink 的部分只在三处：`FepKuaiRandBench.java` 里的反射调用 `configureMegaFlink` 和 `fep.megaflink` 开关、`driver/src/megaflink/`、以及 `scripts/bench.sh` 只在设了 `MEGA_FLINK_CP` 时才加的 `cudf.flink.*` 参数。没设时，`bench.sh` 会拒绝 `gpu*` 模式并给出提示。

## 7a. 一条命令得到 Mega-Flink 构建，一条命令跑六个用例

本节取代第 7 步里的"按它的 README 构建"和"请问负责人"：构建已脚本化并固定了版本。

**固定版本。** `build_megaflink.sh` 构建 Mega-Flink 提交 `f7f06845cb6b457c83d424d6424632fbf929e230`（[https://github.com/cb521/Mega-Flink](https://github.com/cb521/Mega-Flink) 的 `main`，提交说明 "Merge Spark expression-fusion plugin into main"；该提交在公开历史中，全新 `git clone` 即可取到）。依据：已发表的数字是用恰好这个提交的一次构建测得的，构建在 L20 上、CUDA 12.4，native 库按 sm_89 编译（`CUDF_FLINK_CUDA_ARCH=89`），运行方式是 classpath（三个模块的 Maven `target/classes` 加依赖 jar），没有装进 Flink。该工作树相对此提交唯一的源码差异，是 native 构建脚本里把临时构建目录改为移走而非删除，不影响编译出的代码。未核实：全新构建的 class 和库没有与当年的逐字节比对；比对的是最终表 SHA-256（见下）。

**补丁：不需要。** 上游只放行名字含 `H20`（sm_90）的设备，除非编译时定义 `CUDF_FLINK_EXPERIMENTAL_DEVICE_MAJOR`（`native/gpu_arch_compat.h`）；否则 L20 构建运行时会报 "does not admit selected device"。上游 native 构建脚本在给定其他架构时自己会定义这个宏，所以脚本只需传 `CUDF_FLINK_CUDA_ARCH=89`（默认 `ARCHS=89`），此时放行所有计算能力主版本为 8 的设备。要为 H20 构建请用 `ARCHS=90`（保持原有按名字的规则）。`benchmarks/patches/` 是放源码补丁的位置（按文件名顺序应用，先检查，可重复执行），此提交下该目录不存在。

**构建**（需要：git、能访问 GitHub 和 Maven Central 的网络、带 `javac` 的 JDK 11、带 `nvcc` 的 CUDA 12.x 工具包；仓库自带的 Maven wrapper 会把 Maven 3.9.11 和 Flink 2.2.1 的 jar 下载到克隆目录里，`jni.h` 取自 JDK，只有 JRE 时会带校验和下载；CUB 随 CUDA 工具包提供）：

```bash
WORK_DIR=$WORK/megaflink CUDA_TOOLKIT_ROOT=/usr/local/cuda JAVA_HOME=<JDK 11 目录> \
    $BENCH/scripts/build_megaflink.sh
```

变量：`MEGA_FLINK_REF`（默认上面的提交）、`MEGA_FLINK_REPO`、`WORK_DIR`、`CUDA_TOOLKIT_ROOT`、`ARCHS`（一个架构号，默认 89）、`JAVA_HOME`。联网的机器没有 CUDA、GPU 节点又不能联网时，用 `--java-only` 和 `--native-only` 分两步（先 `--java-only`，再用同一个 `WORK_DIR` 跑 `--native-only`）。已有的克隆会在校验提交后复用。脚本以 `[build_megaflink] wrote .../megaflink.env` 结束；native 步骤还会打印 `MEGA_FLINK_NATIVE_BUILD_OK native_dsos=14 ...`（`nvcc` 的编译警告属正常）。

**不安装任何东西。** 你的 Flink 安装（`FLINK_HOME`）不会被读取或修改：基准驱动在自己的 JVM 里跑 Flink `MiniCluster`，Mega-Flink 来自 `megaflink.env` 里的类路径（`MEGA_FLINK_CP`、`MEGA_FLINK_NATIVE_DIR`、`CUDA_LIB_DIR`），Flink 2.2.1 的 jar 来自克隆目录里的 Maven 缓存。Mega-Flink 的 planner shim 会覆盖 Flink 内部的 planner 类，装进真实集群需要它自己的安装器；那是另一件可选的事，本脚本不做。如需：`target/distribution` 由 Mega-Flink 的 `scripts/build-distribution.sh` 生成（H20/sm_90 的打包），用其中的 `install-mega-flink.sh` 安装（`install`、`verify`，并保存精确备份供其回滚命令恢复）；使用前请读它的 README。此处未试过。

**跑六个用例**（CPU 基线和 Mega-Flink GPU，对拍最终表 SHA-256，汇总成一张表）：

```bash
source $WORK/megaflink/megaflink.env
export FEP_DATA_DIR=<第 4 步准备的数据> FEP_TMP_DIR=<快的本地盘>
$BENCH/scripts/run_all_six.sh --quick --out $OUT/six      # 冒烟：100,000 预热 + 300,000 条，即第 5 步那份切片
$BENCH/scripts/run_all_six.sh --out $OUT/six-full         # 打满档一轮：300,000 预热 + 3,000,000 条
```

每个用例在 `cpu-tuned` 和各自的 Mega-Flink GPU 模式下各跑一次：`gpu-batch-tuned`（c1、c3、c4、c5-join）、`gpu-1p-tuned`（c2-hop；一阶段聚合，成熟的窗口路径）、`gpu-tuned`（c6-lookup）。`--cases` 选子集，`--records`、`--warmup` 改规模；需要时会先编译驱动。只有每个用例都 MATCH、GPU 算子确实替换了 CPU 算子（`result.json` 里的 `shim_status`），并且在标准数据上用 `--quick` 时摘要等于第 5 步的参考值，退出码才是 0。标准数据上 `--quick` 的预期输出（吞吐写在各 `result.json` 里，不是基准）：

```
| case | CPU final rows | GPU final rows | CPU sha256 (16) | GPU sha256 (16) | CPU vs GPU | GPU operator replaced | vs REPRODUCE reference |
| c1-filter | 199,254 | 199,254 | 2f29e3f499b32bbf | 2f29e3f499b32bbf | MATCH | yes | equal |
| c2-hop | 85,727 | 85,727 | 528ec5b56b375126 | 528ec5b56b375126 | MATCH | yes | equal |
| c3-topn | 58,028 | 58,028 | 21d7d0915cc5edfa | 21d7d0915cc5edfa | MATCH | yes | equal |
| c4-dedup | 392,746 | 392,746 | 8593325de68ad677 | 8593325de68ad677 | MATCH | yes | equal |
| c5-join | 157,446 | 157,446 | d9600de61e056926 | d9600de61e056926 | MATCH | yes | equal |
| c6-lookup | 400,000 | 400,000 | 07800d9c2c117f9f | 07800d9c2c117f9f | MATCH | yes | equal |
```

**已验证：** 这一对脚本在一张 L20（驱动对应 CUDA 12.4、CUDA 工具包 12.4、JDK 11）上从空目录运行：从固定提交的全新克隆构建 Java 与 native，再跑 `run_all_six.sh --quick`：六个用例全部 MATCH，且等于参考摘要。**未验证：** 其他 GPU、其他 CUDA 或 JDK 版本、H20 构建（`ARCHS=90`）、没有网络的节点上单独跑 `--native-only`、`run_all_six.sh` 的完整（非 quick）运行、安装器路径。

## 8. 核对结果

每次运行写一个 `result.json`，其中 `final_sha256` 是最终表（应用 changelog 后、行排序）的 SHA-256。按用例和档位对比：

```bash
python3 $BENCH/scripts/summarize.py $OUT/run                # 先出中位数表，再按用例和档位给出 MATCH / MISMATCH
```

已发表的结果全部 MATCH。**MISMATCH 的不要报速度。** 对比对象是最终表；CPU 与 GPU 的中间 changelog 行数不同（见 README 第 3 节）。CPU 和 GPU 的运行放在同一个结果根目录下才会互相对比（上面的例子都用 `$OUT/run`）。需要逐行对比时，第 1 轮还保留了 `final.tsv`。

## 9. 读结果

- `summarize.py` 按 用例 x 档位 x 模式 输出各轮吞吐的中位数、各轮 p50 / p95 / p99 的中位数、changelog 行数和最终表行数，并列出每轮数值。统计口径同 README（3 轮中位数；分位数先每轮算、再取跨轮中位数）。
- 吞吐 = 测量条数 /（最后一条 sink 到达时间 - 第一条测量输入的发送时间）。延迟 = sink 到达时间 - 使该输出可见的输入行的计划发送时间；窗口用例以窗口结束时刻之后的第一行为参照；`c5-join-agg` 以该用户第 k 个匹配对为参照。打满档 source 被反压，延迟只适合相对比较。见 README 第 3 节。
- `plan.txt` 是 `EXPLAIN` 文本（`bench.sh` 对 `cpu*` 模式设了 `--forbid-gpu true`，计划里出现 GPU 节点就报错）；`plan-full.txt` 是 JSON 执行计划，`jobgraph.txt` 列出算子链。`result.json` 还有 `gpu_in_plan`、`shim_status` 和 `metrics`（Mega-Flink 的阶段计时；开源 Flink 下为空）。
- Breakdown：`scripts/breakdown.sh ROOT TIER "case:modes"` 每个模式做一次 JFR 采样运行（CPU 也能用；Mega-Flink 模式下附加阶段计时），`scripts/digest.py ROOT` 打印汇总。`scripts/nsys-study.sh`、`scripts/nsys_digest.py`（Nsight Systems 交叉核对）和 `scripts/c3-study.sh`、`scripts/c3_digest.py`（c3 专项调查）需要 Mega-Flink。JFR 占比是采样估计；阶段计时是实测（README 第 3、6 节）。

## 10. 已知局限

- 单卡，几乎都是并行度 1；没有多卡运行。并行度 2、4 只有 c2、c3、c6（README 第 5 节）。
- 没有 Kafka 等消息系统：日志由一个 source map 函数在进程内回放，sink 在内存中捕获行。
- 已发表数字来自一份改写后的内部报告，**没有独立复核**（见 README 的状态说明）。作者以外没有人重跑过。
- CPU 基线用开源 Flink 只在冒烟切片上核对过（计划里没有 GPU 节点，最终表与 Mega-Flink CPU 模式一致）；完整矩阵没有在两者之间对比。
- `prepare_stream_data.py` 的转换器只在小的合成文件上与 Java 参考转换器对过，没有在下载的压缩包上对过。
- 绝对数字随 CPU、磁盘、JDK、GPU 变化；请在独占机器上跑。筛选时，同一节点并发作业会让吞吐变化几个百分点。
- 作业读取脚本时不要编辑它（bash 增量读脚本，曾因此把几轮误标为失败）。脚本为每次运行在 `FEP_TMP_DIR` 下建一个临时目录，不会清理；Flink 自己的文件由 Flink 清理。
- 主机 CPU 型号：此处没有记录。已发表 GPU 数字用的 Mega-Flink commit 和 native 编译参数见 7a 节（全新构建的产物没有与当年的逐字节比对）。
