<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Streaming: how it works and why it is safe to switch on

English | [中文](how-it-works.zh-CN.md)

This page describes what [Mega-Flink](https://github.com/cb521/Mega-Flink) does **today** (label **Exists**). Section 7 says what the operator library in this directory took over and what it did not. Where the Mega-Flink docs say nothing, this page says "not documented" and does not guess.

## 1. One-page overview

```
 SQL / Table API -> Flink planner -> optimized StreamExec* nodes
   plan time, per node:  all expressions supported?  latency-safe?  experimental switch on?
      no  -> original Flink CPU node         explain: GpuNativeFallback*(reason=...)
      yes -> GPU operator                    explain: GpuNativeStreamExec*(mode=gpu-native, ...)

 run time, per batch:
   RowData -> copy needed columns -> pinned buffer -> H2D -> CUDA kernel -> D2H -> pinned buffer
           -> JVM rebuilds RowData -> merge into Flink state / emit
   a batch is cut by: batchRows | maxWaitMs | watermark | checkpoint barrier | end of input

 ledger:  Flink keyed state + timers (checkpointed)      GPU memory: a cache that can be rebuilt
```

Three rules carry the design:

- Admission is decided per physical node and is all-or-nothing. A node that is not fully supported stays on Flink CPU.
- Flink state and timers stay the source of truth. Checkpoints and recovery do not depend on GPU memory.
- Every GPU path is checked against the CPU result, and paths that did not win are off by default.

## 2. One batch, from input to output

1. **Copy in.** The operator (for example `CudfTumblingWindowOperator`, `CudfAggregateOperator`) copies only the referenced columns of each `RowData` into a pinned buffer that the native side created (kernel wrappers such as `DenseGroupbyKernel`).
2. **H2D, kernel, D2H.** The wrapper submits through JNI. Native code lives in `.cu` files such as `dense_groupby_jni.cu`. Device buffers, pinned buffers, streams and events are allocated once in `open()` and reused; a batch does no `cudaMalloc`/`cudaFree`.
3. **Rebuild rows.** Results are decoded in the JVM into `BinaryRowData` (`PackedBinaryRowBatch` holds many rows in one `byte[]`), merged into Flink state, then emitted.
4. **Timing.** Each batch is split into `row_to_native`, `h2d`, `compute`, `d2h`, `native_to_row`.

There is no Arrow or columnar exchange. Except for the listed deep fusions, data goes back to CPU row format after each node, and the GPU does not cross an Exchange.

**When a batch is cut.** A batch runs when the first of these happens (architecture doc, "batching boundaries"):

- the buffer reaches `cudf.flink.batch.rows` (default 4096), or `cudf.flink.batch.max-wait-ms` (default 5) passes after the first row;
- a watermark arrives (for a two-input join, a watermark on either side);
- a checkpoint barrier is about to snapshot (`prepareSnapshotPreBarrier`);
- bounded input ends (`endInput`/`finish`).

Window operators batch inside the window and watermark boundaries. They never emit earlier than the watermark allows. Operators that emit per record do not batch under the default `cudf.flink.batch.policy=latency-safe` (section 4). A fixed `Calc` shape has a per-record `persistent` path: a mapped ring of 1,024 slots, no batching.

## 3. State, checkpoints, watermarks, failures

### 3.1 State

Flink state is the checkpoint ledger. GPU memory is a cache that can be thrown away and rebuilt.

| Operator | In Flink state | On the GPU |
|---|---|---|
| Group aggregate | `ValueState` accumulators, `cudfAggregateAverageStateV1` | batch: this batch's partials. persistent: cached; reseeded from Flink state |
| Window aggregate | `ValueState` (`cudfTumblingWindowStateV2`) + event-time timers | this batch's partials |
| Inner join | `ListState` of both sides (`gpuNativeJoinLeftRows`, `gpuNativeJoinRightRows`) | persistent: hash cache, lazily reloaded per key after restore |
| Interval / temporal join | `MapState` as ledger + timers | optional hot cache (temporal cache is off by default) |

`GpuDeviceBatch` handles never enter a checkpoint (island design doc).

**Exception: bounded lookup join.** The dimension table lives only in GPU memory (unique BIGINT key, read-only, at most `cudf.flink.lookup-join.dimension.max-rows`, default 1,048,576). Flink stores an identity and a SHA-256 content digest in operator state `cudfLookupJoinDimensionSnapshotV1`. On restore, every subtask reloads the snapshot from the connector. If the identity or digest differs, it refuses to start. Rebuilds are double-buffered and atomic, so a failed rebuild never exposes a half-built table.

**State names and savepoints.** Docs say operator UID and state serializer do not change when the engine (CPU, persistent, batch) changes. A fused island keeps the anchor operator's original UID, state descriptor name and serializer. Restoring from a savepoint written by an older plugin version is on the open list; only one narrow join-calc-aggregate savepoint gate exists. Whether a CPU-planned job's savepoint restores onto the GPU operator: not documented.

### 3.2 Barriers, watermarks, timers

- **Barrier.** The operator flushes pending rows, merges results into Flink state, drains any ring, and only then lets the barrier pass. Window operators also write cached partials to state and register their timers first, so recovery does not depend on the transient cache. Stateless `Calc` buffers are not checkpointed.
- **Watermark.** `processWatermark` flushes first, then forwards the watermark. If a watermark already covers a window end, the window result is emitted directly without writing state.
- **Timers.** Flink's timer service owns all timers (`registerEventTimeTimer`, `onEventTime`). The plugin adds a processing-time timer for `max-wait-ms`, and a 1 ms timer that collects the tail of the `Calc` ring. Late rows are counted (`cudfWindowLateRowsDropped`).

### 3.3 When the GPU fails: two different things

- **Before anything is visible:** a stateless island keeps its host batch until device output is published, and can replay the whole batch on CPU. Aggregates reduce a batch that holds a value the kernel cannot handle in the JVM (`CpuAggregateFallback`) and merge it into the same Flink state.
- **After output is visible or state or timers changed:** the paper says a CUDA error fails the task, and Flink restores from a completed checkpoint. This avoids a batch that is part GPU and part CPU. Real CUDA OOM and an illegal launch were injected once each; the task recovered and the final table was identical. GPU reset, ECC and illegal-address faults are not tested.
- Going over the GPU memory budget fails before `cudaMalloc`. Persistent cache full: `adaptive` releases it and switches to batch; fixed `persistent` fails.

## 4. Admission and fallback

| Layer (`cudf.flink.*`) | Default | Effect |
|---|---|---|
| `mode` | `cpu-row` | GPU is never used unless set to `gpu-native`. The master switch |
| `fallback.enabled` | `true` | unsupported node returns to the Flink node; `false` (benchmarks) throws |
| `batch.policy` | `latency-safe` | a per-record node with only a batch GPU path stays on CPU; `throughput` accepts the wait; Flink mini-batch counts as accepted |
| `experimental.operators.enabled` | `false` | paths without default performance/stability admission need it |
| `experimental.generic-calc.enabled` | `false` | generic Calc interpreter (also needs `throughput`) |
| `cross-operator-fusion.enabled`, `operator-islands.enabled` | `true`, `false` | fusion sub-paths such as `calc.local-aggregate.native.enabled` default `false` |

- **Where a fallback lands.** The same `StreamExec*` class calls the original Flink implementation, and explain shows `GpuNativeFallbackCalc(reason=...)`. If a fusion proof fails, the whole segment goes back to separate operators. There is no half fusion.
- **Compiled plans.** `mode`, batch, fallback and memory settings are written into the compiled plan. A restored plan keeps `gpu-native` even if the new process defaults to CPU. How to turn an already compiled plan back to CPU: not documented. Installation patches the planner loader; `rollback` restores the original bytes (SHA-256 checked).
- **Runtime controller.** It never changes which operator was chosen. It only switches between paths that operator already has. `engine=adaptive` samples pressure (EWMA), uses separate enter and exit thresholds, needs several confirmations plus a minimum residence time, and drains the old path first. Example aggregate defaults: 100 ms sampling, batch in/out at 200k/150k rows/s; `aggregate.adaptive.persistent-enabled=false`, so it moves between CPU and batch. Calc `auto` and Over `auto` choose statically at plan time.
- **Small batches stay in the JVM.** Lookup join: below `cudf.flink.lookup-join.gpu.min-batch-rows` (default 1,024) a partial batch uses the exact JVM lookup (`cudfLookupJoinCpuBatches`). Window join: at most 64 candidate pairs per key and window run in a JVM task. Temporal sort: groups under 256 rows stay in the JVM. A global `gpuMinRows` option: not documented.

## 5. Why results match the CPU

- **Fail-closed semantic checks.** A node must be fully supported before it moves. The checks cover row kinds (INSERT-only for most stateful paths), key and type, NULL and NaN rules, timestamp precision and watermark rules.
- **Independent oracle.** Each benchmark workload has its own CPU expected-value function, so CPU and GPU cannot be wrong the same way. Checks use exact counts and an order-independent digest, a final dynamic table, or an exact changelog. Recovery tests inject one failure after a completed checkpoint. A GPU run is void if the GPU operator did not really run (explain plus runtime status must both show it).
- **SHA-256.** Docs name SHA-256 for the lookup snapshot, installer files and snapshot trailers. The long-run oracle states that a hash does not prove set equality. The digest algorithm of the final-table check: not documented.
- **Known differences and limits.** Stock Flink 2.2.1 compares NaN with raw `>`/`<`, which can make an illegal retract, so Rank/TopN with NaN is checked against an independent reference. Stock Flink rejects streaming RANK/DENSE_RANK, so its CPU path is the plugin's own reference. Not supported (stay on CPU): UDF, DECIMAL columns, DATE/TIME/BINARY, nested types (except UNNEST prototype), retract with non-zero TTL, MIN/MAX multiset retract, multiple DISTINCT, processing-time windows. Many paths are experimental; the project calls itself a research plugin until its P0 list is done.

## 6. Observability

- **Explain.** `GpuNativeStreamExecCalc(mode=gpu-native, engine=batch, batchRows=4096, maxWaitMs=5)` or `GpuNativeFallback*(reason=...)`. The text plan can still show a fused AVG `Calc`; trust `calc_planner_status=gpu-fused-window-avg-finalize:gpu-native` instead.
- **Flink metrics, per operator.** `cudfBatches`, `cudfInputRows`, `cudfOutputRows`, `cudfRowToColumnNanos`, `cudfComputeNanos`, `cudfColumnToRowNanos`; `cudfAggregatePersistentCommands`, `gpuJoinPersistentCommands`, `cudfOverPersistentCommands`; `*AdaptiveMode`, `*AdaptiveSwitches`. `h2d` and `d2h` are in `RuntimeMetrics`; their export as Flink metrics: not documented.
- **Fallback counters.** `gpuJoinJavaFallbacks`, `cudfFusedCalcNativeFallbackBatches`, `cudfOverExactJvmFallbackBatches`, `gpuJoinPersistentCapacityFallbacks`, `cudfTopNAdaptivePersistentCapacityFallbacks`.
- **Memory.** `cudfGpuMemoryReservedBytes`, `cudfGpuTaskManagerMemoryLimitBytes`, `cudfGpuDeviceMemoryFreeBytes`/`UsedBytes` (`-1` if telemetry fails). Per-operator CUDA allocation attribution does not exist yet.
- **Dashboards.** Prometheus rules and a Grafana board are a static candidate: not deployed, not run on a GPU. Native/JNI error counts are an open item.
- **Tools.** `qualification` ranks jobs `high` (GPU-candidate share of compute at least 60%), `medium` (25 to 60%), `low`, `unknown`, and flags risky shapes. `profiling` reports busy/backpressure, skew, `pending_work_is_growing`, `checkpoint_health` and CPU-fallback markers. Read-only HTTP GET, no GPU, no plugin. They see plan shape and Flink metrics, not kernel time.

## 7. The library and Mega-Flink: what came along, what did not

The operator library in [`../library`](../library) (design: [DESIGN](DESIGN.md), checked range: [LIMITS](LIMITS.md)) carries the operators of sections 2 to 3 for window aggregation and the bounded lookup join. Sections 1 and 4 to 6 describe the planner side of Mega-Flink and do not apply to it as written.

| Topic | In the library |
|---|---|
| Operators, batching, barrier and watermark flush | **Came along.** The window operators (one phase, local and global) and the lookup join operator are the Mega-Flink classes, with the planner-fusion branches removed. Batch size and wait time are builder options (`batchRows`, `maxWaitMs`; defaults 4096 rows and 5 ms for windows, 1,048,576 rows for the lookup join). A run through a JVM was checked against CPU output only for four cases (LIMITS) |
| State | **Came along.** Flink state stays the ledger, with the same recovery model. Recovery, savepoint and restore were not tested on the library |
| Admission | **Now a build-time check.** The planner's proof became shape checks in `GpuWindows.build()` and `GpuJoins.build()`, taken from Mega-Flink's own admission code. A failed check throws `GpuUnsupportedShapeException`; there is no fallback in the middle of a run. Append-only input cannot be proven at build time, so the operators throw on the first non-INSERT row |
| Two-phase aggregation | **Came along as a switch.** `.twoPhase()` builds the local combiner and the global merge; the caller does not write them. It is experimental in Mega-Flink |
| Plan classes, key selector, snapshot provider | **Redone.** Calcite types are replaced by plain serializable classes (`AggregateCallSpec`, `FieldRowDataKeySelector`, `LookupDimension`) |
| Device acquisition | **Redone.** A builder option, `gpuResource`; the default is the Flink external resource `gpu` |
| System properties | **Not redone.** Cache sizes and the native library path are still read as `cudf.flink.*` system properties |
| Installer, planner patch, `mode=cpu-row` master switch, `latency-safe` and experimental switches | **Not carried.** There is no planner override; the job calls the library on purpose |
| Adaptive controller, persistent paths, deep fusions, Calc interpreter | **Not carried.** The library runs in batch mode only |
| Shadow comparison | **Not carried.** Nothing compares the GPU result with a CPU result while a job runs. The comparison is done offline: the example jobs run on CPU (Flink SQL) or GPU (library) and print the SHA-256 of the final table |
| State names | Follow Mega-Flink, so they are not compatible with Flink-native operator savepoints. Moving a CPU SQL job to the library means a fresh start |
| Metrics, dashboards | Operator metrics code (for example the late-row counter `cudfWindowLateRowsDropped`) came with the operators; whether and how it appears in a library job was not checked. Prometheus rules and the Grafana board are not carried |
| Table bridge | **Not started.** Not in the first version; conversion cost unverified |
