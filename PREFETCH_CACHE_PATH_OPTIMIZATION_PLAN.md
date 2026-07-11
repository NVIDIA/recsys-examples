# Fused Sparse Cache–Storage Prefetch: Production Baseline

## Status

The fused cache prefetch path is the production baseline on
`perf/fused-sparse-cache-prefetch`, consolidated to a single production path
with no experimental variants and no legacy fallback.

- Cache lookup/insertion, storage metadata processing, bidirectional value
  exchange, admission, initialization, and rejected-slot reclamation are
  integrated.
- The value path is the symmetric compact exchange kernel with a depth-3
  pipeline, 24 sparse rows per warp, eight warps per block, and rolled sparse
  bookkeeping loops.
- The exchange launch grid is capped at ~SM/4 blocks. The exchange is
  latency-bound on the sparse mapped-host (PCIe) access path, not SM-bound, so a
  small grid saturates the request path while freeing the remaining SMs for the
  forward/backward compute the prefetch overlaps in the training pipeline. EOS
  measured this no slower -- slightly faster -- than an all-SM launch (38.8 vs
  37.1 GB/s at the 132-block peak; at or above the all-SM baseline down to 16
  SMs), because over-subscribing all SMs adds DRAM bank/row contention.
- All experimental variants explored during development are REMOVED:
  split-overlap/HBM-snapshot directional kernels, independently sorted
  worklists, Green-Context writer-SM partitioning, host-staged CPU
  gather/scatter (and its TMA specialization), and the VMM host-NUMA large-page
  probe. Verified 2 MiB GMMU pages did not reduce the sparse-address penalty
  (see the VMM control below), and every scheduling/staging variant regressed
  against the fused kernel, so none are retained.
- The sections below are the historical investigation record. Some describe the
  removed variants and the env flags that gated them
  (`DEMB_EXCHANGE_SPLIT_OVERLAP`, `DEMB_EXCHANGE_HOST_STAGED`, ...); those flags
  and code paths no longer exist.

## Runtime Flow

`_prefetch_cache_path` performs:

1. Fused cache `find_or_insert` over the full input layout.
2. Backing-storage lookup and insertion metadata processing.
3. One symmetric CUDA value-exchange kernel for storage-to-cache and
   cache-to-storage traffic on the default path.
4. Admission over the full tensors using `founds` as the sparse mask.
5. Direct initialization of admitted rows.
6. Reclamation of rejected provisional cache slots by known slot.

There is no legacy-prefetch fallback for deterministic mode or external
storage. Deterministic mode uses stable waves within the same fused API.

With the split experiment enabled, step 3 instead performs an input-aligned HBM
snapshot of eviction victims, then launches independent cache-to-storage and
storage-to-cache kernels in separate streams. The inbound reader may overwrite
cache rows only after the snapshot completes. Both directional streams are
event-joined back to the caller stream before dependent work may run.

## API Contract

- `Cache.find_or_insert(...)` returns input-aligned cache slots, cache-hit
  flags, and eviction metadata.
- `Cache.decrement_counter(...)` releases aligned cache references.
- Every caching `Storage` implementation provides `exchange(...)` and
  `release_cache_exchange_refs(...)`.
- `Storage.prepare_exchange(...)` preflights optional exchange resources before
  cache insertion mutates metadata or acquires references.
- `CacheExchangeRequest` carries the typed `DynamicEmbTableState`; opaque
  cache-state payloads are not supported.
- Admission and initialization APIs consume full-layout boolean masks.
- Built-in mapped storage, deterministic mode, and the reference external
  storage implementation follow the same semantic contract.

Configuration fails immediately when a caching backend does not implement the
required exchange and reference-release methods.

Split mode owns a persistent, input-aligned
`DynamicEmbTableState.exchange_evicted_values` HBM workspace. It grows
geometrically, is serialized before reuse or resize, and is registered on each
caller stream so the caching allocator observes the final Green-stream event
joins. It is distinct from a forward/backward embedding staging buffer. Split
mode rejects CUDA graph capture. The extension exchange ABI receives this
workspace and its maximum row width even when the default fused path is
selected.

## Correctness Invariants

- Cache hits acquire a reference before publication or score replacement.
- Lookup protection completes before misses may evict cache rows.
- A backing lookup row remains acquired until its storage-to-cache read
  completes.
- A backing insertion destination remains acquired until the mapped-host write
  is system-visible.
- An evicted cache row is captured before the paired incoming value overwrites
  that physical cache row.
- In split mode, every active eviction victim is copied to HBM before either
  the mapped-storage writer consumes it or the inbound reader may overwrite its
  cache row.
- Backing source and destination rows for one exchange are disjoint.
- Failed backing insertion is rolled back on device before value exchange.
- Cache-provisioning failure uses aligned direct-storage metadata rather than a
  temporary embedding buffer.
- Backing expansion is deferred while retained storage references are live.
- Rejected provisional cache rows become reclaimable tombstones and their
  aligned slot is set to `-1` before forward/backward.
- Sparse masks remain device-resident; the hot path does not use `.item()` or
  Python `any()` checks.
- Split epochs serialize snapshot-workspace reuse, use explicit cross-context
  events, join both directional kernels to the caller stream, and retain the
  system-scope fence before releasing a backing insertion reference.

## Default Fused Exchange Kernel

The production kernel uses:

- eight warps per block;
- 24 input rows per warp;
- fixed 512-byte value tiles;
- a depth-3 shared-memory ring for both directions;
- per-warp compaction of active rows without changing API layout;
- flattened row/tile work so arbitrary physical row widths do not require
  row-sized shared-memory allocations;
- classic `cp.async` on SM80+ and the intrinsic's synchronous fallback on
  SM75;
- a per-lane system fence before insertion-reference release.

The fixed shared-memory footprint is checked at compile time against the SM75
48-KiB limit. Device properties used to size the launch grid are captured when
the table state is constructed, not queried on the hot path.

## Experimental Snapshot/Green-Context Split

The opt-in path uses three kernels:

1. A cache-to-HBM snapshot over active eviction victims.
2. A depth-4 storage-to-cache reader with 32 directional rows per warp.
3. An HBM-snapshot-to-storage writer with 32 directional rows per warp.

An additional `DEMB_EXCHANGE_SORTED_WORKLISTS=1` experiment builds independent
reader-source and writer-destination address orders. In that mode,
`DEMB_EXCHANGE_SORTED_ROWS_PER_WARP` selects 8, 12, 16, 24, or 32; K=8 is the
measured optimum for the production replay. The unsorted split remains on its
original 32-row kernels.

`DEMB_EXCHANGE_WRITER_SMS` requests the writer partition and defaults to 16.
A positive value constructs two explicit Green Contexts from one exact SM
resource split; zero uses an ordinary nonblocking writer stream. On the
132-SM EOS H100, the best tested split was 16 writer SMs and 116 reader SMs.
`DEMB_EXCHANGE_SPLIT_STRICT=1` turns Green Context setup failure into an error;
otherwise the implementation falls back to the fused path. Green Contexts
isolate physical SM assignment, but they do not partition L2, PCIe, HostVMM
translation, the IOMMU, or root-complex arbitration.

## Validation

Required production validation is:

1. Build all extension objects for SM75, SM80, SM90, and SM100.
2. Run the existing batched dynamic-embedding table suite, including native
   and external storage, deterministic mode, mixed dimensions, SGD/Adam,
   admission, overflow, flush, and forward/backward coverage.
3. Run the existing table-operation, admission, and LFU score suites.
4. Run the standalone production-volume exchange test:

   ```bash
   cd corelib/dynamicemb
   python benchmark/benchmark_exchange_cache_storage_values.py
   python benchmark/benchmark_exchange_cache_storage_values.py \
     --split-overlap --writer-sms 16 --split-strict
   python benchmark/benchmark_exchange_cache_storage_values.py \
     --split-overlap --writer-sms 0 --split-strict \
     --sorted-worklists --sorted-rows-per-warp 8
   ```

   The test verifies both copy directions, source preservation, direct-row
   metadata normalization, and reference release before reporting logical
   HostVMM-aperture GB/s. Split mode additionally verifies the HBM snapshot
   signatures and reports its workspace, exact SM resources, and launch grids.
   It does not claim physical PCIe wire bandwidth.
5. Run `benchmark_batched_dynamicemb_tables.sh -k TestCaching` in fresh
   processes and compare five-run medians.
6. Confirm the default steady-state built-in path contains fused cache insertion, at
   most two storage metadata kernels, one value-exchange kernel, no
   prefetch-side compaction kernel, and no temporary embedding buffers.

EOS job 5581630 rebuilt the split-capable extension for SM75, SM80, SM90, and
SM100, imported the resulting 140,388,608-byte extension from its isolated
build directory, and passed the strict 16/116-SM snapshot, bidirectional copy,
metadata-normalization, and reference-release checks.

## Benchmark Result

The production comparison was run on EOS with one NVIDIA H100 80GB HBM3,
driver 535.129.03, and
`gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest`.
The pinned revisions were `origin/main` at `5dc46a217d59` and the consolidated
runtime at `fffda9fae6ca`. The unchanged TestCaching harness ran in five fresh
processes per revision with alternating order; all 40 configuration executions
passed.

Final warmed `dyn_forward_ms` medians:

| Configuration | `origin/main` | Production baseline | Change |
|---|---:|---:|---:|
| Adam, `cfr=0.8` | 22.833490 ms | 21.325342 ms | 6.60% faster |
| Adam, `cfr=1.0` | 10.860436 ms | 10.460384 ms | 3.68% faster |
| SGD, `cfr=0.8` | 19.429918 ms | 19.715805 ms | 1.47% slower |
| SGD, `cfr=1.0` | 9.650535 ms | 9.731634 ms | 0.84% slower |

TorchRec forward controls moved by at most 0.42%, so the DynamicEmb deltas are
not explained by whole-node timing drift. The first reporting-iteration Adam
`cfr=0.8` diagnostic has a five-run median of 45.148 ms on `origin/main` and
31.644 ms on the production baseline (29.91% faster), but first-launch samples
are noisy and are not the warmed JSON benchmark metric.

Direct job-5575102 telemetry from the first timed Adam `cfr=0.8` train
iteration measured a 434,397-row kernel input with 162,725 inbound rows,
162,725 outbound rows, and complete overlap. The standalone test reproduces
the exact ten-table counts with a 499,891,200-byte logical payload, the real
24.112-GB cache-value allocation, and the real 412.318-GB HostVMM-value
allocation. Five fresh processes produced a median kernel time of 13.719200 ms
and 36.437344 GB/s logical duplex bandwidth. This is logical mapped-aperture
payload, not physical PCIe wire bandwidth. `origin/main` has no fused exchange
binding, so there is no direct standalone-kernel A/B comparison.

### Snapshot/Green-Context A/B

The current production-shape benchmark uses 10 warmups and 100 measured
iterations with 434,397 input rows, 162,725 rows in each direction, complete
directional overlap, a 499,891,200-byte logical payload, a 24.112-GB cache
allocation, and a 412.318-GB HostVMM allocation. Valid EOS results are:

| Variant | Median | Logical duplex | Latency vs fused |
|---|---:|---:|---:|
| Fused default | 13.503136 ms | 37.020378 GB/s | baseline |
| Split, ordinary writer stream | 15.099168 ms | 33.107201 GB/s | 11.820% slower |
| Green writer 8 SMs | 14.853872 ms | 33.653933 GB/s | 10.003% slower |
| Green writer 16 SMs | 14.772992 ms | 33.838183 GB/s | 9.404% slower |
| Green writer 16 SMs, B=2 load ablation | 14.713600 ms | 33.974771 GB/s | 8.964% slower |
| Green writer 20 SMs | 14.949664 ms | 33.438290 GB/s | 10.713% slower |
| Green writer 24 SMs | 14.926480 ms | 33.490227 GB/s | 10.541% slower |
| Green writer 32 SMs | 14.885264 ms | 33.582957 GB/s | 10.236% slower |
| Green writer 96 SMs | 15.798896 ms | 31.640894 GB/s | 17.002% slower |

The best retained split is therefore 16 writer SMs and 116 reader SMs, but it
is still 9.404% slower than the fused default. Its persistent HBM snapshot
workspace is 667,233,792 bytes. The B=2 writer-load ablation was only 0.402%
faster by cross-node median and 0.078% faster by mean, below the 1% retention
threshold, so it was reverted. The SM90 FP32 cubin did emit the intended
`LD.E.128`, `LD.E.128`, `ST.E.128`, `ST.E.128` order and every architecture and
dtype remained spill-free at 39–44 registers, so the rejection is a measured
performance decision rather than a failed code-generation experiment.
Increasing the reader pipeline from depth 4 to depth 6 produced 14.823904 ms
and 33.721967 GB/s, 0.3446% slower than depth 4, so depth 6 was also rejected.
Writer partitions of 48, 64, and 112 SMs that failed during the 412-GB HostVMM
setup with `SIGBUS` produced no timing and are not included.

A contiguous synthetic duplex control reached 9.523872 ms and 78.918 GB/s
without Green partitioning. Exact Green splits reached 56.281 GB/s with a
16-SM writer, 57.165 GB/s with 20, and 58.246 GB/s with 24; independent
single-direction controls were approximately 51 GB/s read and 52 GB/s write.
An SM-ID probe found zero physical-SM intersection between the reader and
writer Green Contexts. The control therefore confirms that the real sparse
HostVMM address pattern, not the nominal link rate alone, sets the workload
roofline, and that fixed SM isolation reduces useful dynamic scheduling even
when it successfully prevents SM co-scheduling.

### One-pass NCU PC sampling

EOS job 5581679 used one NCU invocation. Every split kernel reports
`profiler__replayer_passes=1`; `smsp__pcsamp_aggregated_passes=2` is the PC
sampler's internal aggregation count, not a second application replay, and the
report recorded zero overflow and zero dropped bytes.

| Kernel | PC samples | Short scoreboard | Long scoreboard |
|---|---:|---:|---:|
| HBM snapshot | 21,999 | 0.4000% | 64.7984% |
| HostVMM writer | 95,553 | 13.1749% | 70.1265% |
| HostVMM reader | 804,382 | 24.0264% | 10.7909% |

Source/SASS attribution identifies the dependencies rather than merely naming
the stall class:

- Reader short stalls: `ST.E.128` immediately after `LDS.128` contributes
  34.71%; producer-address `LEA` contributes 23.96%; and `SHFL.IDX`
  contributes 21.25%. The top five PCs contribute 90.04%.
- Writer short stalls: the row-active predicate immediately after its shuffle
  contributes 71.95%, and the predicate after source/destination/dimension
  shuffles contributes 26.60%. Together they contribute 98.55%; the mapped
  store is not a writer short-scoreboard source.
- Snapshot short stalls: the active predicate consuming its shuffle contributes
  95.45%.
- Three snapshot `STG.E.128` consumers contribute 78.56% of snapshot long
  stalls. Three writer `ST.E.128` consumers immediately following HBM loads
  contribute 44,621 of 67,008 raw writer long samples, or 66.59%.
- The reader PC immediately before `DEPBAR.LE SB0,2` contributes 53.01% of
  reader long stalls. NCU charges the asynchronous completion wait to that
  preceding PC; the explicit wait itself has only nine samples.

Consequently, the earlier aggregate “short scoreboard: 47.31%” is not a
source-level diagnosis of this implementation. Per-kernel PC sampling shows
that the reader's short stalls are a mixture of the shared-load/store chain and
row-address production, while the writer's dominant sampled dependency is the
HBM-load-to-mapped-store chain in the long-scoreboard class.

### Nsight Systems overlap proof

EOS job 5581695 captured the timed split invocation in two distinct streams and
two explicit Green Contexts with one parent context:

| Kernel | Duration | Stream | Green Context | Assigned SMs |
|---|---:|---:|---:|---:|
| HBM snapshot | 0.222078 ms | 18 | 3 | 116 |
| Storage-to-cache reader | 8.684611 ms | 18 | 3 | 116 |
| Snapshot-to-storage writer | 14.680792 ms | 17 | 2 | 16 |

The reader and writer start only 1.664 microseconds apart and overlap for
8.682947 ms: 99.981% of the reader and 59.145% of the writer. The complete GPU
span from snapshot start through writer completion is 14.906486 ms, and the
profiled benchmark reports 14.954368 ms. Thus, the split regression is not a
failure to launch concurrently. The writer remains active for 5.997845 ms
after the reader finishes; its payload-only directional rate is 17.025 GB/s,
versus 28.780 GB/s for the reader. Giving the writer 20–32 SMs did not improve
the 100-iteration result, which points to poor scaling in the sparse
mapped-write and shared system-memory path rather than merely an insufficient
writer grid.

### Root-cause isolation: sparse HostVMM address coverage

The exchange benchmark now accepts an exact `--storage-row-stride` and a
`--storage-access-order` of `randomized` or `address_sorted`. The production
default remains stride 65,537 with randomized assignment. Every point below
uses the same 412.318-GB registered HostVMM allocation, 434,397 input rows,
499,891,200-byte logical payload, launch geometry, and correctness checks; only
the active storage-row addresses change.

The EOS randomized sweep used 10 warmups and 100 measured iterations:

| Row stride | Byte stride | Selected 4-KiB pages | Mean active span/table | Median | Logical duplex |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.5 KiB | 122,054 | 0.050 GB | 8.733088 ms | 57.241058 GB/s |
| 3 | 4.5 KiB | 366,132 | 0.150 GB | 8.695888 ms | 57.485932 GB/s |
| 11 | 16.5 KiB | 406,812 | 0.550 GB | 8.721776 ms | 57.315299 GB/s |
| 43 | 64.5 KiB | 406,812 | 2.149 GB | 12.207152 ms | 40.950681 GB/s |
| 683 | 1.0005 MiB | 406,812 | 34.142 GB | 13.385184 ms | 37.346607 GB/s |
| 65,537 | 96.0015 MiB | 406,814 | 41.230 GB | 13.601840 ms | 36.751734 GB/s |

The stride-11 and stride-43 points are the main discriminator: they touch the
same number of OS pages and transfer the same rows in randomized execution
order, but widening the active address coverage by 3.91x increases latency by
39.96%. Page count alone is therefore not the limiting variable. Sorting the
production address set recovers only 5.49% (12.894208 ms and 38.768663 GB/s),
because it improves local assignment but does not impose one globally ordered
CUDA-block request stream.

Nsight Systems split traces provide a directional control. Moving from the
production stride to stride 1 reduces the reader from 8.745982 to 5.212493 ms
(1.678x faster) and the writer from 14.719600 to 8.843658 ms (1.664x faster),
while the HBM snapshot remains approximately 0.22 ms. Both mapped directions
therefore suffer the same address-coverage penalty; it is not specific to PCIe
read completions, mapped-write fencing, or the cache-row alias dependency.

One-application-replay-pass NCU endpoint counters show that the compact and
sparse cases perform the same explicit memory work:

| NCU metric | Stride 1 | Stride 65,537 | Change |
|---|---:|---:|---:|
| Kernel duration | 8.713216 ms | 13.519744 ms | +55.16% |
| L2 sysmem read requests | 3,905,400 | 3,905,400 | 0% |
| L2 sysmem read sectors | 7,810,800 | 7,810,800 | 0% |
| L2 sysmem write sectors | 7,828,482 | 7,828,292 | -0.0024% |
| L2 sysmem read-miss sectors | 7,810,800 | 7,810,800 | 0% |
| `pcie__read_bytes.sum` | 296.840704 MB | 296.893952 MB | +0.0179% |
| `pcie__write_bytes.sum` | 344.051712 MB | 344.101888 MB | +0.0146% |
| HBM read bytes | 272.051456 MB | 332.549376 MB | +22.24% |
| HBM write bytes | 254.921472 MB | 267.053312 MB | +4.76% |

The L2 request count, average sectors/request, L2 read-miss payload, and PCIe
data volume are invariant. Sparse addressing does not create smaller
transactions, additional useful traffic, PCIe-data amplification, or a new L2
data-cache miss-rate problem. The same counter-labelled 640.9 MB of PCIe data
simply takes longer to service. These PCIe counters are device-level data
counters, not complete protocol-wire byte counts.

At the equal-page-count stride-11/43 knee, L1TEX device-read-miss sectors also
remain invariant (8,430,144 versus 8,429,984), while HBM reads increase from
303.207424 to 328.414720 MB and HBM writes increase from 265.210880 to
267.225088 MB. Thus approximately 25.21 MB of additional HBM reads appear
outside the measured L1TEX demand-read path at the exact latency knee. Across
stride 1 to 65,537, the HBM-read residual after subtracting L1TEX device-miss
sectors grows from 2.334 to 62.736 MB. This is strong circumstantial evidence
for GPU-internal address-dependent metadata or request-generation traffic; the
TEX-qualified subtraction is not itself a direct page-walk counter.

A second one-pass pack measured aggregate device-aperture L2 read misses in the
same invocation as the TEX-qualified misses. Non-TEX read-miss traffic grows
from 136.044 MB at stride 1 to 153.089 MB at stride 11, 177.443 MB at stride
43, and 182.103 MB at stride 65,537. At the stride-11/43 knee, TEX traffic
changes by only 6 KiB, while the non-TEX L2 client adds 24.354 MB and total HBM
reads add 26.181 MB. The non-TEX client therefore explains 93.0% of the added
HBM-read traffic at the performance knee. This narrows the signal to an
address-dependent GPU-internal L2 client rather than the exchange kernel's SM
demand loads. Page-table walking/GMMU request generation is the leading source
given the span sensitivity, but the exposed aggregate metric does not name the
private client directly. GH100 exposes only GCC, L1-class, and LTC-fabric
non-TEX source selectors for this event, not an MMU/page-walker selector; fully
decomposing the knee would require six additional one-pass production runs and
would still not prove a page walk, so those runs were not taken.

A separate one-pass stall pair recorded a 55.22% duration increase, an 11.65%
increase in accumulated long-scoreboard stalled warps, and a 109.06% increase
in `lg_throttle` stalled warps. Queue/issue backpressure therefore participates
in the sparse ceiling, but the invariant transaction shape shows that it is a
consequence of longer address-dependent service rather than an independent
payload-bandwidth cause.

### Independently sorted directional worklists

The snapshot/split experiment now has an additional opt-in control behind
`DEMB_EXCHANGE_SORTED_WORKLISTS`. One CUDA key builder creates independent
64-bit HostVMM-address keys for the reader source and writer destination, and
two CUB radix sorts pair those keys with original input IDs. The input IDs,
rather than gathered metadata, are consumed by the directional kernels. This
keeps the HBM snapshot, output metadata, and reference-counter ownership in the
original input layout while permitting different reader and writer orders.
The production fused path is unchanged.

The first implementation used 32 sorted rows per warp. The following values
are medians of three independent 10-warmup, 100-iteration EOS run medians; all
split cases in this table use ordinary nonblocking streams (`writer_sms=0`):

| Variant | Median | Logical duplex | Latency vs fused |
|---|---:|---:|---:|
| Fused, randomized | 13.601824 ms | 36.751776 GB/s | baseline |
| Split, randomized | 15.362144 ms | 32.540458 GB/s | 12.942% slower |
| Split, rows preordered outside timing | 14.926192 ms | 33.490873 GB/s | 9.737% slower |
| Split, runtime independent sorts, 32 rows/warp | 16.304480 ms | 30.659746 GB/s | 19.870% slower |

The free-preordering control demonstrates a real but modest locality benefit.
It preserves the randomized active-position mask and excludes ordering cost.
The initial runtime sort instead packs 162,725 valid rows into 5,086 dense warp
chunks, compared with approximately 13,575 active chunks at about 12 rows per
warp in the original schedule. It therefore changes memory-level parallelism,
metadata access, and stream arbitration in addition to changing HostVMM order.

One-pass Nsight Systems controls isolate those effects. For the 32-row runtime
sort, key generation took 0.009 ms, both radix sorts spanned 0.230 ms, and the
snapshot took 0.214 ms. The timed reader improved from 11.872 ms in the
unsorted ordinary-stream control to 10.673 ms, proving that address order helps
the PCIe-read direction. The writer, however, moved from 9.092 to 11.353 ms,
and ordinary-stream scheduling delayed the reader by 4.859 ms. The resulting
profiled total was 16.071 ms. Thus radix cost alone was not the regression;
dense per-warp serialization and unstable ordinary-stream residency were the
larger costs.

The sorted directional kernels therefore accept an experimental
`DEMB_EXCHANGE_SORTED_ROWS_PER_WARP` of 8, 12, 16, 24, or 32. Lanes above the
selected count remain copy helpers but do not own another row. A one-run EOS
sweep gave:

| Sorted rows/warp | Median | Logical duplex |
|---:|---:|---:|
| 8 | 13.746144 ms | 36.365921 GB/s |
| 12 | 14.164528 ms | 35.291765 GB/s |
| 16 | 15.015632 ms | 33.291386 GB/s |
| 24 | 15.493904 ms | 32.263734 GB/s |
| 32 | 15.942688 ms | 31.355515 GB/s |

Three independent K=8 runs were 13.746144, 13.750416, and 13.801872 ms. Their
median is 13.750416 ms and 36.354624 GB/s: 10.519% faster than the unsorted
split and only 1.092% slower than fused. In the K=8 one-pass trace the reader
and writer were balanced at 8.164076 and 8.145356 ms, respectively. Ordinary
streams still delayed the reader by 5.572198 ms and overlapped the directions
for only 2.573158 ms, so the directional span was 13.736274 ms despite both
kernels individually becoming much shorter.

Two follow-up controls were rejected:

- Writing the snapshot in sorted outbound-work order produced 14.247424 ms at
  K=12 versus 14.164528 ms input aligned, and 13.780448 ms at K=8 versus the
  13.750416-ms input-aligned median. The 0.218% K=8 regression shows that random
  HBM snapshot-source order is not the remaining writer bottleneck. The code
  was reverted.
- Exact Green Context points with K=8 measured 13.877968 ms at 16 writer SMs,
  13.738848 ms at 20, and 13.805680 ms at 24. The best point is only 0.084%
  faster than ordinary K=8 and remains 1.007% slower than fused. Fixed SM
  overlap loses enough per-direction capacity, while L2, GMMU, and PCIe remain
  shared, to offset its scheduling benefit. Wider tuning was stopped because
  the measured gain was immaterial.

Jobs 5583149 and 5583183 built the experiment for SM75, SM80, SM90, and SM100
and passed production-volume value, snapshot, metadata-normalization, and
reference-counter checks. Jobs 5583154--5583165 provide the three-run baseline
matrix; 5583198--5583202 provide the K sweep; 5583237 and 5583238 complete the
K=8 repetitions; and one-pass traces are in jobs 5583171, 5583180, and
5583244. Runtime sort tensors are currently allocated inside the opt-in
exchange path. Shipping this as a production default would first require
persistent workspace allocation before cache/hash mutation so an allocation
failure cannot strand acquired storage references.

The result validates the locality hypothesis but does not justify replacing
the fused kernel: independent sorting plus sufficient warp-level MLP nearly
closes the sparse-address penalty, yet sort/snapshot/control overhead leaves it
about 1% behind. The next higher-upside work remains physical value packing or
coarse region binning; more fixed-SM tuning is not warranted.

### Sorted snapshot plus one dual-queue kernel

The proposed single-kernel follow-up was implemented and measured as a
temporary opt-in control. It retained the input-aligned HBM eviction snapshot
and the two independent CUB address sorts, then replaced the two directional
kernels with one persistent grid. Each 256-thread block began with four reader
warps and four writer warps. Warp-private padded queue heads handed out four
adjacent K=8 chunks per atomic claim, and a warp could switch to the other
direction after its preferred full worklist had been claimed. The kernel ran
on the caller stream without a Green Context or fixed SM partition.

EOS job 5583348 built the control for SM75, SM80, SM90, and SM100 and passed
the production-volume value, HBM-snapshot, metadata-normalization, and
reference-counter checks. On H100, the float32 K=8 specialization used 48
registers/thread and 18,176 bytes of shared memory per block, with zero stack
or local spill storage. Registers permit approximately five 256-thread blocks
per SM, so lack of resident warps was not an immediate invalidating artifact.

Matched exclusive-H100 points used 10 warmups, 100 measured iterations,
randomized stride 65,537, and NUMA node 0:

| Variant | EOS job | Median | Logical duplex | Latency vs fused |
|---|---:|---:|---:|---:|
| Fused | 5583349 | 13.662816 ms | 36.587714 GB/s | baseline |
| Independently sorted two-kernel K=8 | 5583350 | 13.805680 ms | 36.209097 GB/s | 1.046% slower |
| Independently sorted dual-queue K=8 | 5583351 | 15.383440 ms | 32.495411 GB/s | 12.594% slower |

The dual queue is 11.428% slower than the already marginal two-kernel sorted
path. This rejects the premise that forcing the balanced directional work into
one CTA schedule will convert the separate kernels' similar durations into
useful full overlap. Reader and writer warps still share the GMMU/L2/system
request path and PCIe/root-complex resources, so forced concurrency increases
arbitration while each direction starts with only half the issue slots. Every
warp also pays the combined kernel's reader-path register/shared allocation
and queue-control instructions, although the resource dump proves that spills
are not the cause. Because both replay directions contain the same active-row
count and both queues traverse the complete permutation (including the
invalid-key suffix needed for metadata normalization), work stealing provides
almost no active-work rebalancing in this benchmark.

The loss was large and stable: the build-validation replay independently
measured 15.395744 ms over three iterations. No SM-count sweep or additional
multi-pass profile was run. The temporary dual-queue kernel, environment flag,
and benchmark switch were reverted; the validated EOS binary and job outputs
remain the experiment artifact. The independently sorted two-kernel K=8 path
is preserved.

### EOS system-path controls

GPU 0, the benchmark CPUs, and the HostVMM allocation are all on NUMA node 0.
Every hop from the H100 through the PCIe switches to the CPU root port is
negotiated at PCIe Gen5 x16, and the GPU reports zero replay errors. The kernel
boots with `iommu=pt`; there are no IOMMU groups and the GPU has no
`iommu_group`. A translated host-IOMMU path is therefore not active.

Live `smaps` inspection found ten fully locked 40,265,472-KiB HostVMM mappings.
The ordinary allocator uses 4-KiB pages and had no anonymous huge pages. An
experimental preload added `MADV_HUGEPAGE` before first touch; after
`cudaHostRegister`, almost every byte of every mapping was backed by THP
(`AnonHugePages` 40,261,632--40,263,680 KiB per mapping). Performance was
unchanged within 0.006%: 13.602624 ms and 36.749616 GB/s versus the ordinary
13.601840 ms and 36.751734 GB/s. This rules out Linux 4-KiB physical
fragmentation and CPU page-table overhead as material causes. It does not prove
that CUDA built 2-MiB GPU mappings: `smaps` describes the CPU VMA, and
`cudaHostRegister` may still construct base-page GMMU mappings.

Uncore counters provide the bandwidth controls:

| Case | Logical duplex | Host IMC read | Host IMC write | Corrected PCIe inbound |
|---|---:|---:|---:|---:|
| Stride 1 | 57.053 GB/s | 28.67--28.72 GB/s | 28.63--28.67 GB/s | about 28.59 GB/s |
| Stride 65,537 | 36.703 GB/s | 18.51--18.57 GB/s | 18.48--18.52 GB/s | about 18.40 GB/s |

Host-DRAM traffic tracks useful payload almost exactly, with no meaningful
read or write amplification. Aggregate traffic is approximately 57 or 37
GB/s, only 18.7% or 12.0% of one socket's theoretical 307.2-GB/s DDR5-4800
bandwidth. The corrected PCIe inbound event agrees with the logical and IMC
rates. Host-DRAM bandwidth, the Gen5 x16 link bandwidth, NUMA placement,
PCIe replay, and host-IOMMU translation are not the primary sparse limiter.
Random-access host-DRAM latency cannot be excluded completely, but low IMC
utilization, one-to-one traffic, and the no-op THP control make it a secondary
possibility.

The best-supported diagnosis is therefore an address-coverage-sensitive
GPU-side mapping/translation and request-generation limit, with finite
L1TEX/L2 system queues and PCIe completion capacity determining how much of its
latency can overlap. NCU exposes no named H100 TLB/page-walk counter, so the
exact private GMMU cache or page-table level cannot be proven from public
counters. PCIe credits and L2 queues are real secondary mechanisms, but they
cannot independently explain why identical request sizes and counts slow down
only when virtual coverage expands.

This also separates two rooflines. Packing the same active rows into compact
coverage raises logical duplex bandwidth from about 36.8 to 57.3 GB/s; that
approximately 20.5-GB/s gap is the sparse-address penalty. The remaining gap
between the compact control and nominal PCIe peak is the next zero-copy mapped
access ceiling: protocol overhead, finite tags/credits, request latency, and
kernel MLP. Logical duplex payload is not physical PCIe wire utilization.

The next optimization should target the value layout rather than another SM
partition:

1. Decouple backing-value row placement from the hash slot and allocate active
   values in page-/region-local extents.
2. Bin exchange work by HostVMM region and give CTAs region-coherent work; the
   simple address-sorted assignment is not a global scheduler.
3. Investigate a CUDA allocation/registration path with verified large GMMU
   mappings. Linux THP advice alone is not sufficient.
4. Retune outstanding-request depth and directional scheduling only after
   reducing translation coverage. Green Contexts do not partition the GMMU,
   L2, PCIe tags/credits, root complex, or host memory controllers.

### Verified 2 MiB GMMU mapping control (resolves item 3): translation refuted

Item 3 above was executed. A gated `DEMB_HOSTVMM_USE_VMM=1` path in
`src/vmm_tensor.cu` backs the host store through
`cuMemCreate(CU_MEM_LOCATION_TYPE_HOST_NUMA) + cuMemAddressReserve + cuMemMap +
cuMemSetAccess` instead of `mmap + mlock + cudaHostRegister`. Activation was
verified, not assumed: the driver reported an allocation granularity of
2,097,152 bytes (2 MiB) for each of the ten 41.2-GB tables, so the full
412-GB store is GPU-mapped at 2 MiB rather than 4 KiB. This shrinks the
GPU-side page-table/TLB working set by 512x. Correctness passed. (The VMM
branch must force CUDA runtime initialization before the driver `cuMem*` calls,
or they fail with `CUDA_ERROR_NOT_INITIALIZED`.)

The EOS A/B on one exclusive H100, NUMA node 0, same build, 10 warmups and 100
measured iterations, was:

| Storage row stride | Default 4 KiB | Verified 2 MiB VMM | Change |
|---:|---:|---:|---:|
| 1 (compact) | 57.30 GB/s | 57.26 GB/s | -0.1% |
| 65,537 (production sparse) | 37.08 GB/s | 36.85 GB/s | -0.6% |

The sparse-address penalty is -35.3% with 4 KiB pages and -35.6% with 2 MiB
pages: unchanged. A 512x larger GPU page produces no improvement. This is
decisive evidence that GPU-side address translation (TLB misses / GMMU
page walks) is **not** the dominant cause of the sparse penalty; the earlier
translation-coverage diagnosis is refuted, and the ambiguous THP null result is
now settled with the GMMU granularity confirmed. The span sensitivity is
therefore best explained by host-memory random-access latency (DRAM
row-buffer/bank locality over a wider active span, consistent with the ~18%
IMC utilization latency-bound signature) and PCIe read-completion latency on a
latency-bound zero-copy path, not translation. Large GMMU pages are not a
lever; do not pursue them further.

## Active Follow-up: Host-Staged Compact Exchange

The next experiment moves sparse HostVMM gathering and scattering to native
CPU workers so the GPU transfers only through compact mapped-host rings. Its
approved design, synchronization protocol, tuning matrix, and production gates
are recorded in `HOST_STAGED_COMPACT_EXCHANGE_PLAN.md`.

The first implementation is gated by `DEMB_EXCHANGE_HOST_STAGED`, restricted
to SM90 and built-in HostVMM storage. EOS job 5584965 rejected its row-granular
protocol at production volume. Three matched fused fresh-process medians
produced a 13.606720-ms cross-process median and 36.738553 GB/s logical duplex.
Two completed 16-worker host-staged runs produced 199.698219 and 200.104538 ms,
for a 199.901379-ms median and 2.500689 GB/s: 14.691x slower and only 6.807% of
fused throughput. A third full staged process held the GPU at 100% while CPU
workers busy-polled until a 12-minute guard terminated it. A 32-worker quick
point remained at 196.222488 ms.

The required packed-v2 follow-up is now implemented. A 32-KiB slot carries up
to 64 records with 16-byte-aligned spans; the production 1,536-byte row packs
21 records per full slot. Each CTA prepares a depth-4 window and publishes it
in two-slot prefixes, while native CPU workers gather or scatter the entire
observed prefix before one release-counter update. One fused kernel handles
outbound capture, prefix publication, inbound consumption, cache scatter, and
reference release. Rings grow only between drained epochs and remain under the
256-MiB mapped-memory cap.

EOS job 5585814 ran the exact 412.318-GB replay with 16 workers, one channel per
SM, depth 4, two-slot publications, two warmups, and five measured launches.
The matched fused median was 13.654784 ms and 36.609235 GB/s. Packed-v2 vector
was 14.172512 ms and 35.271884 GB/s: 3.792% slower, but about 14x faster than
the rejected row-granular revision and at 96.35% of fused throughput. Each
direction represented 162,725 records with 7,815 slots and 3,940 publications,
reducing slot and publication traffic by 20.82x and 41.30x respectively versus
one slot and one release per row. Eight epochs completed with exact payloads,
metadata, reference release, 1,866 ring wraps, 65 partial publications, and
balanced final counters; the historical hang did not recur.

The one-record-slot control in EOS job 5586419 retained the new depth-4,
two-slot publication protocol but disabled multi-row packing. It took
110.872803 ms and 4.508691 GB/s, with 162,725 slots, 81,391 publications, and
40,593 wraps per direction. Packed-v2 vector is 7.823x faster and uses 20.82x
fewer slots and 20.66x fewer publications. Coarser publication helps, but true
multi-row packing is the change that closes most of the gap.

Mapped-host Hopper TMA is also implemented behind `DEMB_EXCHANGE_HOST_TMA=1`.
The isolated bidirectional probe passed 24,576 size/tail/reuse cases and
402,964,480 bytes per direction. The integrated production replay in EOS job
5586419 remained correct but measured 18.025728 ms and 27.732095 GB/s: 27.188%
slower than packed vector and 32.010% slower than fused. The additional
HBM/shared staging and per-slot CTA synchronization cost more than bulk issue
saves. Vector therefore remains the experimental copy mode, TMA remains an
explicit diagnostic, and the fused kernel remains the production default. The
detailed protocol, controls, and raw-result locations are in
`HOST_STAGED_COMPACT_EXCHANGE_PLAN.md`.

## Acceptance

- Functional/API consolidation: complete.
- Requested benchmark validation: the fused, split, sorted-worklist, and
  rows-per-warp revision builds passed; all 40 E2E cases and every retained
  standalone exchange point passed bidirectional value, snapshot, metadata,
  and reference-release checks.
- Strict origin/main warmed-forward no-regression guardrail: not met for SGD.
- Original 15% warmed Adam `cfr=0.8` target: not met. The noisy first-iteration
  cold diagnostic exceeds 15%.
- The fused kernel remains the production default. The optional split path is
  retained for controlled experiments. The stable independently sorted K=8
  ordinary-stream result is 1.092% slower; a 20-SM Green sample is 1.007%
  slower and not materially better. Neither is enabled by default.
- The sorted single-kernel dual-queue control is rejected and reverted. Its
  matched 15.383440-ms result is 12.594% slower than fused and 11.428% slower
  than the sorted two-kernel control despite zero spills, so forced 4R:4W
  co-residency does not solve the shared system-path bottleneck.
- The cache-row alias dependency is removable: the HBM snapshot makes the two
  directional kernels independent after snapshot completion. Independent
  address sorting plus K=8 warp scheduling closes most of the split penalty but
  does not beat fused, so aliasing is not the dominant bandwidth limiter.
  Controlled locality and system-path measurements localize the remaining
  production penalty to address-coverage-sensitive GPU mapping/translation and
  request generation, with shared L2/PCIe queues acting as secondary latency
  multipliers. The exact private GMMU level remains unobservable with the
  exposed counters.
- Packed-v2 resolves the row-granular host-staged failure and removes its
  observed hang, but misses the promotion threshold by 3.792%. The optional
  TMA specialization is correct and slower. Keep both host-staged variants
  benchmark-only and disabled by default.
