# HSTU End-to-End Training Performance Benchmark

This document describes the HSTU end-to-end training benchmark: a set of **progressive experiments** that incrementally enable optimizations to quantify each one's contribution to training throughput (MFU).

## 1. Background

### Embedding in Large-Scale Recommendation

Production recommendation models use massive embedding tables (tens of millions to billions of rows). DynamicEmb stores these tables in host memory and serves lookups to GPU during training.

### Optimization Space

Each experiment below adds **one** optimization on top of the previous, so the speedup is cumulative:

| # | Optimization | What It Does |
|---|-------------|-------------|
| 0 | Baseline | Triton attention, DynamicEmb without caching, no shuffler, DP-only |
| 1 | **Workload-Balanced Shuffler** | Redistribute variable-length sequences across GPUs so that each GPU's total attention FLOPs are balanced. Eliminates GPU idle time caused by HSTU's O(n²) attention on skewed sequence lengths. |
| 2 | **CUTLASS Attention** | Replace Triton attention with a hand-tuned CUTLASS kernel optimized for HSTU's causal+context mask. Better register allocation and H100 utilization. |
| 3 | **DynamicEmb Caching** | Cache hot DynamicEmb rows in GPU HBM while keeping the full table in host memory. |
| 4 | **Hash-RoundRobin Sharding** | Use row-wise `hash_roundrobin` placement for DynamicEmb tables to distribute IDs more evenly across ranks. |
| 5 | **Prefetch Pipeline** | Enable the prefetch pipeline on top of caching and Hash-RoundRobin sharding. |
| 6 | **Hierarchical Dist** | Move the DynamicEmb tables' keys and rows with DynamicEmb's hierarchical dist instead of TorchRec's dists. Inside a node they move over NVLink; between nodes, one copy of each (table, key) per node moves over InfiniBand. Needs DynamicEmb built against NCCL 2.30.7 or newer. |

### Benchmark Configuration

**Hardware**: H100-SXM5-80GB (single-node 8 GPU or multi-node)

**Model hyperparameters** (fixed across all experiments):

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Num HSTU layers | 8 |
| Num attention heads | 4 |
| Head dimension (kv_channels / dim_per_head) | 256 |
| Item embedding dim | 128 |
| Contextual embedding dim | 128 |
| Prediction head | [512, 8] × 8 tasks |
| Optimizer | Adam (lr=1e-3) |

**Embedding tables**:

| Table | Rows | Dim | Type |
|-------|------|-----|------|
| item | 50M | 128 | DynamicEmb |
| action | 100 | 128 | Static (DP sharded) |
| user_id | 50M | 128 | DynamicEmb |
| user_age | 100 | 128 | Static (DP sharded) |
| item_category_l1 | 50 | 128 | Static (DP sharded) |

`experiments.txt` omits the contextual features, so its runs have only `item` and `action`. `--include-contextual` adds the other three tables.

**Data distribution**:

| Parameter | Value |
|-----------|-------|
| Batch size per GPU | 32 |
| Max sequence length | 4096 |
| Sequence length distribution | Zipf (α=1.2), jagged |
| Key value distribution | Zipf (α=1.05) for `item` and `user_id` IDs |
| Training iterations | 1000 |
| Generated batches | 20, used in turn |
| Log interval | 20 |
| Eval interval | 1001 (disabled for 1000-iteration benchmark runs) |
| Profiling window | iterations 150-170 |

Synthetic data with Zipf-distributed sequence lengths simulates the heavy-tailed user-history patterns seen in production.

---

## 2. Results

**Hardware**: 2× H100-SXM5-80GB nodes (16 GPUs total) of Eos, with 8 InfiniBand NICs of 400 Gb/s per node.

**Run**: `e2e_hier_20261108_165847`, commit `8ba1969a`: exp0-exp6 in one batch on the same 2 nodes. The training logs the job's TFLOPS for each interval of 20 iterations, and the table divides them by the 16 GPUs. The averages are over the 41 intervals logged at iterations 199-999, and the peak columns give the best interval. MFU uses 989 BF16 dense Tensor Core TFLOPS per H100 GPU.

| Exp | Name | Avg TFLOPS/GPU | Avg MFU (%) | Peak TFLOPS/GPU | Peak MFU (%) | Speedup vs Baseline | Notes |
|-----|------|---------------:|------------:|-----------------:|-------------:|---------------------:|-------|
| 0 | Baseline | 84.7 | 8.57 | 84.9 | 8.58 | 1.00× | Triton attention, no shuffler/caching |
| 1 | +Shuffler | 131.9 | 13.33 | 133.2 | 13.47 | 1.56× | Balances Zipf-distributed sequence lengths |
| 2 | **+CUTLASS** | **341.5** | **34.53** | **348.1** | **35.20** | **4.03×** | Attention kernel swap, largest single-step gain |
| 3 | +DynamicEmb Caching | 346.1 | 35.00 | 352.3 | 35.62 | 4.09× | Enables HBM cache for DynamicEmb hot rows |
| 4 | +Hash-RoundRobin | 345.1 | 34.89 | 352.0 | 35.59 | 4.07× | Row-wise `hash_roundrobin` DynamicEmb placement |
| 5 | +Prefetch Pipeline | 349.5 | 35.34 | 355.9 | 35.99 | 4.13× | Prefetch pipeline on top of caching + Hash-RoundRobin |
| 6 | +Hierarchical Dist | 353.3 | 35.72 | 357.1 | 36.11 | 4.17× | DynamicEmb's hierarchical dist in place of TorchRec's dists |

From exp1 on, the averages include one cluster of 1-4 slow intervals, at iterations 460-580 with TorchRec's dists and 880-900 with the hierarchical dist. In it, Python's garbage collector stops each rank for 0.2-0.4 s ([Profile of the Hierarchical Dist](#profile-of-the-hierarchical-dist)).

### Key Takeaways

1. **CUTLASS attention is the main jump**: Replacing Triton attention with CUTLASS raises average throughput from 131.9 TFLOPS/GPU to 341.5 TFLOPS/GPU after the shuffler step (2.59×), and reaches 348.1 peak TFLOPS/GPU.

2. **Workload-balanced shuffler gives a clear baseline lift**: Zipf-distributed sequence lengths create load imbalance with O(n²) attention. The shuffler improves average throughput from 84.7 TFLOPS/GPU to 131.9 TFLOPS/GPU, a 1.56× speedup.

3. **DynamicEmb caching keeps the throughput**: Caching keeps the `item` table in host memory with 10% of its rows in an HBM cache. Average throughput goes from 341.5 to 346.1 TFLOPS/GPU.

4. **Hash-RoundRobin is flat in this run**: `hash_roundrobin` gives 345.1 TFLOPS/GPU, against 346.1 with `roundrobin`.

5. **The prefetch pipeline gains 1.3%**: from 345.1 to 349.5 TFLOPS/GPU, with a peak of 355.9 TFLOPS/GPU. The pipeline runs the input dist and the cache prefetch of the next batch on side streams. In the nsys profile below, the GPU idles 4.9 ms per step with it and 7.0 ms without it.

6. **The hierarchical dist gains 1.1%**: from 349.5 to 353.3 TFLOPS/GPU, with a peak of 357.1 TFLOPS/GPU (36.11% MFU). About 0.4-0.5% comes from a shorter steady step (fewer host syncs), and the rest from cheaper garbage-collection stalls, which depend on when each rank's Python collector runs. The profile below gives the details.

### Profile of the Hierarchical Dist

The profile compares exp5 with exp6, and exp4 with exp4h (exp4 with `--output_dist hierarchical`, not in the table). Each configuration ran:

- once under nsys with the benchmark's `--nsys` options, on all 16 ranks, iterations 150-169;
- once with `MEM_DEBUG=1`;
- two or three times without nsys (the batch above and two more runs), and exp5 and exp6 once more for 3,000 iterations, with Python's garbage collections logged.

The nsys numbers are means over the 16 ranks and over iterations 151-168 without the log step 159. nsys stretches the parts of the step where the GPU waits for the CPU, so the step times come from the runs without nsys. "Steady step" is the mean step over iterations 199-999 without the garbage-collection cluster. [PERF_ANALYSIS.md](./PERF_ANALYSIS.md) breaks down the GPU time of an earlier exp4 run.

| | exp4 | exp4h | exp5 | exp6 |
|---|---:|---:|---:|---:|
| Steady step, no nsys (ms) | 104.97 | 104.35 | 103.73 | 103.31 |
| Mean step, iterations 199-999, no nsys (ms) | 107.47 | 106.11 | 106.39 | 104.34 |
| Blocking host syncs per step | 23.6 | 16.0 | 27.1 | 15.6 |
| GPU idle per step (ms) | 6.97 | 6.11 | 4.94 | 4.53 |
| GPU idle while it waits for the CPU (ms) | 6.02 | 5.16 | 4.09 | 3.56 |
| Embedding-path communication not overlapped (ms) | 4.72 | 3.90 | 4.47 | 4.13 |
| Memory used per GPU, max, `MEM_DEBUG` run (GiB) | 65.2 | 66.7 | 65.4 | 67.0 |

- **Host syncs**: The hierarchical dist removes the blocking syncs of TorchRec's input dist (the split sizes of the KJT all-to-alls and the lengths of the owner's KJT) and of the index dedup (`num_uniques.item()`). It adds one sync per call: the copy of its status. With the prefetch pipeline, the `input_dist` and `wait_sparse_data_dist` phases have no sync left. The other syncs come from the shuffler (8 per step), the optimizer (2.5-3), DynamicEmb's cache prefetch (3) and the token count (1).
- **Host-boundness**: The GPU waits for the CPU only in one part of the step, from the optimizer's first sync (where the CPU waits for the backward) to the first HSTU layer. Elsewhere the CPU runs ahead: it waits for the GPU for about 73% of the step. The hierarchical dist makes that part shorter on every rank and in every step: by about 1.4 ms without a pipeline and 0.6 ms with the prefetch pipeline. Without nsys, the steady step is 0.6 ms (exp4) and 0.4 ms (exp5) shorter. The prefetch pipeline gains less, because it already takes part of TorchRec's input dist out of the forward. There, the hierarchical input dist's CUDA calls cost about as much CPU time as TorchRec's input dist.
- **Garbage collection**: About once every 1,150 iterations, Python's gen-2 garbage collection stops each rank for 0.2-0.4 s, with either dist. The ranks start their collections at different moments, and every collective then waits for the paused rank, so each start moment adds about one pause to the step time. With TorchRec's dists the 16 ranks start at 8-10 moments, and a cluster costs 2.0-3.0 s. In exp6 they start at 3 moments, and a cluster costs 0.8-1.0 s. This gives 71-91% of exp6's gain in the table. It comes from each rank's Python allocations, not from the communication, and under nsys and `MEM_DEBUG` most of it is lost. A collection at the same iteration on all ranks (not tested here) would remove most of these stalls with either dist.
- **Load balance**: In each step, the ranks' GPU work differs by 6.3-6.6 ms, almost all of it in the attention kernels. The sparse work is small: the owners' DynamicEmb kernels take 0.16-0.25 ms per step. The index dedup already removes the duplicate keys of each rank, so the node dedup removes only 5.3% of the rows that an owner receives and does not change its unique keys. In this workload the hierarchical dist does not change the balance. On more nodes the node dedup removes more: an owner then receives a key from at most local_size + num_nodes - 1 ranks instead of from all of them.
- **Communication**: The hierarchical dist replaces TorchRec's 5 embedding NCCL kernels per step with 11 kernels of its own. Its planner kernels take about as long as the bucketize and permute kernels that they replace. Most of the embedding communication that is not overlapped is the backward output dist, where the ranks wait for the rank with the most attention work. After it, the critical path is the backward of the data-parallel `action` table on a side stream (0.4-0.5 ms). On the slowest rank the hierarchical dist's backward kernel makes that path about 0.1 ms longer.
- **Memory**: At the default capacity (2,097,152 keys, see [Experiment definitions](#experiment-definitions)), the hierarchical dist adds 1.5 GiB per GPU. Of this, 1.35 GiB is outside PyTorch's allocator: its 1.6 GiB window, less about 0.25 GiB that the runs on TorchRec's dists keep outside PyTorch (most likely NCCL's buffers for their all-to-alls). The other 0.2 GiB is PyTorch memory for its working buffers. A call of this workload needs at most about 23,400 keys. With `output_dist_capacity = 131072`, exp6 uses 84 MiB less per GPU than exp5, and its step time does not change. TorchRec's row buffers are small here (about 30 MiB per step); most of the memory is the HSTU activations.

---

## 3. Reproducing the Benchmark

### Prerequisites

- Docker image built from `docker/Dockerfile`, or an equivalent environment with HSTU kernels and DynamicEmb compiled.
- Experiment 6 needs DynamicEmb built against, and run with, NCCL 2.30.7 or newer. If DynamicEmb was built against an older NCCL, the training raises `ImportError` when it shards the model. [Section 4](../../../../corelib/dynamicemb/docs/hier_output_dist.md#4-using-it) of DynamicEmb's hierarchical dist document lists its requirements and NCCL settings. Export any NCCL setting you use (such as `NCCL_CROSS_NIC=0` across nodes) before you start the batch, so that every experiment runs with it, not exp6 alone.
- All commands below assume **working directory** = `recsys-examples/examples/hstu`.

```bash
cd recsys-examples/examples/hstu
```

### Experiment definitions

Experiments are listed in `training/benchmark/experiments.txt`:

```
exp0_baseline,--value_dist zipf --value_dist_alpha 1.05
exp1_shuffler,--balanced_shuffler --value_dist zipf --value_dist_alpha 1.05
exp2_cutlass,--balanced_shuffler --kernel_backend cutlass --value_dist zipf --value_dist_alpha 1.05
exp3_caching,--balanced_shuffler --kernel_backend cutlass --caching --value_dist zipf --value_dist_alpha 1.05
exp4_caching_hr,--balanced_shuffler --kernel_backend cutlass --caching --dist_type hash_roundrobin --value_dist zipf --value_dist_alpha 1.05
exp5_prefetch,--balanced_shuffler --kernel_backend cutlass --caching --dist_type hash_roundrobin --pipeline_type prefetch --value_dist zipf --value_dist_alpha 1.05
exp6_hier,--balanced_shuffler --kernel_backend cutlass --caching --dist_type hash_roundrobin --pipeline_type prefetch --output_dist hierarchical --value_dist zipf --value_dist_alpha 1.05
```

Each line is `exp_name,options_for_generate_gin_config.py`. The script `generate_gin_config.py` produces a complete gin config file from these flags.

`--output_dist hierarchical` sets `output_dist` on every DynamicEmb table. When the config sets no `DynamicEmbeddingArgs.output_dist_capacity` (the generator sets none), the training sizes the capacity so that every call fits: world size × batch size per GPU × the most keys a sample has, summed over the DynamicEmb tables' features. In `experiments.txt`, `item` is the only DynamicEmb table, so that is 16 × 32 × 4096 keys on 2 nodes; `--include-contextual` adds `user_id`, one key a sample.

This bound is safe but large. On 2 nodes the dist's window takes about 1.6 GiB per GPU: `lookup_rows` alone is 1 GiB, and it grows by 64 MiB for each GPU in the job. The calls of exp6 need at most about 23,400 keys: with a capacity of 131,072 the window takes about 0.13 GiB, and exp6 uses no more memory than exp5 ([Profile of the Hierarchical Dist](#profile-of-the-hierarchical-dist)). For larger runs, set a smaller `output_dist_capacity` on every DynamicEmb table (`item_embedding/DynamicEmbeddingArgs.output_dist_capacity`, plus `user_id_emb` with `--include-contextual`). [Section 5.2](../../../../corelib/dynamicemb/docs/hier_output_dist.md#52-how-to-choose-them) of the hierarchical dist document shows how to choose it. The scripts generate the config on every run, so add the line to the template in `generate_gin_config.py`. A call that needs more than the capacity raises `RuntimeError`, which names the capacity to raise.

In some cases DynamicEmb only warns and takes TorchRec's dists, for example when GIN does not connect the rails on some rank, or when the window cannot be allocated ([section 6](../../../../corelib/dynamicemb/docs/hier_output_dist.md#6-when-a-sharding-takes-torchrecs-dists) of the document lists the cases). The training turns that warning into an error, so exp6 stops with the reason instead of measuring TorchRec's dists.

### Debug environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM_DEBUG` | `0` | Log GPU physical memory (including NCCL buffers) after each optimizer step on all ranks |
| `CUDA_MEM_WATCHDOG` | `0` | Auto-call `torch.cuda.empty_cache()` when caching allocator fragmentation exceeds threshold |

Set before launching training, e.g. `export CUDA_MEM_WATCHDOG=1` in the SLURM job script or shell.

### Option A: Single experiment (local)

```bash
# Run one experiment on 8 GPUs
./training/benchmark/scripts/run_single_experiment_local.sh exp2_cutlass \
    --kernel_backend cutlass --nproc=8

# Dry-run (prints generated config, does not train)
./training/benchmark/scripts/run_single_experiment_local.sh exp2_cutlass \
    --kernel_backend cutlass --dry-run
```

### Option B: All experiments (local)

```bash
# Run every experiment in experiments.txt sequentially
./training/benchmark/scripts/run_all_experiments_local.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nproc=8

# With nsys profiling
./training/benchmark/scripts/run_all_experiments_local.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nproc=8 --nsys
```

### Option C: SLURM cluster

```bash
# Submit all experiments as SLURM jobs
./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nodes=2 --ranks-per-node=8 --nsys

# Sequential execution (each job waits for the previous)
./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nodes=2 --ranks-per-node=8 --nsys --sequential

# Dry-run
./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments.txt --dry-run
```

Key `submit_all_experiments_slurm.sh` options:

| Flag | Default | Description |
|------|---------|-------------|
| `--exp-file=FILE` | *(required)* | Experiment list |
| `--nodes=N` | 2 | SLURM nodes |
| `--ranks-per-node=N` | 8 | GPUs per node |
| `--nsys` | off | Enable nsys profiling |
| `--sequential` | parallel | Chain jobs with dependencies |
| `--container-image=IMG` | *(see script)* | Override container image |
| `--partition=NAME` | batch | SLURM partition |
| `--time=HH:MM:SS` | 00:30:00 | Wall-time limit |
| `--wait-and-analyze` | off | Poll jobs and auto-run analysis |
| `--dry-run` | off | Print commands only |

### Running a subset

Create a custom experiment file:

```bash
cat > quick_test.txt << 'EOF'
exp0_baseline,--value_dist zipf --value_dist_alpha 1.05
exp2_cutlass,--balanced_shuffler --kernel_backend cutlass --value_dist zipf --value_dist_alpha 1.05
EOF

./training/benchmark/scripts/run_all_experiments_local.sh --exp-file=quick_test.txt --nproc=8
```

### Output directory structure

```
training/benchmark/results/
└── {batch_timestamp}/
    ├── exp0_baseline/
    │   ├── exp0_baseline_{timestamp}.gin     # generated config
    │   ├── exp0_baseline_{timestamp}.log     # training log
    │   └── exp0_baseline_*.nsys-rep          # nsys profiles (if --nsys)
    ├── exp1_shuffler/
    │   └── ...
    ├── exp2_cutlass/
    │   └── ...
    ├── exp3_caching/
    │   └── ...
    ├── exp4_caching_hr/
    │   └── ...
    ├── exp5_prefetch/
    │   └── ...
    ├── exp6_hier/
    │   └── ...
    ├── summary.txt                           # batch summary
    └── comparison.png                         # TFLOPS + MFU comparison chart
```

### Analyzing results

```bash
# Parse MFU from training logs
python training/benchmark/scripts/analyze_results.py \
    training/benchmark/results/{batch_timestamp}/

# Nsight Systems CLI stats
nsys stats training/benchmark/results/{batch_timestamp}/exp0_baseline/*.nsys-rep
```
