#!/bin/bash
# Run the dynamicemb benchmark via pytest.
#
# The benchmark is organized into three test classes (suites):
#   TestGpu       -- full table in HBM, gpu_ratio=1.0
#   TestCaching   -- 10% HBM with LRU caching
#   TestNoCaching -- 10% HBM without caching (UVM / eviction)
#
# Each suite sweeps batch_size x optimizer x pooling_mode (8 configs).
#
# Usage:
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh                       # all suites
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh -k TestGpu            # gpu only
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh -k TestCaching        # caching only
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh -k "adam and sum"     # filter
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile torch       # with profiling
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile ncu-gen --ncu-kernel-regex 'segmented_unique'  # print ncu commands (no tests; regex optional, omit for all kernels in nvtx scope)
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile ncu         # run all iters (wrap with ncu)
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile ncu --num-iterations 3  # fewer iters to keep ncu tractable
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile ncu --ncu-iterations 90:  # only capture late (warm) iters; supports 0,3,7 or begin:end:step
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --correctness          # force-enable forward-only correctness check on every config
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --correctness -k TestGpu  # correctness on gpu suite only
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --num-tables 1000      # override table count (total cap/batch held fixed -> cap/table = total//1000)
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --sparse-key-range 4096  # sample keys from [.., 4096) per table (controls dup rate, independent of cap)
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --no-torchrec          # DynamicEmb only; skip TorchRec/TBE baseline (trc_* metrics null)
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh -k "TestGpu and sgd" --num-tables 1000 --no-torchrec  # 1000-table dyn-only run
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --co                  # list configs
#
# Correctness can also be enabled per-config by setting `correctness=True` on
# the BenchmarkConfig (in _gpu_configs / _caching_configs / etc.).  The check
# runs only in the default (no --profile) path and compares forward outputs
# against the TorchRec/FBGEMM TBE baseline.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export BENCHMARK_RESULTS_FILE=${BENCHMARK_RESULTS_FILE:-benchmark_results.json}

rm -f "$BENCHMARK_RESULTS_FILE"

torchrun --nnodes 1 --nproc_per_node 1 \
    -m pytest ./benchmark/benchmark_batched_dynamicemb_tables.py \
    -svv "$@"
