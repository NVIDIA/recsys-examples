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
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile ncu-gen     # print ncu commands (no tests)
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile ncu-run     # single-iter run (wrap with ncu)
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --co                  # list configs

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export BENCHMARK_RESULTS_FILE=${BENCHMARK_RESULTS_FILE:-benchmark_results.json}

rm -f "$BENCHMARK_RESULTS_FILE"

torchrun --nnodes 1 --nproc_per_node 1 \
    -m pytest ./benchmark/benchmark_batched_dynamicemb_tables.py \
    -svv "$@"
