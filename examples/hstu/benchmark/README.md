# Fused HSTU layer benchmark

In hstu example, we have provided a set of performance optimization guidelines, including
1. Fast and memory-efficient hstu attention integration.
2. Kernel fusions (with triton).
3. Seletive forward recompute.

You can run script `run.sh` to see the performance over native implementation. The baseline (native implementation) is 
1. With triton-based hstu attention kernels
2. No kernel fusions.
3. No recompute.

# How to run

```bash
RECOMPUTE_INPUT_SILU=True RECOMPUTE_INPUT_LAYERNORM=True bash run.sh <num_layers>
```
Since recompute helps reduce activation memory usage but incurs latency increase, you can use env `RECOMPUTE_INPUT_SILU, RECOMPUTE_INPUT_SILU` to decide whether to enable the input layernorm and the first silu following uvqk linear.


# results

We cover sequence from 1k~8k, other hyper-params are as followed:
| Item          | Value |
| ------------- | ----- |
| Batchsize     | 32    |
| dim per head  | 256   |
| num_heads     | 4     |
| embedding dim | 1024  |


## latency


| seqlen | Baseline (ms) | Optimized (ms) | +layer norm recompute (ms) | +silu recompute (ms) |
| ------ | ------------- | -------------- | -------------------------- | -------------------- |
| 1K     |               |                |                            |                      |
| 2K     |               |                |                            |                      |
| 4K     |               |                |                            |                      |
| 8K     |               |                |                            |                      |

The numbers of last 2 columns are incrementally tested based on the previous column.

## PEAK memory

| seqlen | Baseline (MB) | Optimized (MB) | +layer norm recompute (MB) | +silu recompute (MB) |
| ------ | ------------- | -------------- | -------------------------- | -------------------- |
| 1K     |               |                |                            |                      |
| 2K     |               |                |                            |                      |
| 4K     |               |                |                            |                      |
| 8K     |               |                |                            |                      |



