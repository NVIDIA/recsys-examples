# Fused HSTU layer benchmark

In hstu example, we have provided a set of performance optimization guidelines for single HSTU layer, including
1. Fast and memory-efficient hstu attention integration.
2. Kernel fusions (with triton).
3. Seletive forward recompute.

You can run script `run.sh` to see the performance over native implementation. The baseline (native implementation) is 
1. With triton-based hstu attention kernels
2. No kernel fusions.
3. No recompute.

# How to run
The test entry is `python ./benchmark/fused_hstu_layer_benchmark.py run`, you can type `python ./benchmark/fused_hstu_layer_benchmark.py run --help` to get the input arguments. 2 important arguments are :
1. --layer-type: whether to enable fusions. Could be `fused` or `native`.
2. --kernel-backend: select the hstu mha backend. Could be `triton` or `cutlass`.

Our baseline cmd example (1K): 
```bash
python ./benchmark/fused_hstu_layer_benchmark.py run \
  --iters 100 \
  --warmup-iters 50 \
  --layer-type native \
  --kernel-backend triton \
  --dim-per-head 256 \
  --num-heads 4 \
  --num-layers 1 \
  --dtype bfloat16 \
  --max-seqlen 1024 \
  --full-sequence True \
  --batchsize 32 
```

You can also run a set of arguments with run.sh:
```bash
RECOMPUTE_INPUT_SILU=True RECOMPUTE_INPUT_LAYERNORM=True bash run.sh <num_layers>
```
Since recompute helps reduce activation memory usage but incurs latency increase, you can use env `RECOMPUTE_INPUT_SILU, RECOMPUTE_INPUT_SILU` to decide whether to enable the input layernorm and the first silu following uvqk linear.

After one run is done, a memory snapshot file in current working directory is generated, you can trace the memory usage with the file. Please refer to [PyTorch docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) on how to visualize the memory trace.

# results

We cover sequence from 1k~8k, other hyper-params are as followed:
| Item          | Value |
| ------------- | ----- |
| Batchsize     | 32    |
| dim per head  | 256   |
| num_heads     | 4     |
| embedding dim | 1024  |

All results are conducted on single H100-SXM5-80G
## Latency

| seqlen | Baseline (ms) | + cutlass kernel | +fusion | +layer norm recompute (ms) | +silu recompute (ms) |
| ------ | ------------- | ---------------- | ------- | -------------------------- | -------------------- |
| 1K     | 6.6515        | 5.8640           | 3.8854  | 3.9271                     | 4.1149               |
| 2K     | 16.0452       | 12.9900          | 9.1797  | 9.2780                     | 9.7622               |
| 4K     | 44.3293       | 31.7074          | 24.5428 | 24.7954                    | 25.5000              |
| 8K     | 137.9320      | 88.3084          | 74.7734 | 74.8163                    | 76.3875              |

The columns other than the first column are incrementally tested based on the previous column.

## Peak memory
We trace the peak memory with the help of torch memory snapshot. To better identify the boundary forward and backward process, we have run 2 HSTU layers.
Below are the memory usage for seqlen=4K:

![image](./memory_snapshot.png)