# Benchmark

We provide two sets of benchmarks:

* [HSTU layer benchmarks](#hstu-layer-benchmark)
* [Inference benchmarks](#hstu-inference-benchmark) on end-to-end inference and paged HSTU inference layer 

# HSTU layer benchmark

In hstu example, we have provided a set of performance optimization guidelines for single HSTU layer, including
1. Fast and memory-efficient hstu attention integration.
2. Kernel fusions: layer norm + multiplication + dropout 
3. Seletive forward recompute.

You can run script `run_hstu_benchmark.sh` to see the performance over the base implementation. The baseline is from [Meta's open source HSTU implementation](https://github.com/meta-recsys/generative-recommenders/tree/bb389f9539b054e7268528efcd35457a6ad52439), which features in:

1. Triton-based HSTU attention kernels with the remaining operations using PyTorch ops.
2. No kernel fusions.
3. No recompute.

## How to run

The test entry is `python ./benchmark/hstu_layer_benchmark.py run`, you can type `python ./benchmark/hstu_layer_benchmark.py run --help` to get the input arguments. 4 important arguments are :

1. --kernel-backend: select the hstu mha backend. Could be `triton` or `cutlass`.
2. --fuse-norm-mul-dropout: knob of  `layer norm + multiplication + dropout ` fusion. Could be `False` or `True`
3. --recompute-input-silu: knob of silu recompute. Could be `False` or `True`
4. --recompute-input-layernorm: knob of input layer norm recompute. Could be `False` or `True`

Our baseline cmd example (1K): 

```bash
python ./benchmark/hstu_layer_benchmark.py run \
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
bash run_hstu_layer_benchmark.sh <num_layers>
```

After one run is done, a memory snapshot file in current working directory is generated, you can trace the memory usage with the file. Please refer to [PyTorch docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) on how to visualize the memory trace.

## Benchmark results

We cover sequence from 1k~8k, other hyper-params are as followed:
| Item          | Value |
| ------------- | ----- |
| Batchsize     | 32    |
| dim per head  | 256   |
| num_heads     | 4     |
| embedding dim | 1024  |

All results are conducted on single H100-SXM5-80G

### Throughput

![hstu_layer_perf](./hstu_layer_perf.png)

The columns other than the first column are incrementally tested based on the previous column.

### Peak memory

We trace the peak memory with the help of torch memory snapshot. To better identify the boundary forward and backward process, we have run 3 HSTU layers.
Below are the memory usage for seqlen=4K:

![image](./memory_snapshot.png)

# HSTU Inference benchmark

## Key Features

1. Cache for KV data

We use GPU memory and host storage for KV data cache., as in `GpuKVCacheManager` and `HostKVStorageManager`. This can help to reduce the recomputation of KV data.

The GPU KV cache is organized as a paged KV-data table, and supports KV data adding/appending, lookup and eviction. When appending new data to the GPU cache, we will evict data from the oldest users according to the LRU policy if there is no empty page. The HSTU attention kernel also accepts KV data from a paged table.

The host KV data storage support adding/appending and lookup. We only present an example implementation, since this can be built over other database and can vary widely in the deployment.

2. Asynchronous H2D transfer of host KV data 

By using asynchronous data copy on the side CUDA stream, we overlap the host-to-device KV data transfer with HSTU computation layer-wise, to reduce the latency of HSTU inference.


3. Optimization with CUDA graph

We utilize the graph capture and replay support in Torch for convenient CUDA graph optimization on the HSTU layers. This decreases the overhead for kernel launch, especially for input with a small batch size. The input data (hidden states) fed to HSTU layers needs paddding to pre-determined batch size and sequence length, due to the requirement of static shape in CUDA graph.

## How to run

1. Build TensorRT-LLM (with HSTU KV cache extension):

The HSTU inference utilize customized KV cache manager from TensorRT-LLM.
The current version is based on the HSTU specialized implementation based on TensorRT-LLM v0.19.0.

```bash
~$ cd ${WORKING_DIR}
~$ git clone -b hstu-kvcache-recsys-examples https://github.com/geoffreyQiu/TensorRT-LLM.git tensorrt-llm-kvcache && cd tensorrt-llm-kvcache
~$ git submodule update --init --recursive
~$ make -C docker release_build CUDA_ARCHS="80-real;86-real"
# This will build a docker image with TensorRT-LLM installed.
```

2. Install the dependencies for Recsys-Examples.

Turn on option `INFERENCEBUILD=1` to skip Megatron installation, which is not required for inference.

```bash
~$ cd ${WORKING_DIR}
~$ git clone --recursive -b ${TEST_BRANCH} ${TEST_REPO} recsys-examples && cd recsys-examples
~$ TRTLLM_KVCACHE_IMAGE="tensorrt_llm/release:latest" docker build \
    --build-arg BASE_IMAGE=${TRTLLM_KVCACHE_IMAGE} \
    --build-arg INFERENCEBUILD=1 \
    -t recsys-examples:inference \
    -f docker/Dockerfile .
```

3. Run the benchmark.

```bash
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ python3 ./benchmark/inference_benchmark.py
~$ python3 ./benchmark/paged_hstu_with_kvcache_benchmark.py
```

## Benchmark results

Here we present the benchmark results of the HSTU layers with KV cache on single L20 gpu.

HSTU Setup for benchmark:

| Parameter                   | Value |
| --------------------------- | ----- |
| Number of HSTU layers       | 8     |
| Hidden Dim Size             | 1024  |
| Number of Heads             | 4     |
| Head Dim Size               | 256   |
| Max Batchsize               | 16    |
| Max Per Sequence Length     | 4096  |
| Per Sequence Targets Number | 256   |

### 1. End-to-end inference performance

Here we benchmarked with a synthetic input dataset:

* Each user's input sequence starts from 256 tokens to 4096 in increments of 256.
* Each input request has 256 item candidates for ranking.
* Generate data for 1, 2, 4 and 8 users to benchmark with different batch size. 

We can achieve **1.4x ~ 2.7x** performance speedup for inference (with batch size ranging from 1 to 8), after utilizing the KV cache and CUDA graph optimization.

Performance results:

![Local Image](inference_benchmark_l20.png)

Note:

1. The baseline performance is based on our implementation without KVCache support and CUDA Graph optimization.
2. The end-to-end performance includes the embedding part, which utilizes both native `EmbeddingCollection` from TorchRec and `DynamicEmbedding`.
3. The number of input sequences from the synthetic dataset increases according to the batch size.

### 2. HSTU block performance

Performance Results:

![Local Image](hstu_inference_l20_batch1.png)
![Local Image](hstu_inference_l20_batch8.png)

When the input sequence has 4096 tokens in which 3968 tokens have KV data cached, we can achieve on HSTU block a speedup of **3x ~ 20x** without candidate items, and a speedup of **3x ~ 8x** with extra 256 candidates for each sequence. (for batch size = 1 and 8 respectively).