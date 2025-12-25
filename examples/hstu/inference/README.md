# HSTU Inference

## Key Features

1. Cache for KV data

We use GPU memory and host storage for KV data cache., as in `GpuKVCacheManager` and `HostKVStorageManager`. This can help to reduce the recomputation of KV data.

The GPU KV cache is organized as a paged KV-data table, and supports KV data adding/appending, lookup and eviction. When appending new data to the GPU cache, we will evict data from the oldest users according to the LRU policy if there is no empty page. The HSTU attention kernel also accepts KV data from a paged table.

The host KV data storage support adding/appending and lookup. We only present an example implementation, since this can be built over other database and can vary widely in the deployment.

2. Asynchronous H2D transfer of host KV data 

By using asynchronous data copy on the side CUDA stream, we overlap the host-to-device KV data transfer with HSTU computation layer-wise, to reduce the latency of HSTU inference.


3. Optimization with CUDA graph

We utilize the graph capture and replay support in Torch for convenient CUDA graph optimization on the HSTU layers. This decreases the overhead for kernel launch, especially for input with a small batch size. The input data (hidden states) fed to HSTU layers needs paddding to pre-determined batch size and sequence length, due to the requirement of static shape in CUDA graph.

4. Kernel fusion

## KVCache Manager for Inference

### KVCache Usage

1. KVCache Manager supports the following operations:
* `get_user_kvdata_info`: to get current cached length and index of the first cached tokens in the history sequence
* `prepare_kv_cache`: to allocate the required cache pages. The input history sequence need to be 
* `paged_kvcache_ops.append_kvcache`: the cuda kernel to copy the `K, V` values into the allocated cache pages
* `offload_kv_cache`: to offload the KV data from GPU KVCache to Host KV storage.
* `evict_kv_cache`: to evict all the KV data in the KVCache Manager.

2. Currently, the KVCache manager need to be access from a single thread.

3. For different requests, the call to `get_user_kvdata_info` and `prepare_kv_cache` need to be in order and cannot be interleaved. Since the allocation in `prepare_kv_cache` may evict the cached data of other users, which changes the user kvdata_info.

4. The KVCache manager does not support uncontinuous user history sequence as input from the same user. The overlapping tokens need to be removed before sending the sequence to the inference model. Doing the overrlapping removal in the upstream stage should be more performant than in the inference model.

```
[current KV data in cache] userId: 0, starting position: 0, cached length: 10
[next input] {userId: 0, starting position: 10, length: 10}
# Acceptable input

[current KV data in cache] userId: 0, starting position: 0, cached length: 10
[next input] {userId: 0, starting position: 20, length: 10}
                         ^^^^^^^^^^^^^^^^^^^^^
ERROR: The input sequence has missing tokens from 10 to 19 (both inclusive).

[current KV data in cache] userId: 0, starting position: 0, cached length: 10
[next input] {userId: 0, starting position: 5, length: 20}
                         ^^^^^^^^^^^^^^^^^^^^^
ERROR: The input sequence has overlapping tokens from 5 to 9 (both inclusive).
```

## How to Setup

1. Install the dependencies for Recsys-Examples.

Turn on option `INFERENCEBUILD=1` to skip Megatron installation, which is not required for inference.

```bash
~$ cd ${WORKING_DIR}
~$ git clone --recursive -b ${TEST_BRANCH} ${TEST_REPO} recsys-examples && cd recsys-examples
~$ docker build \
    --build-arg INFERENCEBUILD=1 \
    -t recsys-examples:inference \
    -f docker/Dockerfile .
```

## Example: Kuairand-1K

```
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ 
~$ # Proprocess the dataset for inference:
~$ python3 ./preprocessor.py --dataset_name "kuairand-1k" --inference
~$
~$ # Run the inference example
~$ python3 ./inference/inference_gr_ranking_async.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ${PATH_TO_CHECKPOINT}  --mode eval
```

## Consistency Check for Inference

Currently, we use the evaluation metrics results (e.g. AUC) to check the consistency between training and inference.

1. Evaluation metrics from training

* Add evaluation output in training configs. Make sure `max_train_iters` is a multiple of `max_train_iters`.

```
# File: examples/hstu/training/configs/
...
TrainerArgs.eval_interval = 50
TrainerArgs.max_train_iters = 550
TrainerArgs.ckpt_save_interval = 550
...
```

* Get eval metrics from training
```
/workspace/recsys-examples$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 ./training/pretrain_gr_ranking.py --gin-config-file ./training/configs/kuairand_1k_ranking.gin
... [training output] ...
[eval] [eval 296 users]:
    Metrics.task0.AUC:0.557266
    Metrics.task1.AUC:0.801949
    Metrics.task2.AUC:0.599034
    Metrics.task3.AUC:0.666739
    Metrics.task4.AUC:0.555904
    Metrics.task5.AUC:0.582272
    Metrics.task6.AUC:0.620481
    Metrics.task7.AUC:0.556170
... [training output] ...
```

2. Evaluation metrics from inference
```
/workspace/recsys-examples$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) python3 ./inference/inference_gr_ranking_async.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ${PATH_TO_CHECKPOINT} --mode eval
... [inference output] ...
[eval]:
    Metrics.task0.AUC:0.556894
    Metrics.task1.AUC:0.802019
    Metrics.task2.AUC:0.599779
    Metrics.task3.AUC:0.666891
    Metrics.task4.AUC:0.559471
    Metrics.task5.AUC:0.580227
    Metrics.task6.AUC:0.620498
    Metrics.task7.AUC:0.556064
... [inference output] ...
