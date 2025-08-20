# KVCache Manager Guidelines for Inference

## KVCache Usage

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


## Dataloader Example: Kuairand-1K

```
~$ # Proprocess the dataset for inference:
~$ python3 ./preprocessor_inference.py --dataset_name "kuairand-1k"
~$
~$ # Run the inference example
~$ python3 ./benchmark/inference_end2end_example.py 
```
