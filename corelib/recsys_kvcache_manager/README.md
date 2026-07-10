# Recsys KVCache Manager

Recsys KVCache Manager is a Python package that provides LLM-compatible KV data caching, storage, and lookup for generative recommender model inference.
It supports KV-cache management based on recommender-system **user IDs**, enabling KV-cache reuse across requests from the same user.

Recsys KVCache Manager is based on the **PyTorch** ecosystem. It manages KV data in GPU memory and host memory backed by lower-tier storage.
It supports lookup, offloading to lower-tier storage, onboarding to GPU memory for inference, and read/write APIs for KV-cache data.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Basic APIs](#basic-apis)
- [FlexKV Backend Options](#flexkv-backend-options)
- [Important Notes](#important-notes)
- [Examples](#example)
- [Future Plans](#future-plans)

## Features

- **User ID Based Caching**:
For recommender systems, behavior sequences that generate KV data vary greatly across users, and there is usually little common prefix between input sequences.
Recsys KVCache Manager supports data lookup by user ID instead of comparing sequence token values.

<div style="margin-left: 2em;"> 

**Recsys KVCache Manager** modules:

</div>

```
    + KVCacheManager :  Interface for KV-cache operations
    |
    ├---- GPUKVCacheManager :  Manager for GPU KV-cache table
    |
    └---- HostKVStorageManagerBase :  Interface of manager for host memory/SSD/remote KV cache
        |
        ├---- NativeHostKVCacheManager :  Wrapper for pinned-host-memory-only KV cache
        |
        └---- FlexKVCacheManager :  Wrapper to FlexKV cache system
```

- **Paged GPU KVCache Table**:
The GPU KV-cache table is organized as a paged KV-data table and supports KV data add/append, lookup, and eviction. When appending new data to the GPU cache, it evicts data from the oldest users by LRU policy if there is no empty page. The HSTU attention kernel from FBGEMM-HSTU can load KV data directly from a paged table, avoiding additional data copies.

- **Asynchronous Onboarding/Offloading**:
By using asynchronous data copy on the side CUDA stream, we overlap the KV data transfer between GPU memory and host storage (onboarding/offloading) with embedding lookup, sequence pre-/post-processing, and inference for other requests (in some cases) to reduce the latency of HSTU inference.
Furthermore, the `NativeHostKVCacheManager` backend supports layerwise KV data onboarding, overlapping the H2D data transfer with computation from the previous HSTU layers.

- **Extension for Multiple Backend**: 
`HostKVStorageManagerBase` is provided as an interface for other LLM-compatible KV-cache systems for host memory, storage, and remote data pools.
This can be extended to integrate other KV-cache systems. Currently, we provide integration with [`FlexKV`](https://github.com/taco-project/FlexKV/tree/main) as the lower-tier KV storage backend.


## Installation

To install, please use the following command:

```bash
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" pip3 install --no-build-isolation .
```

**Note**: To enable the FlexKV backend, install the FlexKV package according to the [FlexKV doc](https://github.com/taco-project/FlexKV/tree/main#how-to-use).


## Basic APIs

-   `lookup_kvcache`: gets the cached sequence length from both the GPU cache table and host KV storage.

-   `allocate_kvcache`: assigns required cache pages in the GPU cache table for inference.
This generates page IDs for reading/writing data and metadata for appending KV data into the GPU table.
It also evicts used cache pages when running out of empty pages. **No offloading** is performed upon eviction in the current implementation.

-   `onboard_launch`: launches async host-to-GPU KV-cache transfer.

-   `onboard_try_wait`, `onboard_wait`: performs non-blocking/blocking waiting for KV data onboarding.

-   `offload_launch`: launches async GPU-to-host offload and records the task into ongoing offload queue.

-   `offload_try_wait`: polls ongoing offload tasks, finishes ready tasks, cancels failed or timed-out tasks, and unlocks host KV storage state.

-   `evict`, `evict_all`: explicitly evicts cached data from the GPU cache table and/or lower-tier storage if supported.

## FlexKV Backend Options

When `host_kvstorage_backend` is set to `flexkv`, additional FlexKV options can be passed through `extra_configs`.

| Option | Default | Description |
| --- | --- | --- |
| `flexkv_as_batch` | `1` | Enables batched FlexKV launch for multi-user onboard/offload requests. Set to `0` / `false` to disable. |
| `flexkv_config_path` | `None` | Optional FlexKV tier config path. Use this to enable SSD-backed cache tiers or other FlexKV user config settings. |

Example:

```python
extra_configs = {
    "flexkv_mode": "direct",
    "flexkv_as_batch": 1,
    "flexkv_config_path": "corelib/recsys_kvcache_manager/configs/flexkv_ssd.yml",
}
```

The sample SSD config is:

```yaml
cpu_cache_gb: 0.2
ssd_cache_gb: 0.5
ssd_cache_dir: /path/to/your/ssd_cache
enable_gds: false
enable_p2p_cpu: false
enable_p2p_ssd: false
enable_3rd_remote: false
```

Before using SSD cache, update `ssd_cache_dir` to a writable local SSD path and set `ssd_cache_gb` > 0.

## Important Notes

There are some **limitations** in the current implementation. These are expected to be resolved for broader use cases and better performance.

1. API `allocate_kvcache` is host blocking, and cannot overlap with other operations.

2. Only **one** GPU KV-cache manager is allowed per device, and only one inference instance is allowed for each GPU KV-cache manager.

3. The `native` host backend is limited to at most **one** GPU KV-cache manager and one inference instance. Use it with `user_id`-based routing and inference instance isolation.


## Example

**Typical Usage of KVCache Manager**:
```python
    # Input User IDs
    user_ids: torch.Tensor
    # Sequence lengths per user
    sequence_lengths: torch.Tensor

    # Lookup 
    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)

    # strip cached tokens from input sequences
    [[ ... strip cached tokens ... ]]

    # Allocate in GPU cache table
    kvcache_mgr.offload_try_wait()  # [Optional] Try to free up GPU cache space by completing offloading.
    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    # Onboard to GPU cache table, non-blocking, if necessary
    kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)

    [[ ... embedding lookup ... ]]
    [[ ... preprocess ... ]]


    # Dense Module computation
    # Note: Here we show two possible ways to synchronize with onboard completion:
    #       (1) blocking wait for onboard completion, for non-layerwise backends
    #       (2) non-blocking CUDA stream wait with layerwise onboard events

    # Case (1): Blocking total wait. [ Note: not supported with "native" backend. ]
    kvcache_mgr.onboard_wait(index_meta, kvcache_metadata.kv_onload_handle)
    for layer_idx in range(num_layers):

        [[ ... write new KV data through `kvcache_metadata.kv_cache_table` ]]  # See `kvcache_mgr.gpu_kvcache_mgr.put`

        # Case (2): Layerwise stream wait. [ Note: only active with layerwise backends such as "native". ]
        kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)

        [[ ... attention computation, loading data using `kv_indices`, `kvkv_indptr`, etc. ... ]]
        

    # Offloading to host and lower tiers, non-blocking, if necessary
    kvcache_mgr.offload_try_wait() # [Optional] Try to free up host buffer/sync host caching status by completing offloading.
    kvcache_mgr.offload_launch(index_meta)

    [[ ... postprocessing ... ]]
    [[ ... return inference results ... ]]
```

**Refer to** HSTU model inference in RecSys Examples for details: [InferenceRankingGR](../../examples/hstu/model/inference_ranking_gr.py), [InferenceDenseModule](../../examples/hstu/modules/inference_dense_module.py).


## Future Plans

1. Support concurrent KV-cache operations across inference instances.
2. Broaden Torch Export and AOTInductor coverage for recommender-model KV-cache inference paths in the Torch C++ runtime.
