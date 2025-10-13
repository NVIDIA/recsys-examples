# Dynamicemb Example Introduction

In short, **dynamicemb** provides distributed, high-performance dynamic embedding storage and related functions for training.

**dynamicemb** provides a high-performance **hash table** to support dynamic embedding and leverages **torchrec** to implement sharding logic on multiple GPUs. This explains why dynamicemb largely reuses the user interface of torchrec while adding some new configuration options related to dynamic embedding. This document and `example.py` will help you understand the usage of dynamicemb and how it works from a code perspective.


## Requirement
> torchrec >= v1.2.0

Thanks to the torchrec team for their [support](https://github.com/meta-pytorch/torchrec/commit/6aaf1fa72e884642f39c49ef232162fa3772055e), torchrec v1.2.0 added support for custom embedding lookup module.

# EmbeddingCollection

```python
eb_configs = [
  EmbeddingConfig(
    name="user_id",
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings,
    feature_names=["user_id"], # a list, means different features can share the same table
    data_type=DataType.FP32,  # weight or embedding's data type.
  ),
]
ec = EmbeddingCollection(
    tables=eb_configs,
    device=torch.device("meta"),  # set device to 'meta
)
```

`EmbeddingCollection` is a collection of multiple logical tables. It does not allocate memory for embedding tables(device is "meta"). `num_embeddings` in `EmbeddingConfig` is the sum of all slices on all GPUs for a table.

***dynamicemb** supports not only `EmbeddingCollection` but also `EmbeddingBagCollection`. However, due to the requirements of generative recommendations, dynamicemb focuses on performance optimization of `EmbeddingCollection` while providing full functional support for `EmbeddingBagCollection`.*

# DynamicEmbeddingCollectionSharder

After configuring the `EmbeddingCollection`, you need to configure `DynamicEmbeddingCollectionSharder`. It will create an instance of `ShardedDynamicEmbeddingCollection`.

The actual sharding operation occurs during the initialization of the `ShardedDynamicEmbeddingCollection`, but the parameters used to initialize `DynamicEmbeddingCollectionSharder`  will play a key role in the sharding process.

```python

optimizer_kwargs = {
  "optimizer": EmbOptimType.ADAM,
  "learning_rate": learning_rate,
  "beta1": beta1,
  "beta2": beta2,
  "weight_decay": weight_decay,
  "eps": eps,
}

fused_params = {}
fused_params.update(optimizer_kwargs)
fused_params["output_dtype"] = SparseType.FP32 # data type of the output after lookup, and can differ from the stored.
fused_params["prefetch_pipeline"] = args.prefetch_pipeline  # whether enable prefetch for embedding lookup module

# precision of all-to-all
qcomm_codecs_registry = (
  get_qcomm_codecs_registry(
    qcomms_config=QCommsConfig(
      # pyre-ignore
      forward_precision=CommType.FP32,
      # pyre-ignore
      backward_precision=CommType.FP32,
    )
  )
  if backend == "nccl"
  else None
)

sharder = DynamicEmbeddingCollectionSharder(
  qcomm_codecs_registry=qcomm_codecs_registry,
  fused_params=fused_params,
  use_index_dedup=True,
)
```
- **fused_params**: items in fused_params will be finally passed to embedding lookup module. But before that:  
logic tables in `EmbeddingCollection` will be divided into multiple groups in the `ShardedDynamicEmbeddingCollection`, and the fused_params are equal for tables in the same group. 
However, we only provide the common for all tables here, but some fields in `DynamicEmbTableOptions` will be merged into fused_params and then be used to group tables(please refer [DynamicEmbTableOptions](#options) for more details).
**Performance** issue: Embedding lookup within the same group can be executed in parallel, while embedding lookup between different groups can only be executed sequentially.
- **use_index_dedup**: Unlike `EmbeddingBagCollection`, there is no reduction operation at the jagged dimension in the input `KeyedJaggedTensor` for `EmbeddingCollection`.
Therefore, we can deduplicate the input's indices in the input distributor before sparse feature's all-to-all, then it will reduce the bandwidth pressure of NVLink or PCIe when perform embedding's all-to-all, and restore them using inverse information finally.
- **qcomm_codecs_registry**: used to configure the embeddings(forward) or gradients(backward)' precision when perform all-to-all operation across different ranks in distributed environment. 

## ShardedDynamicEmbeddingCollection

*This section focus on internal implementation, and can skip it if just for usage.*

`ShardedDynamicEmbeddingCollection` provides customized embedding lookup module base on [HKV](https://github.com/NVIDIA-Merlin/HierarchicalKV), a GPU hash table which can utilize both device and host memory, support automatic eviction based on score(per key) while provide a better performance.

Besides, due to differences in deduplication between hash tables and array based static tables, `ShardedDynamicEmbeddingCollection` also provide customized input distributor to support deduplication when `use_index_dedup=True`.

By the way, `DynamicEmbeddingCollectionSharder` inherits `EmbeddingCollectionSharder`, and its main job is return an instance of `ShardedDynamicEmbeddingCollection`.


## DynamicEmbTableOptions {#options}

`DynamicEmbTableOptions` is used to control each table's configuration and behaviors.
I will first introduce `DynamicEmbTableOptions` and explain the features supported by dynamicemb.
*Since there already exists API doc, I will not cover it in detail.*

**Only** need to configure these parameters(except max_capacity):
- training: Is it in training mode. **dyanmicemb** store embeddings and optimizer states together in the underlying key-value table. e.g. 
```python
key:torch.int64
value = torch.concat(embedding, opt_states, dim=1)
```
Therefore, if `training=True` dynamicemb will allocate memory for optimizer states whose consumption is decided by the `optimizer_type` which got from `fused_params`. 

- initializer_args: initializer arguments for training, and dynamicemb supports multiple initialization modes(refer `DynamicEmbInitializerMode`). Fothermore, for `UNIFORM` and `TRUNCATED_NORMAL`, the `lower` and `upper` will set to $\pm {1 \over \sqrt{EmbeddingConfig.num\_embeddings}}$.

- eval_initializer_args: initializer arguments for evaluation, and will return torch.zeros(...) as embedding by default if index/sparse feature is missing.

- caching: When the device memory on a single GPU is insufficient to accommodate a single shard of the dynamic embedding table, HKV supports the mixed use of device memory and host memory(pinned memory).
But by default, the values of the entire table are concatenated with device memory and host memory. This means that the storage location of one embeddng is determined by `hash_function(key)`, and mapping to device memory will bring better lookup performance. However, sparse features in training are often with temporal locality.
In order to store hot keys in device memory, dynamicemb creates two HKV instances, whose values are stored in device memory and memory respectively, and store hot keys on the GPU table priorily. If the GPU table is full, the evicted keys will be inserted into the CPU table. If the CPU table is also full, the key granularity will be evicted(all the eviction is based on the score per key). The original intention of eviction is based on this insight: features that only appear once should not occupy memory(even host memory) for a long time.
In short, set **`caching=True`** will create a GPU table and a CPU table, and make GPU table serves as a cache; set **`caching=False`** will create a hybrid table which use GPU and CPU memory in a concated way to store value. All keys and other meta data are always stored on GPU for both cases.

- max_capacity: It is not configurable, but it's important for the total memory consumption. `max_capacity` is the size of the shard of the embedding table on a single GPU. This field is inferred from `EmbeddingConfig.num_embeddings` and world size, rounded up to the power of 2, and minimized to the size of bucket capacity of the HKV.
- init_capacity: If `init_capacity` is provided, it will serve as the initial table capacity on a single GPU. As the `load_factor` of the table increases, its capacity will gradually double (rehash) until it reaches `max_capacity`. Rehash will be done implicitly.
- score_strategy: dynamicemb gives each key-value pair a score to represent its importance. Once there is insufficient space, the key-value pair will be evicted based on the score. The `score_strategy` is used to configure how to set the scores for keys in each batch.
- bucket_capacity: Capacity of each bucket in HKV(except cache). A key will only be mapped to one bucket. When the bucket is full, the key with the smallest score in the bucket will be evicted, and its slot will be used to store a new key. The larger the bucket capacity, the more accurate the score based eviction will be, but it will also result in performance loss.
- safe_check_mode: Used to check if all keys in the current batch have been successfully inserted into the table.
- global_hbm_for_values: It has different meanings under `caching=True` and  `caching=False`.
When `caching=False`, it decides how much GPU memory is in the total memory to store value in a single hybrid table.
When `caching=True`, it decides the table capacity of the GPU table.

- external_storage: dynamicemb supports external storage once `external_storage` inherits the `Storage` interface under [key_value_table.py](../dynamicemb/key_value_table.py). 
Refer to demo `PyDictStorage` in [uint test](../test/test_batched_dynamic_embedding_tables_v2.py).


## DynamicEmbParameterSharding

The final step of preparation is to generate a `ParameterSharding` for each table, describe (configure) the sharding of a parameter. For dynamic embedding table, `DynamicEmbParameterSharding` will be generated, which includes the parameters required for our embedding lookup module.

*We will not expand `DynamicEmbParameterSharding` here. The following steps demonstrate how to obtain `DynamicEmbParameterSharding` by `DynamicEmbeddingShardingPlanner`.*

```python
const = DynamicEmbParameterConstraints(
  sharding_types=[
    ShardingType.ROW_WISE.value, # dynamicemb embedding table only support to be sharded in row-wise.
  ],
  use_dynamicemb=True, # indicate using dynamicemb.
  dynamicemb_options=DynamicEmbTableOptions(
    global_hbm_for_values=total_hbm_need * cache_ratio
    if caching
    else total_hbm_need,
    initializer_args=DynamicEmbInitializerArgs(
        mode=DynamicEmbInitializerMode.NORMAL
    ),
    score_strategy=DynamicEmbScoreStrategy.STEP,
    caching=caching,
    training=training,
  ),
)

dict_const["user_id"] = const

topology = Topology(
  local_world_size=get_local_size(),
  world_size=dist.get_world_size(),
  compute_device=device.type,
  hbm_cap=hbm_cap,
  ddr_cap=ddr_cap,
  intra_host_bw=intra_host_bw,
  inter_host_bw=inter_host_bw,
)

# Same usage of  torchrec's EmbeddingEnumerator
enumerator = DynamicEmbeddingEnumerator(
  topology=topology,
  constraints=dict_const,
)

# Almost same usage of  torchrec's EmbeddingShardingPlanner, except to input eb_configs, as dynamicemb need EmbeddingConfig info to help to plan. 
planner = DynamicEmbeddingShardingPlanner(
  eb_configs=eb_configs,
  topology=topology,
  constraints=dict_const,
  batch_size=batch_size,
  enumerator=enumerator,
  storage_reservation=HeuristicalStorageReservation(percentage=0.05),
  debug=True,
)

# get plan for all ranks.
# ShardingPlan is a dict, mapping table name to ParameterSharding/DynamicEmbParameterSharding.
plan: ShardingPlan = planner.collective_plan(ec, [sharder], dist.GroupMember.WORLD)
```

# DistributedModelParallel

The final step is to input the `sharder` and `ShardingPlan` to the `DistributedModelParallel`, who will implement the sharded plan through `sharder` and hold the `ShardedDynamicEmbeddingCollection` after sharding. Then you can use `dmp` for model **training** and **evaluation**, just like using `EmbeddingCollection`.

```python
dmp = DistributedModelParallel(
  module=ec,
  device=device,
  # pyre-ignore
  sharders=[sharder],
  plan=plan,
)
```

# Dump/Load and Incremental dump

Dump/Load and incremental dump is different from general module in PyTorch, because dynamicemb's underlying implementation is a hash table instead of a dense `torch.Tensor`.

So dynamicemb provides dedicated interface to load/save models' states, and provide conditional dump to support online training.

Please see `DynamicEmbDump`, `DynamicEmbLoad`, `incremental_dump` in [APIs Doc](../DynamicEmb_APIs.md) for more information.