# Gin Configurable Interfaces Documentation

This document provides comprehensive documentation for all configurable hypara-params that used by both inference and training


## 1. TrainerArgs - Trainer Configuration

Training-related parameters and settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_batch_size` | int | - | **Required**. Batch size per GPU. When TP is enabled, the theoretical batch size is (train_batch_size × tp_size) |
| `eval_batch_size` | int | - | **Required**. Evaluation batch size |
| `eval_interval` | int | 100 | Evaluation interval in iterations |
| `log_interval` | int | 100 | Logging interval in iterations |
| `max_train_iters` | Optional[int] | None | Maximum training iterations |
| `max_eval_iters` | Optional[int] | None | Maximum evaluation iterations |
| `seed` | int | 1234 | Random seed |
| `profile` | bool | False | Enable profiling |
| `profile_step_start` | int | 100 | Profiling start step |
| `profile_step_end` | int | 200 | Profiling end step |
| `ckpt_save_interval` | int | -1 | Checkpoint save interval, -1 means no checkpoint saving |
| `ckpt_save_dir` | str | "./checkpoints" | Checkpoint save directory |
| `ckpt_load_dir` | str | "" | Checkpoint load directory |
| `pipeline_type` | str | "native" | Pipeline overlap type: `none` (no overlap), `native` (overlap h2d, input dist, fwd+bwd), `prefetch` (includes prefetch overlap) |

---


## 2. EmbeddingArgs - Embedding Configuration

Base embedding layer configuration parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_names` | List[str] | - | **Required**. List of feature names |
| `table_name` | str | - | **Required**. Embedding table name |
| `item_vocab_size_or_capacity` | int | - | **Required**. For dynamic embedding: capacity; for static embedding: vocabulary size |
| `sharding_type` | str | "None" | Sharding type, must be "data_parallel" or "model_parallel" |

---

## 3. DynamicEmbeddingArgs - Dynamic Embedding Configuration

Extends `EmbeddingArgs` with dynamic embedding-specific parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_hbm_for_values` | Optional[int] | None | Global HBM size in bytes (highest priority) |
| `item_vocab_gpu_capacity` | Optional[float] | None | Item vocabulary GPU capacity (second priority) |
| `item_vocab_gpu_capacity_ratio` | Optional[float] | None | Item vocabulary GPU capacity ratio (lowest priority) |
| `evict_strategy` | str | "lru" | Eviction strategy: "lru" or "lfu" |
| `caching` | bool | False | Enable caching on HMB. When caching is enabled, the global_hbm_for_values indicates the cache size |

**Note**: `sharding_type` is automatically set to "model_parallel"

**Precedence**: The first 3 params can be used for setting the HBM size for dynamic embedding, but there is a precedence relationship:   `global_hbm_for_values` > `item_vocab_gpu_capacity` > `item_vocab_gpu_capacity_ratio`. When only `item_vocab_gpu_capacity_ratio` is given, `item_vocab_gpu_capacity = item_vocab_gpu_capacity_ratio * item_vocab_size_or_capacity` and `global_hbm_for_values` are deduced based on the optimizer and embedding dims.

**Note**: A table could be only one of type EmbeddingArgs or DynamicEmbeddingArgs.

**Note**: When movielen\* or kuairand\* dataset are used,  DynamicEmbeddingArgs/EmbeddingArgs are predefined. See [get_dataset_and_embedding_args() func](../hstu/training/trainer/utils.py)

---

## 4. DatasetArgs - Dataset Configuration

Dataset-related configuration parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | - | **Required**. Dataset name |
| `max_sequence_length` | int | - | **Required**. Maximum sequence length |
| `dataset_path` | Optional[str] | None | Path to dataset |
| `max_num_candidates` | int | 0 | Maximum number of candidates |
| `shuffle` | bool | False | Whether to shuffle data |

**Note**: `dataset_path` could be none if your dataset is preprocessed and moved under <root-to-project>/hstu/tmp_data folder or you're running with `BenchmarkDatasetArgs` which is a in-memory random data generator. Please refer to [example](../hstu/training/configs/benchmark_ranking.gin).


---

## 5. FeatureArgs - Feature Configuration

Feature-specific configuration parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_names` | List[str] | - | **Required**. List of feature names |
| `max_sequence_length` | int | - | **Required**. Maximum sequence length |
| `is_jagged` | bool | False | Whether features are jagged (variable length) |
 FeatureArgs and DatasetArgs

**Note**: `FeatureArgs` are only used when the dataset is of `BenchmarkDatasetArgs`.

---
## 6. BenchmarkDatasetArgs - Benchmark Dataset Configuration

Configuration for benchmark datasets combining features and embeddings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_args` | List[FeatureArgs] | - | **Required**. List of feature arguments |
| `embedding_args` | List[Union[EmbeddingArgs, DynamicEmbeddingArgs]] | - | **Required**. List of embedding arguments |
| `item_feature_name` | str | - | **Required**. Item feature name |
| `contextual_feature_names` | List[str] | - | **Required**. List of contextual feature names |
| `action_feature_name` | Optional[str] | None | Action feature name |
| `max_num_candidates` | int | 0 | Maximum number of candidates |

---

## 7. NetworkArgs - Network Architecture Configuration

Neural network architecture parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | int | - | **Required**. Number of layers |
| `hidden_size` | int | - | **Required**. Hidden layer size |
| `num_attention_heads` | int | - | **Required**. Number of attention heads |
| `kv_channels` | int | - | **Required**. Key-value channels |
| `hidden_dropout` | float | 0.2 | Hidden layer dropout rate |
| `norm_epsilon` | float | 1e-5 | Normalization epsilon |
| `is_causal` | bool | True | Use causal attention mask |
| `dtype_str` | str | "bfloat16" | Data type: "bfloat16" or "float16" |
| `kernel_backend` | str | "cutlass" | Kernel backend: "cutlass", "triton", or "pytorch" |
| `target_group_size` | int | 1 | Target group size |
| `num_position_buckets` | int | 8192 | Number of position buckets |
| `recompute_input_layernorm` | bool | False | Recompute input layer normalization |
| `recompute_input_silu` | bool | False | Recompute input SiLU activation |
| `item_embedding_dim` | int | -1 | Item embedding dimension |
| `contextual_embedding_dim` | int | -1 | Contextual embedding dimension |

---

## 8. OptimizerArgs - Optimizer Configuration

Optimizer-related parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_str` | str | - | **Required**. Optimizer name |
| `learning_rate` | float | - | **Required**. Learning rate |
| `adam_beta1` | float | 0.9 | Adam optimizer beta1 parameter |
| `adam_beta2` | float | 0.999 | Adam optimizer beta2 parameter |
| `adam_eps` | float | 1e-8 | Adam optimizer epsilon parameter |

---

## 9. TensorModelParallelArgs - Tensor Model Parallelism Configuration

Tensor model parallelism settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_model_parallel_size` | int | 1 | Tensor model parallel size (number of GPUs for model sharding) |

**Note**: The data parallel size is deduced based on the `world_size` and `tensor_model_parallel_size`.

---

## 10. RankingArgs - Ranking Task Configuration

Configuration specific to ranking tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction_head_arch` | List[int] | None | **Required**. Prediction head architecture (list of layer sizes) |
| `prediction_head_act_type` | str | "relu" | Prediction head activation type: "relu" or "gelu" |
| `prediction_head_bias` | bool | True | Whether to use bias in prediction head |
| `num_tasks` | int | 1 | Number of tasks (for multi-task learning) |
| `eval_metrics` | Tuple[str, ...] | ("AUC",) | Evaluation metrics tuple |

---

## 11. RetrievalArgs - Retrieval Task Configuration

Configuration specific to retrieval tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_negatives` | int | -1 | Number of negative samples |
| `temperature` | float | 0.05 | Temperature parameter for similarity scoring |
| `l2_norm_eps` | float | 1e-6 | Epsilon value for L2 normalization |
| `eval_metrics` | Tuple[str, ...] | ("HR@10", "NDCG@10") | Evaluation metrics tuple (Hit Rate, NDCG) |

---

## Usage Examples

### Example 1: Basic Configuration

```python
# In your .gin config file

# Trainer configuration
TrainerArgs.train_batch_size = 256
TrainerArgs.eval_batch_size = 512
TrainerArgs.max_train_iters = 10000
TrainerArgs.pipeline_type = "prefetch"

# Network configuration
NetworkArgs.num_layers = 4
NetworkArgs.hidden_size = 256
NetworkArgs.num_attention_heads = 8
NetworkArgs.kv_channels = 32
NetworkArgs.dtype_str = "bfloat16"

# Optimizer configuration
OptimizerArgs.optimizer_str = "adam"
OptimizerArgs.learning_rate = 0.001
```

### Example 2: Ranking Task Configuration

```python
# Dataset
DatasetArgs.dataset_name = "criteo"
DatasetArgs.max_sequence_length = 128

# Ranking model
RankingArgs.prediction_head_arch = [512, 256, 1]
RankingArgs.prediction_head_act_type = "relu"
RankingArgs.eval_metrics = ("AUC")

# Embeddings
EmbeddingArgs.feature_names = ["item_id", "category"]
EmbeddingArgs.table_name = "item_table"
EmbeddingArgs.item_vocab_size_or_capacity = 1000000
EmbeddingArgs.sharding_type = "data_parallel"
```

### Example 3: Retrieval Task with Dynamic Embedding

```python
# Retrieval configuration
RetrievalArgs.num_negatives = 100
RetrievalArgs.temperature = 0.05
RetrievalArgs.eval_metrics = ("HR@10", "HR@50", "NDCG@10")

# Dynamic embedding
DynamicEmbeddingArgs.feature_names = ["user_id", "item_id"]
DynamicEmbeddingArgs.table_name = "user_item_table"
DynamicEmbeddingArgs.item_vocab_size_or_capacity = 10000000
DynamicEmbeddingArgs.item_vocab_gpu_capacity_ratio = 0.1
DynamicEmbeddingArgs.evict_strategy = "lru"
DynamicEmbeddingArgs.caching = True
```

---

