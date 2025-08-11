# Dynamic Embedding Benchmark

## Overview

This folder contains benchmarks about dynamicemb.

## 1.Benchmark EmbeddingCollection

In this benchmark, we provide a simple performance test for dynamic embedding using 8 GPUs. The test utilizes the embedding table from DLRM and performs embedding table fusion to create a large embedding table, followed by lookups for 26 features.

### How to run

```bash
bash ./benchmark/benchmark_embedding_collection.sh <use_index_dedup> <use_dynamic_embedding> <batch_size>
```

#### Parameters

- `<use_index_dedup>`: A boolean flag to enable or disable index deduplication before data distribution.
  - **True**: Enables index deduplication, reducing communication overhead.
  - **False**: Disables index deduplication.
  - **Default**: True.

- `<use_dynamic_embedding>`: A boolean flag to enable or disable the use of dynamic embedding tables.
  - **True**: Enables dynamic embedding tables.
  - **False**: Uses static embedding tables from TorchREC.
  - **Default**: True.

- `<batch_size>`: The global batch size for processing during the benchmark.
  - **Default**: 65536.

### Test Results

In this benchmark, we primarily focus on the performance of embedding collection and deduplication. The tests were conducted on a single node with 8 H100 GPUs connected via NVSwitch. Below are the performance results:

| Configuration               | TorchREC Raw Table (ms) | Dynamic Embedding Table (ms) |
|-----------------------------|-------------------------|-------------------------------|
| Open Dedup, Batch Size 65536 | 14.88                   | 21.56                         |
| Close Dedup, Batch Size 65536 | 23.99                   | 28.47                         |

These results indicate the time taken to perform the embedding collection and deduplication operations under the specified configuration.

During the embedding lookup process, dynamic embedding incurs some performance overhead compared to TorchREC's raw table. However, these overheads diminish when considered within the context of the entire end-to-end model.

## 2.Benchmark BatchedDynamicEmbeddingTables

In this benchmark, we test the forward and backward overhead of `BatchedDynamicEmbeddingTables` (torch.nn.Module contains batched embedding tables) on a single GPU.

### How to run

```bash
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh
```

### Test Results

We test the `BatchedDynamicEmbeddingTables` under different capacities with the same HBM consumption for embeddings, leading to different HBM proportion.

The overhead(ms) on H100 80GB HBM3:

| use_index_dedup | batch_size | num_embeddings_per_feature | hbm_for_embeddings | optimizer_type | feature_distribution-alpha | embedding_dim | num_iterations | cache_algorithm | eval(torchrec) | forward(torchrec) | backward(torchrec) | train(torchrec) | eval(dynamicemb) | forward(dynamicemb) | backward(dynamicemb) | train(dynamicemb) |
| --------------- | ---------- | -------------------------- | ------------------ | -------------- | -------------------------- | ------------- | -------------- | --------------- | -------------- | ----------------- | ------------------ | --------------- | ---------------- | ------------------- | -------------------- | ----------------- |
| False           | 65536      | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | 0.4965         | 0.4972            | 0.4929             | 0.9901          | 0.2463           | 0.2488              | 0.4163               | 0.6651            |
| False           | 65536      | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | 0.5000         | 0.4999            | 1.1617             | 1.6616          | 0.2517           | 0.2514              | 0.4319               | 0.6833            |
| False           | 65536      | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | 0.5124         | 0.5124            | 0.5376             | 1.0499          | 1.2037           | 1.2038              | 1.2844               | 2.4882            |
| False           | 65536      | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | 0.5158         | 0.5157            | 1.2876             | 1.8033          | 1.2541           | 1.2537              | 1.4530               | 2.7068            |
| False           | 1048576    | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | 7.5263         | 7.5274            | 3.6960             | 11.2234         | 1.4483           | 1.4542              | 1.6121               | 3.0662            |
| False           | 1048576    | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | 7.5300         | 7.5305            | 10.2640            | 17.7945         | 1.4518           | 1.4552              | 1.8184               | 3.2736            |
| False           | 1048576    | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | 7.8093         | 7.8095            | 4.4519             | 12.2614         | 11.8942          | 11.9063             | 12.3404              | 24.2467           |
| False           | 1048576    | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | 7.8124         | 7.8129            | 12.5192            | 20.3321         | 12.3072          | 12.3132             | 13.0427              | 25.3560           |