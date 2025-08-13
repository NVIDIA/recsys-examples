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
| False           | 65536      | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | 0.4965         | 0.4972            | 0.4929             | 0.9901          | 0.0687           | 0.1951              | 0.4059               | 0.5999            |
| False           | 65536      | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | 0.5000         | 0.4999            | 1.1617             | 1.6616          | 0.0691           | 0.2001              | 0.4339               | 0.6347            |
| False           | 65536      | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | 0.5124         | 0.5124            | 0.5376             | 1.0499          | 1.0508           | 1.1495              | 1.282                | 2.4302            |
| False           | 65536      | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | 0.5158         | 0.5157            | 1.2876             | 1.8033          | 1.0916           | 1.2015              | 1.4509               | 2.6543            |
| False           | 1048576    | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | 7.5263         | 7.5274            | 3.6960             | 11.2234         | 0.6011           | 0.8402              | 1.6120               | 2.4558            |
| False           | 1048576    | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | 7.5300         | 7.5305            | 10.2640            | 17.7945         | 0.6012           | 0.8596              | 1.8197               | 2.6794            |
| False           | 1048576    | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | 7.8093         | 7.8095            | 4.4519             | 12.2614         | 15.0906          | 10.8440             | 11.8741              | 22.7194           |
| False           | 1048576    | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | 7.8124         | 7.8129            | 12.5192            | 20.3321         | 15.5863          | 11.2428             | 12.6806              | 23.9257           |