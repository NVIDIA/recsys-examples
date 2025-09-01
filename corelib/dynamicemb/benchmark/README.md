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

| caching | table_version | batch_size | num_embeddings_per_feature | hbm_for_embeddings | optimizer_type | feature_distribution-alpha | embedding_dim | num_iterations | cache_algorithm | use_index_dedup | eval(torchrec) | forward(torchrec) | backward(torchrec) | train(torchrec) | eval(dynamicemb) | forward(dynamicemb) | backward(dynamicemb) | train(dynamicemb) |
| ------- | ------------- | ---------- | -------------------------- | ------------------ | -------------- | -------------------------- | ------------- | -------------- | --------------- | --------------- | -------------- | ----------------- | ------------------ | --------------- | ---------------- | ------------------- | -------------------- | ----------------- |
| True    | 2             | 65536      | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 0.2908         | 0.2897            | 0.2335             | 0.5232          | 0.2395           | 0.2604              | 1.0207               | 1.2811            |
| False   | 2             | 65536      | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 0.4966         | 0.4965            | 0.4949             | 0.9914          | 0.1687           | 0.2971              | 0.4552               | 0.7523            |
| False   | 1             | 65536      | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 0.4967         | 0.4971            | 0.4917             | 0.9888          | 0.0692           | 0.1957              | 0.4099               | 0.6056            |
| True    | 2             | 65536      | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 0.2943         | 0.2943            | 0.9045             | 1.1989          | 0.2397           | 0.2605              | 1.2278               | 1.4883            |
| False   | 2             | 65536      | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 0.4994         | 0.5003            | 1.1593             | 1.6596          | 0.1673           | 0.2949              | 0.4750               | 0.7698            |
| False   | 1             | 65536      | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 0.5001         | 0.4998            | 1.1651             | 1.6650          | 0.0691           | 0.2008              | 0.4333               | 0.6342            |
| True    | 2             | 65536      | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 0.2685         | 0.2689            | 0.3429             | 0.6118          | 0.2420           | 0.2653              | 1.1501               | 1.4153            |
| False   | 2             | 65536      | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 0.5125         | 0.5124            | 0.5370             | 1.0494          | 1.1706           | 1.2602              | 1.3229               | 2.5830            |
| False   | 1             | 65536      | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 0.5125         | 0.5125            | 0.5409             | 1.0534          | 1.0534           | 1.1528              | 1.2851               | 2.4379            |
| True    | 2             | 65536      | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 0.2677         | 0.2681            | 1.0812             | 1.3493          | 0.2430           | 0.2643              | 1.3368               | 1.6011            |
| False   | 2             | 65536      | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 0.5154         | 0.5152            | 1.2836             | 1.7988          | 1.2187           | 1.3053              | 1.4750               | 2.7803            |
| False   | 1             | 65536      | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 0.5155         | 0.5155            | 1.2864             | 1.8018          | 1.0910           | 1.1998              | 1.4233               | 2.6231            |
| True    | 2             | 1048576    | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 2.6006         | 2.6006            | 0.7010             | 3.3016          | 1.0290           | 1.0483              | 3.6262               | 4.6745            |
| False   | 2             | 1048576    | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 7.5234         | 7.5251            | 3.6853             | 11.2103         | 0.9815           | 1.2196              | 1.6465               | 2.8661            |
| False   | 1             | 1048576    | 8388608                    | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 7.5269         | 7.5277            | 3.6901             | 11.2178         | 0.6010           | 0.8442              | 1.6158               | 2.4600            |
| True    | 2             | 1048576    | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 2.6029         | 2.6029            | 7.1274             | 9.7304          | 1.0307           | 1.0503              | 4.3007               | 5.3510            |
| False   | 2             | 1048576    | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 7.5282         | 7.5296            | 10.2938            | 17.8234         | 0.9841           | 1.2214              | 1.8516               | 3.0730            |
| False   | 1             | 1048576    | 8388608                    | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 7.5275         | 7.5280            | 10.2635            | 17.7914         | 0.6015           | 0.8576              | 1.8129               | 2.6705            |
| True    | 2             | 1048576    | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 2.7751         | 2.7762            | 0.7709             | 3.5471          | 3.1945           | 3.2183              | 7.4745               | 10.6928           |
| False   | 2             | 1048576    | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 7.8065         | 7.8077            | 4.4502             | 12.2580         | 16.4718          | 11.8067             | 12.4042              | 24.2109           |
| False   | 1             | 1048576    | 67108864                   | 4294967296         | sgd            | pow-law-1.05               | 128           | 100            | lru             | False           | 7.8107         | 7.8114            | 4.4515             | 12.2629         | 15.8456          | 11.3021             | 12.3373              | 23.6394           |
| True    | 2             | 1048576    | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 2.7659         | 2.7680            | 8.6259             | 11.3939         | 3.3010           | 3.3228              | 8.1585               | 11.4812           |
| False   | 2             | 1048576    | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 7.8090         | 7.8114            | 12.5489            | 20.3603         | 16.9604          | 12.1736             | 13.1217              | 25.2953           |
| False   | 1             | 1048576    | 67108864                   | 12884901888        | adam           | pow-law-1.05               | 128           | 100            | lru             | False           | 7.8099         | 7.8106            | 12.5173            | 20.3279         | 16.3222          | 11.6924             | 13.0711              | 24.7636           |