# DynamicEmb

DynamicEmb is a Python package that provides model-parallel dynamic embedding tables and embedding lookup functionalities for TorchREC, specifically targeting the sparse training aspects of recommendation systems. DynamicEmb uses a GPU-optimized scored hash table backend to store key-value (feature-embedding) pairs in the high-bandwidth memory (HBM) of GPUs as well as in host memory.

The lookup kernel algorithms implemented in DynamicEmb primarily leverage portions of the algorithms from the [EMBark](https://dl.acm.org/doi/abs/10.1145/3640457.3688111) paper (Embedding Optimization for Training Large-scale Deep Learning Recommendation Systems with EMBark).


## Table of Contents

- [Features](#features)
- [Pre-requisites](#pre-requisites)
  - [Version Compatibility](#version-compatibility)
- [Installation](#installation)
- [DynamicEmb APIs](#dynamicemb-apis)
- [Usage Notes](#usage-notes)
  - [DynamicEmb Insertion Behavior Checking Modes](#dynamicemb-insertion-behavior-checking-modes)
- [Getting Started](#getting-started)
- [Future Plans](#future-plans)
- [Acknowledgements](#acknowledgements)

## Features

- **Dynamic Embedding Table Support**: DynamicEmb supports embedding tables backed by hash tables, allowing for optimal utilization of both GPU memory and host memory within the system. Hash tables can accept any specified `indices` type values, unlike static tables which only support index values.

- **Seamless Integration with TorchREC**: DynamicEmb inherits the API from TorchREC, ensuring that its usage is largely consistent with TorchREC. Users can easily modify their existing code to run recommendation system models with dynamic embedding tables alongside TorchREC.
**dynamicemb** provides a high-performance **hash table** to support dynamic embedding and leverages **torchrec** to implement sharding logic on multiple GPUs. This explains why dynamicemb largely reuses the user interface of torchrec while adding some new configuration options related to dynamic embedding.

- **Embedded in DistributedGR Repository Supporting Generative-Recommenders(GR) Models**: Currently, DynamicEmb is integrated into the DistributedGR repository, serving as an embedding backend for GR models.

- Support for creating dynamic embedding tables within `EmbeddingBagCollection` and `EmbeddingCollection` in TorchREC, allowing for embedding storage and lookup, and enabling coexistence with native Torch embedding tables within Torch models.

- **Pooling Mode Support**: DynamicEmb supports `SUM`, `MEAN`, and `NONE` (sequence) pooling modes with fused CUDA kernels for both forward and backward passes. Tables with different embedding dimensions (mixed-D) are fully supported in pooling mode.

- Support for optimizer types: `EXACT_SGD`,`ADAM`,`EXACT_ADAGRAD`,`EXACT_ROWWISE_ADAGRAD`.

- Support for automatically parallel `dump`/`load` of embedding weights in dynamic embedding tables.


## Pre-requisites

Currently, dynamicemb is integrated into latest TorchRec main branch, while TorchRec requires FBGEMM_GPU main branch, both of which are not packaged. Temporarily, installing from source code is required. Before installing the 2 libraries, make sure you have PyTorch CUDA version installed (refer to [PyTorch documentation](https://pytorch.org/get-started/locally/)).

1. **FBGEMM_GPU**

Please follow below instructions to build fbgemm_gpu from source code. It may take minutes to finish. 

```bash
# install setup tools
pip install --no-cache setuptools==69.5.1 setuptools-git-versioning scikit-build
git clone --recursive -b main https://github.com/pytorch/FBGEMM.git fbgemm
cd fbgemm/fbgemm_gpu
git checkout 642ccb980d05aa1be00ccd131c5991b0914e2e64
# please specify the proper TORCH_CUDA_ARCH_LIST for your ENV
python setup.py bdist_wheel --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="8.0 9.0"
python setup.py install --package_variant=cuda -DTORCH_CUDA_ARCH_LIST="8.0 9.0"
```

Once above processing is done, please execute `python -c 'import fbgemm_gpu'` to make sure it's properly installed.

2. **TorchRec**

> torchrec >= v1.2.0

Thanks to the torchrec team for their [support](https://github.com/meta-pytorch/torchrec/commit/6aaf1fa72e884642f39c49ef232162fa3772055e), torchrec v1.2.0 added support for custom embedding lookup module.

After fbgemm_gpu is installed, you can install TorchRec with below commands.

```bash
# torchrec depends on below 2 libs
pip install --no-deps tensordict orjson
git clone --recursive -b main https://github.com/pytorch/torchrec.git torchrec
cd torchrec && git checkout 6aaf1fa72e884642f39c49ef232162fa3772055e
# with --no-deps to prevent from installing dependencies
pip install --no-deps .
```

Once above processing is done, please execute `python -c 'import torchrec'` to make sure it's properly installed.

## Installation

To install DynamicEmb, please use the following command:

```bash
python setup.py install
```

## DynamicEmb APIs

Regarding how to use the DynamicEmb APIs and their parameters, please refer to the [DynamicEmb_APIs.md](./DynamicEmb_APIs.md) file in the same folder as this document.

## Usage Notes

1. Only the following optimizer types are supported: `EXACT_SGD`, `ADAM`, `EXACT_ADAGRAD`,`EXACT_ROWWISE_ADAGRAD`. This behavior is to maintain consistency with TorchREC.
2. The sharding method for dynamic embedding tables is always `row-wise sharding`, which will be evenly distributed across all GPUs within the TorchREC scope, unlike the `table-wise` and other sharding methods in TorchREC.
3. The allocated memory for dynamic embedding tables may have slight differences from the specified `num_embeddings` because each dynamic embedding table must set a capacity as a power of 2. This will be automatically calculated by the code, so please ensure that `num_embeddings` is aligned to a power of 2 when applying.
4. The lookup process for each dynamic embedding table incurs additional overhead from unique or radix sort operations. Therefore, if you request a large number of small dynamic embedding tables for lookup, the performance will be poor. Since the lookup range of dynamic embedding tables is particularly large (using the entire range of `int64_t`), it is recommended to create one large embedding table and perform a fused lookup for multiple features.
5. Although dynamic embedding tables can be trained together with TorchREC tables, they cannot be fused together for embedding lookup. Therefore, it is recommended to select dynamic embedding tables for all model-parallel tables during training.
6. DynamicEmb supports training with TorchREC's `EmbeddingBagCollection` (pooling mode: SUM/MEAN) and `EmbeddingCollection` (sequence mode). Both modes use fused CUDA kernels for embedding lookup and gradient reduction. Tables with different embedding dimensions are supported in pooling mode.

### DynamicEmb Insertion Behavior Checking Modes

DynamicEmb uses a hashtable as the backend. If the embedding table capacity is small and the number of indices in a single feature is large, it is easy for too many indices to be allocated to the same hash table bucket in one lookup, resulting in the inability to insert indices into the hashtable. DynamicEmb resolves this issue by setting the lookup results of indices that cannot be inserted to 0.

Fortunately, in a hashtable with a large capacity, such insertion failures are very rare and almost never occur. This issue is more frequent in hashtables with small capacities, which can affect training accuracy. Therefore, we do not recommend using dynamic embedding tables for very small embedding tables.

To prevent this behavior from affecting training without user awareness, DynamicEmb provides a safe check mode. Users can set whether to enable safe check when configuring `DynamicEmbTableOptions`. Enabling safe check will add some overhead, but it can provide insights into whether the hash table frequently fails to insert indices. If the number of insertion failures is high and the proportion of affected indices is large, it is recommended to either increase the dynamic embedding capacity or avoid using dynamic embedding tables for small embedding tables.

#### Example

```python
from dynamicemb import DynamicEmbTableOptions, DynamicEmbCheckMode

# Configure the DynamicEmbTableOptions with safe check mode enabled
table_options = DynamicEmbTableOptions(
    safe_check_mode=DynamicEmbCheckMode.WARNING
)

# Use the table_options in your dynamic embedding setup
# ...
```

## Getting Started

We provide benchmark and unit test code to demonstrate how to use DynamicEmb. Please visit the benchmark and test folders. Below is a pseudocode example demonstrating how to convert TorchREC code to use DynamicEmb.

To get started with DynamicEmb, we highly recommend checking out the [example.py](./example/example.py). It walks you through the entire process of modifying your code and setting up a training script with model parallelism. You can quickly experiment with DynamicEmb and see its benefits in a practical setting.

## Future Plans

1. Support the latest version of TorchREC and continuously follow TorchREC's version updates.
2. Support the separation of backward and optimizer update (required by certain large language model frameworks like Megatron), to better support large-scale GR training.
3. Add more shard types for dynamic embedding tables, including `table-wise`, `table-row-wise` and `column-wise`.

## Acknowledgements

We would like to thank the Meta team and specially [Huanyu He](https://github.com/TroyGarden) for their support in [TorchRec](https://github.com/pytorch/torchrec).

We also acknowledge the [HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV) project, which inspired the scored hash table design used in DynamicEmb.