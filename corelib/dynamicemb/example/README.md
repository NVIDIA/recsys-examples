# Dynamicemb Example Introduction

In short, **dynamicemb** provides distributed, high-performance dynamic embedding storage and related functions for training.

How to run:
```shell
export NGPU=1
bash ./run_example.sh
```

- The [example.py](./example.py) will show you how to train and evaluate the embedding module, as well as dump, load and incremental dump the module, and this example also demonstrates how to customize embedding admissions.

- Input distribution examples:
```shell
export NGPU=2

# default input distribution: roundrobin
torchrun --standalone --nproc_per_node=${NGPU} example.py --train

# explicit roundrobin
torchrun --standalone --nproc_per_node=${NGPU} example.py --train --dist_type roundrobin

# opt-in hash-based routing
torchrun --standalone --nproc_per_node=${NGPU} example.py --train --dist_type hash_roundrobin

# continuous routing
torchrun --standalone --nproc_per_node=${NGPU} example.py --train --dist_type continuous

# run through the helper script (arguments are forwarded to example.py)
bash ./run_example.sh --dist_type hash_roundrobin
```

`hash_roundrobin` is an opt-in routing mode intended to reduce sensitivity to pathological raw-key patterns that can break plain modulo-based `roundrobin`.


- For detailed explanations of specific APIs and parameters, please refer to [API Doc](../DynamicEmb_APIs.md).

- For usage of external storage, Refer to demo `PyDictStorage` in [uint test](../test/test_batched_dynamic_embedding_tables_v2.py).

***dynamicemb** supports not only `EmbeddingCollection` but also `EmbeddingBagCollection`. However, due to the requirements of generative recommendations, dynamicemb focuses on performance optimization of `EmbeddingCollection` while providing full functional support for `EmbeddingBagCollection`. And we use `EmbeddingCollection` as an example.*
