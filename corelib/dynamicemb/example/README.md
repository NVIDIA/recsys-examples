# Dynamicemb Example Introduction

In short, **dynamicemb** provides distributed, high-performance dynamic embedding storage and related functions for training.

How to run:
```shell
export NGPU=1
bash ./run_example.sh
```

- The [example.py](./example.py) will show you how to train and evaluate the embedding module, as well as dump, load and incremental dump the module.


- For detailed explanations of specific APIs and parameters, please refer to [API Doc](../DynamicEmb_APIs.md).

- For usage of external storage, Refer to demo `PyDictStorage` in [uint test](../test/test_batched_dynamic_embedding_tables_v2.py).

***dynamicemb** supports not only `EmbeddingCollection` but also `EmbeddingBagCollection`. However, due to the requirements of generative recommendations, dynamicemb focuses on performance optimization of `EmbeddingCollection` while providing full functional support for `EmbeddingBagCollection`. And we use `EmbeddingCollection` as an example.*