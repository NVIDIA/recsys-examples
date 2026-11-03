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

- Multi-worker data loading (`--num_workers`):
```shell
export NGPU=2

# default: load data in the main process (no worker subprocesses)
torchrun --standalone --nproc_per_node=${NGPU} example.py --train

# offload dataset reads + collate to 4 background worker processes per rank
torchrun --standalone --nproc_per_node=${NGPU} example.py --train --num_workers 4

# also works through the helper script
bash ./run_example.sh --num_workers 4
```

`--num_workers` controls how many subprocesses each rank uses to prefetch and
collate batches (default `0`, i.e. loading happens in the main process).

When `--num_workers > 0` the example builds the `DataLoader` with the **`spawn`**
multiprocessing start method (plus `persistent_workers=True`). This is required
because the main process has already initialized a CUDA context
(`torch.cuda.set_device`): the default `fork` start method would hand each worker
an unusable copy of that context and trigger `RuntimeError: initialization error`
on the worker's first CUDA touch. `spawn` instead launches a fresh interpreter
per worker with no inherited CUDA state. The workers only do CPU work (dataset
reads and `collate_fn`); the host-to-device copy stays in the main process, so
the workers never need their own CUDA context.

Notes:
- `spawn` re-imports `example.py` in every worker, so its startup is heavier than
  `fork`; for the small MovieLens dataset the speedup from extra workers is
  usually marginal. Treat `--num_workers > 0` mainly as a reference for safely
  enabling multi-worker loading in CUDA + distributed setups.
- Pick a value no larger than the CPU cores available per rank.

- Erasing keys that have not been seen for a while (`--erase_interval`, `--erase_retention`):
```shell
export NGPU=2

# every 100 batches, erase the keys that were not seen in the last 1000 batches
torchrun --standalone --nproc_per_node=${NGPU} example.py --train --erase_interval 100 --erase_retention 1000

# the same by the device clock: keys not seen for the last 60 seconds
torchrun --standalone --nproc_per_node=${NGPU} example.py --train --score_strategy timestamp --erase_interval 100 --erase_retention 60
```

`--erase_interval N` asks, every N batches, for an [erase_by_score_async](../DynamicEmb_APIs.md#erase_by_score_async)
of the keys whose score is below a threshold (default `0`: off). The call returns at once and the scan runs in the
background of the training. `--erase_retention K` sets the threshold: a key is kept if it was seen within the last `K`
batches (`--score_strategy step`, the default) or `K` seconds (`--score_strategy timestamp`). The example reads the
tables' current score with `get_score` and erases what is below `score - K`, so nothing is erased until the run is
older than `K`.

Notes:
- The MovieLens tables are small and nearly every key appears in every epoch, so with a `--erase_retention` longer than
  an epoch nothing is old enough to erase. To see keys erased, use a retention shorter than an epoch.
- Each epoch ends by waiting for the erase in flight, since an erase must not overlap an evaluation or a dump, and prints
  what this rank erased (the call is local: each rank erases what it holds).
- `--erase_interval` needs `--score_strategy step` or `timestamp` (not `lru_lfu`, whose `(TIMESTAMP, LFU)` tables cannot
  be erased by either score), and cannot be combined with `--external_storage` or with `--incremental_dump`: an erase by
  score is not recorded, so the deltas of a table it removed keys from cannot be replayed.

- For detailed explanations of specific APIs and parameters, please refer to [API Doc](../DynamicEmb_APIs.md).

- For usage of external storage, refer to the `PyDictStorage` demo in the [unit test](../test/test_batched_dynamic_embedding_tables_v2.py).

***dynamicemb** supports not only `EmbeddingCollection` but also `EmbeddingBagCollection`. However, due to the requirements of generative recommendations, dynamicemb focuses on performance optimization of `EmbeddingCollection` while providing full functional support for `EmbeddingBagCollection`. And we use `EmbeddingCollection` as an example.*
