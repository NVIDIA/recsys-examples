# HostLayer examples

This directory tests real NVE HostLayer collections: loading, lookup, AOT export,
replay and CPU incremental updates. It covers **both CPU and CUDA callers**.
The CPU-only package is one supported deployment of these examples.

**CPU-only build and test: follow [CPU_BUILD_AND_TEST.md](CPU_BUILD_AND_TEST.md), steps 1–3.**
It uses stock NVE's minimal driverless configuration and DynamicEmb's CPU
inference targets, then checks export/replay, updates and related CPU unit tests.

| File | Purpose | Execution device |
|---|---|---|
| [example.py](example.py) | Load, verify, export and replay a collection | `--device cpu` (default), or `--device cuda:0` with a regular CUDA build |
| [replay.cpp](replay.cpp) | C++ AOT replay; also consumes saved CPU updates | Device index `-1` for CPU; `0` for CUDA with a regular CUDA build |
| [incremental_cpu.py](incremental_cpu.py) | Verify CPU lookup during incremental updates | CPU only |
| [CMakeLists.txt](CMakeLists.txt) | Build the C++ runner | `DYNAMICEMB_CPU_ONLY=ON` for CPU-only PyTorch |

**Storage and execution are separate choices.** `nve_layer_type="host"` keeps
embedding values in CPU memory in every case. `device="cpu"` also runs indexing
and returns outputs on CPU. `device="cuda:0"` uses CUDA keys, indexing and outputs
with NVE's host lookup.

For **CPU-only PyTorch + NVE + cudart**, use the [CPU-only guide](CPU_BUILD_AND_TEST.md).
With the [normal DynamicEmb installation](../../README.md#installation) and NVE
inference bindings already installed, run these examples directly. There is no
separate Host-specific DynamicEmb build. These examples do not depend on HSTU.

## Collection API and supported features

Supported: fused identity, linear hash, single-table identity; FP32/FP16;
plain, sum and mean lookup; embedded or sidecar index state. BitConcat is
unsupported by the current dense pynve HostLayer. HSTU's ranking exporters accept
`--host_embeddings`; HostLayer lookup cannot be captured in a CUDA graph.

```python
import torch
from dynamicemb.exportable_tables import (
    EmbeddingCollectionIndexerType, InferenceEmbeddingCollectionConfig,
    create_inference_embedding_collection,
)

collection = create_inference_embedding_collection(
    tables,  # TorchRec configs or objects with name/num_embeddings/embedding_dim/feature_names
    config=InferenceEmbeddingCollectionConfig(
        indexer_type=EmbeddingCollectionIndexerType.LINEAR_HASH_MAP,
        nve_layer_type="host",
    ),
    pooling_mode=-1,
    device=torch.device("cpu"),
)
collection.load_from_dynamicemb_file(checkpoint_dir)
result = collection(keys, feature_offsets)
```

Loading reads CPU batches, builds the index on CPU, and fills host weights.
CUDA execution receives the completed index; embedding values stay on the host.
The initial source loads once. CPU collections also support incremental updates
through the coordinator/subscriber below. Duplicate source
keys reuse their row and the last value wins. Capacity overflow, reserved hash
keys (-1 through -4), and invalid dense source IDs raise errors. Inference
misses return zero. A full checkpoint load includes every saved shard.

## Build and run

Use the [CPU-only build-to-test guide](CPU_BUILD_AND_TEST.md) for CPU-only PyTorch. It uses
`example.py`, `incremental_cpu.py`, `replay.cpp` and `CMakeLists.txt` from this
directory; `../../setup.py` selects CPU inference with `DYNAMICEMB_CPU_ONLY=1`.

With a normal CUDA-enabled DynamicEmb/NVE installation, `example.py` also accepts
`--device cuda:0`. Export CPU and CUDA graphs into separate package directories.
A C++ runner built against that normal installation accepts device index `0` for
CUDA and `-1` for CPU. Keep NVE resources/layers and indexer directories alive
until the AOTI loader is destroyed, as the example shows.

Host collections require NVE 26.09 or later; use the repository-pinned NVE 26.10
and matching headers/libraries. See [NVE compatibility](../../docs/nve_tier_updates.md).

## Incremental updates on CPU

| Indexer | Loading | Mapping |
| --- | --- | --- |
| Identity, single table | Inference continues (experimental; see below) | Overwrite fixed rows. |
| FusedIdentity | Inference continues (experimental; see below) | Overwrite within reserved table ranges. |
| LinearHashMap | Pause and drain CPU readers | Update/insert/erase mappings; reuse free rows within capacity. Sidecar required. |

The intended fixed-mapping contract permits stale/mixed lookup values during
updates and checks final values after completion. With stock NVE 26.10,
overlapping lookup/update uses ordinary `memcpy`, which has a C++ data race.
Concurrent runs are an experiment, not a guarantee of safe execution or
final-value visibility. Identity/FusedIdentity examples run lookup concurrently
with updates; LinearHashMap updates require paused/drained readers.
See [the explanation, word-level example, and validation results](../../docs/host_cpu_update_concurrency.md).
Keep the storage allocation fixed and use one update writer. Raw weight writes,
serialization, and accumulation must happen outside concurrent serving.

```text
CPU delta → UpdateCoordinator → row/value message (+ paused hash snapshot)
          → UpdateSubscriber → HostLayer.update
CPU inference → unchanged indexer → HostLayer.lookup, concurrently for fixed mappings
```

`EmbeddingCollectionUpdateCoordinator.open(..., device=cpu)` reads
the exported package. Call `apply_delta(collection_id, delta)` with a CPU
`DeltaDumpResult`, then pass `update.to_json()` to the subscriber. Use one update
writer and apply messages once, in snapshot order. Each subscriber has its own
host storage; the coordinator does not need Redis. Exported packages must include
per-table capacity metadata.

```python
update = coordinator.apply_delta(collection_id, delta)
subscriber.apply_incremental_load(update.to_json())  # Identity / FusedIdentity
# For LinearHashMap, after the caller pauses and drains inference:
subscriber.apply_incremental_load(update.to_json(), inference_paused=True)
```

Duplicate upserts use the last value. A final dumped value wins over an earlier
eviction/erase of the same key. Other removals zero fixed rows or remove hash
mappings. Invalid keys, malformed values and capacity overflow are rejected
before publication. A hash update can reuse a row freed by that same delta;
surviving keys keep their rows. CPU hash publication and acknowledgement complete
without waiting for a subsequent inference call.

Run [step 3 of the CPU-only guide](CPU_BUILD_AND_TEST.md#3-test)
for Python/C++ update checks with all three indexers. `--embedded` works for fixed
mappings; `--dtype float16 --pooling 2` selects pooled FP16. CUDA HostLayer
incremental loading is outside this CPU update path.
