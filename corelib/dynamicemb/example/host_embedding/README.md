# HostLayer collections

`nve_layer_type="host"` stores embedding values on the host. `device="cpu"`
executes the whole collection on CPU; `device="cuda:0"` uses CUDA keys,
indexing and outputs with NVE's host lookup. HSTU's dense model remains on CUDA.

Supported: fused identity, linear hash, single-table identity; FP32/FP16;
plain, sum and mean lookup; embedded or sidecar index state. BitConcat is
unsupported by the current dense pynve HostLayer.

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

Use the repository-pinned NVE build (26.10 plus the concurrent host-row patch)
and the repository's inference bindings. For a machine without an
NVIDIA driver, build NVE with `PYNVE_DRIVERLESS_BUILD=1 PYNVE_DISABLE_MPI=1`
and keep DynamicEmb inference updates enabled (the default). Set
`-DNVE_LIB_DIR` to that driverless build.
A CUDA toolkit and CUDA-capable PyTorch build are still used to build these
bindings; CPU execution does not require a GPU or driver. Importing the
collection does not import TorchRec or the DynamicEmb training extension.

From `corelib/dynamicemb`, with NVE on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
# Set TORCH_CUDA_ARCH_LIST to the deployment GPU architecture when building.
cmake -S . -B torch_binding_build -DNVE_ROOT=/path/to/nv-embedding-cache \
  -DNVE_LIB_DIR=/path/to/pynve -DNVTX_INCLUDE_DIR=/path/to/NVTX/c/include
cmake --build torch_binding_build --parallel
python example/host_embedding/example.py --device cpu --package-dir /tmp/host-cpu
python example/host_embedding/example.py --device cpu --package-dir /tmp/host-cpu --replay
```

Use `--device cuda:0` to export a separate CUDA package. CUDA callers must
select the same current device with `torch.cuda.set_device(device)` before
construction and replay; the example does this once at startup. `--embedded`,
`--indexer fused_identity`, `--dtype float16`, and `--pooling 1` select other
supported modes. The example checks several input sizes against a direct
reference, exports, and verifies Python replay.

For C++ replay, configure `example/host_embedding` with the same NVE paths and
`-DDYNAMICEMB_LIB_DIR=/path/to/torch_binding_build`, build it, then run:

```bash
example/host_embedding/build/replay_host /tmp/host-cpu -1
```

Pass `0` for CUDA device 0. Keep the NVE layer directory, resource directory,
and indexer directory alive until the AOTI loader is destroyed, as the example
shows. Export CPU and CUDA packages separately.

HSTU's ranking exporters accept `--host_embeddings`; the existing default
storage choices are unchanged. HostLayer lookup itself cannot be captured in
a CUDA graph; capture only the dense part of the pipeline.

## Incremental updates on CPU

| Indexer | Loading | Mapping |
| --- | --- | --- |
| Identity, single table | Inference continues | Overwrite fixed rows. |
| FusedIdentity | Inference continues | Overwrite within reserved table ranges. |
| LinearHashMap | Pause and drain CPU readers | Update/insert/erase mappings; reuse free rows within capacity. Sidecar required. |

During fixed-mapping updates, lookups may return stale or mixed values, including
within a row or pooled result. After `apply_incremental_load` returns, new lookups
see the updated rows. The dense host table uses relaxed, lock-free word accesses;
there is no new lookup/update mutex, retry loop, fence or CUDA synchronization.
Keep the storage allocation fixed. Raw weight writes, serialization, and
accumulation must happen outside concurrent serving.

```text
CPU delta → UpdateCoordinator → row/value message (+ paused hash snapshot)
          → UpdateSubscriber → HostLayer.update
CPU inference → unchanged indexer → HostLayer.lookup, concurrently for fixed mappings
```

The existing `EmbeddingCollectionUpdateCoordinator.open(..., device=cpu)` reads
the exported package. Call `apply_delta(collection_id, delta)` with a CPU
`DeltaDumpResult`, then pass `update.to_json()` to the subscriber. Use one update
writer and apply messages once, in snapshot order. Each subscriber has its own
host storage; the coordinator does not need Redis. Packages need the new
per-table capacity metadata, so re-export older packages.

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

Run the Python/C++ AOT example (CPU only):

```bash
python example/host_embedding/incremental_cpu.py --indexer fused_identity \
  --package-dir /tmp/host-updates
example/host_embedding/build/replay_host /tmp/host-updates -1
```

Use `--indexer identity` or `--indexer linear_hash_map` for the other modes;
`--embedded` works for fixed mappings, and `--dtype float16 --pooling 2` tests
pooled FP16. The example keeps AOT inference running during fixed-mapping updates
and checks final values afterwards. CUDA HostLayer incremental loading is outside
this CPU update path.
