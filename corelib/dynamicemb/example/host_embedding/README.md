# HostLayer collections

`nve_layer_type="host"` stores embedding values on the host. `device="cpu"`
executes the whole collection on CPU; `device="cuda:0"` uses CUDA keys,
indexing and outputs with NVE's host lookup. Standalone CPU collections run
without a GPU or CUDA driver when installed with driverless NVE as described below.
They do not depend on HSTU.

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

Host collections support NVE 26.09 and later; zero-filled misses require the
26.09 Python binding. Earlier Host bounds checks do not provide that behavior.
Compatible later releases use the same path. Rebuild the native adapter against
matching NVE headers and libraries when changing versions; see
[NVE compatibility and rebuild requirements](../../docs/nve_tier_updates.md).

Use the repository-pinned official NVE 26.10 build (`803e278`)
and the repository's inference bindings. For a machine without an
NVIDIA driver, build NVE with `PYNVE_DRIVERLESS_BUILD=1 PYNVE_DISABLE_MPI=1`
and keep DynamicEmb inference updates enabled (the default). Set
`-DNVE_LIB_DIR` to that driverless build.
A CUDA toolkit and CUDA-capable PyTorch build are still used to build these
bindings; CPU execution does not require a GPU or driver.
The inference-only installation avoids the training extension, which links
`-lcuda`. Importing the collection does not import TorchRec or the DynamicEmb
training extension.

### Host-only installation

With CUDA-enabled PyTorch and driverless NVE installed, build a DynamicEmb wheel
from `corelib/dynamicemb`. No GPU, CUDA driver, or TorchRec is needed to build it;
the builder still needs the CUDA toolkit, CMake, a C++ compiler, and NVTX headers.

```bash
# Explicit targets avoid GPU detection during the build; these kernels are unused
# by CPU Host collections. Choose a target supported by the installed toolkit.
export TORCH_CUDA_ARCH_LIST=8.0
export CMAKE_CUDA_ARCHITECTURES=80
DYNAMICEMB_BUILD_TRAINING=0 NVE_ROOT=/path/to/nv-embedding-cache \
  python3 -m pip wheel --no-build-isolation --no-deps . -w dist/host
```

`DYNAMICEMB_BUILD_TRAINING=0` builds and packages the inference bindings, including
Host incremental updates. It skips TorchRec, the training extension, and its
GPU eviction kernels. It uses `torch_binding_build_inference` and `build_inference`
to keep previously built training artifacts out of the wheel.
`python setup.py install` also accepts this environment setting; the default
training build is unchanged.

Install that wheel in the deployment image/environment alongside matching
CUDA-enabled PyTorch and driverless NVE:

```bash
python3 -m pip install --no-deps /wheels/dynamicemb-*.whl
```

CPU lookup and updates use the installed package; they need neither a GPU, a
CUDA driver, TorchRec, nor the source checkout. CUDA runtime libraries remain
required. The CUDA toolkit is needed for building native bindings, not for
ordinary lookup/update with the installed wheel. AOT export additionally needs
PyTorch's C++ compilation tools.

### Build directly from source

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
| Identity, single table | Inference continues (experimental; see below) | Overwrite fixed rows. |
| FusedIdentity | Inference continues (experimental; see below) | Overwrite within reserved table ranges. |
| LinearHashMap | Pause and drain CPU readers | Update/insert/erase mappings; reuse free rows within capacity. Sidecar required. |

The intended fixed-mapping contract permits stale/mixed lookup values during
updates and checks final values after completion. The sixth commit removes the
atomic host-row patch and pins stock NVE 26.10: overlapping lookup/update now uses
ordinary `memcpy`, which has a C++ data race. Concurrent runs are an experiment,
not a guarantee of safe execution or final-value visibility. The examples keep
their existing concurrency and synchronization; paused LinearHashMap is unchanged.
See [the explanation, word-level example, and validation results](../../docs/host_cpu_update_concurrency.md).
Keep the storage allocation fixed and use one update writer. Raw weight writes,
serialization, and accumulation must happen outside concurrent serving.

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
