# Tier updates with NVE 26.10

DynamicEmb provides `TierUpdateBinding<StockBinding>` for NVE's GPU, LinearUVM
and Hierarchical bindings. It exposes `update_tier(count, keys, values, tier,
stream)` by forwarding to the existing `EmbeddingLayerBase::update` API. It
compiles and runs with official NVE 26.10; the customized five-argument
`NVEmbedBinding::update` overload is not used.

```text
create_inference_embedding_collection
  → TierNVEmbedding / TierNVEmbeddingBag
  → TierUpdateBinding<stock NVE binding>
      lookup → existing NVE lookup/pooling
      update_tier → existing NVE layer.update(..., table_id)

Python/C++ AOTI reload → NveRuntimeDirectory → the same binding types
```

The binding is constructed once, with its real caches. There is no replacement
cache, unsafe downcast, monkey patch, or modified NVE header. The Python wrappers
reuse NVE's forward/export behavior. The native directory reads standard v2
metadata/weights, preserves shared resources within a package, and registers
markers for both AOTI constant buffers. It supports this PR's inference layer
configurations; distributed MemBlock topology is not supported.

Use `load_embedding_collection_aot` in Python or `NveRuntimeDirectory` in C++.
Keep the returned layers/directory alive until the AOTI loader is destroyed.
Stock NVE loaders create stock bindings; the incremental subscriber rejects
those bindings for CUDA updates instead of silently losing tier selection.
Inference-only builds with `DYNAMICEMB_ENABLE_NVE_UPDATE=OFF` continue to use
stock NVE construction/replay. The separate NVE 26.05 path is unchanged.

For hierarchical incremental updates the coordinator first writes Redis. The
subscriber then updates each changed row in host-cache tier 1 (when present),
followed by GPU-cache tier 0, on the existing inference stream. Tier IDs identify
cache/storage levels, not logical embedding tables. Update ordering, context
selection, snapshot retirement and demo synchronization remain unchanged.

| Companion NVE change | Status |
| --- | --- |
| Binding overload exposing a selected tier | Replaced by this local adapter. |
| Atomic Host row reads/writes | Removed by the sixth commit; concurrent CPU Host updates now exercise stock non-atomic copies. |
| Construction/export/reload | Previously unmodified NVE code. Local construction/reload is needed to select the new derived bindings. |

Host uses its stock binding and original four-argument update method. The
repository now pins official NVE 26.10 (`803e278`), with neither companion patch.
Concurrent CPU Host updates remain available as an experiment, without a
data-race safety guarantee; see [the rationale and experiment](host_cpu_update_concurrency.md).

Validation: `tests/test_nve_tier_binding.py` checks tier isolation, GPU/UVM
lookup/pooling, and fresh-process AOTI subscriber updates. The original
`example/exportable_embedding` Python/C++ workflow exercises paused hash and
continuous BitConcat updates against Redis without changing its scheduling.

The application image packages `_C` (default NVE updates) and `_C_legacy`
(indexers only, no NVE linkage). Python selects the latter when importing NVE
26.05, so ordinary installed-package imports cannot load the 26.10 libraries
into the legacy process. No alternate DynamicEmb `PYTHONPATH` is required.
