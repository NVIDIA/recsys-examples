# Exportable embedding-collection workflow

This example creates deterministic TorchRec embedding collections, writes one
complete checkpoint plus the required DynamicEmb dumps, converts the
collections to configurable indexer/NVE pairs, exports the AOTI package with
out-of-band indexer state, and verifies eager and Python AOTI replay. With NVE
26.06 or later and Redis enabled, it also applies two incremental rounds
through the reusable coordinator subprocess and per-runtime subscribers; the
second round updates only the BitConcat/Redis collection while inference
continues. An optional C++ replayer stays alive for the complete workflow.

The Python and C++ runtimes use the same four rounds:

- Round 1 verifies 100 requests against the initial state, then globally
  synchronizes.
- Inference pauses while the update worker applies the first update.
- Round 2 verifies 100 requests against the first incremental state, then
  globally synchronizes.
- Round 3 runs at least 1,000 unchecked requests while the BitConcat/Redis
  update runs concurrently, followed by 100 post-update requests and a global
  synchronization.
- Round 4 verifies 100 requests against the second incremental state, then
  globally synchronizes.

Normal inference calls the model directly on the device's default CUDA stream.
It does not acquire a publication lock or record a CUDA event; a replacement
snapshot records one retirement event only when inference adopts it.

The default workflow covers:

- `FusedIdentityIndexer + GPULayer`
- `LinearHashMapIndexer + LinearUVM`
- `LinearHashMapIndexer + Hierarchical(GPU cache -> NVHashMap host cache -> Redis)`
- `BitConcatIndexer + Hierarchical(GPU cache -> NVHashMap host cache -> Redis)`
- one paused and one concurrent Redis incremental-load round

NVE 26.05 runs the two non-hierarchical combinations. For NVE 26.06 or later,
`run_example.sh` starts and stops a local standalone Redis server, launches the
coordinator subprocess, and runs the in-process DynamicEmb delta producers.
The NVHashMap host caches are empty when loaded and are populated by inference;
their contents are not exported.

Build the optional C++ replayer for the selected NVE installation:

```bash
cmake -S /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp \
  -B /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp/build \
  -DNVE_ROOT=/workspace/deps/nve \
  -DNVE_LIB_DIR=/opt/nve/default/python/pynve \
  -DDYNAMICEMB_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build
cmake --build /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp/build -j 8
```

Run all four combinations with the repository NVE (26.07 or later):

```bash
/workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example \
  --cpp-replayer /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp/build/replay_and_verify
```

To use an existing Redis server instead, run the complete workflow with local
Redis disabled:

```bash
START_LOCAL_REDIS=0 \
/workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example \
  --redis-address 127.0.0.1:6379 \
  --cpp-replayer /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp/build/replay_and_verify
```

Run the two NVE 26.05 combinations:

```bash
NVE_VERSION=26.05 \
/workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example-2605
```
