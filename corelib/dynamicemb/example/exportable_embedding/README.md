# Exportable embedding-collection workflow

This example creates deterministic TorchRec embedding collections, writes one
complete checkpoint plus the required DynamicEmb dumps, converts the
collections to configurable indexer/NVE pairs, exports them with indexer and
NVHashMap sidecars, and verifies eager and Python AOTI replay. With NVE 26.07
and Redis enabled, it also applies one DynamicEmb delta through the reusable
coordinator subprocess and per-runtime subscribers. An optional C++ replayer
stays alive across the initial and updated inference rounds to verify the same
AOTI package and its incremental update path.

The default workflow covers:

- `FusedIdentityIndexer + GPULayer`
- `LinearHashMapIndexer + LinearUVM`
- `LinearHashMapIndexer + Hierarchical(NVHashMap)`
- `BitConcatIndexer + Hierarchical(NVHashMap)`
- the two corresponding Redis hierarchical collections and incremental load

NVE 26.05 runs the four non-Redis combinations. For NVE 26.07,
`run_example.sh` starts and stops a local standalone Redis server, launches the
coordinator subprocess, and runs the in-process DynamicEmb delta producers.

Build the optional C++ replayer for the selected NVE installation:

```bash
cmake -S /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp \
  -B /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp/build \
  -DNVE_VERSION=26.07 \
  -DNVE_ROOT=/workspace/deps/nve \
  -DNVE_LIB_DIR=/opt/nve/default/python/pynve \
  -DDYNAMICEMB_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build
cmake --build /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp/build -j 8
```

Run all six NVE 26.07 combinations:

```bash
NVE_VERSION=26.07 \
/workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example \
  --cpp-replayer /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp/build/replay_and_verify
```

To use an existing Redis server instead, run the complete workflow with local
Redis disabled:

```bash
START_LOCAL_REDIS=0 \
NVE_VERSION=26.07 \
/workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example \
  --redis-address 127.0.0.1:6379 \
  --cpp-replayer /workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/cpp/build/replay_and_verify
```

Run the four NVE 26.05 combinations:

```bash
NVE_VERSION=26.05 \
/workspace/recsys-examples/corelib/dynamicemb/example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example-2605
```
