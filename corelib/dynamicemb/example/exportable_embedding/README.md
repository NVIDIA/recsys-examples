# Exportable embedding-collection workflow

The demo creates deterministic embedding tables, exports an AOTI package with
indexer sidecars, and serves it in **separate eager, Python AOTI and optional
C++ AOTI processes**. Each runtime owns its Subscriber. The orchestrator creates
deltas, asks the Coordinator to prepare each update once, delivers it to every
compatible runtime, and forwards retirement ACKs.

```text
export_and_verify.py: source data, delta producers, process/connection handles
  delta -> update_coordinator_main.py -> one update payload
                                             |
                  same payload delivered to all runtimes
                       |              |                |
                eager worker    Python AOTI worker   C++ AOTI replayer
                Python model    exported graph      exported graph
                Subscriber      Subscriber          Subscriber
                       |              |                |
                       +------------- ACKs ------------+
                                      |
                         orchestrator -> Coordinator
```

## Code map and ownership

| File | Responsibility |
|---|---|
| [run_example.sh](run_example.sh) | Select NVE environment and manage the local Redis used by CI |
| [export_and_verify.py](export_and_verify.py) | Orchestrate processes, generate deltas, deliver payloads, forward ACKs and verify reports |
| [inference_worker.py](inference_worker.py) | One eager or Python AOTI runtime; local inference thread and passive update thread |
| [workflow_utils.py](workflow_utils.py) | Shared configurations, checkpoint/delta fixtures, expected values, assertions and IPC helpers |
| [update_coordinator_main.py](update_coordinator_main.py) | Receive delta/ACK requests and call the library Coordinator |
| [cpp/replay_and_verify.cpp](cpp/replay_and_verify.cpp) | Optional C++ AOTI runtime, local Subscriber and four-round loop |
| [cpp/CMakeLists.txt](cpp/CMakeLists.txt) | Build the C++ replayer against the selected NVE installation |

The orchestrator owns no Subscriber. It creates the checkpoint/dumps and retains
the delta producers. The eager worker constructs and exports its actual serving
model; the AOTI workers load that package, so all runtimes start with the same
mapping. Package and snapshot files must be accessible to all workers.

Python workers receive batches of existing `EmbeddingCollectionUpdate` JSON
strings over an update socket and return ACK JSON. Separate test-control sockets
and expected-value files coordinate verification. C++ uses its existing
stdin/stdout protocol: update JSON, `RUN` with expected sums, `ACK`, `READY`,
and `STOP`. Each C++ update round follows the Python ACKs. The Coordinator tracks
all runtime Subscribers, including C++ when enabled.

## Configurations and four rounds

| Indexer | NVE storage | Incremental rounds |
|---|---|---|
| FusedIdentity | GPU | Initial values remain unchanged |
| LinearHashMap | LinearUVM | Initial values remain unchanged |
| LinearHashMap | Hierarchical: GPU cache → NVHashMap host cache → Redis | First, paused update |
| BitConcat | Hierarchical: GPU cache → NVHashMap host cache → Redis | First update and continuous-inference update |

| Round | Work per runtime | Expected result |
|---|---|---|
| 1. Initial | 100 checked requests, then phase synchronization | Checkpoint values |
| 2. First update | Apply LinearHashMap + BitConcat while paused; resume after staging, before retirement wait; 100 checked requests and phase synchronization | Updated/inserted rows; Python also checks LinearHashMap eviction misses |
| 3. Continuous inference | Start inference before the BitConcat update; at least 1,000 unchecked requests, continue until applied, then 100 further requests and phase synchronization | Execution continues through intermediate values |
| 4. Final | Python waits for all update ACKs, including C++; C++ joins its local update thread. Each then runs 100 checked requests and phase synchronization | Second updated values |

Python checks complete output vectors; C++ checks output count and tensor sums.
The C++ checks are weaker than full-vector comparisons. Checked phases include
CPU comparisons; the continuous phase has no per-request numerical checks or
extra CUDA/stream synchronization. There is one Coordinator call per delta:
three payloads total, independent of the number of runtimes. Retirement ACKs for
BitConcat are not GPU-completion fences; the phase synchronizations still precede
final verification.

NVE 26.05, or `--disable-redis-incremental`, runs only initial load/export/replay
for the two GPU/LinearUVM configurations, using the same process separation.
No Redis, Coordinator or update Subscriber is started in that mode. Modern
unpooled lookup requires NVE 26.06+; pooled collections require 26.07+ (or legacy
26.05), and Host collections require 26.09+. See
[NVE compatibility](../../docs/nve_tier_updates.md).

## Build and run

From `corelib/dynamicemb`, in the application image with training and inference
bindings installed, build the optional C++ replayer for the selected NVE:

```bash
cmake -S example/exportable_embedding/cpp \
  -B example/exportable_embedding/cpp/build \
  -DNVE_ROOT=/workspace/deps/nve \
  -DNVE_LIB_DIR=/opt/nve/default/python/pynve \
  -DDYNAMICEMB_LIB_DIR="$PWD/torch_binding_build"
cmake --build example/exportable_embedding/cpp/build -j 8

bash example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example \
  --cpp-replayer "$PWD/example/exportable_embedding/cpp/build/replay_and_verify"
```

The default is one eager and one Python AOTI process. Omit `--cpp-replayer` for
Python-only verification. `--replicas N` selects the number of Python AOTI
processes; the eager process is always included.

The launcher manages a local Redis by default. To use a dedicated existing
Redis instead:

```bash
START_LOCAL_REDIS=0 bash example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example \
  --redis-address 127.0.0.1:6379 \
  --cpp-replayer "$PWD/example/exportable_embedding/cpp/build/replay_and_verify"
```

The fixture clears its two fixed Redis test namespaces. Use a short work path:
the Coordinator creates its Unix socket underneath it. When invoking
`export_and_verify.py` directly in an already-configured environment, omitting
`--redis-address` starts a private Redis on a free local port.

Run the NVE 26.05 compatibility path with the installed legacy bindings:

```bash
NVE_VERSION=26.05 bash example/exportable_embedding/run_example.sh \
  --work-dir /tmp/exportable-embedding-example-2605
```

A C++ executable must match the selected NVE installation; the default modern
binary is not the legacy 26.05 binary.

## Results and scope

`verification.json` records distinct process IDs, Coordinator calls, payload
hashes, matching Python output hashes and numerical errors. The optional
`cpp_aoti` entry records checked rounds and ACKs; C++ assertions also appear in
stdout. Worker counters measure inference submissions at update receipt,
application return and ACK preparation, not completed GPU work. Logs and
reference fixtures remain in the work directory.

This local-process demo uses ordered delivery and shared snapshot files. It does
not implement retry/deduplication, discovery, arbitrary multi-stream readers or
uninterrupted LinearHashMap replacement. Standalone Host demos live in
[host_embedding](../host_embedding/README.md). The process separation needs no
library API or implementation changes.
