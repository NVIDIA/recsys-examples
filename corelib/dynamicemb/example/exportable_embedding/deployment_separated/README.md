# Separate orchestrator and inference processes

This version of the adjacent `export_and_verify.py` demo moves each loaded AOTI
runtime and its Subscriber into a separate process. It reuses the original
demo's four collection configurations, deterministic weights, real DynamicEmb
delta producers, export helpers and numerical assertions.

```text
orchestrator.py
  original delta -> Coordinator subprocess -> one update payload
                                             |
                    same payload sent to every inference process
                        |                                |
                 inference_worker.py              inference_worker.py
                 local update thread              local update thread
                     -> Subscriber                    -> Subscriber
                     -> local runtime state           -> local runtime state
                 inference thread                  inference thread
                     -> AOTI graph                     -> AOTI graph
                        |                                |
                        +--------- ACKs -----------------+
                                    |
                       orchestrator -> Coordinator
```

The orchestrator holds process/connection handles, not Subscribers. The update
receiver sees batches of existing `EmbeddingCollectionUpdate` JSON payloads,
never the original delta. It calls its local Subscriber and returns the existing
ACK JSON. A separate control socket and reference-value files belong only to
the test harness: they coordinate verification phases, not embedding updates.

There is one Coordinator call per collection/delta, regardless of replica count.
Two replicas consume the same three payloads: LinearHashMap and BitConcat in the
first batch, then BitConcat alone in the second. This is explicit ordered
delivery, not a retry/deduplication or subscriber-discovery service. Replicas
load the same package and can access the shared snapshot directory.

## Run

Use the same GPU/NVE environment as the original demo (NVE 26.06 or later),
with DynamicEmb's training and inference bindings installed. From
`corelib/dynamicemb`:

```bash
export DYNAMICEMB_OPS_LIB_DIR="$PWD/torch_binding_build"
export PYTHONPATH="/opt/nve/default/python:$PWD:${PYTHONPATH:-}"
python3 example/exportable_embedding/deployment_separated/orchestrator.py \
  --work-dir /tmp/embedding-separated --replicas 2
```

By default the demo starts an isolated Redis on a free local port. Alternatively
pass `--redis-address HOST:PORT` for a dedicated test Redis; the reference setup
clears its two fixed test namespaces. Use a short work path because the reused
Coordinator launcher puts its Unix socket under that directory.

## Verification and ordering

Each replica runs the same four phases as the original demo:

1. Verify 100 initial requests; synchronize at the phase boundary.
2. While inference is paused, apply batch one. Resume inference **after staging,
   before waiting for retirement** so the graph can adopt the new mapping.
   Verify 100 requests and LinearHashMap eviction misses; synchronize.
3. Start inference before producing batch two. Run at least 1,000 unchecked
   requests, keep running until the batch is applied, then run 100 more and
   synchronize. Only BitConcat changes during this continuous phase.
4. Verify 100 requests against the second expected state; synchronize.

No additional CUDA/stream synchronization is inserted into inference or update
calls. Test-control IPC replaces the original cross-thread phase signals.
Retirement ACKs retain their existing meaning: for BitConcat they do not signal
GPU completion; final verification follows the existing phase synchronization.

`verification.json` checks delivery hashes and identical output vectors across
replicas. Per-worker reports include process IDs, numerical errors, and inference
call counters at update receipt, application return and ACK preparation. These
counters measure submitted inference calls, not completed GPU work. Worker logs
and reference fixtures remain in the work directory.

This example adds process/transport orchestration only. It does not change
library APIs, Subscriber behavior, Coordinator behavior, or the original demo.
