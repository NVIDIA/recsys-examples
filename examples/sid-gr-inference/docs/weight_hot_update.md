<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Weight Hot-Update User Guide

This feature lets the GR HTTP serving engine swap model weights **in place, without
restarting the process**, so it can act as the rollout engine for RL training
frameworks. The API surface mirrors SGLang's weight-sync protocol.

See PR: [NVIDIA/recsys-examples#444](https://github.com/NVIDIA/recsys-examples/pull/444)

## Overview

Two weight transport paths are available:

| Path | Endpoint | Weight source | When to use |
| --- | --- | --- | --- |
| Disk load | `POST /update_weights_from_disk` | HF checkpoint read from disk by the **server process** | Cross-process update; the client only supplies a path |
| Tensor push (colocate) | `POST /update_weights_from_tensor` | Checkpoint read by the **client**, pushed zero-copy via CUDA IPC | Colocated trainer and server on the same GPU |

Both paths share the same coordination protocol (the weight-sync flow used by RL
frameworks such as slime):

```
pause_generation → flush_cache (retry until 200) → update_weights_from_* → continue_generation
```

## Enabling the Feature

Weight update is opt-in and disabled by default. Enable it at server startup in
either of two ways:

```bash
# Environment variable (read by scripts/serve_qwen3_gr_http.sh)
GR_ALLOW_WEIGHT_UPDATE=1 bash scripts/serve_qwen3_gr_http.sh

# CLI flag (tools/serve_qwen3_gr_http.py)
python tools/serve_qwen3_gr_http.py --allow-weight-update ...
```

When disabled, every weight update/lookup endpoint returns **403 `route_disabled`**
(`/flush_cache` is exempt, matching SGLang).

If the server is configured with an API key (`--api-key`, or `GR_HTTP_API_KEY` at
the shell-script layer), all requests must carry an `X-GR-API-Key: <key>` or
`Authorization: Bearer <key>` header.

## The Update Flow

### 1. pause_generation

```http
POST /pause_generation
{"mode": "abort"}
```

- `mode` is optional; one of `abort` (default) / `retract` / `in_place`. `abort`
  fails all in-flight requests; `in_place` / `retract` only freeze worker progress
  (`retract` currently degrades to `in_place`).
- Success response: `{"paused": true, "mode": "abort", "num_aborted_requests": N}`.
- **No service while paused**: `/generate`, `/submit`, and `/submit_many` are
  rejected immediately with **503, `error.code = "paused"`, `retryable = true`**.
  Callers should retry shortly instead of queuing requests at the server.

### 2. flush_cache (retry until 200)

```http
GET /flush_cache
```

- GET and POST are both accepted.
- Success (HTTP 200): `{"success": true, "entries_cleared": N}`.
- While requests are still running/waiting (abort is not instantaneous), it
  returns **HTTP 400** `{"success": false, ...}` — retry until 200 before
  updating, so no KV cache computed with the old weights survives the swap.
  (slime, for example, retries once per second, 60 attempts by default.)
- A `timeout` argument is accepted (SGLang compatibility) but the server returns
  immediately and never blocks.

### 3. Update the weights

Pick one of the two transports described below.

### 4. continue_generation

```http
POST /continue_generation
```

No parameters; success response `{"paused": false}`. Inference resumes.

## Endpoint Reference

All error responses share one shape:

```json
{"error": {"code": "...", "message": "...", "retryable": false, "status": 400}}
```

### `POST /update_weights_from_disk`

Loads an HF checkpoint from disk and replaces the weights in place.

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `model_path` | yes | — | Checkpoint directory, readable by the **server process**; alias `model_dir` is also accepted |
| `weight_version` | no | keep previous | Version label used for bookkeeping and verification |
| `token_step` | no | keep previous | Training-step label |
| `flush_cache` | no | `true` | Clear the prefill cache and reset hit/miss counters after the update |
| `abort_all_requests` | no | `false` | Allow the update without pausing, even with in-flight requests |

Success response:

```json
{
  "success": true,
  "message": "Succeeded to update model weights from disk.",
  "model_dir": "/models/Qwen3-GR-plus1",
  "tensors_loaded": 310,
  "flushed_cache": true,
  "num_aborted_requests": 0,
  "weight_version": "plus1",
  "token_step": null,
  "elapsed_ms": 12345.6
}
```

Checkpoint requirements and validation:

- The directory must contain `config.json`. Weights may be safetensors
  (`model.safetensors`, or shards with `model.safetensors.index.json`);
  `pytorch_model.bin` / `*.pt` are also supported.
- A structural mismatch in `num_layers` / `hidden_size` / `num_attention_heads` /
  `num_kv_heads` / `head_dim` / `vocab_size` / `tie_word_embeddings` /
  `intermediate_size` → 400 `validation_error`.
- **Validate-then-load atomicity**: a full dry-run validation passes before any
  copy happens; if validation fails, the previous weights are left untouched.
- The whole checkpoint is materialized in host memory during loading —
  **peak CPU memory ≈ checkpoint size**.

### `POST /update_weights_from_tensor`

Receives weights pushed by the client over CUDA IPC (SGLang wire format, zero-copy).

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `serialized_named_tensors` | yes | — | Per-TP-rank list; each item is a base64 string (ForkingPickler-serialized; single-GPU uses index 0) |
| `load_format` | no | direct | `flattened_bucket` (SGLang `FlattenedTensorBucket`) or omitted (a plain `[(name, tensor)]` list) |
| `weight_version` | no | keep previous | Same as the disk endpoint |
| `token_step` | no | keep previous | Same as the disk endpoint |
| `flush_cache` | no | `true` | Clients pushing in chunks should pass `false` and flush once at the end |
| `abort_all_requests` | no | `false` | Same as the disk endpoint |

Protocol notes:

- **Chunked push**: the client (e.g. slime) may POST one chunk at a time. The
  server applies each chunk independently with validate-then-copy — it does not
  accumulate and does not require the full model in one request. Trailing empty
  alignment buckets are treated as successful no-ops.
- **CUDA IPC requires the same machine and GPU**: payloads carry IPC handles
  (remapped by device UUID), so the client must run on the server's GPU. The HTTP
  body stays tiny, so `--max-http-request-bytes` does not need raising.
- **Name mapping**: payloads use HF split names (`q/k/v_proj`, `gate/up_proj`);
  the server packs them into the fused `qkv_proj` / `gate_up_proj` parameters.
  Dtype mismatches are converted to the parameter dtype automatically.
- **Safety**: deserialization uses a module allowlist (torch / numpy / builtins
  only); malicious or corrupt payloads are rejected with 400.

#### Preparing `serialized_named_tensors`

Each payload is a base64 string produced by staging a chunk of tensors on the
server's GPU, flattening them into a `FlattenedTensorBucket`, and serializing it
with `MultiprocessingSerializer`. Both helpers ship with the server package
(`gr_inference.gr_serving.weight_ipc`), so client and server are guaranteed to
speak the same wire format.

For a working example of building the payload and pushing it in chunks, see
`serialize_bucket` in `tests/_weight_update_helpers.py:160` and its use in
`tests/test_weight_update_tensor.py` (including a real cross-process CUDA IPC
round-trip).

Keep the staged CUDA tensors alive until the response returns: the payload only
carries IPC handles, and the server reads the client's GPU memory during the
call.

### Coordination and Probe Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/pause_generation` | POST | See above; `mode` ∈ `abort` / `retract` / `in_place` |
| `/continue_generation` | POST | Resume service; responds `{"paused": false}` |
| `/flush_cache` | GET / POST | See above; 400 while requests are in flight |
| `/get_weight_version` | GET | Returns `{"weight_version", "token_step", "weight_update_count"}`; `weight_update_count` increments on every successful update |
| `/get_weights_by_name` | GET / POST | Params: `name` (module parameter name, e.g. `embed_tokens.weight`), `truncate_size` (default 100, truncated along dim 0); returns `{"parameter": [[...]]}`; unknown name → 404 `not_found` |

`GET /status` also exposes `weight_version` / `token_step` /
`weight_update_count` / `last_weight_update_ms` under its `weights` field.

### In-Flight Request Protection

Both update endpoints refuse to run **while unpaused with in-flight requests**,
returning **HTTP 409 `conflict`** with a hint to call
`pause_generation(mode="abort")` first or pass `abort_all_requests=true`. This
prevents old KV cache from being mixed with new weights mid-update. Weights are
copied in place, so captured CUDA graphs remain valid.

## RL Framework Integration Notes

- The full colocate flow used by RL frameworks (e.g. slime) — `pause_generation →
  flush_cache (retry) → chunked update_weights_from_tensor (flush_cache=false,
  trailing empty bucket) → continue_generation` — is supported end to end.
- `GET /ready` reports `paused` in its reasons while an update is in progress.
- Only `weight_version` is required by the SGLang-compatible protocol (slime, for
  instance, sends just that); `token_step` and `weight_update_count` are a
  superset provided by this implementation, readable via `/get_weight_version`
  and `/status`.
