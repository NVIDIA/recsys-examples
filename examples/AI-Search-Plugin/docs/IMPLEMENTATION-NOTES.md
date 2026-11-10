# Implementation notes

What the plugin ships, what it leaves out, and what has not been run.

## Scope and conventions

- **Naming**: the Python package is `ai_search_plugin` (`ai_search_plugin.adapters.X`, `ai_search_plugin.response_logits`).
  Environment variables keep the `ASRT_*` prefix and log lines keep the `[asrt]` prefix, so existing job scripts and
  log parsers keep working. `ai_search_plugin.adapters.verl_plugin` stays as an alias of `ai_search_plugin.verl_plugin`.
- **No C++ extension**: only the pure-Python tool cache and partitioner (`toolcache_py`, `partition_py`) are shipped.
  A C++ tool cache, partitioner and cost router are not part of the plugin. Expert placement uses the pure-Python
  `placement_py`; the verl `route` feature uses its own pure-Python picker.
- **Deprecated components are not included** (tried, no measured gain): out-of-order batch assembly
  (`ASRT_STALE_MODE=ooo`), straggler slip (`ASRT_TICKET_SLIP`), slow-tool cut (`ASRT_TICKET_SLOWTOOL_S`), closed-loop
  extra fill (`ASRT_STALE_FILL`), verl-window cut (`ASRT_TICKET_DROP=verl4`), elastic GPU lending, cross-engine
  rebalancing and expert replication. Their switches are not read (`ASRT_STALE_MODE` accepts `ticket` (default) /
  `count`). Resuming from a checkpoint written by an earlier version of the admission component is not supported;
  a checkpoint without the ticket book (slime's own fully-async state, or count mode) is still resumed by adopting its
  completed groups as the next tickets.
- **No bundle that selects kernels for one specific model structure.** In `adapters/verl.py` the `trainperf` feature is
  replaced by a `response_logits` feature that installs only the response_logits verl adapter and its guard. The opt-in
  FSDP response_logits hook is not included.
- **No default tool targets**: `slime_toolcache` and `tool_hedge` need the tool targets to be listed explicitly
  (`ASRT_TOOLCACHE_GENERATE`, `ASRT_TOOLCACHE_TARGETS`, `ASRT_TOOL_HEDGE_TARGETS`, or the `toolcache_generate` /
  `tool_targets` options).

## Tests

- Script-style tests (module-level asserts or `__main__` runners) live in `tests/scripts/` without the `test_` prefix and
  are run each in a fresh interpreter by `tests/test_ported_scripts.py`; plain pytest files are in `tests/cpu/`.
- `slime_admit_cpu.py` and `slime_admit_resume_cpu.py` load slime's real `rollout/fully_async_rollout.py`. They read
  `$SLIME_SRC` (or an installed slime) and are skipped otherwise.
- `cpu/test_slime_bigbucket_cpu.py`: two tests failed with the original fake engine because it lacked the `_actor_id`
  attribute that Ray handles have; the fixture now has it. No library change.
- `slime_admit_cpu.py::test_flush1_cut_requeue_e2e`: whether the cut+requeue path is hit is timing dependent (about 1 of
  4 isolated runs missed it). The test retries up to 4 times for path coverage; every attempt still checks all
  correctness assertions.
- Tests of components that are not included were dropped. `verl_rlogits_guard_cpu.py` and `verl_adapter_cpu.py` fake
  `ai_search_plugin.response_logits` instead of a top-level `response_logits` package; tool-hedge tests use explicit
  targets.
- **Not run here**: `tests/gpu/*` (needs CUDA and Megatron, `torchrun`) and `tests/integration/slime_rshard_ray.py`
  (needs slime, Ray and a tokenizer).

## Present but not registered

Benefit undecided, kept off: verl `route`, `keepkv` (as a stand-alone feature; `verl_inplace` turns it on), `tokcache`,
`balance`, `profile`, `offpol` (measurement only, used by `verl_ceiling`'s tests). They can be enabled by hand through
`ASRT_VERL=<list>`; registering one means adding a `Component(...)` entry.

## `expert_placement`, `prefix_tree`, `r3_ep_plan`, `hybridep_overlap`

CPU-tested only; the GPU paths were reviewed but not run in this repository.

- `expert_placement`: planner without numpy (same algorithm, checked against the numpy version on random inputs).
  Hot-expert replication is not included. Not valid with group-limited routing (`moe_router_num_groups`), which is not
  checked yet.
- `prefix_tree`: paths that fit only one model structure (sparse attention, PLE, cuDNN-GDN) are not included. The RoPE
  mode is an explicit argument (inferring it from a tensor length can collide under CP>=2). Silent fallbacks in the GDN
  state path and the verl packer raise. The per-token log-prob difference against the flat path (dense model, CP2)
  comes from bf16 rounding: under CP>=2 each sample is zigzag-split by its own length, while the tree reuses sibling
  0's prompt for all siblings (bit-exact at CP1; about 1e-5 in fp32). Checked for the forward pass only.
- `r3_ep_plan`: patches Megatron's alltoall dispatcher (`preprocess`) from a slime init hook. The default on a mismatch
  is to warn and fall back to Megatron's path (`ASRT_R3PLAN_STRICT=1` makes it raise). The first step is verified
  exactly. The HybridEP version and the recompute metadata cache are not included. A single unexplained grad-norm spike
  has been seen and is not ruled out.
- `hybridep_overlap`: only the HybridEP chunked path; Megatron args are written in the init hook; version gate (the
  overlap patch only on Megatron-Core 0.19) with an `auto` fallback (plain HybridEP dispatch, or alltoall when HybridEP
  is missing). verl is not wired.

## Stubs (not implemented)

Each has a registry entry and a module in `src/ai_search_plugin/stubs/` whose `install()` raises `NotPortedError`.

### `mtp_draft_loader` (models with an MTP head)

- What it would do: SGLang `--speculative-algorithm NEXTN` with fixed steps and
  `--sglang-speculative-draft-load-format auto`, so the draft head follows weight syncs, plus the R3 audit
  (`ASRT_R3AUDIT`, already in `slime_r3check`).
- Left: decide whether this is a component or plain framework configuration; if a component, emit the SGLang arguments
  from `frameworks/slime.py` and document the draft-head sync.

### `dsa_replay` (DSA sparse attention)

- What it would do: replay the per-token top-k selection between rollout and training (a codec and a store, with
  rollout and slime hooks).
- Left: the rollout-side capture needs SGLang to expose the per-token top-k selection (not available in the SGLang we
  tested); never run end to end. The codec and store are pure Python and could be implemented first.

### `gdn_cudnn` (GDN; slime only)

- What it would do: a cuDNN kernel binding for GDN with a loader overlay.
- Left: the kernel binding and overlay, the slime install hook, and the jitcache `cudnn` store integration (already in
  `adapters/jitcache.py`). The verl use is deprecated (no gain, more memory).

### `fla_nb_bucket` (FLA / GDN)

- What it would do: an FLA bucket kernel and the `ASRT_FLA_NB_PIN` pinning.
- Left: a stand-alone install hook for slime and verl.

### `gdn_convfuse_ep` (GDN / MoE)

- What it would do: a fused GDN convolution kernel and expert-parallel dispatch items.
- Left: re-validate on an open GDN/MoE model before implementing.

## Not part of the plugin

- Components that only fit one specific model structure.
- Framework tuning that the baseline can enable itself (SGLang prefill graphs, concurrency / prefetch factor, bucket
  size, recompute / PP overlap settings, NCCL CTA count).
- The deprecated components listed above.
