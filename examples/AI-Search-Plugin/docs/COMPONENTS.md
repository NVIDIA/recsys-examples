# Components

Every component is opt-in and listed in one registry, `src/ai_search_plugin/registry.py`; this table is generated
from it (`python -m ai_search_plugin list` prints the same data).  Removing a component later = deleting its registry
entry and module.

- **category**: *model-agnostic* works for any model; *model-family* needs a model-family feature (MoE with rollout
  routing replay (R3), an MTP head, GDN / FLA linear attention, DSA sparse attention).
- **default**: member of `components="default"`.  Stubs are never in the default set.
- **switch**: what `ai_search_plugin.install()` sets for you.  Env variables keep the `ASRT_` prefix of the original
  code base; logs are prefixed `[asrt]` and go to `$ASRT_LOG_DIR/*.jsonl` when that is set.
- **status**: *ported* = the code is in this package; *stub* = registry entry + stub module only (see
  [PORTING.md](PORTING.md)).

Sparse-attention kernels and other components specific to one proprietary model are not part of this plugin.

### model-agnostic

| component | category | what it does | frameworks | default | switch | status |
|---|---|---|---|---|---|---|
| `sampler_nosync` | model-agnostic | Remove SGLang's per-step GPU->CPU sync on return_top_p_log_probs so the overlap scheduler works with logprobs. | slime, verl | on | engine hook: ASRT_SGL_HOOKS+=nosync (verl: ASRT_VERL+=sglhooks) | ported |
| `kvepoch` | model-agnostic | Tag radix-cache keys with a weight-update epoch so no new request reuses prefix KV computed by older weights. | slime | off | engine hook: ASRT_SGL_HOOKS+=kvepoch | ported |
| `kvfresh` | model-agnostic | On every weight sync, retract in-flight requests and flush the prefix cache (checked) so all KV is rebuilt with the new weights. | slime | off | ASRT_WS_KV=fresh + engine hook kvfresh | ported |
| `verl_kvfresh` | model-agnostic | verl: checked prefix-cache flush after each weight sync + earliest-version accounting for resumed requests. | verl | off | ASRT_VERL+=kvfresh (ASRT_VERL_KVFRESH=1) | ported |
| `ticket_admission` | model-agnostic | slime fully-async: hard staleness ceiling S (default 2) with ticket admission, deferral and a budgeted straggler cut; next batch prebuilt while training. | slime | on | --rollout-function-path ...slime_admit.generate_rollout_fully_async; ASRT_MAX_STALENESS=2, ASRT_ADMIT=cap,eager,profile | ported |
| `verl_ceiling` | model-agnostic | verl async trainers: the same hard staleness ceiling S in verl's own unit (ticket admission, deadline-ordered sampling, budgeted cut). | verl | on | ASRT_VERL+=ceiling; ASRT_MAX_STALENESS=2 | ported |
| `pool_fill` | model-agnostic | Size slime's in-flight group pool from batch and engines (not group size N) and keep every engine at max_running_requests within the staleness ceiling. | slime | on | ASRT_POOL=auto, ASRT_POOL_FILL=engines (runs inside the slime_prefetch rollout wrapper) | ported |
| `rshard` | model-agnostic | Shard slime's rollout client over worker processes that talk to their engine directly, with per-engine admission (auto: only for large trajectory pools). | slime | on | --rollout-function-path ...slime_rshard.generate_rollout_fully_async; ASRT_RS=auto,profile | ported |
| `toolcache` | model-agnostic | TTL cache + single-flight for read-only tool calls (identical concurrent queries share one call). | slime, verl | on | slime: --custom-generate-function-path ...slime_toolcache.generate + ASRT_TOOLCACHE_GENERATE/TARGETS; verl: ASRT_VERL+=toolcache | ported |
| `tool_hedge` | model-agnostic | Hedged tool calls: send a duplicate request when a call is slower than the recent tail; first answer wins. | slime | on | ASRT_TOOL_HEDGE=auto + ASRT_TOOL_HEDGE_TARGETS (under toolcache) | ported |
| `verl_inplace` | model-agnostic | verl: weight sync without stopping generation (no abort/resume, KV kept, kvepoch on), engine-side in-place apply + fastload. | verl | on | ASRT_VERL+=inplace | ported |
| `weight_sync` | model-agnostic | Trainer->SGLang weight sync: flattened buckets, pipelined send, parallel PP stages / big buckets, direct HTTP control calls, in-flight requests kept, engine-side record/replay loading (fastload). | slime, verl | on | slime: ASRT_WEIGHTSYNC=pipeline,flatten,parsync,profile + engine hook fastload; verl: ASRT_VERL+=sglhooks (fastload) | ported |
| `prefetch` | model-agnostic | Overlap the trainer's rollout-data fetch (data_preprocess) with the weight sync; prebuild the next batch. | slime | on | driver: python -m ai_search_plugin.adapters.slime_train <train.py>; --rollout-function-path ...slime_prefetch.generate_rollout_fully_async; ASRT_PREFETCH=overlap,profile | ported |
| `response_logits` | model-agnostic | Compute the output layer (and log-prob/entropy) only on response rows, chunked so full [n, V] logits are never materialised. | slime, verl | on | slime: chained in megatron_init; verl: ASRT_VERL+=response_logits (Megatron engine) | ported (GPU code, untested here) |
| `jitcache` | model-agnostic | Persistent Triton binary + autotune cache across runs (two-level, atomic publish): saves step-0 compile time. | slime, verl | off | ASRT_JITCACHE_DIR=<dir> (verl: ASRT_VERL+=jitcache) | ported |
| `prefix_tree` | model-agnostic | Train a GRPO group's shared prompt prefix once (prefix-tree packing) instead of once per sample. | slime, verl | off | `ASRT_PREFIX_TREE=1`; slime: megatron_init chain + rollout wrapper `prefix_tree.adapters.slime_pack`; verl: `VERL_USE_EXTERNAL_MODULES+=ai_search_plugin.prefix_tree.verl_plugin` | ported (GPU code untested here) |

### model-family

| component | category | model family | what it does | frameworks | default | switch | status |
|---|---|---|---|---|---|---|---|
| `r3_transfer` | model-family | MoE with rollout routing replay (R3) | Routing replay (R3) data path: routed-expert ids stored as int16/uint8 for transfer, copy-free R3 validation. | slime | on | ASRT_PREFETCH+=r3c, ASRT_R3CHECK=fast, ASRT_NUM_EXPERTS=<E> | ported |
| `r3pack` | model-family | MoE with rollout routing replay (R3) | R3-aware micro-batch packing: balance per-micro-step expert load from the replayed routes. | slime | off | ASRT_R3PACK=1 | ported |
| `jitfix` | model-family | MoE / GDN | Stop torch.compile recompile storms of Megatron's @jit_fuser helpers under dynamic token counts (MoE/GDN only). | slime | on | ASRT_JITFIX=eager | ported |
| `prefix_tree` | `cpu/test_prefix_tree_plan_cpu.py`, `cpu/test_prefix_tree_toy_cpu.py` (tree vs flat path on a float64 toy model: GQA + RoPE, GDN, MLP) | Megatron / slime / verl on GPU; CP>=2 bf16 behaviour |
| `expert_placement` | `cpu/test_placement_py_cpu.py`, `cpu/test_megatron_placement_cpu.py`, `cpu/test_slime_placement_cpu.py`, `cpu/test_verl_placement_cpu.py`, `scripts/place_decision_mp_cpu.py` | real all-to-all weight moves with EP>1, distributed-optimizer state, real slime / verl export |
| `r3_ep_plan` | `cpu/test_r3plan_cpu.py`, `cpu/test_megatron_r3plan_cpu.py` | bit-exact check on GPU, real slime runs |
| `hybridep_overlap` | `cpu/test_hybridep_cpu.py`, `cpu/test_hybridep_chunk_cpu.py` | real HybridEP kernels, CUDA-stream overlap, multi-rank |
| `mtp_draft_loader` | model-family | models with an MTP head | MTP/NEXTN speculative decoding in the rollout engines + draft-head weight-loading fix. | slime | off | (not ported) | stub |
| `dsa_replay` | model-family | DSA sparse attention | Sparse-attention selection replay: record the rollout's top-k selection and replay it in training. | slime | off | (not ported) | stub |
| `r3_ep_plan` | model-family | MoE with rollout routing replay (R3) | Plan the EP all-to-all from the replayed routes once per step (no per-layer count all-gather / host sync); runtime plan-vs-routing guard. | slime | off | `ASRT_CHAIN_MEGATRON_INIT+=slime_r3plan.megatron_init` (R3 + alltoall dispatcher); conflicts with `hybridep_overlap` | ported (GPU code untested here) |
| `expert_placement` | model-family | MoE | Re-place experts across EP ranks from R3 routing counts when imbalance is high (auto-off otherwise). | slime, verl | off | slime: `ASRT_CHAIN_MEGATRON_INIT+=slime_placement.megatron_init`; verl: `ASRT_VERL+=place` | ported (GPU code untested here) |
| `hybridep_overlap` | model-family | MoE | HybridEP dispatch/combine (Megatron flex dispatcher) with 2-chunk compute/communication overlap. | slime | off | `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep` + megatron_init chain (`ASRT_HYBRIDEP=auto`, `ASRT_HYBRIDEP_CHUNKS=2`); conflicts with `r3_ep_plan` | ported (GPU code untested here) |
| `gdn_cudnn` | model-family | GDN (gated delta net) linear attention | cuDNN gated-delta-net kernels in training (slime only). | slime | off | (not ported) | stub |
| `fla_nb_bucket` | model-family | FLA / GDN linear attention | Stop FLA conv1d/l2norm kernels from recompiling for every new packed length (varlen RL). | slime, verl | off | (not ported) | stub |
| `gdn_convfuse_ep` | model-family | GDN / MoE | GDN conv fusion + four EP communication items from the training-speed line. | slime | off | (not ported) | stub |

## Measured effect and whether a baseline can get it by config

**Reading the effect column.** Leave-one-out from our full set, same node, both orders. Small models: ratio (above 1 = helps). Qwen3-Next: % gain. ✓ = significant at 95%; n.s. = not significant; *not active* = the component never ran there, so no conclusion. SR1 = Search-R1, ASR = ASearcher, Q3N = Qwen3-Next. Measured with the original implementation; current defaults differ slightly (e.g. `ws direct` is off).

### Model-agnostic components

| Component | What it does | Measured effect | Baseline can get it by config? |
|---|---|---|---|
| `ticket_admission` (slime) | Bounds staleness without capping concurrency. DEFER moves slow groups to a later batch; a budgeted cut drops stragglers. | DEFER: ASR ×1.333 ✓ (first 2 pairs reviewed: ×1.286); Q3N +0.38% n.s. Ticket itself: SR1·slime ×0.996 n.s.; Q3N −2.32% ✓ (off is faster and still compliant); ASR has no compliant run without it. | No. verl drops stale samples instead. |
| `verl_ceiling` (verl) | The same staleness ceiling, in verl's units | SR1·verl ×1.000 n.s. (never triggered) | Yes |
| `pool_fill` (slime) | Keeps every engine full within the ceiling | Not ablated | No |
| `rshard` (slime) | Shards the rollout client over processes | ASR ×1.051 n.s. (`--use-distributed-post` on in both arms); Libra n.s. | Partly: `--use-distributed-post` covers LLM requests, not tool calls |
| `toolcache` + `tool_hedge` | Merges identical concurrent tool calls; sends a backup request for slow ones | ASR ×1.079 n.s. (not reviewed); Libra n.s.; SR1·verl not active (a positional-argument call bypassed the cache) | No |
| `sampler_nosync` + `fastload` ("engine hooks") | Removes SGLang's per-step GPU→CPU sync; faster weight loading in the engine | ASR ×1.116 ✓ (first 2 pairs reviewed: ×1.171); Libra ×1.151 ✓; SR1 n.s.; Q3N n.s. | No |
| `weight_sync` (+ `verl_inplace`) | Faster weight push: buckets, pipelined send, parallel PP stages | n.s. everywhere (Q3N +1.2%, one order only) | Partly: slime baselines got the same 2 GB buffer; verl already buckets |
| `prefetch` (slime) | Overlaps rollout-data fetch with weight sync; prebuilds the next batch | SR1·slime ×1.033 ✓ (ratio not reviewed); ASR ×1.066 (weak ✓); Libra n.s.; Q3N +13.51% ✓ (with `r3_transfer`) | No |
| `response_logits` | LM head, log-prob and entropy only on response tokens, chunked | SR1·slime ×1.116 ✓; ASR ×1.090 ✓; Libra n.s.; Q3N n.s. | No. verl's fused kernel saves memory but still computes every row |
| `kvepoch` / `kvfresh` / `verl_kvfresh` | Correctness: after a sync, no request reuses prefix KV from old weights | Cost n.s. (SR1·verl ×0.993; Libra vs. prefix cache off ×1.065) | Partly: flush or disable the prefix cache |
| `jitcache` | Keeps Triton compile and autotune results across runs | Not ablated | Mostly |
| `prefix_tree` | Trains a group's shared prompt once | Not in the measured configs (code verified on CPU only) | — |

### Model-family components

| Component | Family | What it does | Measured effect | Baseline by config? |
|---|---|---|---|---|
| `expert_placement` | MoE | Moves experts across EP ranks to even out load | Q3N +2.70% ✓ | No |
| `r3_ep_plan` | MoE + routing replay | Plans a whole step's MoE all-to-all before training, removing per-layer syncs; checked every step, falls back on mismatch | Not in the measured configs | No |
| `hybridep_overlap` | MoE | HybridEP MoE dispatch with compute/communication overlap in two chunks; cannot be combined with `r3_ep_plan` | Not in the measured configs | No |
| `r3_transfer` | MoE + routing replay | Sends routed-expert ids in compact form | Only measured as part of prefetch | No |
| `r3pack` | MoE + routing replay | Balances expert load when packing micro-batches | Not active on Q3N (turned itself off) | No |
| `jitfix` | MoE / GDN | Prevents torch.compile recompile storms | Not ablated | No |
| `mtp_draft_loader` (stub) | MTP | MTP speculative decoding in rollout | Q3N: slower in both tries (−3.4%; −1.6% n.s.); kept off | — |
| `dsa_replay`, `gdn_cudnn`, `fla_nb_bucket`, `gdn_convfuse_ep` (stubs) | DSA / MoE / GDN | See COMPONENTS.md | Not measured (`gdn_cudnn` was slower than FLA in an earlier test) | — |

`ws direct` (the direct-HTTP path of `weight_sync`) is **off by default**: it caused rare hangs on the 80B model (1 in 60 syncs); root cause not confirmed.

## What is tested here (CPU only, no GPU, no framework installed)

`pytest tests/` runs on a CPU-only machine.  Tests marked *needs slime source* run only when `SLIME_SRC` points at a
slime checkout's `slime/` directory (or slime is installed); otherwise they are reported as skipped.

| component | CPU tests | not covered here |
|---|---|---|
| registry / install / CLI | `tests/test_registry.py` | real slime / verl launches |
| post-import hooks, engine sitecustomize | `scripts/hooks_reentry_cpu.py`, `scripts/sgl_boot_chain_cpu.py` | |
| `sampler_nosync` | hook selection only (`scripts/sglhooks_sets_cpu.py`) | the SGLang sampler patch itself (needs SGLang) |
| `kvepoch` | `scripts/kvepoch_cpu.py` | real SGLang radix cache |
| `kvfresh` | `scripts/kvfresh_cpu.py`, `scripts/kvfresh_r3keep_cpu.py` | real engines |
| `verl_kvfresh` | `scripts/verl_kvfresh_cpu.py`, `scripts/sglhooks_sets_cpu.py` | real verl |
| `ticket_admission` | `scripts/slime_admit_cpu.py`, `scripts/slime_admit_resume_cpu.py` (*needs slime source*), `scripts/slime_admit_dist_cpu.py`, `scripts/slime_admit_poll_cpu.py`, `scripts/vstart_cpu.py`, `scripts/startup_lines_cpu.py` | real slime run |
| `verl_ceiling` | `scripts/verl_ceiling_cpu.py`, `scripts/verl_offpol_cpu.py` | real verl |
| `pool_fill` | `cpu/test_slime_pool_cpu.py` | |
| `rshard` | none on CPU; `tests/integration/slime_rshard_ray.py` needs slime + Ray + a tokenizer and was **not run** here | everything |
| `toolcache` | `scripts/toolcache_nocache_err_cpu.py`, `scripts/verl_adapter_cpu.py` (verl path), `scripts/startup_lines_cpu.py` | |
| `tool_hedge` | `scripts/tool_hedge_cpu.py`, `scripts/startup_lines_cpu.py` | |
| `verl_inplace` | `scripts/verl_rollout_cpu.py`, `scripts/verl_keep_attrs_cpu.py`, `scripts/sglhooks_sets_cpu.py` | engine-side in-place apply (needs SGLang + GPU) |
| `weight_sync` | `cpu/test_slime_bigbucket_cpu.py`, `scripts/slime_ws_direct_cpu.py`, `scripts/sgl_fastload_cpu.py` | NCCL transfer, real engines |
| `prefetch` | `cpu/test_slime_prefetch_prebuild_cpu.py`, `cpu/test_prefetch_race_cpu.py` | real slime driver loop |
| `response_logits` | the verl guard only (`scripts/verl_rlogits_guard_cpu.py`, `scripts/verl_adapter_cpu.py`) | **the op and both adapters**: GPU tests in `tests/gpu/` (torchrun + Megatron) were **not run** in this repo |
| `jitcache` | `scripts/jitcache_cudnn_cpu.py` (cuDNN store part) | Triton cache manager (needs Triton) |
| `r3_transfer` | `cpu/test_slime_r3check_cpu.py`, `scripts/r3audit_cpu.py` | r3c dtype compaction in a real run |
| `r3pack` | `cpu/test_r3pack_cpu.py`, `cpu/test_slime_r3pack_cpu.py` | |
| `jitfix` | `scripts/jitfix_gate_cpu.py` (model gate) | the Dynamo replacement on real Megatron |
