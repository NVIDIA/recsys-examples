"""Component registry.

Every component is an opt-in module.  The registry is the single source of truth for

* what a component is (name, category, one-line description, status),
* which frameworks it applies to,
* whether it is in the ``"default"`` set, and
* how it is switched on: the environment variables / framework feature names / CLI arguments that the framework
  entry points (:mod:`ai_search_plugin.frameworks.slime`, :mod:`ai_search_plugin.frameworks.verl`) emit for it.

Categories
    ``model-agnostic`` (③): works for any model.
    ``model-family`` (②): needs a model family feature (MoE + routing replay, MTP head, GDN / FLA linear attention,
    DSA sparse attention).

Status
    ``ported``: code is in this package (see ``modules``); ``stub``: only a registry entry and a stub module that
    raises on install - see docs/PORTING.md for where the source lives and what is left.

Dropping a component later = deleting its ``Component(...)`` entry (and its module).  Nothing else refers to a
component by name except the registry, the framework entry points (which read the registry) and the tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

MODEL_AGNOSTIC = "model-agnostic"
MODEL_FAMILY = "model-family"
PORTED = "ported"
STUB = "stub"
FRAMEWORKS = ("slime", "verl")


@dataclass(frozen=True)
class Component:
    name: str
    category: str                      # MODEL_AGNOSTIC | MODEL_FAMILY
    summary: str                       # one line, plain English
    frameworks: Tuple[str, ...]        # subset of FRAMEWORKS
    default: bool                      # member of the "default" set
    switch: str                        # human-readable: how it is turned on (env / feature / CLI)
    status: str                        # PORTED | STUB
    modules: Tuple[str, ...] = ()      # implementing modules (relative to ai_search_plugin)
    family: str = ""                   # model family, for MODEL_FAMILY components
    needs: Tuple[str, ...] = ()        # required user options (see frameworks.*.plan), e.g. "toolcache_generate"
    conflicts: Tuple[str, ...] = ()    # components that must not be enabled together with this one
    notes: str = ""

    def applies_to(self, framework: str) -> bool:
        return framework in self.frameworks


_C = Component
COMPONENTS: List[Component] = [
    # ------------------------------------------------------------------ (3) model-agnostic
    _C("sampler_nosync", MODEL_AGNOSTIC,
       "Remove SGLang's per-step GPU->CPU sync on return_top_p_log_probs so the overlap scheduler works with logprobs.",
       ("slime", "verl"), True, "engine hook: ASRT_SGL_HOOKS+=nosync (verl: ASRT_VERL+=sglhooks)", PORTED,
       ("adapters.slime_sglang_hooks", "adapters.sgl_boot"),
       notes="Installs nothing on SGLang builds without that check."),
    _C("kvepoch", MODEL_AGNOSTIC,
       "Tag radix-cache keys with a weight-update epoch so no new request reuses prefix KV computed by older weights.",
       ("slime",), False, "engine hook: ASRT_SGL_HOOKS+=kvepoch", PORTED,
       ("adapters.slime_sglang_hooks",),
       notes="Correctness fix when the prefix cache survives weight syncs (flush 0 / keep_inflight / inplace). "
             "verl gets it through verl_inplace (keepkv installs it)."),
    _C("kvfresh", MODEL_AGNOSTIC,
       "On every weight sync, retract in-flight requests and flush the prefix cache (checked) so all KV is rebuilt "
       "with the new weights.",
       ("slime",), False, "ASRT_WS_KV=fresh + engine hook kvfresh", PORTED,
       ("adapters.slime_weightsync", "adapters.slime_sglang_hooks"),
       notes="Correctness fix (train/rollout logp gap); measured cost 2-4% step time."),
    _C("verl_kvfresh", MODEL_AGNOSTIC,
       "verl: checked prefix-cache flush after each weight sync + earliest-version accounting for resumed requests.",
       ("verl",), False, "ASRT_VERL+=kvfresh (ASRT_VERL_KVFRESH=1)", PORTED,
       ("adapters.verl_rollout", "adapters.verl"), conflicts=("verl_inplace",),
       notes="Correctness, not a speed-up. Refuses keepkv/inplace."),
    _C("ticket_admission", MODEL_AGNOSTIC,
       "slime fully-async: hard staleness ceiling S (default 2) with ticket admission, deferral and a budgeted "
       "straggler cut; next batch prebuilt while training.",
       ("slime",), True,
       "--rollout-function-path ...slime_admit.generate_rollout_fully_async; ASRT_MAX_STALENESS=2, ASRT_ADMIT=cap,eager,profile",
       PORTED, ("adapters.slime_admit", "adapters.staleness_core", "adapters.slime_resume"),
       notes="DEFER and the cut rule (ASRT_TICKET_DEFER / ASRT_TICKET_DROP*) are on by default in the code but their "
             "benefit is cluster dependent (status: undecided)."),
    _C("verl_ceiling", MODEL_AGNOSTIC,
       "verl async trainers: the same hard staleness ceiling S in verl's own unit (ticket admission, deadline-ordered "
       "sampling, budgeted cut).",
       ("verl",), True, "ASRT_VERL+=ceiling; ASRT_MAX_STALENESS=2", PORTED,
       ("adapters.verl_ceiling", "adapters.staleness_core", "adapters.verl_offpol")),
    _C("pool_fill", MODEL_AGNOSTIC,
       "Size slime's in-flight group pool from batch and engines (not group size N) and keep every engine at "
       "max_running_requests within the staleness ceiling.",
       ("slime",), True, "ASRT_POOL=auto, ASRT_POOL_FILL=engines (runs inside the slime_prefetch rollout wrapper)",
       PORTED, ("adapters.slime_pool",)),
    _C("rshard", MODEL_AGNOSTIC,
       "Shard slime's rollout client over worker processes that talk to their engine directly, with per-engine "
       "admission (auto: only for large trajectory pools).",
       ("slime",), True, "--rollout-function-path ...slime_rshard.generate_rollout_fully_async; ASRT_RS=auto,profile",
       PORTED, ("adapters.slime_rshard",)),
    _C("toolcache", MODEL_AGNOSTIC,
       "TTL cache + single-flight for read-only tool calls (identical concurrent queries share one call).",
       ("slime", "verl"), True,
       "slime: --custom-generate-function-path ...slime_toolcache.generate + ASRT_TOOLCACHE_GENERATE/TARGETS; "
       "verl: ASRT_VERL+=toolcache", PORTED,
       ("adapters.slime_toolcache", "adapters.common", "toolcache_py"), needs=("toolcache_generate", "tool_targets"),
       notes="slime needs the user's generate function and tool functions (module.attr); skipped if not given."),
    _C("tool_hedge", MODEL_AGNOSTIC,
       "Hedged tool calls: send a duplicate request when a call is slower than the recent tail; first answer wins.",
       ("slime",), True, "ASRT_TOOL_HEDGE=auto + ASRT_TOOL_HEDGE_TARGETS (under toolcache)", PORTED,
       ("adapters.tool_hedge",), needs=("tool_targets",),
       notes="Costs extra tool requests (~15% measured); turn off for rate-limited or paid APIs. Read-only tools only."),
    _C("verl_inplace", MODEL_AGNOSTIC,
       "verl: weight sync without stopping generation (no abort/resume, KV kept, kvepoch on), engine-side in-place "
       "apply + fastload.",
       ("verl",), True, "ASRT_VERL+=inplace", PORTED, ("adapters.verl_engine", "adapters.verl_rollout"),
       conflicts=("verl_kvfresh",)),
    _C("weight_sync", MODEL_AGNOSTIC,
       "Trainer->SGLang weight sync: flattened buckets, pipelined send, parallel PP stages / big buckets, direct HTTP "
       "control calls, in-flight requests kept, engine-side record/replay loading (fastload).",
       ("slime", "verl"), True,
       "slime: ASRT_WEIGHTSYNC=pipeline,flatten,parsync,profile + engine hook fastload; verl: ASRT_VERL+=sglhooks "
       "(fastload)", PORTED,
       ("adapters.slime_weightsync", "adapters.slime_sglang_hooks", "adapters.slime_resume"),
       notes="slime in-place sync (no pause) is an option: ai_search_plugin.install(..., options={'ws_inplace': True})."),
    _C("prefetch", MODEL_AGNOSTIC,
       "Overlap the trainer's rollout-data fetch (data_preprocess) with the weight sync; prebuild the next batch.",
       ("slime",), True,
       "driver: python -m ai_search_plugin.adapters.slime_train <train.py>; "
       "--rollout-function-path ...slime_prefetch.generate_rollout_fully_async; ASRT_PREFETCH=overlap,profile",
       PORTED, ("adapters.slime_prefetch", "adapters.slime_train")),
    _C("response_logits", MODEL_AGNOSTIC,
       "Compute the output layer (and log-prob/entropy) only on response rows, chunked so full [n, V] logits are "
       "never materialised.",
       ("slime", "verl"), True,
       "slime: chained in megatron_init; verl: ASRT_VERL+=response_logits (Megatron engine)", PORTED,
       ("response_logits",),
       notes="GPU/Megatron code; ported verbatim, NOT tested in this repo (no GPU tests run here)."),
    _C("jitcache", MODEL_AGNOSTIC,
       "Persistent Triton binary + autotune cache across runs (two-level, atomic publish): saves step-0 compile time.",
       ("slime", "verl"), False, "ASRT_JITCACHE_DIR=<dir> (verl: ASRT_VERL+=jitcache)", PORTED,
       ("adapters.jitcache",), needs=("jitcache_dir",), notes="Off unless a cache directory is given."),
    _C("prefix_tree", MODEL_AGNOSTIC,
       "Train a GRPO group's shared prompt prefix once (prefix-tree packing) instead of once per sample.",
       ("slime", "verl"), False,
       "ASRT_PREFIX_TREE=1; slime: megatron_init chain + rollout wrapper prefix_tree.adapters.slime_pack; "
       "verl: VERL_USE_EXTERNAL_MODULES+=ai_search_plugin.prefix_tree.verl_plugin",
       PORTED, ("prefix_tree",),
       notes="Megatron backend. slime: CP1/PP1; verl: THD + zigzag CP. Under CP>=2 siblings other than the first "
             "differ from the flat path by bf16 rounding of the shared prompt. GPU code NOT tested in this repo."),
    # ------------------------------------------------------------------ (2) model-family
    _C("r3_transfer", MODEL_FAMILY,
       "Routing replay (R3) data path: routed-expert ids stored as int16/uint8 for transfer, copy-free R3 validation.",
       ("slime",), True, "ASRT_PREFETCH+=r3c, ASRT_R3CHECK=fast, ASRT_NUM_EXPERTS=<E>", PORTED,
       ("adapters.slime_prefetch", "adapters.slime_r3check"), family="MoE with rollout routing replay (R3)",
       needs=("num_experts",)),
    _C("r3pack", MODEL_FAMILY,
       "R3-aware micro-batch packing: balance per-micro-step expert load from the replayed routes.",
       ("slime",), False, "ASRT_R3PACK=1", PORTED, ("adapters.slime_r3pack", "r3pack"),
       family="MoE with rollout routing replay (R3)", notes="No end-to-end gain measured yet."),
    _C("jitfix", MODEL_FAMILY,
       "Stop torch.compile recompile storms of Megatron's @jit_fuser helpers under dynamic token counts (MoE/GDN only).",
       ("slime",), True, "ASRT_JITFIX=eager", PORTED, ("adapters.slime_jitfix",), family="MoE / GDN",
       notes="Skips itself on dense models (measured slower there)."),
    _C("mtp_draft_loader", MODEL_FAMILY,
       "MTP/NEXTN speculative decoding in the rollout engines + draft-head weight-loading fix.",
       ("slime",), False, "(not ported)", STUB, ("stubs.mtp_draft_loader",), family="models with an MTP head"),
    _C("dsa_replay", MODEL_FAMILY,
       "Sparse-attention selection replay: record the rollout's top-k selection and replay it in training.",
       ("slime",), False, "(not ported)", STUB, ("stubs.dsa_replay",), family="DSA sparse attention"),
    _C("r3_ep_plan", MODEL_FAMILY,
       "Plan the EP all-to-all from the replayed routes once per step: no per-layer count all-gather / host sync in "
       "Megatron's alltoall MoE dispatcher under R3; runtime plan-vs-routing guard.",
       ("slime",), False, "ASRT_CHAIN_MEGATRON_INIT+=slime_r3plan.megatron_init (R3 + alltoall dispatcher)", PORTED,
       ("adapters.slime_r3plan", "adapters.megatron_r3plan", "r3plan"), family="MoE with rollout routing replay (R3)",
       conflicts=("hybridep_overlap",),
       notes="First step verified exactly, per-step device check, falls back to Megatron's path on mismatch. "
             "GPU code NOT tested in this repo."),
    _C("expert_placement", MODEL_FAMILY,
       "Re-place experts across EP ranks from R3 routing counts when imbalance is high (auto-off otherwise).",
       ("slime", "verl"), False,
       "slime: ASRT_CHAIN_MEGATRON_INIT+=slime_placement.megatron_init; verl: ASRT_VERL+=place", PORTED,
       ("adapters.slime_placement", "adapters.verl_placement", "adapters.megatron_placement",
        "adapters.place_verify", "placement_py"), family="MoE",
       notes="Megatron backend only; not valid with group-limited routing (moe_router_num_groups). Auto-off below "
             "ASRT_PLACE_MIN_IMB (1.15). GPU / multi-rank paths NOT tested in this repo."),
    _C("hybridep_overlap", MODEL_FAMILY,
       "HybridEP dispatch/combine (Megatron flex dispatcher) with 2-chunk compute/communication overlap in the MoE layer.",
       ("slime",), False,
       "--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep + megatron_init chain; "
       "ASRT_HYBRIDEP=auto, ASRT_HYBRIDEP_CHUNKS=2", PORTED,
       ("adapters.megatron_hybridep", "adapters.slime_hybridep"), family="MoE",
       conflicts=("r3_ep_plan",),
       notes="Needs Megatron-Core with the HybridEP flex backend and DeepEP's hybrid-ep build; overlap patch checked "
             "on Megatron-Core 0.19 only (auto mode falls back otherwise). GPU code NOT tested in this repo."),
    _C("gdn_cudnn", MODEL_FAMILY,
       "cuDNN gated-delta-net kernels in training (slime only).",
       ("slime",), False, "(not ported)", STUB, ("stubs.gdn_cudnn",), family="GDN (gated delta net) linear attention"),
    _C("fla_nb_bucket", MODEL_FAMILY,
       "Stop FLA conv1d/l2norm kernels from recompiling for every new packed length (varlen RL).",
       ("slime", "verl"), False, "(not ported)", STUB, ("stubs.fla_nb_bucket",), family="FLA / GDN linear attention"),
    _C("gdn_convfuse_ep", MODEL_FAMILY,
       "GDN conv fusion + four EP communication items from the training-speed line.",
       ("slime",), False, "(not ported)", STUB, ("stubs.gdn_convfuse_ep",), family="GDN / MoE"),
]

_BY_NAME: Dict[str, Component] = {c.name: c for c in COMPONENTS}
assert len(_BY_NAME) == len(COMPONENTS), "duplicate component name"


def get(name: str) -> Component:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown component {name!r}; known: {sorted(_BY_NAME)}") from None


def list_components(framework: Optional[str] = None, category: Optional[str] = None,
                    status: Optional[str] = None) -> List[Component]:
    out = []
    for c in COMPONENTS:
        if framework and not c.applies_to(framework):
            continue
        if category and c.category != category:
            continue
        if status and c.status != status:
            continue
        out.append(c)
    return out


def resolve(framework: str, components="default") -> List[Component]:
    """Turn a component selection into Component objects for ``framework``.

    ``components``: ``"default"``, ``"all"`` (every ported component of the framework), ``"none"``, or an iterable /
    comma string of names; a name prefixed with ``-`` removes it (``"default,-tool_hedge"``).  Explicitly named stubs,
    components of another framework and conflicting pairs raise ``ValueError``; ``default``/``all`` never include stubs.
    """
    if framework not in FRAMEWORKS:
        raise ValueError(f"framework must be one of {FRAMEWORKS}, got {framework!r}")
    if components is None:
        components = "default"
    items: Iterable[str] = components.split(",") if isinstance(components, str) else list(components)
    chosen: List[str] = []
    for raw in items:
        n = (raw or "").strip()
        if not n:
            continue
        if n.startswith("-"):
            n = n[1:].strip()
            get(n)
            chosen = [x for x in chosen if x != n]
            continue
        if n in ("default", "all", "none"):
            if n == "none":
                chosen = []
                continue
            for c in COMPONENTS:
                if c.applies_to(framework) and c.status == PORTED and (n == "all" or c.default) and c.name not in chosen \
                        and not any(o in chosen for o in c.conflicts):
                    chosen.append(c.name)
            continue
        c = get(n)
        if not c.applies_to(framework):
            raise ValueError(f"component {n!r} does not apply to {framework} (frameworks: {c.frameworks})")
        if c.status != PORTED:
            raise ValueError(f"component {n!r} is a stub (not ported yet, see docs/PORTING.md)")
        if n not in chosen:
            chosen.append(n)
    out = [get(n) for n in chosen]
    names = set(chosen)
    for c in out:
        for other in c.conflicts:
            if other in names:
                raise ValueError(f"components {c.name!r} and {other!r} cannot be enabled together")
    return out
