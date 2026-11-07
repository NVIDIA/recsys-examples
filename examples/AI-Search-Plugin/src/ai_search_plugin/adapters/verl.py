"""verl adapter (verl v1 engine workers, 2026-10 main branch).

Turn on without changing verl or the launch command, in every verl process
(driver/TaskRunner, AgentLoopWorkers, actor workers):

    pip install ai-search-plugin   (or PYTHONPATH=<repo>/src:$PYTHONPATH)
    VERL_USE_EXTERNAL_MODULES=ai_search_plugin.adapters.verl_plugin   # verl imports it in every process
    ASRT_VERL=toolcache,ceiling,sglhooks                  # pick features ('' or 'off' = plain verl)

Features (each also callable directly from Python):

  toolcache  `install_tool_cache()` wraps `ToolAgentLoop._call_tool`: identical
             tool calls (same tool name + same arguments, string values
             case/space-normalized) are served from a TTL cache, concurrent
             duplicates share one real call (single-flight); failures are not
             cached. Only tools matching ASRT_VERL_TOOLCACHE_TOOLS (fnmatch
             patterns, default '*search*,*retriev*') are cached - stateful
             tools must not be. Custom AgentLoops that call a module-level
             search function: ASRT_TOOLCACHE_TARGETS=mod.attr,... (same as the
             slime adapter). Cache is per AgentLoopWorker process.
  balance    `install_balanced_partitions()` replaces
             `get_seqlen_balanced_partitions` in verl.utils.seqlen_balancing and
             in every module that imported it by name (ray_trainer, v1
             trainer_base, sft_trainer_ray): DP balancing (`_balance_batch`),
             dynamic micro-batching (`rearrange_micro_batches`) and group
             balancing all go through it. verl already passes FLOP-like
             workloads (24576*L + L^2, 7B-dense constants); with
             ASRT_VERL_COST="a,b" we also replace `calculate_workload` by
             a*L + b*L^2 (model-specific: small b for linear/sparse attention).
             Mode ASRT_VERL_BALANCE=best (default) runs verl's Karmarkar-Karp
             and asrt's LPT+swap and keeps the lower max load (never worse);
             'asrt' always uses ours.
  place      Megatron backend only: expert re-placement from R3 routing counts (verl_placement); skips itself
             when the EP-rank imbalance is below ASRT_PLACE_MIN_IMB (1.15) and on dense models.
  response_logits  Megatron backend only: output layer on response rows only (ai_search_plugin.response_logits).
  route      prefix/group-affinity cost routing of new requests (verl_rollout / verl_route).
  keepkv     keep SGLang KV + prefix cache across weight syncs (verl_rollout).
  tokcache   tokenize each prompt once per AgentLoopWorker (verl_rollout).
  profile    timing only: weight-sync phases + agent-loop tokenization (verl_rollout).
  offpol     measurement only: v1 replay buffer submitted / trained / dropped (max_off_policy_threshold) per step
             (verl_offpol; verl-offpol-*.jsonl + "[asrt] verl offpol" lines). Fine on the plain verl baseline.
  sglhooks   load asrt's SGLang engine hooks (ASRT_SGL_HOOKS, default nosync,fastload) into verl's SGLang
             processes (verl_engine).
  inplace    weight sync without stopping generation: no abort / resume around the sync, keepkv, and the
             engine-side inplace + fastload hooks (verl_engine).
  ceiling    hard staleness ceiling S (ASRT_MAX_STALENESS, default 2) in verl's own unit
             training/off_policy/trajectory_staleness_worst, for the v1 async trainers (verl_ceiling): ticket admission
             of prompts, deadline-ordered sampling + wait at the deadline, budgeted cut; verl's own
             max_off_policy_threshold drop stays underneath.  Not in `full` (it is a fairness rule, not a speed-up).

Not applicable / not done in verl (reasons):
  weight-sync bucketing+pipelining: verl's NCCL checkpoint engine already packs
     each bucket into one flat buffer and double-buffers send/recv
     (`update_weights_bucket_megabytes`); the slime gain came from exactly that.
  preprocess prefetch: verl's v1 trainer moves batches with TransferQueue /
     DataProto dispatch, there is no slime-style per-rank `ray.get` of the
     rollout batch to overlap; R3 routes are already int16 in the agent loop
     (`agent_loop.py`, `routed_experts` int16).
"""
import fnmatch
import json
import os
import time

from .common import CachedTool, query_key  # noqa: F401
from ._hooks import patch_method, when_imported

solvers = None  # no C++ extension in ai_search_plugin (pure-Python partitioner / router only)

_STATS = {"balance_calls": 0, "balance_ours": 0, "balance_gain_sum": 0.0, "toolcache": None}
_INSTALLED = set()

_BALANCE_IMPORTERS = [
    "verl.trainer.ppo.ray_trainer",
    "verl.trainer.ppo.v1.trainer_base",
    "verl.trainer.sft_trainer_ray",
    "verl.experimental.separation.ray_trainer",
]
_PATCH_TARGETS = ["verl.utils.seqlen_balancing"] + _BALANCE_IMPORTERS


def cached(search_fn, **kw) -> CachedTool:
    return CachedTool(search_fn, **kw)


def balanced_partitions(values, k, equal_size=True, quad_coef=0.0):
    """asrt partitioner; C++ when the extension is built, else pure Python."""
    if solvers is not None and hasattr(solvers, "balanced_partitions"):
        from .common import balanced_partitions as cpp
        return cpp(values, k, equal_size, quad_coef)
    from ..partition_py import balanced_partitions as py
    return py(values, k, equal_size, quad_coef)


def _log(kind, rec):
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"verl-{kind}-{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}) + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------- balance

def _make_partitioner(orig, quad_coef, mode):
    def get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size):
        vals = [float(x) for x in (seqlen_list.tolist() if hasattr(seqlen_list, "tolist") else seqlen_list)]
        ours = balanced_partitions(vals, k_partitions, equal_size, quad_coef)
        if any(len(p) == 0 for p in ours):
            ours = None
        _STATS["balance_calls"] += 1
        if mode == "asrt" and ours is not None:
            _STATS["balance_ours"] += 1
            return ours
        base = orig(seqlen_list, k_partitions, equal_size)
        if ours is None:
            return base

        def mx(parts):
            return max(sum(vals[i] + quad_coef * vals[i] * vals[i] for i in p) for p in parts)
        mo, mb = mx(ours), mx(base)
        if mo < mb:
            _STATS["balance_ours"] += 1
            _STATS["balance_gain_sum"] += (mb - mo) / mb
            out = ours
        else:
            out = base
        if (_STATS["balance_calls"] - 1) % int(os.environ.get("ASRT_VERL_BALANCE_LOG_EVERY", 50)) == 0:
            rec = {"n": len(vals), "k": k_partitions, "equal": bool(equal_size), "max_verl": mb, "max_asrt": mo,
                   **{k: v for k, v in _STATS.items() if k.startswith("balance")}}
            print(f"[asrt] verl balance {rec}", flush=True)
            _log("balance", rec)
        return out
    get_seqlen_balanced_partitions.__asrt_patched__ = True
    return get_seqlen_balanced_partitions


def install_balanced_partitions(quad_coef: float = 0.0, mode: str = None, cost=None) -> list:
    """Patch verl's partitioner (+ optionally the workload model). Modules not yet
    imported are patched on import. Returns modules patched so far."""
    mode = mode or os.environ.get("ASRT_VERL_BALANCE", "best")
    cost = cost or os.environ.get("ASRT_VERL_COST")
    patched = []
    state = {"impl": None}

    def on_sb(mod):
        orig = mod.get_seqlen_balanced_partitions
        if getattr(orig, "__asrt_patched__", False):
            state["impl"] = orig
            return
        state["impl"] = _make_partitioner(orig, quad_coef, mode)
        state["orig"] = orig
        mod.get_seqlen_balanced_partitions = state["impl"]
        if cost:
            a, b = (float(x) for x in str(cost).split(","))

            def calculate_workload(seqlen_list):
                return a * seqlen_list + b * seqlen_list ** 2
            state["cw_orig"] = mod.calculate_workload
            state["cw"] = calculate_workload
            mod.calculate_workload = calculate_workload
        patched.append(mod.__name__)

    def on_importer(mod):
        import verl.utils.seqlen_balancing  # noqa: F401  (fires on_sb first)
        if getattr(mod, "get_seqlen_balanced_partitions", None) is state.get("orig"):
            mod.get_seqlen_balanced_partitions = state["impl"]
            patched.append(mod.__name__)
        if "cw" in state and getattr(mod, "calculate_workload", None) is state["cw_orig"]:
            mod.calculate_workload = state["cw"]

    import importlib.util
    try:
        have_verl = importlib.util.find_spec("verl") is not None
    except Exception:
        have_verl = False
    if not have_verl:
        return []
    when_imported("verl.utils.seqlen_balancing", on_sb)
    for name in _BALANCE_IMPORTERS:
        when_imported(name, on_importer)
    _INSTALLED.add("balance")
    return patched


# ----------------------------------------------------------------------------- tool cache

def _norm_args(args):
    if isinstance(args, str):
        return " ".join(args.lower().split())
    if isinstance(args, dict):
        return {k: _norm_args(v) for k, v in sorted(args.items())}
    if isinstance(args, list):
        return [_norm_args(v) for v in args]
    return args


class VerlToolCache:
    """Cache in front of ToolAgentLoop._call_tool (keyed by tool name + args)."""

    def __init__(self, patterns=None, capacity=None, ttl_s=None):
        pats = patterns or os.environ.get("ASRT_VERL_TOOLCACHE_TOOLS", "*search*,*retriev*")
        self.patterns = [p.strip() for p in pats.split(",") if p.strip()]
        self._pending = {}
        cap = int(capacity or os.environ.get("ASRT_TOOLCACHE_CAPACITY", 1 << 20))
        ttl = float(ttl_s or os.environ.get("ASRT_TOOLCACHE_TTL_S", 7 * 86400))
        self.tool = CachedTool(self._run, capacity=cap, ttl_s=ttl, key_fn=query_key)
        self.bypass = 0

    def wants(self, name):
        return any(fnmatch.fnmatch(name, p) for p in self.patterns)

    async def _run(self, key):
        call = self._pending[key]
        resp, reward, res = await call()
        if any(getattr(resp, a, None) is not None for a in ("image", "video")):
            raise _Uncacheable((resp, reward, res))
        return json.dumps({"text": resp.text, "reward": reward, "res": res}, default=str)

    def stats(self):
        s = self.tool.stats()
        n = s["hits"] + s["joins"] + s["misses"]
        s["real_call_ratio"] = round(s["misses"] / n, 4) if n else None
        s["bypass"] = self.bypass
        return s


class _Uncacheable(Exception):
    def __init__(self, value):
        self.value = value


def install_tool_cache(patterns=None) -> bool:
    """Wrap verl's ToolAgentLoop._call_tool (now or on import)."""
    tc = VerlToolCache(patterns)
    _STATS["toolcache"] = tc
    every = int(os.environ.get("ASRT_TOOLCACHE_LOG_EVERY", 200))

    def on_import(mod):
        from verl.tools.schemas import ToolResponse

        def make(orig):
            async def _call_tool(self, tool_call, tools_kwargs, agent_data):
                name = getattr(tool_call, "name", "")
                if not tc.wants(name):
                    tc.bypass += 1
                    return await orig(self, tool_call, tools_kwargs, agent_data)
                try:
                    args = json.loads(tool_call.arguments)
                except Exception:
                    return await orig(self, tool_call, tools_kwargs, agent_data)
                key = name + "\x00" + json.dumps(_norm_args(args), sort_keys=True)
                tc._pending.setdefault(key, lambda: orig(self, tool_call, tools_kwargs, agent_data))
                try:
                    out = json.loads(await tc.tool(key))
                except _Uncacheable as u:
                    return u.value
                finally:
                    tc._pending.pop(key, None)
                n = sum(tc.tool.stats()[k] for k in ("hits", "joins", "misses"))
                if every and n % every == 0:
                    st = tc.stats()
                    print(f"[asrt] verl toolcache {st}", flush=True)
                    _log("toolcache", st)
                return ToolResponse(text=out["text"]), out["reward"], out["res"]
            return _call_tool
        patch_method(mod.ToolAgentLoop, "_call_tool", make)

    when_imported("verl.experimental.agent_loop.tool_agent_loop", on_import)
    targets = os.environ.get("ASRT_TOOLCACHE_TARGETS")
    if targets:
        from .slime_toolcache import install as install_targets  # generic module.attr patcher
        for t in targets.split(","):
            mod = t.strip().rpartition(".")[0]
            when_imported(mod, lambda _m, _t=t.strip(): install_targets([_t]))
    _INSTALLED.add("toolcache")
    return True


def stats() -> dict:
    out = {k: v for k, v in _STATS.items() if k != "toolcache"}
    if _INSTALLED & {"route", "keepkv", "tokcache", "profile"}:
        from .verl_rollout import stats as rstats
        out["rollout"] = rstats()
    if _INSTALLED & {"sglhooks", "inplace"}:
        from .verl_engine import stats as estats
        out["engine"] = estats()
    if _STATS["toolcache"] is not None:
        out["toolcache"] = _STATS["toolcache"].stats()
    try:
        from .slime_toolcache import stats as target_stats
        ts = target_stats()
        if ts:
            out["toolcache_targets"] = ts
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- entry

# `ASRT_VERL=full`: everything that applies to verl and has shown a gain somewhere, each one self-guarding
# ("nothing to fix -> costs nothing"). `ASRT_VERL_DROP=a,b` removes items (ablation).
#   profile   timing only (sync phases, tokenization)                           cost ~0
#   route     group/prefix-affinity engine choice                               no shared prefix -> plain least-loaded
#   keepkv    keep KV/prefix cache across weight syncs                          no request in flight at sync -> no-op
#   tokcache  tokenize each prompt once per agent worker                        n=1 -> one dict lookup
#   balance   best-of(verl KK, asrt LPT+swap) DP/micro-batch partition          never worse than verl's own
#   toolcache cache identical search calls (tools matching *search*/*retriev*)  no repeats -> single-flight only
#   response_logits  output layer only on response rows in the Megatron actor; falls back to the stock path when its
#             preconditions fail (e.g. use_fused_kernels)
#   sglhooks  SGLang engine hooks inside verl's rollout servers (ai_search_plugin.adapters.verl_engine; default
#             nosync,fastload: nosync installs nothing where SGLang has no per-step check, fastload falls back to
#             the stock loader for layouts it cannot replay). `engine` is an alias.
FULL = ["profile", "offpol", "route", "keepkv", "inplace", "tokcache", "balance", "toolcache", "response_logits", "sglhooks", "jitcache"]
# NOTE (ai_search_plugin): `full` is the original code base's everything-on list and contains items whose benefit is
# still undecided (route / keepkv / tokcache / balance).  ai_search_plugin.install() never uses `full`; it passes an
# explicit list built from the component registry.
# jitcache (): no-op unless ASRT_JITCACHE_DIR is set
_ALIAS = {"engine": "sglhooks"}


def expand(features) -> list:
    """'full' -> FULL, minus env ASRT_VERL_DROP; keeps order, drops duplicates/'off'."""
    if isinstance(features, str):
        features = [f.strip() for f in features.split(",")]
    out = []
    for f in features:
        f = (f or "").strip()
        if not f or f == "off":
            continue
        for g in (FULL if f == "full" else [_ALIAS.get(f, f)]):
            if g not in out:
                out.append(g)
    drop = {_ALIAS.get(d.strip(), d.strip()) for d in os.environ.get("ASRT_VERL_DROP", "").split(",") if d.strip()}
    return [f for f in out if f not in drop]


def _install_response_logits():
    """response_logits in verl's Megatron actor processes (ai_search_plugin.response_logits.adapters.verl), installed
    when verl's Megatron engine module is imported (actor workers only: rollout servers / agent workers never import
    it, so they pay nothing), plus the per-micro-batch guard below (_guard_response_logits).  The adapter itself falls
    back to the stock path when its preconditions fail (e.g. use_fused_kernels=True, value model, MTP training)."""

    def on_engine(mod):
        res = {}
        try:
            from ..response_logits.adapters import verl as RV
            res["response_logits"] = RV.install()
        except Exception as e:  # torch / verl / megatron missing or incompatible -> stock path
            res["response_logits_error"] = repr(e)[:200]
        res["rlogits_guard"] = _guard_response_logits()
        print(f"[asrt] verl response_logits pid={os.getpid()} {res}", flush=True)
        _log("response_logits", res)

    when_imported("verl.workers.engine.megatron.transformer_impl", on_engine)
    return True


_RL = {"compact": 0, "stock": 0}


def _rlogits_kept_rows(loss_mask):
    """Response rows that carry loss in this micro-batch, for nested loss_mask only (None otherwise): loss_mask sum,
    i.e. tool / search-result tokens inside the response are not counted (multi-turn responses are mostly tool
    output). Padded loss_mask ([b, R], verl v1's left_right_2_no_padding path) returns None = keep the compact path,
    as before: that is the common path (measured: compact about -10% update_actor vs off) and the guard
    never acted on it; changing it needs data first."""
    if loss_mask is None or not getattr(loss_mask, "is_nested", False):
        return None
    return int((loss_mask.values() != 0).sum())


def _guard_response_logits():
    """'Nothing to fix -> costs nothing' for response_logits in verl: per micro-batch, when the response
    rows that carry loss (_rlogits_kept_rows: loss_mask sum, tool / search tokens not counted; nested loss_mask only,
    padded input always compact) are more than ASRT_VERL_RLOGITS_MAX_FRAC (0.5) of all rows (multi-turn: the whole conversation after the first
    prompt is 'response'), call verl's own forward instead (the compact path only adds index work there; measured
    +3% update_actor on a small multi-turn run). Wraps response_logits.adapters.verl._wrap_forward_fn (looked up at call time by
    its getter wrapper). Returns True when installed."""
    try:
        import inspect
        from ..response_logits.adapters import verl as RV
    except Exception:
        return False
    orig_wrap = getattr(RV, "_wrap_forward_fn", None)
    if orig_wrap is None or getattr(orig_wrap, "_asrt_guard", False):
        return orig_wrap is not None
    thr = float(os.environ.get("ASRT_VERL_RLOGITS_MAX_FRAC", 0.5))
    every = int(os.environ.get("ASRT_VERL_RLOGITS_LOG_EVERY", 200))

    cache = {}

    def _wrap_forward_fn(fn):
        if fn in cache:
            return cache[fn]
        wrapped = orig_wrap(fn)
        if wrapped is fn:
            return fn
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return wrapped

        def forward_fn(*args, **kwargs):
            use = True
            try:
                a = sig.bind(*args, **kwargs).arguments
                lpa = a.get("logits_processor_args") or {}
                kept = _rlogits_kept_rows(lpa.get("loss_mask"))
                ids = a.get("input_ids")
                if kept is not None and getattr(ids, "is_nested", False):
                    use = kept <= thr * max(1, ids.values().numel())
            except Exception:
                pass
            _RL["compact" if use else "stock"] += 1
            n = _RL["compact"] + _RL["stock"]
            if every and n % every == 1:
                _log("rlogits", dict(_RL))
            return wrapped(*args, **kwargs) if use else fn(*args, **kwargs)
        forward_fn._response_logits = True
        cache[fn] = forward_fn
        return forward_fn

    _wrap_forward_fn._asrt_guard = True
    RV._wrap_forward_fn = _wrap_forward_fn
    return True


def install(features=None) -> set:
    """Install the features listed in `features` (iterable or comma string,
    default env ASRT_VERL; 'full' = FULL minus ASRT_VERL_DROP). Safe to call in
    every process, idempotent."""
    if features is None:
        features = os.environ.get("ASRT_VERL", "")
    raw = features.split(",") if isinstance(features, str) else list(features)
    explicit = {(x or "").strip() for x in raw}
    feats = expand(features)
    from .verl_rollout import kvfresh_wanted
    if kvfresh_wanted():  # ASRT_VERL_KVFRESH=1
        bad = [f for f in feats if f in ("keepkv", "inplace")]
        if bad:
            raise RuntimeError(f"asrt verl kvfresh: {bad} keep old-weight KV on purpose; set ASRT_VERL_DROP=keepkv,inplace")
        if "kvfresh" not in feats:
            feats.append("kvfresh")
        explicit.add("kvfresh")
    for f in feats:
        if f in _INSTALLED:
            continue
        try:
            _install_one(f)
        except Exception as e:
            if f in explicit:  # asked for by name -> fail loudly
                raise
            # came in through 'full': a component that cannot load here is skipped (plain verl for that part)
            print(f"[asrt] verl full: {f} not installed ({e!r}) -> skipped", flush=True)
            _log("skipped", {"feature": f, "error": repr(e)[:300]})
    return set(_INSTALLED)


def _install_one(f):
    if f in ("response_logits", "rlogits"):
        _install_response_logits()
        _INSTALLED.add("response_logits")
    elif f == "jitcache":
        from . import jitcache
        jitcache.install()
        _INSTALLED.add(f)
    elif f == "toolcache":
        install_tool_cache()
    elif f == "balance":
        q = float(os.environ.get("ASRT_VERL_QUAD", 0.0))
        install_balanced_partitions(q)
    elif f == "place":  # expert re-placement (verl_placement; Megatron engine, MoE only, auto-off when balanced)
        from .verl_placement import install as install_place
        install_place()
        _INSTALLED.add(f)
    elif f == "sglhooks":
        from .verl_engine import install_sglhooks
        install_sglhooks()
        _INSTALLED.add(f)
    elif f == "inplace":
        from .verl_engine import install_inplace
        install_inplace()
        _INSTALLED.update({"inplace", "keepkv", "sglhooks"})
    elif f == "ceiling":  # staleness ceiling S (verl_ceiling)
        from .verl_ceiling import install as install_ceiling
        install_ceiling()
        _INSTALLED.add(f)
    elif f == "offpol":  # measurement only: submitted / trained / dropped per step (verl_offpol)
        from .verl_offpol import install as install_offpol
        install_offpol()
        _INSTALLED.add(f)
    elif f == "kvfresh":  # checked flush + earliest version (verl_rollout.install_kvfresh)
        if _INSTALLED & {"keepkv", "inplace"}:
            raise RuntimeError("asrt verl kvfresh: keepkv/inplace keep old-weight KV on purpose; drop them "
                               "(ASRT_VERL_DROP=keepkv,inplace)")
        from . import verl_rollout
        verl_rollout.install_kvfresh()
        _INSTALLED.add(f)
    elif f == "kvprobe":  # measurement only (kvsync / kvcall logs)
        from . import verl_rollout
        verl_rollout.install_kvprobe()
        _INSTALLED.add(f)
    elif f in ("route", "keepkv", "tokcache", "profile"):
        from . import verl_rollout
        getattr(verl_rollout, "install_" + f)()
        _INSTALLED.add(f)
    else:
        raise ValueError(f"unknown asrt verl feature {f!r}")


