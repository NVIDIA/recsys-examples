"""verl: engine-side (SGLang) components, switched on through the verl adapter.

    VERL_USE_EXTERNAL_MODULES=ai_search_plugin.adapters.verl_plugin
    ASRT_VERL=...,sglhooks            # load asrt's SGLang hooks into verl's SGLang processes
    ASRT_SGL_HOOKS=nosync,fastload    # which ones (default when unset: nosync,fastload)
    ASRT_VERL=...,inplace             # weight sync without stopping generation (implies sglhooks + keepkv,
                                      # adds inplace,fastload to ASRT_SGL_HOOKS)

How the hooks get into SGLang: verl's SGLangHttpServer Ray actor runs SGLang's tokenizer manager in-process and
spawns the scheduler (TP worker) processes with the actor's environment.  The plugin runs in that actor (verl
imports it in every process), so ``install_sglhooks`` (1) registers the import hooks there (tokenizer side) and
(2) prepends ``ai_search_plugin/adapters/sgl_boot`` (a sitecustomize) + the package root to ``PYTHONPATH`` in
``os.environ`` so every scheduler subprocess loads them at interpreter start.  Nothing in verl or the launch
command changes.

Engine hooks (ai_search_plugin.adapters.slime_sglang_hooks; the same code slime uses):
  nosync    drops SGLang's per-step GPU->CPU check of return_top_p_log_probs.  Only some SGLang builds
            (e.g. the 0.5.21 fork shipped in the slime image) have that check; SGLang 0.5.20 does not,
            and the hook then installs nothing (logs "not needed").
  fastload  record / replay of the parameter copies of model.load_weights.  verl calls update_weights_from_tensor
            with plain named tensors (each its own CUDA-IPC storage, HF names, fused experts), so the tensor-path
            variant records each copy's source as (input index, offset) and coalesces the per-expert copies of a
            fused expert tensor into one strided copy.  ``fastload_check`` runs the original loader once per layout
            and compares fingerprints.
  inplace   an un-paused tensor update skips the tokenizer's writer lock (= drain) and the cache flush and is
            loaded on the scheduler's forward stream between two iterations.
  tokstat   0.5 s token timeline + pause / weight-update events (sgl-tokstat-<pid>.jsonl).
  profile   per-call load time (sgl-weightload-<pid>.jsonl, op "tload").

verl side of ``inplace`` (this module): CheckpointEngineManager.update_weights no longer aborts the replicas
or resumes them afterwards (abort_replicas / resume_generation_replicas are no-ops inside it), plus ``keepkv``
(no KV release / resume, no flush).  Requests keep generating during the transfer; a request that spans the sync
has tokens from old, mixed and new weights (the logprobs SGLang returns are still the behaviour logprobs).
verl's version tags: such a trajectory reports the version at completion (min = max), so the
trajectory_spans metric undercounts spanning requests; the replay buffer's staleness / drop logic uses the
prompt's submit version and is unchanged.
"""
import contextvars
import os

from ._hooks import keep_attrs, when_imported

DEFAULT_HOOKS = "nosync,fastload"
_STATE = {"sglhooks": None, "inplace": {"abort_skipped": 0, "resume_skipped": 0}}
_IN_SYNC = contextvars.ContextVar("asrt_verl_inplace_sync", default=False)


def _here():
    return os.path.dirname(os.path.abspath(__file__))


def install_sglhooks(extra=(), defaults=True) -> set:
    """defaults=False: when ASRT_SGL_HOOKS is unset, install only `extra` (not DEFAULT_HOOKS). 
    kvfresh needs only its own engine hook; pulling in fastload/nosync re-enabled what ASRT_VERL_DROP=sglhooks had
    turned off on purpose (e.g. a model where fastload was seen to drift)."""
    env = os.environ.get("ASRT_SGL_HOOKS")
    base = env if env else (DEFAULT_HOOKS if defaults else "")
    hooks = {h.strip() for h in base.split(",") if h.strip()}
    hooks |= set(extra)
    hooks.discard("off")
    os.environ["ASRT_SGL_HOOKS"] = ",".join(sorted(hooks))
    boot = os.path.join(_here(), "sgl_boot")
    root = os.path.dirname(os.path.dirname(_here()))  # the directory that contains ai_search_plugin/
    pp = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
    new = [boot] + ([root] if root not in pp else []) + [p for p in pp if p != boot]
    os.environ["PYTHONPATH"] = ":".join(new)
    from . import slime_sglang_hooks
    slime_sglang_hooks.register_import_hooks()
    _STATE["sglhooks"] = sorted(hooks)
    print(f"[asrt] verl engine hook set: ASRT_SGL_HOOKS={os.environ['ASRT_SGL_HOOKS']} (pid {os.getpid()})", flush=True)
    return hooks


def install_inplace() -> bool:
    from . import verl_rollout
    verl_rollout.install_keepkv()
    if os.environ.get("ASRT_VERL_KVEPOCH", "1") == "0" and os.environ.get("ASRT_VERL_INPLACE_STALE_KV", "0") != "1":
        # in-place sync keeps the running requests (no abort), so their KV and SGLang's prefix cache stay
        # computed with the old weights and are reused under the new ones (this verl image's SGLang cannot flush the
        # prefix cache while requests are queued) - multi-turn agents then sample from mixed versions (measured:
        # train/rollout logp diff 0.017 -> 0.12-0.33).  keepkv installs the kvepoch engine hook by default (
        # new requests never match old-epoch prefix KV); without it (ASRT_VERL_KVEPOCH=0) inplace is off unless forced.
        print("[asrt] verl inplace: OFF (kvepoch off -> stale prefix KV across weight syncs; ASRT_VERL_INPLACE_STALE_KV=1 "
              "to force) - keepkv stays on", flush=True)
        return False
    # fastload keeps the scheduler thread short during the un-paused load; ASRT_VERL_INPLACE_FASTLOAD=0 = original loader
    fl = ("fastload",) if os.environ.get("ASRT_VERL_INPLACE_FASTLOAD", "1") != "0" else ()
    install_sglhooks(extra=("inplace",) + fl, defaults=False)   # only what inplace needs (no implicit nosync)
    st = _STATE["inplace"]

    def on_ckpt(mod):
        M = mod.CheckpointEngineManager
        try:
            from verl.utils.ray_utils import auto_await
        except Exception:  # pragma: no cover
            return

        def skip_in_sync(name, key):
            fn = M.__dict__.get(name)
            inner = getattr(fn, "__wrapped__", None)
            if fn is None or inner is None or getattr(fn, "__asrt_inplace__", False):
                return

            async def f(self, *a, **k):
                if _IN_SYNC.get():
                    st[key] += 1
                    return None
                return await inner(self, *a, **k)
            new = keep_attrs(auto_await(f), fn)
            new.__asrt_inplace__ = True
            setattr(M, name, new)

        skip_in_sync("abort_replicas", "abort_skipped")
        skip_in_sync("resume_generation_replicas", "resume_skipped")
        uw = M.__dict__.get("update_weights")
        inner = getattr(uw, "__wrapped__", None)
        if uw is None or inner is None or getattr(uw, "__asrt_inplace__", False):
            return

        async def update_weights(self, global_steps=None, *a, **k):
            if self.backend == "naive":  # colocated: weights only move while the engine sleeps
                return await inner(self, global_steps, *a, **k)
            tok = _IN_SYNC.set(True)
            try:
                return await inner(self, global_steps, *a, **k)
            finally:
                _IN_SYNC.reset(tok)
        new = auto_await(update_weights)
        new.__asrt_inplace__ = True
        M.update_weights = new

    when_imported("verl.checkpoint_engine.base", on_ckpt)
    return True


def stats() -> dict:
    return dict(_STATE)
