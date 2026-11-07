"""verl rollout-side components (verl v1 separate_async + SGLang standalone rollout).

Turned on through the same switch as the rest of the verl adapter
(VERL_USE_EXTERNAL_MODULES=ai_search_plugin.adapters.verl_plugin, ASRT_VERL=<list>); each
feature is a post-import patch, no verl source change, absent = plain verl.

  route    Prefix/group-affinity routing (cost router).
           verl routes a new request id to the least-in-flight server, so the
           n samples of one prompt (different request ids, same prompt) land on
           different SGLang engines and each engine prefills the whole prompt
           (measured: 27% prefix hit, 75% possible with n=4). We hash the
           prompt on the client (LLMServerClient._acquire_server) and install a
           GlobalRequestLoadBalancer subclass (ai_search_plugin.adapters.verl_route) that,
           for a request id it has not seen, picks
               argmin_s  inflight(s) + uncached_tokens(s) / ASRT_VERL_ROUTE_TOKEQ
           (uncached = 0 on the server that last got this prompt, else the
           prompt length). Sticky multi-turn routing is unchanged.
           ASRT_VERL_ROUTE_TOKEQ (default 4096): how many prompt tokens of
           re-prefill are worth one extra request in flight.
  keepkv   Keep the KV / prefix cache across weight syncs. verl's
           CheckpointEngineManager.update_weights aborts all requests, frees the
           KV pool (release_kv_cache), and after loading flushes the radix
           cache (SGLang update_weights_from_tensor flush_cache=True per bucket +
           an explicit flush_cache + resume_kv_cache's flush). Every request in
           flight at a sync is then resumed by FullyAsyncLLMServerClient with
           prompt + partial output and re-prefilled from scratch, and every
           cached prefix (group siblings, multi-turn histories) is lost.
           With keepkv the KV pool stays allocated (needs 2 x bucket of free
           GPU memory for the NCCL receive buffers) and nothing is flushed, so a
           resumed request hits the cache for everything it had computed.
           Same trade-off as slime --flush-cache-interval 0 / PipelineRL: the
           reused KV was computed with the previous weights (the behaviour
           log-probs SGLang returns are still those actually used for sampling,
           so bypass-mode / TIS training stays consistent).
  tokcache Per-AgentLoopWorker LRU cache of AgentLoopBase.ct_build_initial_tokens
           keyed by the chat messages: the n samples of a prompt tokenize it once.
  profile  Timing only: per weight sync, the phases of
           CheckpointEngineManager.update_weights (abort / release KV / process
           group / transfer / finalize / resume KV / resume generation), and per
           agent-loop worker the tokenization latency incl. executor queueing.
           Written to $ASRT_LOG_DIR/verl-sync-<pid>.jsonl, verl-tok-<pid>.jsonl.
"""
import asyncio
import contextvars
import functools
import hashlib
import json
import os
import random
import time
from array import array
from collections import OrderedDict

from ._hooks import keep_attrs, patch_method, when_imported

_STATS = {"route": None, "keepkv": {"release_skipped": 0, "resume_skipped": 0, "flush_skipped": 0,
                                    "bucket_flush_off": 0},
          "tokcache": {"hits": 0, "misses": 0}}


def _log(kind, rec):
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"verl-{kind}-{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}, default=str) + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------- route

def prompt_key(prompt_ids, limit=None) -> str:
    """Stable short key of a token-id prompt (first `limit` tokens + length)."""
    limit = limit or int(os.environ.get("ASRT_VERL_ROUTE_PREFIX", 1 << 17))
    ids = prompt_ids[:limit]
    try:
        b = array("q", ids).tobytes()
    except (TypeError, OverflowError):
        b = json.dumps(list(map(int, ids))).encode()
    return hashlib.blake2b(b, digest_size=8).hexdigest()


def affinity_pick(loads: dict, affinity_server, ntok: int, tok_eq: float, rng=random):
    """Cost-model pick (same rule as asrt.solvers.Router 'cost'): load in requests
    in flight; a server that does not hold the prompt pays ntok/tok_eq extra.
    Ties broken at random among the cheapest. Returns server id."""
    best, cands = None, []
    pen = (ntok / tok_eq) if tok_eq > 0 else 0.0
    for sid, n in loads.items():
        c = n + (0.0 if sid == affinity_server else pen)
        if best is None or c < best - 1e-9:
            best, cands = c, [sid]
        elif abs(c - best) <= 1e-9:
            cands.append(sid)
    return rng.choice(cands)


class AffinityState:
    """Prompt-key -> server memory used by the balancer (LRU)."""

    def __init__(self, capacity=None, tok_eq=None):
        self.cap = int(capacity or os.environ.get("ASRT_VERL_ROUTE_CAP", 200000))
        self.tok_eq = float(tok_eq if tok_eq is not None else os.environ.get("ASRT_VERL_ROUTE_TOKEQ", 4096))
        self.aff = OrderedDict()
        self.st = {"new": 0, "sticky": 0, "affinity_follow": 0, "affinity_spill": 0, "no_key": 0}

    def pick(self, loads, key, ntok):
        if key is None:
            self.st["no_key"] += 1
            return None
        a = self.aff.get(key)
        if a is not None and a not in loads:
            a = None
        sid = affinity_pick(loads, a, ntok, self.tok_eq)
        if a is None:
            self.st["new"] += 1
        elif sid == a:
            self.st["affinity_follow"] += 1
        else:
            self.st["affinity_spill"] += 1
        self.aff[key] = sid
        self.aff.move_to_end(key)
        while len(self.aff) > self.cap:
            self.aff.popitem(last=False)
        return sid


def install_route() -> bool:
    every = int(os.environ.get("ASRT_VERL_ROUTE_LOG_EVERY", 2000))

    def on_llm_server(mod):
        C = mod.LLMServerClient

        def make(orig):
            async def _acquire_server(self, request_id, **extra):
                p = extra.get("prompt_ids")
                if p is not None and "asrt_prefix_key" not in extra:
                    try:
                        extra["asrt_prefix_key"] = prompt_key(p)
                        extra["asrt_ntok"] = len(p)
                    except Exception:
                        pass
                return await orig(self, request_id, **extra)
            return _acquire_server
        patch_method(C, "_acquire_server", make)

    def on_router(mod):
        orig = mod.get_router_handle
        if getattr(orig, "__asrt_patched__", False):
            return

        @functools.wraps(orig)
        def get_router_handle(servers, router_config_path=None, full_determinism=False, load_balancer_cls=None, **kw):
            if load_balancer_cls is None and not router_config_path and not full_determinism:
                from .verl_route import AsrtAffinityBalancer
                load_balancer_cls = AsrtAffinityBalancer
                print(f"[asrt] verl route: AsrtAffinityBalancer over {len(servers)} servers", flush=True)
            return orig(servers, router_config_path=router_config_path, full_determinism=full_determinism,
                        load_balancer_cls=load_balancer_cls, **kw)
        get_router_handle.__asrt_patched__ = True
        mod.get_router_handle = get_router_handle

    os.environ.setdefault("ASRT_VERL_ROUTE_LOG_EVERY", str(every))
    when_imported("verl.workers.rollout.llm_server", on_llm_server)
    when_imported("verl.workers.rollout.router", on_router)
    return True


# ----------------------------------------------------------------------------- keepkv

_IN_UPDATE = contextvars.ContextVar("asrt_in_update", default=False)


def install_keepkv() -> bool:
    ks = _STATS["keepkv"]
    # keeping the prefix cache across weight syncs is only correct if no request reuses KV the OLD weights
    # computed: the engine hook kvepoch tags each new request's radix key with a weight-update epoch (slime_sglang_hooks).
    # ASRT_VERL_KVEPOCH=0 turns it off (measurement of the stale-KV effect only).
    if os.environ.get("ASRT_VERL_KVEPOCH", "1") != "0":
        from .verl_engine import install_sglhooks
        install_sglhooks(extra=("kvepoch",), defaults=False)   # only kvepoch; fastload/nosync come from 'sglhooks'

    def on_server(mod):
        S = getattr(mod, "SGLangHttpServer", None)
        if S is None:
            return

        def make_noop(name, key):
            def make(orig):
                async def f(self, *a, **k):
                    ks[key] += 1
                    # the counters live in the SGLang server actor, not in the trainer that writes verl-sync:
                    # dump them here (once per sync event) so a run shows whether keepkv actually took effect
                    _log("keepkv", {"event": key, **ks})
                    return None
                return f
            return make
        def make_resume(orig):
            async def f(self, *a, **k):
                # keep the KV memory (no release/resume), but NOT the prefix cache: its entries were
                # computed with the old weights and the next turn of a multi-turn agent reuses them under the new ones
                # (measured: train/rollout logp diff 0.017 -> 0.12-0.33, policy collapse).  verl's own resume_kv_cache ends
                # with this flush; the requests were aborted before the sync, so the engine is idle and it succeeds.
                # ASRT_VERL_KEEP_PREFIX=1: old behaviour (stale prefix KV reused).
                ks["resume_skipped"] += 1
                if os.environ.get("ASRT_VERL_KEEP_PREFIX", "0") != "1" and getattr(self, "node_rank", 0) == 0:
                    try:
                        r = await self.tokenizer_manager.flush_cache()
                        if getattr(r, "success", True) is False:   # requests still running (e.g. inplace kept them)
                            raise RuntimeError("engine not idle")
                        ks["prefix_flushed"] = ks.get("prefix_flushed", 0) + 1
                    except Exception as e:
                        ks["prefix_flush_failed"] = ks.get("prefix_flush_failed", 0) + 1
                        print(f"[asrt] verl keepkv: prefix-cache flush failed: {e!r}", flush=True)
                _log("keepkv", {"event": "resume_skipped", **ks})
                return None
            return f
        patch_method(S, "release_kv_cache", make_noop("release_kv_cache", "release_skipped"))
        patch_method(S, "resume_kv_cache", make_resume)

    def on_rollout(mod):
        A = getattr(mod, "ServerAdapter", None)
        if A is None:
            return

        def make(orig):
            async def update_weights(self, *a, **k):
                tok = _IN_UPDATE.set(True)
                try:
                    return await orig(self, *a, **k)
                finally:
                    _IN_UPDATE.reset(tok)
            return update_weights
        patch_method(A, "update_weights", make)

    def on_http(mod):
        H = getattr(mod, "AsyncHttpServerAdapter", None)
        if H is None:
            return

        def make_flush(orig):
            async def flush_cache(self, *a, **k):
                if _IN_UPDATE.get():
                    ks["flush_skipped"] += 1
                    _log("keepkv", {"event": "flush_skipped", **ks})
                    return {}
                return await orig(self, *a, **k)
            return flush_cache

        def make_upd(orig):
            async def update_weights_from_tensor(self, req, *a, **k):
                if getattr(req, "flush_cache", False):
                    try:
                        req.flush_cache = False
                        ks["bucket_flush_off"] += 1
                    except Exception:
                        pass
                return await orig(self, req, *a, **k)
            return update_weights_from_tensor
        patch_method(H, "flush_cache", make_flush)
        patch_method(H, "update_weights_from_tensor", make_upd)

    when_imported("verl.workers.rollout.sglang_rollout.async_sglang_server", on_server)
    when_imported("verl.workers.rollout.sglang_rollout.sglang_rollout", on_rollout)
    when_imported("verl.workers.rollout.sglang_rollout.http_server_engine", on_http)
    return True


# ----------------------------------------------------------------------------- kvfresh

# verl side of the slime kvfresh fix.  Path: verl v1 colocate_async + naive (hybrid)
# weight update + SGLang 0.5.20 async server.  What verl already does at every sync (verified in verl main of
# 2026-10-23): pause(mode="abort") on_sample_end -> sleep (release weights + KV) -> train -> wake,
# update_weights_from_tensor per bucket (flush_cache=True) + one more flush_cache (result NOT checked,
# sglang_rollout.py:394) -> set_global_steps -> continue.  Aborted requests come back to FullyAsyncLLMServerClient,
# which resubmits prompt + every token generated so far (max_tokens reduced by them; llm_server.py:296-370): tokens and
# their behaviour logprobs are kept, nothing is resampled, and they are prefilled again under the new weights.  While
# paused they wait in the tokenizer (is_pause_cond) holding tokens and no KV.  SGLang's retract is NOT usable here:
# colocate sleeps the engine every step and release_memory_occupation asserts is_fully_idle() incl. an empty waiting
# queue (0.5.20 scheduler_components/weight_updater.py:212), and so does flush_cache (scheduler.py:4981) - retracted
# requests would sit in that queue.  So the "back to the queue, keep tokens, re-prefill" half stays verl's own, and
# kvfresh adds what is missing:
#   1. checked flush after every weight update (ServerAdapter.update_weights, TP leader of each replica, engine still
#      paused): HTTP /flush_cache must return 200, and with the engine hook kvfresh (installed here) SGLang reports
#      failure unless the radix tree holds 0 tokens afterwards; failed -> pause(abort) again (requests go back to the
#      client with their tokens, as verl's own pause does) + bounded backoff retry (slime_weightsync.kv_flush_checked:
#      ASRT_WS_KV_FLUSH_TRIES 8, 0.05 s doubling to 2 s), still failing -> ERROR + KVFreshError
#      (ASRT_WS_KV_ON_FAIL=continue: counted only).  One record per engine per sync -> verl-kvfresh-<pid>.jsonl.
#   2. per sync, server side: in-flight requests at the abort, pause / resume times -> verl-kvsync-<pid>.jsonl.
#   3. earliest version: each server call records the global_steps it STARTED under (a call that arrives during verl's
#      pause starts at the next resume, so it takes the version current at that resume, not the one at arrival - an
#      earlier version counted such calls as corrections = staleness over-reported by 1); when that differs from the one
#      it returns (a call that ran across an update without being aborted), the client lowers min_global_steps to it
#      (verl takes min over the calls' END versions), so staleness is never under-reported.  Corrections are counted
#      (expected 0 on this path: generation only runs between continue and the next abort).
#   4. probe data: per server call t0/t1/in/out/cached/e2e/session -> verl-kvcall-<pid>.jsonl (offline analysis
#      input).
#   5. R3 rows check (ASRT_VERL_R3CHECK, default on): for every trajectory resumed after an abort, the final
#      routed_experts must have prompt + response - 1 rows and keep the first call's rows unchanged (no rows from the
#      re-prefill under new weights) -> verl-r3rows-<pid>.jsonl, one record per resumed trajectory (r3_rows_check).
# Switch: ASRT_VERL_KVFRESH=1 or feature "kvfresh".  Incompatible with keepkv / inplace (they keep
# old-weight KV on purpose) -> refused.  kvepoch is not installed by kvfresh.

_KVSTART = contextvars.ContextVar("asrt_kv_segstarts", default=None)
_KVSEG = contextvars.ContextVar("asrt_kv_segments", default=None)   # per server call: (prompt_len, n_tokens, routed rows)
_R3 = {"r3_resumed": 0, "r3_ok": 0, "r3_bad": 0, "r3_unverifiable": 0, "r3_no_routes": 0}   # running counters (merged into each record)


def r3_rows_check(segs, out):
    """ R3 routes of a trajectory that needed more than one server call (resumed after a
    kvfresh/verl abort): verl keeps the first call's rows and appends only the rows of each later call's NEW tokens
    (llm_server.py:307-315), so the final array must have prompt + response - 1 rows (SGLang: positions [0, seqlen-1))
    and start with the first call's rows unchanged (not rows re-computed by the re-prefill under new weights).
    segs: [(prompt_len, n_tokens, rows or None)] per call in order.  Returns a record dict or None (not resumed / no R3)."""
    import numpy as np
    if len(segs) < 2:
        return None
    final = getattr(out, "routed_experts", None)
    if final is None:
        _R3["r3_no_routes"] += 1
        return None
    tok_calls = [sg for sg in segs if sg[1] > 0]
    if not tok_calls:
        return None
    # the first call that produced tokens: its rows must be the head of the final array.  If it returned NO rows
    # (first_call_has_rows False) the 'kept' comparison would silently use a later call's rows and pass trivially, so
    # that case is recorded as unverifiable and counted separately - never as ok.
    has_rows = tok_calls[0][2] is not None
    final = np.asarray(final)
    expected = segs[0][0] + len(out.token_ids or []) - 1
    rows_ok = final.shape[0] == expected
    if has_rows:
        first = np.asarray(tok_calls[0][2])
        kept = final.shape[0] >= first.shape[0] and bool(np.array_equal(final[:first.shape[0]], first))
        ok = rows_ok and kept
        first_rows = int(first.shape[0])
    else:
        kept, ok, first_rows = None, False, 0
    _R3["r3_resumed"] += 1
    _R3["r3_ok" if ok else ("r3_bad" if has_rows else "r3_unverifiable")] += 1
    return {"ev": "r3rows", "ok": ok, "first_call_has_rows": has_rows, "calls": len(segs),
            "calls_with_tokens": len(tok_calls), "prompt": segs[0][0], "response": len(out.token_ids or []),
            "rows": int(final.shape[0]), "expected": int(expected), "rows_match_expected": rows_ok,
            "first_call_rows": first_rows, "first_rows_kept": kept, **_R3}
_KV_HTTP = None   # tests: replaces slime_weightsync._kv_http (url, method, endpoint, payload, timeout) -> (status, text)
_KVFRESH = {"syncs": 0, "flush_ok": 0, "flush_fail": 0, "retries": 0, "start_fix": 0, "aborted_inflight": 0}


def kvfresh_wanted() -> bool:
    on = lambda k: os.environ.get(k, "").strip().lower() in ("1", "on", "true", "yes", "fresh")
    return on("ASRT_VERL_KVFRESH")


def _abort_http(http):
    """kv_flush_checked re-pauses failed engines with mode="retract" (slime); verl's colocate path pauses with abort
    (requests go back to FullyAsyncLLMServerClient with their tokens), so the same call is sent as abort here."""
    def f(url, method, endpoint, payload=None, timeout=60.0):
        if endpoint == "pause_generation":
            payload = {"mode": "abort"}
        return http(url, method, endpoint, payload, timeout)
    return f


def kv_flush_verl(urls, http=None, **kw):
    """Checked flush for verl engines (see the block comment); returns kv_flush_or_fail's dict or raises."""
    from .slime_weightsync import _kv_http, kv_flush_or_fail
    return kv_flush_or_fail(urls, _abort_http(http or _kv_http), **kw)


def lower_min_steps(extra: dict, starts) -> bool:
    """min_global_steps := min(itself, every call's start version); True if it changed."""
    vs = [v for v in starts if v is not None]
    if not vs or not isinstance(extra, dict):
        return False
    m = extra.get("min_global_steps")
    lo = min(vs) if m is None else min(m, min(vs))
    if lo != m:
        extra["min_global_steps"] = lo
        return True
    return False


_KVPROBE = {"on": False}


def install_kvprobe() -> bool:
    """Measurement only (also part of kvfresh): server-side per-sync pause/continue records (verl-kvsync) and per-call
    records for the cross-version probe (verl-kvcall).  Probe-only (ASRT_VERL=kvprobe) writes logs and changes nothing
    verl returns; with kvfresh the call's start version also goes into extra_fields ("asrt_start_steps", read by
    kvfresh's client guard)."""
    if _KVPROBE["on"]:
        return True
    _KVPROBE["on"] = True

    def on_server(mod):
        S = getattr(mod, "SGLangHttpServer", None)
        if S is None:
            return

        def make_abort(orig):
            async def abort_all_requests(self, *a, **k):
                n = None
                try:
                    n = len(getattr(self.tokenizer_manager, "rid_to_state", {}) or {})
                except Exception:
                    pass
                t = time.time()
                # paused from the START of the abort: SGLang sets is_pause inside pause_generation, so calls arriving
                # while the abort is still running already wait for resume (059c: 165 calls in that ~1 s window)
                self._asrt_paused = True   # until resume: new calls wait in SGLang's tokenizer (is_pause_cond)
                r = await orig(self, *a, **k)
                if getattr(self, "node_rank", 0) == 0:
                    _KVFRESH["aborted_inflight"] += n or 0
                    _log("kvsync", {"ev": "pause", "mode": "abort", "t0": t, "inflight": n,
                                    "global_steps": getattr(self, "global_steps", None)})
                return r
            return abort_all_requests

        def make_resume(orig):
            async def resume_generation(self, *a, **k):
                # version the held calls will generate under (set_global_steps runs before resume)
                self.__dict__.setdefault("_asrt_resumes", []).append(getattr(self, "global_steps", None))
                self._asrt_paused = False
                r = await orig(self, *a, **k)
                if getattr(self, "node_rank", 0) == 0:
                    _log("kvsync", {"ev": "continue", "global_steps": getattr(self, "global_steps", None)})
                return r
            return resume_generation

        def make_gen(orig):
            async def generate(self, prompt_ids, *a, **k):
                v0, t0 = getattr(self, "global_steps", None), time.time()
                held = getattr(self, "_asrt_paused", False)
                k0 = len(self.__dict__.get("_asrt_resumes", ()))
                out = await orig(self, prompt_ids, *a, **k)
                try:
                    ex = out.extra_fields
                    v1 = ex.get("global_steps")
                    if held:   # arrived during a pause: generation starts at the next resume, under that version
                        rs = self.__dict__.get("_asrt_resumes", [])
                        v0 = rs[k0] if len(rs) > k0 else v1
                    if _KVPROBE.get("guard") and v0 is not None and v1 is not None and v0 != v1:
                        ex["asrt_start_steps"] = v0   # only with kvfresh: probe-only leaves verl's outputs untouched
                    if os.environ.get("ASRT_VERL_KVCALL_LOG", "1") != "0":
                        _log("kvcall", {"t0": t0, "t1": time.time(), "sid": k.get("session_id"),
                                        "rid": k.get("request_id"), "in": len(prompt_ids),
                                        "out": len(out.token_ids or []), "cached": ex.get("num_cached_tokens"),
                                        "v0": v0, "v1": v1, "held": held, "stop": out.stop_reason})
                except Exception:
                    pass
                return out
            return generate

        patch_method(S, "abort_all_requests", make_abort)
        patch_method(S, "resume_generation", make_resume)
        patch_method(S, "generate", make_gen)

    when_imported("verl.workers.rollout.sglang_rollout.async_sglang_server", on_server)
    return True


def install_kvfresh() -> bool:
    from .verl_engine import install_sglhooks
    # engine side: flush reports failure unless the tree is empty.  Only the kvfresh engine hook: the other engine hooks
    # (fastload / nosync) come from the separate 'sglhooks' feature, if the user did not drop it (installed before us).
    install_sglhooks(extra=("kvfresh",), defaults=False)
    _KVPROBE["guard"] = True               # server marks calls whose start version differs (client guard below)
    install_kvprobe()

    def on_server_resume(mod):
        """ separate_async + free_cache_engine=True: the checked flush in
        ServerAdapter.update_weights runs while the KV pools are released; with some site-specific guards that call
        returns True without touching the tree, so it verifies nothing.  Right after verl's resume_kv_cache (pools
        mapped again, engine still paused, nothing can run) one more flush through SGLang's scheduler: the kvfresh
        engine hook records tree size before/after and fails it unless the tree is empty -> per engine per sync."""
        S = getattr(mod, "SGLangHttpServer", None)
        if S is None:
            return

        def make(orig):
            async def resume_kv_cache(self, *a, **k):
                r = await orig(self, *a, **k)
                if getattr(self, "node_rank", 0) != 0 or not getattr(getattr(self, "config", None),
                                                                     "free_cache_engine", False):
                    return r
                tries = int(os.environ.get("ASRT_WS_KV_FLUSH_TRIES", 8))
                delay = float(os.environ.get("ASRT_WS_KV_FLUSH_BASE_S", 0.05))
                ok, att, t = False, 0, time.time()
                while att < max(1, tries):
                    att += 1
                    try:
                        res = await self.tokenizer_manager.flush_cache()
                        ok = bool(getattr(res, "success", False))
                    except Exception:
                        ok = False
                    if ok:
                        break
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, float(os.environ.get("ASRT_WS_KV_FLUSH_CAP_S", 2.0)))
                _KVFRESH["post_resume_ok" if ok else "post_resume_fail"] = \
                    _KVFRESH.get("post_resume_ok" if ok else "post_resume_fail", 0) + 1
                _log("kvfresh", {"ev": "post_resume_flush", "ok": ok, "attempts": att,
                                 "global_steps": getattr(self, "global_steps", None), "s": time.time() - t, **_KVFRESH})
                if not ok:
                    import sys as _sys
                    from .slime_weightsync import KVFreshError
                    msg = (f"[asrt] ERROR kvfresh: prefix cache not verified empty after resume_kv_cache "
                           f"({att} attempts) -> old-weight KV may survive this sync")
                    print(msg, file=_sys.stderr, flush=True)
                    if os.environ.get("ASRT_WS_KV_ON_FAIL", "raise").strip().lower() != "continue":
                        raise KVFreshError(msg)
                return r
            return resume_kv_cache
        patch_method(S, "resume_kv_cache", make)

    def on_rollout(mod):
        A = getattr(mod, "ServerAdapter", None)
        if A is None:
            return

        def make(orig):
            async def update_weights(self, *a, **k):
                r = await orig(self, *a, **k)
                eng = getattr(self, "_engine", None)
                try:
                    leader = self._is_server_tp_leader()
                except Exception:
                    leader = eng is not None
                if eng is None or not leader or getattr(eng, "node_rank", 0) != 0:
                    return r
                sa = getattr(eng, "server_args", None)
                url = f"http://{sa.host}:{sa.port}" if sa is not None else None
                if url is None:   # kv_flush_checked would skip the engine and report ok: refuse (review ab38a7c a)
                    from .slime_weightsync import KVFreshError
                    raise KVFreshError("[asrt] ERROR kvfresh: engine has no server_args (host/port) - flush cannot be checked")
                gs = k.get("global_steps")
                t = time.time()
                loop = asyncio.get_running_loop()
                try:
                    res = await loop.run_in_executor(None, functools.partial(kv_flush_verl, [url], _KV_HTTP))
                except Exception as e:  # KVFreshError (or no URL): stale KV would survive this sync
                    _KVFRESH["syncs"] += 1
                    _KVFRESH["flush_fail"] += 1
                    _log("kvfresh", {"ev": "flush", "ok": False, "global_steps": gs, "error": repr(e)[:300],
                                     "s": time.time() - t, **_KVFRESH})
                    raise
                ok = not res["failed"]
                _KVFRESH["syncs"] += 1
                _KVFRESH["flush_ok" if ok else "flush_fail"] += 1
                _KVFRESH["retries"] += res["retries"]
                _log("kvfresh", {"ev": "flush", "ok": ok, "global_steps": gs, "attempts": res["attempts"],
                                 "retries": res["retries"], "failed": {str(i): m for i, m in res["failed"].items()},
                                 "s": time.time() - t, **_KVFRESH})
                return r
            return update_weights
        patch_method(A, "update_weights", make)

    def on_llm_server(mod):
        C, F = getattr(mod, "LLMServerClient", None), getattr(mod, "FullyAsyncLLMServerClient", None)
        if C is not None:
            def make_base(orig):
                async def generate(self, *a, **k):
                    out = await orig(self, *a, **k)
                    try:
                        segs = _KVSEG.get()
                        if segs is not None:
                            p = k.get("prompt_ids", a[1] if len(a) > 1 else None)
                            segs.append((len(p) if p is not None else 0, len(out.token_ids or []),
                                         getattr(out, "routed_experts", None)))
                    except Exception:
                        pass
                    try:
                        s = out.extra_fields.get("asrt_start_steps")
                        if s is not None:
                            col = _KVSTART.get()
                            if col is not None:
                                col.append(s)   # counted once by the FullyAsync wrapper
                            if lower_min_steps(out.extra_fields, [s]) and col is None:
                                _KVFRESH["start_fix"] += 1
                                _log("kvfresh", {"ev": "start_fix", "start": s, **_KVFRESH})
                    except Exception:
                        pass
                    return out
                return generate
            patch_method(C, "generate", make_base)
        if F is not None:
            def make_full(orig):
                async def generate(self, *a, **k):
                    col, segs = [], []
                    tok = _KVSTART.set(col)
                    tok2 = _KVSEG.set(segs)
                    try:
                        out = await orig(self, *a, **k)
                    finally:
                        _KVSTART.reset(tok)
                        _KVSEG.reset(tok2)
                    if os.environ.get("ASRT_VERL_R3CHECK", "1") != "0":
                        try:
                            rec = r3_rows_check(segs, out)
                            if rec is not None:
                                _log("r3rows", rec)
                        except Exception as e:
                            _log("r3rows", {"ev": "r3rows_error", "error": repr(e)[:200]})
                    try:
                        if col:
                            lower_min_steps(out.extra_fields, col)
                            _KVFRESH["start_fix"] += 1
                            _log("kvfresh", {"ev": "start_fix", "starts": col, **_KVFRESH})
                    except Exception:
                        pass
                    return out
                return generate
            patch_method(F, "generate", make_full)

    when_imported("verl.workers.rollout.sglang_rollout.sglang_rollout", on_rollout)
    when_imported("verl.workers.rollout.sglang_rollout.async_sglang_server", on_server_resume)
    when_imported("verl.workers.rollout.llm_server", on_llm_server)
    print("[asrt] verl kvfresh installed: checked prefix-cache flush per engine per weight sync (abort+resume keeps "
          "tokens), earliest-version guard, kvcall/kvsync logs", flush=True)
    return True


# ----------------------------------------------------------------------------- tokcache (+ tokenize timing)

_AGENT = {"installed": False, "tokcache": False, "profile": False}


def _install_agent_hook():
    """One wrapper on AgentLoopBase.ct_build_initial_tokens serving both
    'tokcache' (dedupe) and 'profile' (latency incl. executor queueing)."""
    if _AGENT["installed"]:
        return
    _AGENT["installed"] = True
    cap = int(os.environ.get("ASRT_VERL_TOKCACHE_CAP", 1024))
    every = int(os.environ.get("ASRT_VERL_TOK_LOG_EVERY", 64))
    ts = _STATS["tokcache"]
    acc = {"n": 0, "sum": 0.0, "max": 0.0}

    def on_agent(mod):
        B = mod.AgentLoopBase
        cache = OrderedDict()
        pending = {}

        def make(orig):
            async def cached(self, messages, tools):
                try:
                    key = (id(type(self)), json.dumps(messages, sort_keys=True, default=str),
                           json.dumps(tools, sort_keys=True, default=str) if tools else "")
                except Exception:
                    return await orig(self, messages, tools)
                if key in cache:
                    cache.move_to_end(key)
                    ts["hits"] += 1
                    return list(cache[key])
                fut = pending.get(key)
                if fut is not None:  # a sibling is tokenizing the same prompt right now
                    ts["hits"] += 1
                    return list(await asyncio.shield(fut))
                fut = asyncio.get_running_loop().create_future()
                pending[key] = fut
                try:
                    ids = await orig(self, messages, tools)
                    fut.set_result(list(ids))
                except BaseException as e:
                    fut.set_exception(e)
                    fut.exception()  # mark retrieved
                    raise
                finally:
                    pending.pop(key, None)
                ts["misses"] += 1
                cache[key] = list(ids)
                while len(cache) > cap:
                    cache.popitem(last=False)
                return ids

            async def ct_build_initial_tokens(self, messages, tools=None, images=None, videos=None, audios=None,
                                              mm_inputs_out=None, *a, **k):
                t0 = time.perf_counter()
                plain = not (images or videos or audios or mm_inputs_out is not None or a or k)
                if _AGENT["tokcache"] and plain:
                    out = await cached(self, messages, tools)
                elif plain:
                    out = await orig(self, messages, tools)
                else:
                    out = await orig(self, messages, tools, images, videos, audios, mm_inputs_out, *a, **k)
                if _AGENT["profile"]:
                    dt = time.perf_counter() - t0
                    acc["n"] += 1
                    acc["sum"] += dt
                    acc["max"] = max(acc["max"], dt)
                    if every and acc["n"] % every == 0:
                        _log("tok", {"n": acc["n"], "mean_s": round(acc["sum"] / acc["n"], 4),
                                     "max_s": round(acc["max"], 3), "tokcache": dict(ts)})
                return out
            return ct_build_initial_tokens
        patch_method(B, "ct_build_initial_tokens", make)

    when_imported("verl.experimental.agent_loop.agent_loop", on_agent)


def install_tokcache() -> bool:
    _AGENT["tokcache"] = True
    _install_agent_hook()
    return True


# ----------------------------------------------------------------------------- profile

def install_profile() -> bool:

    def on_ckpt(mod):
        M = mod.CheckpointEngineManager
        try:
            from verl.utils.ray_utils import auto_await
        except Exception:  # pragma: no cover
            auto_await = None
        phases = ["abort_replicas", "release_kv_cache_replicas", "resume_kv_cache_replicas",
                  "resume_generation_replicas"]
        cur = {}

        def timed_async(name, inner):
            async def f(self, *a, **k):
                t0 = time.perf_counter()
                try:
                    return await inner(self, *a, **k)
                finally:
                    cur[name] = cur.get(name, 0.0) + time.perf_counter() - t0
            return f

        for name in phases:
            fn = M.__dict__.get(name)
            inner = getattr(fn, "__wrapped__", None)
            if fn is None or inner is None or auto_await is None or getattr(fn, "__asrt_patched__", False):
                continue
            new = keep_attrs(auto_await(timed_async(name, inner)), fn)
            new.__asrt_patched__ = True
            setattr(M, name, new)

        bpg = M.__dict__.get("build_process_group")
        if bpg is not None and not getattr(bpg, "__asrt_patched__", False):
            def build_process_group(self, *a, **k):
                t0 = time.perf_counter()
                try:
                    return bpg(self, *a, **k)
                finally:
                    cur["build_process_group"] = cur.get("build_process_group", 0.0) + time.perf_counter() - t0
            keep_attrs(build_process_group, bpg)
            build_process_group.__asrt_patched__ = True
            M.build_process_group = build_process_group

        uw = M.__dict__.get("update_weights")
        inner = getattr(uw, "__wrapped__", None)
        if uw is not None and inner is not None and auto_await is not None and not getattr(uw, "__asrt_patched__", False):
            async def update_weights(self, global_steps=None, *a, **k):
                cur.clear()
                t0 = time.perf_counter()
                out = await inner(self, global_steps, *a, **k)
                tot = time.perf_counter() - t0
                rec = {"step": global_steps, "total_s": round(tot, 3), **{k2: round(v, 3) for k2, v in cur.items()}}
                rec["transfer_finalize_s"] = round(tot - sum(cur.values()), 3)
                rec["keepkv"] = dict(_STATS["keepkv"])
                print(f"[asrt] verl sync {rec}", flush=True)
                _log("sync", rec)
                return out
            new = keep_attrs(auto_await(update_weights), uw)
            new.__asrt_patched__ = True
            M.update_weights = new

    when_imported("verl.checkpoint_engine.base", on_ckpt)
    _AGENT["profile"] = True
    _install_agent_hook()
    return True


def stats() -> dict:
    return {k: v for k, v in _STATS.items() if v is not None}
