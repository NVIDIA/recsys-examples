"""CPU tests for asrt verl kvfresh (verl_rollout.install_kvfresh) on stand-ins of the three verl
classes it patches, written after verl main of 2026-10-23:
  ServerAdapter.update_weights   (sglang_rollout.py:305-396: buckets, then flush_cache() with the result ignored)
  SGLangHttpServer.generate      (async_sglang_server.py:693: extra_fields["global_steps"] = server version at RETURN)
  LLMServerClient / FullyAsyncLLMServerClient.generate  (llm_server.py:150-166, 282-366: min over the calls' versions)
Checks:
  (1) checked flush: an engine whose /flush_cache returns 400 twice is re-paused with mode=abort (verl's pause) and
      flushed again until 200; one ok record per sync.            OLD verl: no check -> FAILS (ASRT_TEST_OLD=1)
  (2) an engine that never flushes -> KVFreshError (ASRT_WS_KV_ON_FAIL=continue: counted, no raise).
  (3) earliest version: a call that runs across an update without abort reports min_global_steps = its START
      version.                                                     OLD verl: end version -> FAILS (ASRT_TEST_OLD=1)
  (4) kvfresh refuses to install next to keepkv/inplace.
  (5) tools/verl_kvprobe.py: windows from pause/continue, held requests anchored at continue, probe flags a
      synthetic cross-version hit and nothing else.
usage: python3 tests/scripts/verl_kvfresh_cpu.py   (ASRT_TEST_OLD=1: same checks without kvfresh = the old behaviour)"""
import asyncio
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
OLD = os.environ.get("ASRT_TEST_OLD") == "1"
os.environ.pop("ASRT_SGL_HOOKS", None)   # as in a run with ASRT_VERL_DROP=sglhooks
LOG = tempfile.mkdtemp(prefix="vkf-")
os.environ["ASRT_LOG_DIR"] = LOG
os.environ.setdefault("ASRT_WS_KV_FLUSH_BASE_S", "0.001")
os.environ.setdefault("ASRT_WS_KV_FLUSH_CAP_S", "0.002")


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class TokenOutput:
    def __init__(self, token_ids, stop_reason="stop", extra_fields=None):
        self.token_ids, self.log_probs, self.stop_reason = list(token_ids), [0.0] * len(token_ids), stop_reason
        self.extra_fields = dict(extra_fields or {})
        self.routed_experts = None
        self.num_preempted = 0


class Engine:   # verl AsyncHttpServerAdapter: node_rank, server_args; flush_cache ignores failures
    def __init__(self, port):
        self.node_rank, self.server_args = 0, types.SimpleNamespace(host="127.0.0.1", port=port)
        self.flush_calls = 0

    async def flush_cache(self):
        self.flush_calls += 1
        return {}


class ServerAdapter:
    def __init__(self, port):
        self._engine = Engine(port)

    def _is_server_tp_leader(self):
        return True

    async def update_weights(self, weights=None, global_steps=None, **kw):
        await asyncio.sleep(0)               # buckets
        await self._engine.flush_cache()     # verl: result not checked
        return None


class SGLangHttpServer:
    def __init__(self):
        self.node_rank, self.global_steps = 0, 4
        self.tokenizer_manager = types.SimpleNamespace(rid_to_state={}, flush_cache=self._tm_flush)
        self.config = types.SimpleNamespace(free_cache_engine=True)
        self.flush_fail = 0                       # fake engine: next N scheduler flushes report failure
        self.bump_during_call = False
        self.abort_s = 0.0

    async def _tm_flush(self):
        ok = self.flush_fail <= 0
        self.flush_fail -= 1
        return types.SimpleNamespace(success=ok)

    async def resume_kv_cache(self):            # verl: resume_memory_occupation + flush_cache (fce=True)
        self.resumed = getattr(self, "resumed", 0) + 1
        return None

    async def abort_all_requests(self):
        await asyncio.sleep(self.abort_s)   # SGLang: is_pause is set at the start, the abort itself takes ~1 s
        return None

    async def resume_generation(self):
        return None

    async def generate(self, prompt_ids, sampling_params=None, request_id=None, session_id=None, **kw):
        while getattr(self, "_asrt_paused", False):   # SGLang tokenizer is_pause_cond gate
            await asyncio.sleep(0.001)
        if self.bump_during_call:        # an update lands while this call runs (no abort)
            self.global_steps += 1
        return TokenOutput([1, 2, 3], extra_fields={"global_steps": self.global_steps, "num_cached_tokens": 0})


SERVER = SGLangHttpServer()


class LLMServerClient:   # llm_server.py:150-166
    async def generate(self, request_id, prompt_ids, sampling_params, **kw):
        output = await SERVER.generate(prompt_ids=prompt_ids, sampling_params=sampling_params, request_id=request_id,
                                       session_id=request_id)
        gs = output.extra_fields.get("global_steps")
        output.extra_fields.setdefault("min_global_steps", gs)
        output.extra_fields.setdefault("max_global_steps", gs)
        return output


class FullyAsyncLLMServerClient(LLMServerClient):   # llm_server.py:282-366 (min over the calls' END versions)
    async def generate(self, request_id, prompt_ids, sampling_params, **kw):
        final, mn, mx = TokenOutput([]), None, None
        while True:
            out = await super().generate(request_id=request_id, prompt_ids=prompt_ids + final.token_ids,
                                         sampling_params=sampling_params)
            final.token_ids.extend(out.token_ids)
            gs = out.extra_fields.get("global_steps")
            mn = gs if mn is None else mn
            mx = gs
            if out.stop_reason not in ("aborted", "abort"):
                break
        final.extra_fields.update(global_steps=gs, min_global_steps=mn, max_global_steps=mx)
        return final


_mod("verl"); _mod("verl.workers"); _mod("verl.workers.rollout"); _mod("verl.workers.rollout.sglang_rollout")
_mod("verl.workers.rollout.sglang_rollout.sglang_rollout", ServerAdapter=ServerAdapter)
_mod("verl.workers.rollout.sglang_rollout.async_sglang_server", SGLangHttpServer=SGLangHttpServer)
_mod("verl.workers.rollout.llm_server", LLMServerClient=LLMServerClient,
     FullyAsyncLLMServerClient=FullyAsyncLLMServerClient)

from ai_search_plugin.adapters import verl_rollout as VR  # noqa: E402

if not OLD:
    VR.install_kvfresh()


class FakeHTTP:
    """SGLang HTTP server: /flush_cache -> 400 while `busy` > 0 (each re-pause drains one), then 200."""
    def __init__(self, busy):
        self.busy, self.calls = busy, []

    def __call__(self, url, method, endpoint, payload=None, timeout=60.0):
        self.calls.append((endpoint, payload))
        if endpoint == "pause_generation":
            self.busy = max(0, self.busy - 1) if self.busy < 1 << 30 else self.busy
            return 200, ""
        if endpoint == "flush_cache":
            return (400, "Cache not flushed because there are pending requests") if self.busy else (200, "ok")
        return 404, ""


def _records(kind):
    out = []
    for f in os.listdir(LOG):
        if f.startswith(f"verl-{kind}-"):
            out += [json.loads(x) for x in open(os.path.join(LOG, f))]
    return out


def test_checked_flush_retries_with_abort():
    h = FakeHTTP(busy=2)
    VR._KV_HTTP = h
    a = ServerAdapter(30000)
    asyncio.run(a.update_weights(global_steps=7))
    flushes = [c for c in h.calls if c[0] == "flush_cache"]
    pauses = [c for c in h.calls if c[0] == "pause_generation"]
    recs = [r for r in _records("kvfresh") if r.get("ev") == "flush" and r.get("global_steps") == 7]
    assert len(flushes) == 3 and all(p[1] == {"mode": "abort"} for p in pauses) and len(pauses) == 2, h.calls
    assert len(recs) == 1 and recs[0]["ok"] and recs[0]["retries"] == 2, recs
    print("checked flush: 2 x 400 -> re-pause(abort) x 2 -> 200; one ok record for the sync")


def test_flush_never_succeeds_raises():
    os.environ["ASRT_WS_KV_FLUSH_TRIES"] = "3"
    try:
        VR._KV_HTTP = FakeHTTP(busy=1 << 30)
        a = ServerAdapter(30001)
        raised = False
        try:
            asyncio.run(a.update_weights(global_steps=8))
        except Exception as e:
            raised = type(e).__name__ == "KVFreshError"
        assert raised, "a sync whose flush never succeeds must raise KVFreshError"
        os.environ["ASRT_WS_KV_ON_FAIL"] = "continue"
        asyncio.run(a.update_weights(global_steps=9))
        rec = [r for r in _records("kvfresh") if r.get("global_steps") == 9]
        assert rec and rec[0]["ok"] is False, rec
    finally:
        os.environ.pop("ASRT_WS_KV_FLUSH_TRIES", None)
        os.environ.pop("ASRT_WS_KV_ON_FAIL", None)
    print("flush never succeeds: KVFreshError; with ASRT_WS_KV_ON_FAIL=continue counted (ok false), no raise")


def test_no_url_raises():
    a = ServerAdapter(30002)
    a._engine.server_args = None
    try:
        asyncio.run(a.update_weights(global_steps=10))
        ok = False
    except Exception as e:
        ok = type(e).__name__ == "KVFreshError"
    assert ok, "an engine whose flush cannot be checked must raise"
    print("no server_args -> KVFreshError")


def test_earliest_version():
    SERVER.global_steps, SERVER.bump_during_call = 4, True
    try:
        out = asyncio.run(FullyAsyncLLMServerClient().generate("r1", [5, 6], {}))
    finally:
        SERVER.bump_during_call = False
    assert out.extra_fields["max_global_steps"] == 5, out.extra_fields
    assert out.extra_fields["min_global_steps"] == 4, ("min_global_steps must be the version the call STARTED under",
                                                       out.extra_fields)
    print("earliest version: call started at 4, returned at 5 -> min_global_steps 4 (verl alone: 5)")


def test_held_call_not_lowered():
    """a call that arrives during verl's pause (resubmitted / next turn) waits in the tokenizer and generates only after
    the update + resume: its start version is the resumed one, min_global_steps must NOT be lowered (059a v1: 307)."""
    if OLD:
        print("held call: n/a without kvfresh (no pause tracking)")
        return

    async def go():
        SERVER.global_steps, SERVER.bump_during_call = 4, False
        await SERVER.abort_all_requests()        # verl's pause
        task = asyncio.ensure_future(FullyAsyncLLMServerClient().generate("r2", [7], {}))
        await asyncio.sleep(0.01)                # the call is waiting at the gate
        SERVER.global_steps = 5                  # weight update + set_global_steps
        await SERVER.resume_generation()
        return await task
    out = asyncio.run(go())
    assert out.extra_fields["min_global_steps"] == 5, ("held call generated under 5 only", out.extra_fields)
    print("held call (arrived during pause, served after update): min_global_steps 5, not lowered to 4")


def test_call_during_abort_not_lowered():
    """059c: calls arriving while verl's abort is still running (SGLang already holds them) were tagged with the old
    version (165 calls).  The pause must count from the start of the abort."""
    if OLD:
        print("call during abort: n/a without kvfresh")
        return

    async def go():
        SERVER.global_steps, SERVER.abort_s = 4, 0.05
        ab = asyncio.ensure_future(SERVER.abort_all_requests())
        await asyncio.sleep(0.01)                # abort in progress
        task = asyncio.ensure_future(FullyAsyncLLMServerClient().generate("r3", [9], {}))
        await ab
        await asyncio.sleep(0.01)
        SERVER.global_steps = 5
        await SERVER.resume_generation()
        return await task
    try:
        out = asyncio.run(go())
    finally:
        SERVER.abort_s = 0.0
    assert out.extra_fields["min_global_steps"] == 5, ("call that arrived during the abort generated under 5 only",
                                                       out.extra_fields)
    print("call arriving during the abort: min_global_steps 5, not lowered to 4")


def test_probe_only_leaves_outputs():
    """ASRT_VERL=kvprobe alone (baseline arms): no extra key in verl's outputs, only logs."""
    if OLD:
        print("probe-only: n/a")
        return
    g = VR._KVPROBE.get("guard")
    VR._KVPROBE["guard"] = False
    try:
        SERVER.global_steps, SERVER.bump_during_call = 4, True
        out = asyncio.run(SERVER.generate(prompt_ids=[1], request_id="p", session_id="p"))
    finally:
        VR._KVPROBE["guard"], SERVER.bump_during_call = g, False
    assert "asrt_start_steps" not in out.extra_fields, out.extra_fields
    print("probe-only: verl outputs unchanged (no asrt_start_steps)")


def test_kvfresh_engine_hooks_only_kvfresh():
    """ with ASRT_SGL_HOOKS unset (sglhooks dropped by the user), kvfresh installs only its own
    engine hook - not fastload/nosync (fastload must not come back behind ASRT_VERL_DROP=sglhooks)."""
    if OLD:
        print("engine hooks: n/a without kvfresh")
        return
    assert set(os.environ.get("ASRT_SGL_HOOKS", "").split(",")) == {"kvfresh"}, os.environ.get("ASRT_SGL_HOOKS")
    print("kvfresh alone: ASRT_SGL_HOOKS=kvfresh (no fastload / nosync)")


def test_post_resume_tree_check():
    """separate_async + fce=True: after resume_kv_cache one checked flush per engine per sync is recorded
    (ev=post_resume_flush); a flush that keeps failing raises KVFreshError.  Old code: no record."""
    if OLD:
        recs = []
    else:
        SERVER.global_steps, SERVER.flush_fail = 11, 1
        asyncio.run(SERVER.resume_kv_cache())
        recs = [r for r in _records("kvfresh") if r.get("ev") == "post_resume_flush" and r.get("global_steps") == 11]
    assert len(recs) == 1 and recs[0]["ok"] and recs[0]["attempts"] == 2, recs
    os.environ["ASRT_WS_KV_FLUSH_TRIES"] = "2"
    try:
        SERVER.global_steps, SERVER.flush_fail = 12, 99
        raised = False
        try:
            asyncio.run(SERVER.resume_kv_cache())
        except Exception as e:
            raised = type(e).__name__ == "KVFreshError"
        assert raised, "an engine whose post-resume flush never succeeds must raise"
    finally:
        os.environ.pop("ASRT_WS_KV_FLUSH_TRIES", None)
        SERVER.flush_fail = 0
    print("post-resume checked flush: 1 record per engine per sync (retry 1x), persistent failure -> KVFreshError")


def test_r3_rows_check():
    """R3 rows of resumed trajectories: verl's merge (keep first call's rows, append new tokens' rows) passes;
    a merge that takes the re-prefill rows fails (first rows changed)."""
    if OLD:
        print("r3 rows: n/a without kvfresh")
        return
    import numpy as np

    def call(prompt_len, new, version):        # SGLang rows over positions [0, seqlen-1), marked with the version
        return np.full((prompt_len + new - 1, 2, 2), version, dtype=np.int32)

    prompt, n1, n2 = 10, 4, 3
    first = call(prompt, n1, 1)                 # aborted first call, old weights
    second = call(prompt + n1, n2, 2)           # resubmit prompt + partial: re-prefill under new weights
    good = TokenOutput(list(range(n1 + n2)))
    good.routed_experts = np.concatenate([first, second[-n2:]])          # verl llm_server.py:307-315
    segs = [(prompt, n1, first), (prompt + n1, n2, second)]
    rec = VR.r3_rows_check(segs, good)
    assert rec["ok"] and rec["rows"] == prompt + n1 + n2 - 1 and rec["first_rows_kept"], rec
    print(f"  r3 case 'verl merge (first call's rows kept, rows = prompt+response-1)': ok={rec['ok']} "
          f"first_call_has_rows={rec['first_call_has_rows']} rows={rec['rows']} expected={rec['expected']}")
    bad = TokenOutput(list(range(n1 + n2)))
    bad.routed_experts = second                                         # rows from the re-prefill
    rec = VR.r3_rows_check(segs, bad)
    assert not rec["ok"] and not rec["first_rows_kept"], rec
    print(f"  r3 case 're-prefill rows replaced the first call's rows': ok={rec['ok']} "
          f"first_rows_kept={rec['first_rows_kept']} (flagged)")
    assert VR.r3_rows_check(segs[:1], good) is None                     # single call: not resumed
    print("  r3 case 'single call (not resumed)': no record")
    # first call returned no rows: must NOT pass trivially on the second call's rows
    rec = VR.r3_rows_check([(prompt, n1, None), (prompt + n1, n2, second)], good)
    assert rec["first_call_has_rows"] is False and rec["ok"] is False and rec["r3_unverifiable"] >= 1, rec
    print(f"  r3 case 'first call with no rows': ok={rec['ok']} first_call_has_rows={rec['first_call_has_rows']} "
          f"counted r3_unverifiable={rec['r3_unverifiable']} (never ok)")
    # aborted in the waiting queue (0 tokens, no rows) then a normal call: first call WITH tokens decides
    rec = VR.r3_rows_check([(prompt, 0, None), (prompt, n1 + n2, call(prompt, n1 + n2, 1))],
                           types.SimpleNamespace(token_ids=list(range(n1 + n2)),
                                                 routed_experts=call(prompt, n1 + n2, 1)))
    assert rec["first_call_has_rows"] and rec["ok"], rec
    print(f"  r3 case 'first call with 0 tokens (waiting-queue abort), second call decides': ok={rec['ok']} "
          f"first_call_has_rows={rec['first_call_has_rows']} calls={rec['calls']} calls_with_tokens={rec['calls_with_tokens']}")
    print("r3 rows: verl merge ok (rows = prompt+response-1, first call's rows kept); re-prefill rows flagged")


def test_refuses_keepkv():
    from ai_search_plugin.adapters import verl as V
    os.environ["ASRT_VERL_KVFRESH"] = "1"
    try:
        try:
            V.install("route,keepkv")
            ok = False
        except RuntimeError as e:
            ok = "kvfresh" in str(e)
        assert ok, "kvfresh + keepkv must be refused"
    finally:
        os.environ.pop("ASRT_VERL_KVFRESH", None)
    print("kvfresh + keepkv refused")



if __name__ == "__main__":
    fails = 0
    for t in (test_checked_flush_retries_with_abort, test_flush_never_succeeds_raises, test_no_url_raises, test_earliest_version, test_held_call_not_lowered,
              test_call_during_abort_not_lowered, test_probe_only_leaves_outputs, test_kvfresh_engine_hooks_only_kvfresh,
              test_post_resume_tree_check, test_r3_rows_check,
              test_refuses_keepkv):
        try:
            t()
        except AssertionError as e:
            fails += 1
            print(f"FAIL {t.__name__}: {e}")
    print("ok" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
