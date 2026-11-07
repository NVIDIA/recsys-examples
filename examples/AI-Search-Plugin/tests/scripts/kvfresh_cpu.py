"""kvfresh: retract before flush, checked flush with re-retract + bounded backoff, loud failure,
engine-side check that the tree is empty.  No SGLang / torch / ray needed.

The fake scheduler follows the slime-cu130 SGLang code (scheduler.py is_fully_idle 5027-5060, flush_cache 5203-5230,
pause_generation 5611-5680; http_server.py /flush_cache 1006-1021: 200 if success else 400; tokenizer generate waits for
un-pause before tokenizing, tokenizer_manager.py 911-919).  Radix keys are (extra_key, tokens) as in radix_cache.py.
Run: python3 tests/scripts/kvfresh_cpu.py"""
import os, tempfile, types, json, glob

os.environ["ASRT_SGL_HOOKS"] = "kvfresh"
LOG = tempfile.mkdtemp(prefix="kvfresh-test-")
os.environ["ASRT_LOG_DIR"] = LOG
from ai_search_plugin.adapters import slime_sglang_hooks as H
from ai_search_plugin.adapters import slime_weightsync as W


class Req:
    def __init__(self, rid, ids, extra_key="", ver=None):
        self.rid, self.ids, self.extra_key, self.out, self.done = rid, list(ids), extra_key, [], False

    def finished(self):
        return self.done


class Tree:
    """prefix tree stand-in: {(extra_key, prefix tuple): weight version that computed the KV}."""

    def __init__(self):
        self.kv = {}
        self.root_node = types.SimpleNamespace(children={})
        self.broken_reset = False

    def insert(self, key, ids, ver):
        for n in range(1, len(ids) + 1):
            self.kv[(key, tuple(ids[:n]))] = ver
        self.root_node.children[(key, ids[0])] = True

    def match(self, key, ids):
        best, ver = 0, None
        for n in range(1, len(ids) + 1):
            if (key, tuple(ids[:n])) in self.kv:
                best, ver = n, self.kv[(key, tuple(ids[:n]))]
        return best, ver

    def evictable_size(self):
        return len(self.kv)

    def protected_size(self):
        return 0

    def reset(self):
        if not self.broken_reset:
            self.kv = {}
            self.root_node = types.SimpleNamespace(children={})


class Batch:
    def __init__(self, reqs):
        self.reqs = reqs

    def is_empty(self):
        return not self.reqs


class Scheduler:
    """Only the state the real is_fully_idle / flush_cache / pause_generation touch."""

    def __init__(self):
        self.tree_cache = Tree()
        self.running_batch = Batch([])
        self.waiting_queue = []
        self.chunked_req = None
        self.last_batch = None
        self.enable_overlap = True
        self.result_queue = []
        self.grammar_manager = types.SimpleNamespace(grammar_queue=[])
        self._engine_paused = False
        self.version = 0

    # --- serving ---
    def prefill(self, req):
        """schedule a waiting request: match the tree, compute the rest with the CURRENT weights, run."""
        n, ver = self.tree_cache.match(req.extra_key, req.ids)
        req.hit = (n, ver)
        self.waiting_queue.remove(req)
        self.running_batch.reqs.append(req)
        self.last_batch = Batch([req])
        self.result_queue.append(("res", req))
        return n, ver

    def decode(self, req, tok):
        req.out.append(tok)

    def release(self, req, ver):  # finished or retracted: its KV goes into the tree (cache_finished/unfinished_req)
        self.tree_cache.insert(req.extra_key, req.ids + req.out, ver)

    # --- SGLang control (real logic, trimmed) ---
    def is_fully_idle(self, ignore_waiting=False):
        idle = (self.running_batch.is_empty() and self.chunked_req is None
                and (self.last_batch is None or self.last_batch.is_empty())
                and (not self.enable_overlap or len(self.result_queue) == 0))
        if not ignore_waiting:
            idle &= len(self.waiting_queue) == 0
        idle &= len(self.grammar_manager.grammar_queue) == 0
        return idle

    def flush_cache(self, empty_cache=True):
        if self.is_fully_idle(ignore_waiting=self._engine_paused):
            self.last_batch = None
            self.tree_cache.reset()
            return True
        return False

    def pause_generation(self, recv_req):
        self._engine_paused = True
        if recv_req.mode == "in_place":
            return
        if self.enable_overlap and self.last_batch:
            self.result_queue.pop(0)  # process ONE pending result (real code: popleft + process_batch_result)
        retract = [r for r in self.running_batch.reqs if not r.finished()]
        if self.chunked_req is not None and self.chunked_req not in retract:
            retract.append(self.chunked_req)
        self.last_batch = None
        for r in retract:
            self.release(r, f"v{self.version}")
            r.ids = r.ids + r.out  # resumes from prompt + generated tokens (no token thrown away)
            r.out = []
        self.running_batch.reqs = []
        self.waiting_queue.extend(retract)
        self.chunked_req = None

    def continue_generation(self, recv_req):
        self._engine_paused = False


mod = types.ModuleType("fake_sched")
mod.Scheduler = Scheduler
H.patch_scheduler(mod)
assert getattr(Scheduler, "_asrt_kvfresh", False), "kvfresh hook not installed"


class Engine:
    """HTTP front of one engine (http_server: /flush_cache 200/400, pause/continue)."""

    def __init__(self):
        self.s = Scheduler()
        self.calls = []

    def http(self, method, endpoint, payload=None):
        self.calls.append(endpoint if endpoint != "pause_generation" else f"pause:{payload['mode']}")
        if endpoint == "flush_cache":
            ok = self.s.flush_cache()
            return (200, "Cache flushed.") if ok else (400, "Flush cache failed.")
        if endpoint == "pause_generation":
            self.s.pause_generation(types.SimpleNamespace(mode=payload["mode"]))
            return 200, ""
        if endpoint == "continue_generation":
            self.s.continue_generation(None)
            return 200, ""
        raise ValueError(endpoint)


def fleet(engines):
    urls = [f"http://e{i}" for i in range(len(engines))]
    by = dict(zip(urls, engines))
    return urls, (lambda url, method, endpoint, payload=None: by[url].http(method, endpoint, payload))


def kv_sync(engines, sleeps, **kw):
    """the else-branch order of update_weights with kv_fresh: pause(retract) -> weights -> checked flush -> continue."""
    urls, http = fleet(engines)
    for e in engines:
        e.http("POST", "pause_generation", {"mode": "retract"})
    for e in engines:
        e.s.version += 1  # buckets loaded
    r = W.kv_flush_or_fail(urls, http, sleep=sleeps.append, log=lambda s: None, err=lambda s: None, **kw)
    for e in engines:
        e.http("POST", "continue_generation", {})
    return r


def jl():
    out = []
    for f in glob.glob(os.path.join(LOG, "sgl-kvfresh-*.jsonl")):
        out += [json.loads(x) for x in open(f)]
    return out


# 1. case (1) in-flight request + case (2) tokenized-before / prefilled-after, normal path: 0 cross-version hits
e = Engine(); s = e.s
a = Req("A", [1, 2, 3]); s.waiting_queue.append(a); s.prefill(a); s.result_queue.clear(); s.last_batch = None
s.decode(a, 9); s.decode(a, 9)                      # A in flight with v0 KV for [1,2,3,9,9]
old = Req("O", [5, 6]); s.waiting_queue.append(old); s.prefill(old); s.running_batch.reqs.remove(old)
old.done = True; s.release(old, "v0"); s.result_queue.clear(); s.last_batch = None   # finished turn: v0 in tree
b = Req("B", [5, 6, 7])                             # tokenized BEFORE the sync, still waiting (not prefilled)
s.waiting_queue.append(b)
s.last_batch = Batch([a]); s.result_queue.append(("res", a))   # overlap: one result still queued at pause time
sleeps = []
r = kv_sync([e], sleeps)
assert r["failed"] == {} and r["attempts"] == 1, r
assert s.tree_cache.kv == {}, s.tree_cache.kv
assert e.calls[:3] == ["pause:retract", "flush_cache", "continue_generation"], e.calls
assert a in s.waiting_queue and a.ids == [1, 2, 3, 9, 9], "in-flight A must be retracted with its tokens"
for q in (a, b):                                    # after the sync both prefill against an EMPTY tree
    n, ver = s.prefill(q)
    assert (n, ver) == (0, None), (q.rid, n, ver)
rec = jl()
assert any(x["op"] == "retract" and x["n"] == 1 for x in rec), rec
assert any(x["op"] == "flush" and x["ok"] and x["tree_before"] > 0 and x["tree_after"] == 0 for x in rec), rec
print("1 ok: in-flight retracted, queued request finds empty tree, flush verified", r)

# 2. retry mechanics: engine not idle at the first flush, idle at the second (SIMULATED drain: in real slime-cu130
#    SGLang a paused engine does not move grammar-queue requests on - that case is case 3, loud failure); checks the
#    hook's ERROR record, the re-retract to that engine only, one backoff wait, success on attempt 2
e2 = Engine(); s2 = e2.s
c = Req("C", [1, 2]); s2.waiting_queue.append(c); s2.prefill(c); s2.decode(c, 4)
g = Req("G", [8, 8]); g.ready_in = 1; s2.grammar_manager.grammar_queue.append(g)
_orig_http = e2.http
def _http2(method, endpoint, payload=None):
    if endpoint == "pause_generation":           # stands in for the paused loop's ingest between two attempts
        for q in list(s2.grammar_manager.grammar_queue):
            q.ready_in -= 1
            if q.ready_in < 0:
                s2.grammar_manager.grammar_queue.remove(q); s2.waiting_queue.append(q)
    return _orig_http(method, endpoint, payload)
e2.http = _http2
sleeps = []
r = kv_sync([e2], sleeps, tries=5, base_s=0.05, cap_s=0.2)
assert r["failed"] == {} and r["attempts"] == 2 and r["retries"] == 1, r
assert sleeps == [0.05], sleeps
assert e2.calls == ["pause:retract", "flush_cache", "pause:retract", "flush_cache", "continue_generation"], e2.calls
assert s2.tree_cache.kv == {} and g in s2.waiting_queue and c in s2.waiting_queue
fails = [x for x in jl() if x["op"] == "flush" and not x["ok"]]
assert fails and "engine not idle" in fails[-1]["why"] and fails[-1]["busy"]["grammar"] == 1, fails
print("2 ok: not idle -> re-retract -> flushed on attempt 2, backoff", sleeps)

# 3. never idle (grammar queue stuck) -> bounded backoff, ERROR, KVFreshError (default), counted when 'continue'
e3 = Engine(); e3.s.grammar_manager.grammar_queue.append("g")
sleeps = []
try:
    kv_sync([e3], sleeps, tries=6, base_s=0.05, cap_s=0.2)
    raise AssertionError("must raise")
except W.KVFreshError as ex:
    assert "NOT flushed" in str(ex)
assert sleeps == [0.05, 0.1, 0.2, 0.2, 0.2], sleeps        # 6 attempts, 5 waits, capped
assert e3.calls.count("flush_cache") == 6
e3.calls.clear(); errs = []
r = W.kv_flush_or_fail(*fleet([e3]), on_fail="continue", sleep=lambda d: None, log=lambda s: None, err=errs.append,
                       tries=3)
assert list(r["failed"]) == [0] and errs and errs[0].startswith("[asrt] ERROR kvfresh"), (r, errs)
print("3 ok: persistent failure -> ERROR + KVFreshError; continue mode returns failed", r["failed"])

# 4. SGLang reset that leaves the tree non-empty -> the engine hook reports failure (HTTP 400), not silent success
e4 = Engine(); s4 = e4.s; s4.tree_cache.insert("", [7, 7], "v0"); s4.tree_cache.broken_reset = True
sleeps = []
try:
    kv_sync([e4], sleeps, tries=2)
    raise AssertionError("must raise")
except W.KVFreshError:
    pass
bad = [x for x in jl() if x["op"] == "flush" and "tree not empty" in x["why"]]
assert bad, jl()
print("4 ok: non-empty tree after reset is a failure")

# 5. several engines, one fails once: only that one is retried; node_rank != 0 (url None) skipped
e5a, e5b = Engine(), Engine()
gq = Req("x", [1]); e5b.s.grammar_manager.grammar_queue.append(gq)
_h5 = e5b.http
def _http5(method, endpoint, payload=None):
    if endpoint == "pause_generation" and e5b.calls.count("pause:retract") >= 1 and gq in e5b.s.grammar_manager.grammar_queue:
        e5b.s.grammar_manager.grammar_queue.remove(gq); e5b.s.waiting_queue.append(gq)   # ready on the 2nd ingest
    return _h5(method, endpoint, payload)
e5b.http = _http5
urls, http = fleet([e5a, e5b])
for x in (e5a, e5b):
    x.http("POST", "pause_generation", {"mode": "retract"})
r = W.kv_flush_checked(urls + [None], http, sleep=lambda d: None, tries=4)
assert r["failed"] == {} and e5a.calls.count("flush_cache") == 1 and e5b.calls.count("flush_cache") == 2, \
    (r, e5a.calls, e5b.calls)
print("5 ok: per-engine retry", r)

# 6. network error counts as a failed attempt (not an exception that skips the check); no URL at all = failure
r0 = W.kv_flush_checked([None, None], lambda *a, **k: (200, ""), sleep=lambda d: None)
assert r0["failed"] and r0["attempts"] == 0, r0
r = W.kv_flush_checked(["http://dead"], lambda *a, **k: (-1, "ConnectionError"), sleep=lambda d: None, tries=2)
assert r["failed"] == {0: "HTTP -1: ConnectionError"}, r
print("6 ok: network error -> failed")

# 7. control: WITHOUT kvfresh (pause in_place, no flush) the same scenario hits old KV (the probe is sensitive)
e7 = Engine(); s7 = e7.s
a = Req("A", [1, 2, 3]); s7.waiting_queue.append(a); s7.prefill(a); s7.decode(a, 9)
s7.pause_generation(types.SimpleNamespace(mode="in_place")); s7.version += 1; s7.continue_generation(None)
s7.running_batch.reqs.remove(a); a.done = True; s7.release(a, "v0")     # finishes after the sync, inserts v0 KV
t2 = Req("A2", [1, 2, 3, 9, 8]); s7.waiting_queue.append(t2)
assert s7.prefill(t2) == (4, "v0")
print("7 ok: control without kvfresh hits 4 tokens of v0 KV")

# 8. cache type without size counters (unknown SWA/hybrid variant): flush still reported, but logged as UNCHECKED
class OpaqueTree:
    def reset(self): pass
e8 = Engine(); e8.s.tree_cache = OpaqueTree()
e8.http("POST", "pause_generation", {"mode": "retract"})
st8, _ = e8.http("GET", "flush_cache")
rec8 = [x for x in jl() if x["op"] == "flush"][-1]
assert st8 == 200 and rec8["checked"] is False and rec8["unchecked"] >= 1, rec8
print("8 ok: unknown cache type -> flush logged as unchecked (ERROR once)")
print("ALL OK", LOG)
