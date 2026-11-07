"""kvfresh retract + routing replay (R3): the routed-expert rows of tokens computed before a retract are
kept (SGLang captures routes per KV slot; the retract frees the slots and the re-prefill under the NEW weights would
overwrite them).  Fake capturer / scheduler module / result processor with the slime-cu130 SGLang semantics
(state_capturer/base.py get_topk: buffer[req_to_token[idx][start:seqlen-1]]; batch_result_processor
_maybe_collect_routed_experts gathers at finish; scheduler.pause_generation(retract) calls retract_all(reqs=...)).
Run: python3 tests/scripts/kvfresh_r3keep_cpu.py"""
import os
import sys
import tempfile
import types

import torch

os.environ["ASRT_SGL_HOOKS"] = "kvfresh"
os.environ["ASRT_LOG_DIR"] = tempfile.mkdtemp(prefix="r3keep-test-")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

L, K = 3, 2   # layers, topk


class Cap:
    def __init__(self):
        self.buffer = torch.zeros(1000, L, K, dtype=torch.int32)

    def get_topk(self, req_pool_idx, seqlen, req_to_token_pool, start_len=0):
        start_len = min(start_len, seqlen - 1)
        idx = req_to_token_pool.req_to_token[req_pool_idx][start_len:seqlen - 1]
        return self.buffer[idx]


CAP = Cap()
m = types.ModuleType("sglang.srt.state_capturer.routed_experts")
m.get_global_experts_capturer = lambda: CAP
for name in ("sglang", "sglang.srt", "sglang.srt.state_capturer"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["sglang.srt.state_capturer.routed_experts"] = m

from ai_search_plugin.adapters import slime_sglang_hooks as H  # noqa: E402


class Pool:
    def __init__(self):
        self.req_to_token = torch.zeros(8, 200, dtype=torch.int64)


class Req:
    def __init__(self, n_in):
        self.origin_input_ids = list(range(n_in))
        self.output_ids = []
        self.return_routed_experts = True
        self.routed_experts_start_len = 0
        self.routed_experts = None
        self.kv = types.SimpleNamespace(req_pool_idx=0, kv_committed_len=None)


def compute(req, pool, lo, hi, slot0, version):
    """forward of tokens [lo, hi) into slots slot0.. with routes that depend on the weight version."""
    for j, pos in enumerate(range(lo, hi)):
        pool.req_to_token[req.kv.req_pool_idx][pos] = slot0 + j
        CAP.buffer[slot0 + j] = (pos * 7) % 97 + version * 1000


def retract_all(*, reqs, req_to_token_pool, **kw):  # SGLang frees the KV (slots), output tokens are kept
    for r in reqs:
        r.kv.req_pool_idx = None


sched = types.ModuleType("fake_scheduler")
sched.retract_all = retract_all
sched.Scheduler = type("Scheduler", (), {"flush_cache": lambda self: True, "pause_generation": lambda self, r: None})
H.patch_scheduler(sched)
brp = types.ModuleType("fake_brp")


class SchedulerBatchResultProcessor:
    def __init__(self, pool):
        self.req_to_token_pool = pool

    def _maybe_collect_routed_experts(self, req):
        seqlen = len(req.origin_input_ids) + len(req.output_ids)
        req.routed_experts = CAP.get_topk(req.kv.req_pool_idx, seqlen, self.req_to_token_pool,
                                          req.routed_experts_start_len)


brp.SchedulerBatchResultProcessor = SchedulerBatchResultProcessor
H.patch_brp(brp)
assert getattr(SchedulerBatchResultProcessor, "_asrt_r3keep", False)
assert getattr(sched.retract_all, "_asrt_r3keep", False)


def run(retract_twice=False):
    pool = Pool()
    p = SchedulerBatchResultProcessor(pool)
    req = Req(10)
    compute(req, pool, 0, 10, 0, version=0)            # prefill under v0
    req.output_ids = list(range(5))                     # 5 tokens sampled under v0 (rows 10..13 computed)
    compute(req, pool, 10, 14, 10, version=0)
    v0_rows = CAP.get_topk(0, 15, pool, 0).clone()      # what R3 holds for tokens 0..13 before the sync
    sched.retract_all(reqs=[req], req_to_token_pool=pool)   # pause(retract): KV freed
    req.kv.req_pool_idx = 1                             # re-prefill of the 15 tokens under v1, then 5 more tokens
    compute(req, pool, 0, 15, 100, version=1)
    if retract_twice:
        req.output_ids += list(range(2))
        compute(req, pool, 15, 16, 120, version=1)
        sched.retract_all(reqs=[req], req_to_token_pool=pool)
        req.kv.req_pool_idx = 2
        compute(req, pool, 0, 17, 300, version=2)
    req.output_ids += list(range(5))
    n = len(req.origin_input_ids) + len(req.output_ids)
    compute(req, pool, 15 + (2 if retract_twice else 0), n - 1, 200, version=2 if retract_twice else 1)
    p._maybe_collect_routed_experts(req)
    return req, v0_rows, n


req, v0_rows, n = run()
assert req.routed_experts.shape[0] == n - 1
assert torch.equal(req.routed_experts[:14], v0_rows), "rows computed before the retract must be the v0 routes"
assert int(req.routed_experts[14, 0, 0]) == (14 * 7) % 97 + 1000, "rows after the retract: re-prefill / decode routes"
print("1 ok: rows before the retract kept, later rows from the re-prefill/decode")

req, v0_rows, n = run(retract_twice=True)
assert torch.equal(req.routed_experts[:14], v0_rows), "first retract's rows survive a second retract"
assert int(req.routed_experts[15, 0, 0]) == (15 * 7) % 97 + 1000, "row computed between the retracts keeps v1"
print("2 ok: two retracts - the oldest computation wins per row")

# 4. retracted in the middle of a chunked prefill: only the committed rows are kept (rest = stale slots of another req)
pool = Pool()
p = SchedulerBatchResultProcessor(pool)
CAP.buffer[500:600] = 99999                               # a previous occupant's routes in the slots req_to_token still names
req = Req(40)
pool.req_to_token[0][:40] = torch.arange(500, 540)      # stale mapping for the whole prompt
compute(req, pool, 0, 16, 0, version=0)                 # first chunk: 16 tokens computed under v0
req.kv.kv_committed_len = 16
sched.retract_all(reqs=[req], req_to_token_pool=pool)
req.kv.req_pool_idx = 1
compute(req, pool, 0, 40, 100, version=1)               # full prefill after the sync
req.output_ids = list(range(3))
compute(req, pool, 40, 42, 140, version=1)
p._maybe_collect_routed_experts(req)
assert int(req.routed_experts[5, 0, 0]) == (5 * 7) % 97, "committed rows keep v0"
assert int(req.routed_experts[20, 0, 0]) == (20 * 7) % 97 + 1000, "uncommitted rows come from the re-prefill, not stale"
assert int((req.routed_experts == 99999).sum()) == 0
print("4 ok: chunked-prefill retract keeps only committed rows")

# 5. retracted again in the middle of its chunked RE-prefill: the longer older stash must survive (found in review)
pool = Pool()
p = SchedulerBatchResultProcessor(pool)
req = Req(60)
compute(req, pool, 0, 60, 0, version=0)                 # full prefill under v0
req.output_ids = list(range(3))
compute(req, pool, 60, 62, 60, version=0)
v0_rows = CAP.get_topk(0, 63, pool, 0).clone()           # 62 rows
req.kv.kv_committed_len = 62
sched.retract_all(reqs=[req], req_to_token_pool=pool)
req.kv.req_pool_idx = 1
compute(req, pool, 0, 16, 100, version=1)                # re-prefill, first chunk only
req.kv.kv_committed_len = 16
sched.retract_all(reqs=[req], req_to_token_pool=pool)    # second sync mid chunk
req.kv.req_pool_idx = 2
req.kv.kv_committed_len = None
compute(req, pool, 0, 63, 200, version=2)
req.output_ids += list(range(4))
compute(req, pool, 63, 66, 300, version=2)
p._maybe_collect_routed_experts(req)
assert torch.equal(req.routed_experts[:62], v0_rows), "the 62 v0 rows must survive the mid-chunk second retract"
assert int(req.routed_experts[63, 0, 0]) == (63 * 7) % 97 + 2000
print("5 ok: second retract mid chunked re-prefill keeps the older, longer stash")
os.environ["ASRT_KVFRESH_R3KEEP"] = "0"
assert not H._r3keep_on()
print("3 ok: ASRT_KVFRESH_R3KEEP=0 disables it")
print("ALL OK")
