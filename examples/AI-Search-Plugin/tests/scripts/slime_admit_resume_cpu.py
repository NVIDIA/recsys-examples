"""slime_admit ticket book across a slime checkpoint resume (run -> save -> resume).

slime's REAL fully-async worker (same stubbed harness as slime_admit_cpu.py) under a source that saves and
restores like slime's RolloutDataSource(WithBuffer): state_dict() pauses + drains the consumers and stores
{cursor, requeue buffer, consumers: {"fully_async": worker.state_dict()}}; a new process builds a fresh source,
load_state_dict() keeps the consumer state, register_consumer() hands it to the new worker before start().

Segment A runs rollouts 0..s, saves after train(s) (train.py order: train -> save -> pause -> update -> resume),
overruns past the save like segstop does, and is thrown away.  Segment B = a fresh import of slime + slime_admit
(new process), restores the pickled state and runs s+1..N-1 with the weight version continuing (slime_resume).

Checks (each fails on an older slime_admit without the resume book):
  * prompt continuity: every group drawn before the save is handed out exactly once, in A (<= s) or B, or is
    still outstanding at the end of B; nothing in between is skipped (old: a whole batch, = Tickets.buf, lost);
  * the same rollout s+1 holds the same groups with the same results as A's (overrun) rollout s+1, for every
    group that was complete at the save;
  * ticket ledger 0 BAD in B; batch counter continues (B's first batch is s+1);
  * versions: B's staleness records continue the train version of A (A's rollout s+1 reference = B's), token-
    level max <= S;
  * a version counter that restarts after the resume is flagged (ledger BAD), not silently accepted;
  * checkpoint without the asrt book (slime's own worker state): the ledger still balances;
  * no resume: the worker's saved state is slime's {"ready": ...} plus the book (no change to the run).
usage: python3 tests/scripts/slime_admit_resume_cpu.py   (or pytest)
"""
import copy
import json
import os
import pickle
import random
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
import slime_admit_cpu as H  # noqa: E402  (stubbed slime + fake generation)

Sample = H.Sample
SIM = H.SIM


class Source:
    """slime RolloutDataSourceWithBuffer, reduced: one prompt per group, requeue buffer popped first, cursor fields
    and state_dict / load_state_dict / register_consumer as in slime/data/data_source.py."""

    def __init__(self, args, n_prompts=100000):
        self.args = args
        self.consumers, self._restored_consumers = {}, {}
        self.epoch_id = self.sample_group_index = self.sample_index = self.sample_offset = 0
        self.metadata, self.buffer = {}, []
        prompts = []
        for i in range(n_prompts):
            p = Sample(None, None)
            p.prompt = f"q{i}"
            prompts.append(p)
        self.dataset = SimpleNamespace(samples=prompts)

    def register_consumer(self, name, consumer):
        if name in self._restored_consumers:
            consumer.load_state_dict(self._restored_consumers.pop(name))
        self.consumers[name] = consumer

    def get_samples(self, k):
        out = []
        while self.buffer and len(out) < k:
            out.append(self.buffer.pop(0))
        for _ in range(k - len(out)):
            p = self.dataset.samples[self.sample_offset]
            self.sample_offset += 1
            g = []
            for _j in range(self.args.n_samples_per_prompt):
                s = copy.deepcopy(p)
                s.group_index, s.index = self.sample_group_index, self.sample_index
                self.sample_index += 1
                g.append(s)
            self.sample_group_index += 1
            out.append(g)
        return out

    def add_samples(self, groups):
        self.buffer.extend(groups)

    def state_dict(self):
        paused = [(c, c.pause()) for c in self.consumers.values()]
        try:
            st = {k: copy.deepcopy(getattr(self, k)) for k in
                  ("sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata", "buffer")}
            st["consumers"] = {**self._restored_consumers, **{n: c.state_dict() for n, c in self.consumers.items()}}
            return st
        finally:
            for c, was in paused:
                if not was:
                    c.resume()

    def load_state_dict(self, st):
        for k, v in st.items():
            if k == "consumers":
                self._restored_consumers = v
            else:
                setattr(self, k, v)


async def _gen(args, group, sampling_params=None, evaluation=False):
    """flush-1 style generation (aborts at every sync, partial kept) + a random reward drawn at completion."""
    out = await H._gen_flush1(group)
    for s in out:
        if s.status == Sample.Status.COMPLETED and getattr(s, "reward", None) is None:
            s.reward = random.random()
    return out


def _env(S, log_dir, extra=None):
    for k in [k for k in os.environ if k.startswith("ASRT_")]:
        os.environ.pop(k)
    os.environ.update({"ASRT_ADMIT": "profile", "ASRT_MAX_STALENESS": str(S), "ASRT_LOG_DIR": log_dir,
                       "ASRT_TICKET_POLL": "0"})
    os.environ.update(extra or {})


def _process(B, n, conc, steps):
    """A fresh 'process': stubbed slime, new slime_admit import, a manager on a resumable source."""
    far = H._stub_slime()
    sys.modules["slime.rollout.sglang_rollout"].generate_and_rm_group = _gen
    far.generate_and_rm_group = _gen
    for k in [k for k in sys.modules if k.startswith("ai_search_plugin.adapters.slime_admit")]:
        del sys.modules[k]
    Mgr = H.make_manager_class(0.0)
    args = SimpleNamespace(rollout_batch_size=B, n_samples_per_prompt=n, sglang_server_concurrency=conc, engines=1,
                           fully_async_group_prefetch_factor=1, rollout_data_transport="ray",
                           dynamic_sampling_filter_path=None, rollout_sample_filter_path=None, eval_interval=None,
                           save_interval=None, num_rollout=steps, offload_rollout=False, colocate=False,
                           debug_rollout_only=False, rollout_shuffle=False)
    mgr = Mgr(args, "ai_search_plugin.adapters.slime_admit.generate_rollout_fully_async")
    mgr.data_source = Source(args)
    import ai_search_plugin.adapters.slime_admit as sa
    return mgr, sa, far


def _step(mgr, i, train_s, out, save=None):
    o = mgr.generate(i)
    o["results"] = {g: None for g in o["groups"]}
    out.append(o)
    time.sleep(train_s)
    if save is not None:
        save.append(pickle.dumps(mgr.data_source.state_dict()))   # = torch.save(global_dataset_state_dict_i.pt)
    was = mgr.pause_rollout_admission()
    SIM.epoch += 1          # flush-cache-interval 1: abort every in-flight request
    SIM.version += 1
    time.sleep(0.01)
    mgr.resume_rollout_admission(was)


def _recs(log_dir):
    with open(os.path.join(log_dir, "admit.jsonl")) as f:
        return [json.loads(x) for x in f]


def _outstanding(sa, mgr):
    """Group indices admitted but not handed at the end (pending / buffered / queued / requeued)."""
    w = mgr.data_source.consumers["fully_async"]
    tk = sa._TK[0]
    gis = set()
    with tk.lock:
        for t, ks in tk.pend_keys.items():
            gis |= {k[1] // w.args.n_samples_per_prompt for k in ks if k[0] == "i"}
        for lst in tk.buf.values():
            gis |= {g[0].group_index for _, g in lst}
    with w.output_queue.mutex:
        gis |= {g[0].group_index for _, g in w.output_queue.queue}
    gis |= {g[0].group_index for g in mgr.data_source.buffer}
    return gis


def run_resume(B=8, n=2, conc=40, s=3, overrun=2, N=8, S=2, train_s=0.08, version_reset=False, no_book=False,
               seed=0):
    """no_book: strip the asrt book from the saved worker state (what plain slime saves)."""
    random.seed(seed)
    SIM.version, SIM.epoch, SIM.gen_s, SIM.flush1, SIM.slow, SIM.slow_s = 0, 0, (0.01, 0.05), True, set(), 0.0
    SIM.slow = {i for i in range(200) if i % 7 == 3}
    SIM.slow_s = 0.25            # stragglers: DEFER moves them across tickets, so buf holds several tickets
    dA, dB = tempfile.mkdtemp(prefix="resA"), tempfile.mkdtemp(prefix="resB")
    # ---------------------------------------------------------------- segment A
    _env(S, dA)
    mgr, sa, far = _process(B, n, conc, N)
    outA, saved = [], []
    for i in range(s + 1 + overrun):
        _step(mgr, i, train_s, outA, save=saved if i == s else None)
        if i == s:
            v_at_save = SIM.version   # after the update that follows the save (B's engines start here)
            stA = pickle.loads(saved[0])
            readyA = {g[0].group_index: [getattr(x, "reward", None) for x in g]
                      for _, g in stA["consumers"]["fully_async"]["ready"]}
            cursorA = stA["sample_group_index"]
            requeuedA = {g[0].group_index for g in stA["buffer"]}
    mgr.data_source.consumers["fully_async"].close()
    staleA = [r for r in _recs(dA) if r.get("event") == "staleness"]
    state = pickle.loads(saved[0])
    if no_book:   # slime's queue only, the ticket buffer and the book missing
        w = state["consumers"]["fully_async"]
        book = w.pop("asrt")
        w["ready"] = w["ready"][book["buf_saved"]:]
    # ---------------------------------------------------------------- segment B (new process)
    _env(S, dB)
    SIM.version = 1 if version_reset else v_at_save
    mgr, sa, far = _process(B, n, conc, N)
    mgr.data_source.load_state_dict(state)
    outB = []
    for i in range(s + 1, N):
        _step(mgr, i, train_s, outB)
    tk = sa._TK[0]
    left = _outstanding(sa, mgr)
    cursorB = mgr.data_source.sample_group_index
    mgr.data_source.consumers["fully_async"].close()
    recsB = _recs(dB)
    handed = [g for o in outA[:s + 1] for g in o["groups"]] + [g for o in outB for g in o["groups"]]
    return SimpleNamespace(outA=outA, outB=outB, handed=handed, left=left, cursorA=cursorA, cursorB=cursorB, tk=tk,
                           readyA=readyA, requeuedA=requeuedA, staleA=staleA,
                           staleB=[r for r in recsB if r.get("event") == "staleness"],
                           ledgerB=[r for r in recsB if r.get("event") == "ticket_ledger"],
                           resumeB=[r for r in recsB if r.get("event") == "resume"], B=B, s=s, S=S, sa=sa)


def _check_continuity(r):
    assert len(r.handed) == len(set(r.handed)), "a group was handed out twice across the resume"
    assert all(len(o["groups"]) == r.B for o in r.outA + r.outB)
    have = set(r.handed) | r.left
    missing = sorted(set(range(r.cursorB)) - have)
    assert not missing, (f"prompts drawn but never handed out nor outstanding (skipped across the resume): "
                         f"{len(missing)} groups, e.g. {missing[:12]}")
    assert max(have) < r.cursorB


def _check_ledger(r):
    bad = [x for x in r.ledgerB if not x.get("ok")]
    assert not bad, f"ticket ledger BAD after the resume: {bad[:2]}"
    assert r.ledgerB and r.ledgerB[0]["batch"] == r.s + 1, \
        f"batch counter restarted: first ledger batch {r.ledgerB[0]['batch'] if r.ledgerB else None} != {r.s + 1}"


def _check_versions(r):
    a = {x["rollout"]: x for x in r.staleA}
    b = {x["rollout"]: x for x in r.staleB}
    assert min(b) == r.s + 1, f"staleness log restarted its rollout count at {min(b)}"
    for k in sorted(set(a) & set(b)):   # A's overrun rollouts vs B's same rollouts
        assert a[k]["train_version"] == b[k]["train_version"], (k, a[k]["train_version"], b[k]["train_version"])
    tv = [b[k]["train_version"] for k in sorted(b)]
    assert tv == list(range(tv[0], tv[0] + len(tv))), f"train versions not continuous: {tv}"
    assert tv[0] == a[r.s]["train_version"] + 1, (tv[0], a[r.s]["train_version"])
    tm = [x["tok_max"] for x in r.staleB if x.get("tok_max") is not None]
    assert tm and max(tm) <= r.S, f"token-level staleness over {r.S}: {tm}"


def test_resume_book():
    r = run_resume()
    _check_continuity(r)
    _check_ledger(r)
    _check_versions(r)
    # the same rollout s+1: same groups, and every group complete at the save keeps its result
    a1 = r.outA[r.s + 1]["groups"]
    b1 = r.outB[0]["groups"]
    assert sorted(a1) == sorted(b1), f"rollout {r.s + 1}: A {sorted(a1)} vs B {sorted(b1)}"
    assert r.resumeB and r.resumeB[0]["mode"] == "book"
    print(f"  book: rollout {r.s + 1} groups identical ({len(b1)}), handed {len(r.handed)} + outstanding "
          f"{len(r.left)} = cursor {r.cursorB}; ledger {len(r.ledgerB)} OK; buf restored "
          f"{r.resumeB[0].get('buffered_restored')}; tok_max {max(x['tok_max'] for x in r.staleB)}")


def test_resume_version_reset_flagged():
    r = run_resume(version_reset=True)
    assert r.tk.resume_version_bad > 0, "a weight version that restarted after the resume went unnoticed"
    assert any(not x.get("ok") for x in r.ledgerB)


def test_resume_no_book_balances():
    r = run_resume(no_book=True)
    _check_ledger(r)
    assert r.resumeB[0]["mode"] == "no_book" and r.resumeB[0].get("adopted", 0) > 0


def test_no_resume_state_is_slime_plus_book():
    random.seed(2)
    SIM.version, SIM.epoch, SIM.flush1, SIM.slow = 0, 0, True, set()
    d = tempfile.mkdtemp(prefix="res0")
    _env(2, d)
    mgr, sa, far = _process(8, 2, 40, 4)
    out = []
    _step(mgr, 0, 0.05, out)
    st = mgr.data_source.state_dict()
    w = st["consumers"]["fully_async"]
    assert set(w) == {"ready", "asrt"} and w["asrt"]["mode"] == "ticket"
    _step(mgr, 1, 0.05, out)   # the run goes on unchanged after a save
    mgr.data_source.consumers["fully_async"].close()
    assert [len(o["groups"]) for o in out] == [8, 8]


def test_resume_seed_version():
    """slime_resume: version continues from start_rollout_id only on a slime resume (data-source state saved)."""
    import types

    from ai_search_plugin.adapters import slime_resume as R
    os.environ.pop("ASRT_RESUME_VERSION", None)
    ck = tempfile.mkdtemp(prefix="resck")
    os.makedirs(os.path.join(ck, "rollout"))
    for i in (1, 49):
        open(os.path.join(ck, "rollout", f"global_dataset_state_dict_{i}.pt"), "w").close()
    base = tempfile.mkdtemp(prefix="resbase")   # converted Megatron base checkpoint: no data-source state

    def ns(**k):
        d = dict(update_weight_start_version=0, start_rollout_id=None, finetune=False, load=ck)
        d.update(k)
        return SimpleNamespace(**d)
    a = ns()
    assert R.seed_version(a, 49) == 50 and a.update_weight_start_version == 50
    a = ns(start_rollout_id=0, finetune=True)
    assert R.seed_version(a, 0) is None and a.update_weight_start_version == 0          # HF load (finetune)
    a = ns(load=base)
    assert R.seed_version(a, 0) is None and a.update_weight_start_version == 0          # Megatron base, iteration 0
    a = ns(update_weight_start_version=7)
    assert R.seed_version(a, 49) is None and a.update_weight_start_version == 7         # straw already seeded
    a = ns(update_weight_mode="full", update_weight_transport="disk")
    assert R.seed_version(a, 49) is None                                                 # driver counts disk versions
    # install on a fake actor module: create_weight_updater sees the seeded value
    key = "slime.backends.megatron_utils.actor"
    saved = sys.modules.get(key)
    mod = types.ModuleType(key)
    seen = {}
    mod.initialize_model_and_optimizer = lambda args, role: ("m", "o", "s", 1)
    mod.create_weight_updater = lambda args, *x, **k: seen.setdefault("v", args.update_weight_start_version)
    sys.modules[key] = mod
    try:
        assert R.install(None)
        a = ns()
        mod.initialize_model_and_optimizer(a, "actor")
        mod.create_weight_updater(a)
        assert seen["v"] == 2, seen    # 929s: loaded rollout 1 -> start 2 -> first sync publishes 3 (segment a: 3)
        os.environ["ASRT_RESUME_VERSION"] = "0"
        assert R.seed_version(ns(), 49) is None
    finally:
        os.environ.pop("ASRT_RESUME_VERSION", None)
        if saved is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = saved


if __name__ == "__main__":
    fails = 0
    for name, fn in [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
