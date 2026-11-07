"""verl staleness ceiling on a fake verl v1 ReplayBufferAsync + trainer (no verl / GPU needed).

The fake mirrors verl HEAD's replay_buffer.py control flow (TransferQueue snapshot per poll, stale/failure eviction +
refill, `_has_enough_samples`, `_select_prompt_uids` oldest-enqueue first, `_materialize_batch`) and the separate_async
order per step g: prepare_step submits one batch tagged g -> sample(g) -> train -> update_weights(g).  A simulated
engine starts every submitted group at once with the current version (g-1); trajectories take a random time with a
long tail; tags carry min/max_global_steps like verl's agent loop.  Staleness = verl's trajectory_staleness_worst.

Checks: (1) uncapped runs reach worst 3-4 (short-tail knobs: warm-up 3.69, threshold 6; long-tail: warm-up 1.35,
threshold 4 with stragglers); (2) with the ceiling worst <= 2 at every step, cuts counted (0 with budget 0); (3) cut
groups are never trained, time cuts never exceed the budget; (4) admission never submits more prompts than verl asked
for; (5) ACTIVE line + admit.jsonl records.
Run: python3 tests/scripts/verl_ceiling_cpu.py
"""
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import types
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))


# ------------------------------------------------------------------ simulated world
class Sim:
    def __init__(self, seed, dur):
        self.t = 0.0
        self.version = 0
        self.rng = random.Random(seed)
        self.dur = dur              # rng -> seconds for one trajectory
        self.run = []               # [finish_t, key, uid, tag_steps, v0]
        self.store = {"train": {}}

    def start(self, uid, tag_steps, n=4):
        self.store["train"][uid] = {"is_prompt": True, "status": "running", "global_steps": tag_steps, "_left": n}
        gd = self.dur(self.rng)
        for i in range(n):
            self.run.append([self.t + gd * (0.6 + 0.4 * self.rng.random()), f"{uid}_{i}_0", uid, tag_steps, self.version])

    def advance(self, dt):
        self.t += dt
        keep = []
        for r in self.run:
            if r[0] <= self.t:
                fin, key, uid, g, v0 = r
                p = self.store["train"].get(uid)
                if p is None:   # prompt cleared (sampled / evicted) - verl's agent loop would still put; ignore
                    continue
                self.store["train"][key] = {"status": "success", "global_steps": g, "min_global_steps": v0,
                                            "max_global_steps": self.version}
                p["_left"] -= 1
                if p["_left"] == 0:
                    p["status"] = "finished"
            else:
                keep.append(r)
        self.run = keep


SIM = [None]


def _tq():
    def kv_list():
        return {pid: {k: {kk: vv for kk, vv in t.items() if kk != "_left"} for k, t in items.items()}
                for pid, items in SIM[0].store.items()}

    def kv_clear(partition_id, keys):
        for k in keys:
            SIM[0].store[partition_id].pop(k, None)
    return SimpleNamespace(kv_list=kv_list, kv_clear=kv_clear)


# ------------------------------------------------------------------ fake verl modules
def make_verl():
    for n in ("verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.v1"):
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    rbm = types.ModuleType("verl.trainer.ppo.v1.replay_buffer")
    rbm.tq = _tq()
    tq = rbm.tq
    from collections import defaultdict

    class ReplayBuffer:
        def __init__(self, max_off_policy_threshold, max_off_policy_strategy, refill_fn, poll_interval=2.0):
            self.max_off_policy_threshold, self.max_off_policy_strategy = max_off_policy_threshold, max_off_policy_strategy
            self.refill_fn, self.poll_interval = refill_fn, poll_interval
            self.partitions = defaultdict(dict); self.pending_keys = defaultdict(set); self.running_keys = defaultdict(set)
            self.finished_keys = defaultdict(set); self.failure_keys = defaultdict(set)
            self.prompt_global_steps = defaultdict(dict)
            self.evicted_stale = 0

        def _sync_metadata_from_transfer_queue(self):
            for d in (self.partitions, self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys,
                      self.prompt_global_steps):
                d.clear()
            for pid, items in tq.kv_list().items():
                for key, tag in items.items():
                    if tag.get("is_prompt", False):
                        self.prompt_global_steps[pid][key] = tag["global_steps"]
                        {"pending": self.pending_keys, "running": self.running_keys, "finished": self.finished_keys,
                         "failure": self.failure_keys}[tag["status"]][pid].add(key)
                    else:
                        self.partitions[pid].setdefault(key, {}).update(tag)

        def _clear_groups(self, pid, uids):
            if not uids:
                return
            tk = {k for k in self.partitions[pid] if k.split("_")[0] in uids}
            tq.kv_clear(partition_id=pid, keys=[*uids, *tk])
            for k in tk:
                del self.partitions[pid][k]
            for s in (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys):
                s[pid].difference_update(uids)
            for u in uids:
                self.prompt_global_steps[pid].pop(u, None)

        def _terminal_eviction_reasons(self, global_steps, pid):
            return set(), set(), set(), None

        def _sampleable_terminal_keys(self, pid, reasons):
            return (self.finished_keys[pid] | self.failure_keys[pid]) - (reasons[0] | reasons[1] | reasons[2])

        def _evict_terminal_groups(self, global_steps, pid, reasons):
            ev = reasons[0] | reasons[1] | reasons[2]
            if not ev:
                return set(), 0, 0, {}
            self.evicted_stale += len(reasons[0])
            self._clear_groups(pid, ev)
            return ev, len(reasons[0]), len(reasons[1]), {}

        def _select_prompt_uids(self, pid, sampleable_keys, batch_size):
            pgs = dict(self.prompt_global_steps[pid]); part = dict(self.partitions[pid])
            return sorted(sampleable_keys, key=lambda k: pgs.get(k, 0))[:batch_size], part, pgs

        def _materialize_batch(self, pid, uids, part):
            tq.kv_clear(partition_id=pid, keys=uids)
            sel = set(uids)
            keys = [k for k in part if k.split("_")[0] in sel]
            return SimpleNamespace(partition_id=pid, keys=keys, tags=[part[k] for k in keys])

        def _wait_for_next_poll(self, pid, last):
            SIM[0].advance(self.poll_interval)
            return last

    class ReplayBufferAsync(ReplayBuffer):
        def _stale_terminal_keys(self, global_steps, pid):
            if self.max_off_policy_threshold is None or self.max_off_policy_strategy != "drop":
                return set()
            pgs = self.prompt_global_steps[pid]
            return {u for u in self.finished_keys[pid]
                    if global_steps - pgs.get(u, global_steps) + 1 > self.max_off_policy_threshold}

        def _terminal_eviction_reasons(self, global_steps, pid):
            return self._stale_terminal_keys(global_steps, pid), set(), set(self.failure_keys[pid]), None

        def _has_enough_samples(self, global_steps, pid, batch_size, sampleable_keys):
            return len(sampleable_keys) >= batch_size

        def get_sampleable_count(self, global_steps, pid):
            self._sync_metadata_from_transfer_queue()
            return len(self._sampleable_terminal_keys(pid, self._terminal_eviction_reasons(global_steps, pid)))

        def wait_for_sampleable(self, global_steps, pid, target):
            n_poll = 0
            while True:
                self._sync_metadata_from_transfer_queue()
                reasons = self._terminal_eviction_reasons(global_steps, pid)
                ev, stale, _d, _m = self._evict_terminal_groups(global_steps, pid, reasons)
                if ev:
                    self.refill_fn(len(ev))
                    continue
                sk = self._sampleable_terminal_keys(pid, reasons)
                if self._has_enough_samples(global_steps, pid, target, sk):
                    return sk, {}
                n_poll += 1
                assert n_poll < 200000, "sampler hung"
                self._wait_for_next_poll(pid, 0)

        def sample(self, global_steps, pid, batch_size):
            sk, m = self.wait_for_sampleable(global_steps, pid, batch_size)
            uids, part, pgs = self._select_prompt_uids(pid, sk, batch_size)
            if self.max_off_policy_threshold is not None and self.max_off_policy_strategy == "drop":
                assert all(global_steps - pgs.get(u, global_steps) + 1 <= self.max_off_policy_threshold for u in uids)
            return self._materialize_batch(pid, uids, part), m

    ReplayBuffer.__module__ = ReplayBufferAsync.__module__ = rbm.__name__
    rbm.ReplayBuffer, rbm.ReplayBufferAsync = ReplayBuffer, ReplayBufferAsync
    sys.modules[rbm.__name__] = rbm

    tbm = types.ModuleType("verl.trainer.ppo.v1.trainer_base")

    class PPOTrainer:
        def __init__(self, B, steps, warmup, threshold, train_s):
            self.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=B),
                                          skip=SimpleNamespace(rollout_tq=SimpleNamespace(enable=False)))
            self.trainer_mode, self.parameter_sync_step, self.total_training_steps = "separate_async", 1, steps
            self.global_steps, self.warmup, self.train_s = 0, warmup, train_s
            self.replay_buffer = ReplayBufferAsync(threshold, "drop", self._add_prompts_to_generate)
            self.asked = 0
            self.n_uid = 0

        def _next_train_batch(self, n=None):
            n = self.config.data.train_batch_size if n is None else n
            assert n > 0
            uids = [f"u{self.n_uid + i:05d}" for i in range(n)]
            self.n_uid += n
            return {"uid": uids}

        def _submit_batch_to_rollout(self, batch):
            for u in batch["uid"]:
                SIM[0].start(u, self.global_steps)
            return len(batch["uid"])

        def _add_prompts_to_generate(self, n):  # verl's; the ceiling wraps it (demand counted in fit / refill)
            return self._submit_batch_to_rollout(self._next_train_batch(n))

        def _add_batch_to_generate(self):
            self._submit_batch_to_rollout(self._next_train_batch())

        def fit(self):
            out = []
            self.global_steps = 1
            self.asked = int(self.warmup * self.config.data.train_batch_size) + self.total_training_steps * self.config.data.train_batch_size
            self._add_prompts_to_generate(int(self.warmup * self.config.data.train_batch_size))  # on_train_begin
            while self.global_steps <= self.total_training_steps:
                g = self.global_steps
                t0 = SIM[0].t
                self._add_batch_to_generate()                                                   # prepare_step
                batch, _ = self.replay_buffer.sample(g, "train", self.config.data.train_batch_size)
                ws = [(g - 1) - t["min_global_steps"] for t in batch.tags]
                SIM[0].advance(self.train_s)                                                     # train
                SIM[0].version = g                                                               # update_weights(g)
                out.append({"step": g, "worst": max(ws), "mean": sum(ws) / len(ws), "dt": SIM[0].t - t0,
                            "uids": {k.split("_")[0] for k in batch.keys}})
                self.global_steps += 1
            return out

    tbm.PPOTrainer = PPOTrainer
    sys.modules[tbm.__name__] = tbm
    return rbm, tbm


RBM, TBM = make_verl()
from ai_search_plugin.adapters import verl_ceiling as vc  # noqa: E402

CLOCK = SimpleNamespace(time=lambda: SIM[0].t)
vc.time = CLOCK
assert vc.install(2)


def run(capped, *, seed, B, steps, warmup, threshold, train_s, dur, env=None):
    SIM[0] = Sim(seed, dur)
    old = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    vc._C[0] = vc.Ceiling(2) if capped else None
    buf = io.StringIO()
    try:
        tr = TBM.PPOTrainer(B, steps, warmup, threshold, train_s)
        with contextlib.redirect_stdout(buf):
            res = tr.fit()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    c = vc._C[0]
    return res, c, tr, buf.getvalue()


def tail_dur(p_slow, slow_s, base):
    def d(rng):
        return slow_s * (0.7 + 0.6 * rng.random()) if rng.random() < p_slow else base * (0.5 + rng.random())
    return d


SHORT_TAIL = dict(B=8, steps=12, warmup=3.69, threshold=6, train_s=60.0, dur=tail_dur(0.03, 150, 40))
LONG_TAIL = dict(B=8, steps=12, warmup=1.35, threshold=4, train_s=100.0, dur=tail_dur(0.06, 700, 300))


def _summary(res):
    return [r["worst"] for r in res], sum(r["dt"] for r in res) / len(res)


def test_uncapped_exceeds():
    for name, sc in (("short_tail", SHORT_TAIL), ("long_tail", LONG_TAIL)):
        mx = max(max(_summary(run(False, seed=s, **sc)[0])[0]) for s in range(3))
        print(f"uncapped {name}: worst max {mx}")
        assert mx >= 3, (name, mx)


def test_ceiling_holds_wait_only():
    with tempfile.TemporaryDirectory() as d:
        for name, sc in (("short_tail", SHORT_TAIL), ("long_tail", LONG_TAIL)):
            for seed in range(4):
                base, _, trb, _ = run(False, seed=seed, **sc)
                res, c, tr, out = run(True, seed=seed, env={"ASRT_TICKET_DROP_BUDGET": "0", "ASRT_LOG_DIR": d}, **sc)
                ws, mean_dt = _summary(res)
                print(f"capped {name} seed {seed}: worst/step {ws} s/step {mean_dt:.0f} (uncapped "
                      f"{_summary(base)[0]} {_summary(base)[1]:.0f}); cuts {c.n_cut} (time {c.n_cut_time}, mandatory "
                      f"{c.n_cut_mandatory}) deferred {c.owed} asked {tr.asked} submitted {tr.n_uid}")
                assert max(ws) <= 2, ws
                assert c.n_cut_time == 0                              # budget 0: stragglers are waited for
                demand = tr.asked + tr.replay_buffer.evicted_stale    # warm-up + one batch per step + verl refills
                assert tr.n_uid + (c.owed if c else 0) == demand, (tr.n_uid, c.owed, demand)  # never more than verl asked
                assert "staleness ceiling ACTIVE: S=2" in out and "reader verl(ticket)" in out
        recs = [json.loads(l) for l in open(os.path.join(d, "admit.jsonl"))]
        steps = [r for r in recs if r.get("event") == "verl_step"]
        assert any(r.get("event") == "ceiling_active" for r in recs)
        assert steps and all(r["worst_max"] <= 2 for r in steps)
        assert all(k in steps[0] for k in ("admitted", "deferred", "dropped", "refused", "wait_s", "worst_max"))
        print("admit.jsonl ok:", {k: steps[-1][k] for k in ("step", "admitted", "deferred", "refused", "dropped", "worst_max")})


def test_cut_budget_and_never_trained():
    with tempfile.TemporaryDirectory() as d:
        env = {"ASRT_TICKET_DROP_BUDGET": "2", "ASRT_TICKET_DROP_S": "200", "ASRT_LOG_DIR": d}
        sc = dict(LONG_TAIL, dur=tail_dur(0.15, 1500, 300))
        res, c, tr, out = run(True, seed=7, env=env, **sc)
        ws, _ = _summary(res)
        trained = set().union(*(r["uids"] for r in res))
        recs = [json.loads(l) for l in open(os.path.join(d, "admit.jsonl"))]
        cuts = [r for r in recs if r.get("event") == "verl_cut"]
        print(f"cut test: worst/step {ws}; cuts {c.n_cut} (time {c.n_cut_time}, mandatory {c.n_cut_mandatory}); "
              f"first cut {cuts[0] if cuts else None}")
        assert max(ws) <= 2
        assert 0 < c.n_cut_time <= 2
        assert not (trained & set(c.cut)), "a cut group was trained"
        assert len(cuts) == c.n_cut and all("uid" in r and "age_steps" in r for r in cuts)
        assert "[asrt] ticket cut:" in out and all(r["age_s"] is not None for r in cuts)


def test_selection_is_deadline_first():
    # two finished groups: enqueued at step 3 but started on version 1 (exact) vs enqueued at step 2 started on 2
    SIM[0] = Sim(0, lambda r: 1.0)
    c = vc._C[0] = vc.Ceiling(2)
    c.B = c.Bs = 1
    rb = RBM.ReplayBufferAsync(4, "drop", lambda n: None)
    rb.prompt_global_steps["train"] = {"a": 2, "b": 3}
    rb.partitions["train"] = {"a_0_0": {"min_global_steps": 2}, "b_0_0": {"min_global_steps": 1}}
    sel, _, _ = rb._select_prompt_uids("train", {"a", "b"}, 1)
    assert sel == ["b"], sel


def test_with_offpol():
    # ASRT_VERL=full includes offpol (wraps sample / _evict_terminal_groups of the same classes): both must work
    from ai_search_plugin.adapters import verl_offpol as vo
    for C in (RBM.ReplayBuffer, RBM.ReplayBufferAsync):
        vo._patch_class(C)
    res, c, tr, out = run(True, seed=1, env={"ASRT_TICKET_DROP_BUDGET": "0"}, **SHORT_TAIL)
    assert max(_summary(res)[0]) <= 2 and "[asrt] verl offpol " in out and "[asrt] staleness rollout 12 " in out
    print("with offpol ok:", _summary(res)[0])


if __name__ == "__main__":
    for f in (test_uncapped_exceeds, test_ceiling_holds_wait_only, test_cut_budget_and_never_trained,
              test_selection_is_deadline_first, test_with_offpol):
        f()
    print("ok")
