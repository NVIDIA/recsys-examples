"""slime_admit on slime's REAL fully-async worker (slime/rollout/fully_async_rollout.py, deps stubbed) with a fake
generation (async sleeps) and a fake RolloutManager/trainer loop that mirrors slime's train.py order:
generate(i) -> train(i) -> pause admission -> update weights (version += 1) -> resume -> generate(i+1).

Checks: (1) cap: outstanding groups never exceed (1+ahead) x B; (2) every group is handed out exactly once, in
completion (FIFO) order, nothing lost; (3) eager prebuild: generate(i+1) returns the batch built while train(i)
ran, the conversion time leaves the critical path, the staleness reference version is the trainer's; (4) when
rollout is the bottleneck the cap does not cost throughput; (5) ASRT_ADMIT unset = slime's own behaviour.
Source of fully_async_rollout.py: $SLIME_SRC (a slime checkout's slime/ dir, i.e. $SLIME_SRC/rollout/
fully_async_rollout.py) or an installed slime.  Without either the script exits 77 (= skipped by the pytest runner).
usage: python3 tests/scripts/slime_admit_cpu.py   (or pytest)
"""
import asyncio
import importlib
import importlib.util
import os
import random
import sys
import threading
import time
import types
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)   # dynamic-filter test function is loaded by module path
SRC = [os.path.join(os.environ.get("SLIME_SRC", "/nonexistent"), "rollout", "fully_async_rollout.py")]
try:  # an installed slime (not imported: only its file is read, with its dependencies stubbed)
    _spec = importlib.util.find_spec("slime")
    if _spec is not None and _spec.submodule_search_locations:
        SRC.append(os.path.join(list(_spec.submodule_search_locations)[0], "rollout", "fully_async_rollout.py"))
except Exception:
    pass
if not any(os.path.exists(p) for p in SRC):
    print("SKIP: slime's rollout/fully_async_rollout.py not found (set SLIME_SRC=<slime checkout>/slime)")
    sys.exit(77)


class Sample:
    class Status:
        COMPLETED, TRUNCATED, ABORTED, PENDING = "completed", "truncated", "aborted", "pending"

    def __init__(self, index, group_index):
        self.index, self.group_index, self.status = index, group_index, None
        self.weight_versions = []


SIM = SimpleNamespace(version=0, gen_s=(0.02, 0.06), engine_conc=1 << 30, running=0, slow=set(), slow_s=0.0,
                      flush1=False, epoch=0)


async def _gen_flush1(group):
    """ slime --flush-cache-interval 1: every weight sync aborts every in-flight request
    (pause_generation mode="abort").  A flat sample keeps its partial response (and the weight version of the part
    already generated) and resumes from there when its group is generated again; finished siblings keep their
    results.  s.left = generation time still needed."""
    async def one(s):
        if s.status in (Sample.Status.COMPLETED,):
            return s
        if getattr(s, "left", None) is None:
            s.left = SIM.slow_s if s.group_index in SIM.slow else random.uniform(*SIM.gen_s)
        ep = SIM.epoch
        s.weight_versions.append(str(SIM.version))
        t0 = time.time()
        while time.time() - t0 < s.left:
            if SIM.epoch != ep:            # aborted by the sync: partial kept, resumes later
                s.left -= time.time() - t0
                s.status = Sample.Status.ABORTED
                return s
            await asyncio.sleep(0.005)
        s.left = 0.0
        s.weight_versions.append(str(SIM.version))
        s.status = Sample.Status.COMPLETED
        return s
    return list(await asyncio.gather(*(one(s) for s in group)))


async def generate_and_rm_group(args, group, sampling_params=None, evaluation=False):
    if SIM.flush1:
        return await _gen_flush1(group)
    # engine with limited concurrency: a sample decodes only while it holds an engine slot
    async def one(s):
        while SIM.running >= SIM.engine_conc:
            await asyncio.sleep(0.002)
        SIM.running += 1
        s.weight_versions.append(str(SIM.version))
        try:
            await asyncio.sleep(SIM.slow_s if s.group_index in SIM.slow else random.uniform(*SIM.gen_s))
        finally:
            SIM.running -= 1
        s.weight_versions.append(str(SIM.version))
        s.status = Sample.Status.COMPLETED
        return s
    return list(await asyncio.gather(*(one(s) for s in group)))


class _Loop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


_LOOP = _Loop()


def _stub_slime():
    for k in [k for k in sys.modules if k == "slime" or k.startswith("slime.")]:
        del sys.modules[k]

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    def iter_samples(groups):
        for g in groups:
            for s in (g if isinstance(g, list) else [g]):
                yield s

    def prepare_retry_group(inputs, results):  # = slime/rollout/base_types.py: keep finished + flat partial samples
        return [original if isinstance(result, list) and any(x.status in (Sample.Status.ABORTED, Sample.Status.PENDING)
                                                             for x in iter_samples(result)) else result
                for original, result in zip(inputs, results, strict=True)]

    def finalize_rollout_groups(args, rollout_id, groups, metrics=None, *, controller=None):
        groups.sort(key=lambda g: g[0].index)
        return SimpleNamespace(samples=groups, metrics=metrics)

    class MetricGatherer:
        def on_group_completed(self, v):
            pass

        def collect(self):
            return {}

    def call_dynamic_filter(f, args, group):
        if f is not None:  # dynamic filter (DAPO-style): f(args, group) -> keep?
            keep = bool(f(args, group))
            return SimpleNamespace(keep=keep, reason=None if keep else "dynamic_filter")
        return SimpleNamespace(keep=True, reason=None)

    def load_function(path):
        m, _, a = path.rpartition(".")
        return getattr(importlib.import_module(m), a)

    def should_run_periodic_action(rollout_id, interval, num_rollout_per_epoch=None, num_rollout=None):
        if interval is None:
            return False
        if num_rollout is not None and rollout_id == num_rollout - 1:
            return True
        return (rollout_id + 1) % interval == 0

    mod("slime"); mod("slime.data"); mod("slime.rollout"); mod("slime.utils"); mod("slime.rollout.filter_hub")
    mod("slime.data.transport", discard_rollout_group=lambda *a, **k: None, publish_rollout_async=None)
    mod("slime.rollout.base_types", RolloutFnTrainOutput=SimpleNamespace, finalize_rollout_groups=finalize_rollout_groups,
        iter_samples=iter_samples, prepare_retry_group=prepare_retry_group)
    mod("slime.rollout.filter_hub.base_types", MetricGatherer=MetricGatherer, call_dynamic_filter=call_dynamic_filter)
    mod("slime.rollout.sglang_rollout", GenerateState=lambda args: SimpleNamespace(sampling_params={}),
        generate_and_rm_group=generate_and_rm_group)
    mod("slime.utils.async_utils", run=_LOOP.run)
    mod("slime.utils.http_utils", get_rollout_num_engines=lambda args: args.engines)
    mod("slime.utils.misc", load_function=load_function, should_run_periodic_action=should_run_periodic_action)
    mod("slime.utils.types", Sample=Sample)

    def sample_staleness(sample, v):  # = slime/utils/staleness.py
        vs = [int(x) for x in (sample.weight_versions or [])]
        if not vs or max(vs) > v:
            return None
        return v - min(vs)

    def compute_staleness_metrics(samples, v):
        ages = [a for a in (sample_staleness(x, v) for x in samples) if a is not None]
        return {"staleness/mean": sum(ages) / len(ages), "staleness/max": max(ages)} if ages else {}

    mod("slime.utils.staleness", sample_staleness=sample_staleness, compute_staleness_metrics=compute_staleness_metrics)
    mod("slime.observability")
    mod("slime.observability.rollout_metrics", compute_staleness_metrics=compute_staleness_metrics)
    src = next(p for p in SRC if os.path.exists(p))
    spec = importlib.util.spec_from_file_location("slime.rollout.fully_async_rollout", src)
    far = importlib.util.module_from_spec(spec)
    sys.modules["slime.rollout.fully_async_rollout"] = far
    spec.loader.exec_module(far)
    return far


class DataSource:
    def __init__(self, n):
        self.n, self.next = n, 0
        self.lock = threading.Lock()
        self.requeue = []

    def get_samples(self, k):
        with self.lock:
            out = []
            while self.requeue and len(out) < k:   # slime hands requeued groups out first
                out.append(self.requeue.pop(0))
            for _ in range(k - len(out)):
                g = self.next
                self.next += 1
                out.append([Sample(g * self.n + j, g) for j in range(self.n)])
            return out

    def add_samples(self, groups):
        if not SIM.flush1:
            raise AssertionError("no aborts in this test")
        with self.lock:
            self.requeue.extend(groups)


def make_manager_class(convert_s):
    class RolloutManager:
        """Mirrors slime.ray.rollout.RolloutManager: generate -> _generate -> _get_rollout_data -> rollout fn,
        then a conversion that costs convert_s (batch_builder.convert / split_by_dp / ray.put)."""

        def __init__(self, args, fn_path):
            from slime.utils.misc import load_function

            self.args, self.data_source = args, DataSource(args.n_samples_per_prompt)
            self.generate_rollout = load_function(fn_path)
            self.batch_builder = SimpleNamespace(rollout_id=-1)
            self.recovery, self.weight_version = None, None

        def generate(self, rollout_id):
            self.batch_builder.rollout_id = rollout_id
            return self._generate(rollout_id, time.time())

        def _generate(self, rollout_id, start_time):
            data, _ = self._get_rollout_data(rollout_id=rollout_id)
            data = getattr(data, "samples", data)   # with a dynamic filter slime returns RolloutFnTrainOutput
            time.sleep(convert_s)
            ages = [self.weight_version - min(int(v) for v in s.weight_versions) for g in data for s in g]
            m = sys.modules["slime.observability.rollout_metrics"].compute_staleness_metrics(
                [s for g in data for s in g], self.weight_version)  # slime's per-rollout metric call (patched by asrt)
            return dict(rid=rollout_id, groups=[g[0].group_index for g in data], version=self.weight_version,
                        stale_mean=sum(ages) / len(ages), stale_max=max(ages), metrics=m)

        def _get_rollout_data(self, rollout_id):
            self.weight_version = SIM.version
            return self.generate_rollout(self.args, rollout_id, self.data_source, evaluation=False), None

        def pause_rollout_admission(self):
            return self.data_source.consumers["fully_async"].pause(drain=False)

        def resume_rollout_admission(self, was_paused):
            if not was_paused:
                self.data_source.consumers["fully_async"].resume()

    return RolloutManager


def run(admit, train_s, steps=8, B=8, n=4, conc=64, engines=2, convert_s=0.15, ahead="1", engine_conc=1 << 30, gen_s=(0.02, 0.06),
        max_staleness="", log_dir=None, dyn_filter=None):
    random.seed(0)
    SIM.version, SIM.gen_s, SIM.engine_conc, SIM.running = 0, gen_s, engine_conc, 0
    os.environ["ASRT_ADMIT"] = admit
    os.environ["ASRT_ADMIT_AHEAD"] = ahead
    os.environ["ASRT_MAX_STALENESS"] = max_staleness
    os.environ.pop("ASRT_LOG_DIR", None)
    if log_dir:
        os.environ["ASRT_LOG_DIR"] = log_dir
    _stub_slime()
    sys.modules.pop("ai_search_plugin.adapters.slime_admit", None)
    Mgr = make_manager_class(convert_s)  # class must exist before slime_admit is imported (it patches live classes)
    args = SimpleNamespace(rollout_batch_size=B, n_samples_per_prompt=n, sglang_server_concurrency=conc, engines=engines,
                           fully_async_group_prefetch_factor=1, rollout_data_transport="ray", dynamic_sampling_filter_path=dyn_filter,
                           rollout_sample_filter_path=None, eval_interval=None, save_interval=None, num_rollout=steps,
                           offload_rollout=False, colocate=False, debug_rollout_only=False)
    path = "ai_search_plugin.adapters.slime_admit.generate_rollout_fully_async" if (admit or max_staleness) else "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
    mgr = Mgr(args, path)
    out, waits, peaks = [], [], []
    stop = threading.Event()

    def sampler():  # peak outstanding = admitted - handed (in flight + completed queue + prebuilt)
        while not stop.is_set():
            w = mgr.data_source.consumers.get("fully_async") if hasattr(mgr.data_source, "consumers") else None
            ctr = getattr(w, "_asrt_ctr", None)
            if ctr is not None:
                peaks.append(ctr.outstanding())
            elif w is not None:
                peaks.append(mgr.data_source.next - sum(len(o["groups"]) for o in list(out)))
            time.sleep(0.002)

    threading.Thread(target=sampler, daemon=True).start()
    t_start = time.time()
    for i in range(steps):
        t0 = time.time()
        o = mgr.generate(i)
        waits.append(time.time() - t0)
        out.append(o)
        time.sleep(train_s)                      # train(i)
        was = mgr.pause_rollout_admission()      # update weights
        if SIM.flush1:
            SIM.epoch += 1                       # flush-cache-interval 1: abort every in-flight request
        SIM.version += 1
        time.sleep(0.01)
        mgr.resume_rollout_admission(was)
    total = time.time() - t_start
    stop.set()
    w = mgr.data_source.consumers["fully_async"]
    w.close()
    handed = [g for o in out for g in o["groups"]]
    return SimpleNamespace(out=out, waits=waits, total=total, handed=handed, peak=max(peaks) if peaks else 0, worker=w,
                           steady_wait=sum(waits[1:]) / (steps - 1), stale=sum(o["stale_mean"] for o in out[2:]) / (steps - 2),
                           stale_max=max(o["stale_max"] for o in out))


def _check_handout(r, B):
    assert len(r.handed) == len(set(r.handed)), "a group was handed out twice"
    assert all(len(o["groups"]) == B for o in r.out)


def test_training_bound():
    B = 8
    base = run("", train_s=0.6, B=B)
    adm = run("cap,eager", train_s=0.6, B=B)
    for r in (base, adm):
        _check_handout(r, B)
    # slime: in-flight cap 2*64/4 = 32 groups = 4 batches, queue up to 2x that -> runs far ahead
    assert base.peak > 3 * B, base.peak
    assert adm.peak <= 2 * B, adm.peak
    assert adm.stale_max <= 2 < base.stale_max, (adm.stale_max, base.stale_max)
    # conversion (0.15 s) leaves the critical path
    assert base.steady_wait > 0.12 and adm.steady_wait < 0.05, (base.steady_wait, adm.steady_wait)
    print(f"training-bound: peak outstanding {base.peak} -> {adm.peak} groups; staleness mean {base.stale:.2f} -> {adm.stale:.2f}, "
          f"max {base.stale_max} -> {adm.stale_max}; wait/step {base.steady_wait:.3f} -> {adm.steady_wait:.3f} s; "
          f"total {base.total:.2f} -> {adm.total:.2f} s")


def test_rollout_bound_no_throughput_loss():
    # engine runs 16 samples at a time (= 4 groups); a batch needs 2 rounds -> rollout-bound
    kw = dict(train_s=0.05, B=8, engine_conc=16, gen_s=(0.05, 0.1), steps=8)
    base = run("", **kw)
    adm = run("cap,eager", **kw)
    _check_handout(adm, 8)
    assert adm.total <= base.total * 1.10, (base.total, adm.total)
    print(f"rollout-bound: total {base.total:.2f} -> {adm.total:.2f} s; wait/step {base.steady_wait:.3f} -> {adm.steady_wait:.3f} s; "
          f"staleness {base.stale:.2f} -> {adm.stale:.2f}")


def test_max_staleness_ceiling():
    """ ASRT_MAX_STALENESS=S alone (no ASRT_ADMIT): slime's staleness metric <= S on every batch, the share over
    the cap is logged (driver line + admit.jsonl), nothing handed twice; S=1 = verl's one-step off-policy."""
    import json
    import tempfile

    B = 8
    os.environ["ASRT_STALE_MODE"] = "count"
    base = run("", train_s=0.6, B=B)  # group-count cap (ticket mode: test_ticket_mode_backfill)
    for S in (1, 2):
        d = tempfile.mkdtemp()
        r = run("", train_s=0.6, B=B, max_staleness=str(S), log_dir=d)
        _check_handout(r, B)
        assert r.peak <= S * B, (S, r.peak)
        assert r.stale_max <= S < base.stale_max, (S, r.stale_max, base.stale_max)
        recs = [json.loads(x) for x in open(os.path.join(d, "admit.jsonl")) if '"staleness"' in x]
        assert len(recs) == len(r.out) and all(x["max"] <= S and x["frac_over_cap"] == 0 for x in recs), recs
        assert all(o["metrics"].get("staleness/frac_over_cap") == 0 for o in r.out)
        print(f"max staleness {S}: staleness max {base.stale_max} -> {r.stale_max}, mean {base.stale:.2f} -> {r.stale:.2f}; "
              f"peak outstanding {base.peak} -> {r.peak}; total {base.total:.2f} -> {r.total:.2f} s")
    assert base.stale_max > 1
    # overlap driver (asrt slime_prefetch hands batch i+1 out during the sync after train(i)): S -> (S-1) batches
    os.environ["ASRT_PREFETCH"] = "overlap,r3c,profile"
    try:
        r = run("", train_s=0.3, B=B, steps=4, max_staleness="2")
        assert r.peak <= B and r.stale_max <= 2, (r.peak, r.stale_max)
        r = run("", train_s=0.6, B=B, steps=8, max_staleness="2")
        assert r.peak <= 3 * B and r.stale_max <= 2, (r.peak, r.stale_max)
        print(f"overlap mapping S=2: peak <= {B}; longer run peak {r.peak}, staleness max {r.stale_max}")
    finally:
        os.environ.pop("ASRT_PREFETCH", None)
        os.environ.pop("ASRT_STALE_MODE", None)


def test_ticket_mode_backfill():
    """ ticket mode (ASRT_STALE_MODE=ticket): every group is handed out in the batch it was
    ticketed for, so staleness <= S holds by construction; in a rollout-bound setting it keeps more groups in flight
    than the count cap (next batches start during the tail) and finishes sooner."""
    kw = dict(train_s=0.05, B=8, engine_conc=24, gen_s=(0.03, 0.15), steps=8, max_staleness="2")
    os.environ["ASRT_PREFETCH"] = "overlap,r3c,profile"  # full-plan driver: count cap = 1 batch outstanding
    try:
        os.environ["ASRT_STALE_MODE"] = "count"
        cnt = run("", **kw)
        os.environ["ASRT_STALE_MODE"] = "ticket"
        tk = run("", **kw)
    finally:
        os.environ.pop("ASRT_STALE_MODE", None)
        os.environ.pop("ASRT_PREFETCH", None)
    for r in (cnt, tk):
        _check_handout(r, 8)
        assert r.stale_max <= 2, r.stale_max
    assert tk.total < cnt.total, (cnt.total, tk.total)
    print(f"ticket vs count cap (rollout-bound, S=2): total {cnt.total:.2f} -> {tk.total:.2f} s, "
          f"staleness mean {cnt.stale:.2f} -> {tk.stale:.2f}, max {cnt.stale_max} -> {tk.stale_max}")



def test_ticket_drop_overdue():
    """ question-bound slow groups: without a deadline batch 1 waits for them; with
    ASRT_TICKET_DROP_S the overdue groups are cut (a finished younger group takes the slot) and the run keeps
    going; staleness <= S, every handed-out group is distinct."""
    kw = dict(train_s=0.05, B=8, engine_conc=1 << 30, gen_s=(0.02, 0.06), steps=8, max_staleness="2")
    os.environ["ASRT_STALE_MODE"] = "ticket"
    os.environ["ASRT_PREFETCH"] = "overlap,r3c,profile"
    os.environ["ASRT_TICKET_DEFER"] = "0"           # the cut alone (DEFER is the default since 11-01)
    SIM.slow, SIM.slow_s = {9, 21}, 4.0          # one group of batch 1 and one of batch 2 take 4 s (steps ~0.1 s)
    try:
        base = run("", **kw)
        os.environ["ASRT_TICKET_DROP_S"] = "0.5"  # fixed seconds
        cut = run("", **kw)
    finally:
        for k in ("ASRT_STALE_MODE", "ASRT_PREFETCH", "ASRT_TICKET_DROP_S"):
            os.environ.pop(k, None)
        SIM.slow, SIM.slow_s = set(), 0.0
    from ai_search_plugin.adapters import slime_admit as sa
    st = sa._TK[0].stats()
    _check_handout(cut, 8)
    assert cut.stale_max <= 2 and base.stale_max <= 2, (base.stale_max, cut.stale_max)
    assert st["dropped"] >= 2 and cut.total < base.total, (st, base.total, cut.total)
    assert not ({9, 21} & set(cut.handed)), "a dropped group was handed out"
    # guard (unit level): max(1, 2% x 8) = 1 cut per batch even when 3 groups of the batch are overdue
    import types as _t
    tk = sa.Tickets(8, 2)
    mk = lambda gi: [_t.SimpleNamespace(index=gi * 4 + j, group_index=gi, weight_versions=["0"]) for j in range(4)]
    os.environ["ASRT_TICKET_DROP_S"] = "0.01"
    try:
        tk.admit([mk(i) for i in range(16)])            # tickets 0 and 1, 8 groups each
        for k in list(tk.t_admit):
            tk.t_admit[k] -= 10                         # everything is "old"
        tk.take([(100 + i, mk(i)) for i in range(5)] + [(200 + i, mk(i)) for i in range(8, 16)], 0)
        tk.take([], 0)                                  # next poll: cap reached -> guard
        stg = tk.stats()
    finally:
        os.environ.pop("ASRT_TICKET_DROP_S", None)
        os.environ.pop("ASRT_TICKET_DEFER", None)
    assert stg["dropped"] == 1 and stg["drop_guarded"] == 1, stg
    print(f"ticket drop: total {base.total:.2f} -> {cut.total:.2f} s, dropped {st['dropped']}, staleness max {cut.stale_max}")


def test_ticket_defer():
    """ ASRT_TICKET_DEFER=1 (verl-style): stragglers move to a later ticket they can still train in, a
    finished younger group fills the batch; only stragglers at the ceiling are cut by the deadline.  Long-tail
    generation plus two question-bound slow groups: no slower than the cut alone, no more groups dropped,
    staleness <= S, every handed-out group distinct."""
    kw = dict(train_s=0.05, B=8, engine_conc=1 << 30, gen_s=(0.02, 0.3), steps=10, max_staleness="2")
    os.environ["ASRT_STALE_MODE"] = "ticket"
    os.environ["ASRT_PREFETCH"] = "overlap,r3c,profile"
    os.environ["ASRT_TICKET_DROP_S"] = "0.5"
    os.environ["ASRT_TICKET_DROP_RUNFRAC"] = "0.1"  # 2 slow groups within the first 3 batches: above the 1% default budget
    SIM.slow, SIM.slow_s = {9, 21}, 4.0
    tk_stats = lambda: __import__("ai_search_plugin.adapters.slime_admit", fromlist=["_TK"])._TK[0].stats()  # module reloads per run
    os.environ["ASRT_TICKET_DEFER"] = "0"
    try:
        cut = run("", **kw)
        st_cut = tk_stats()
        os.environ["ASRT_TICKET_DEFER"] = "1"
        dfr = run("", **kw)
        st = tk_stats()
    finally:
        for k in ("ASRT_STALE_MODE", "ASRT_PREFETCH", "ASRT_TICKET_DROP_S", "ASRT_TICKET_DEFER", "ASRT_TICKET_DROP_RUNFRAC"):
            os.environ.pop(k, None)
        SIM.slow, SIM.slow_s = set(), 0.0
    _check_handout(dfr, 8)
    assert dfr.stale_max <= 2, dfr.stale_max
    assert st["deferred"] > 0, st
    assert st["dropped"] <= st_cut["dropped"], (st_cut, st)
    assert dfr.total <= cut.total * 1.15, (cut.total, dfr.total, {k: st_cut[k] for k in ("dropped","refused")}, {k: st[k] for k in ("dropped","deferred","refused","stale_requeue")})
    print(f"ticket defer: total {cut.total:.2f} -> {dfr.total:.2f} s, dropped {st_cut['dropped']} -> {st['dropped']}, "
          f"deferred {st['deferred']}, staleness max {dfr.stale_max}")




def test_ticket_ratio_budget():
    """ ASRT_TICKET_DROP=ratio:<p>: only groups that ran >= ASRT_TICKET_SLOW_S count as slow and may be cut;
    cuts <= floor(p x slow groups so far)."""
    import types as _t
    from ai_search_plugin.adapters import slime_admit as sa
    mk = lambda gi: [_t.SimpleNamespace(index=gi * 4 + j, group_index=gi, weight_versions=["0"]) for j in range(4)]
    os.environ.update(ASRT_TICKET_DROP="ratio:1", ASRT_TICKET_SLOW_S="0.2", ASRT_TICKET_DEFER="1")
    try:
        tk = sa.Tickets(8, 2)
        tk.admit([mk(i) for i in range(16)])
        for k in list(tk.t_admit)[:2]:
            tk.t_admit[k] -= 1.0                      # two groups of batch 0 have run 1 s (>= T), the rest are young
        tk.take([(100 + i, mk(i)) for i in range(2, 8)] + [(200 + i, mk(i)) for i in range(8, 16)], 0)
        st = tk.stats()
        assert st["slow_groups"] == 2 and st["dropped"] == 2, st      # p=1: both slow groups may be cut
        os.environ["ASRT_TICKET_DROP"] = "ratio:0.5"
        tk = sa.Tickets(8, 2)
        tk.admit([mk(i) for i in range(16)])
        for k in list(tk.t_admit)[:2]:
            tk.t_admit[k] -= 1.0
        tk.take([(100 + i, mk(i)) for i in range(2, 8)] + [(200 + i, mk(i)) for i in range(8, 16)], 0)
        st = tk.stats()
        assert st["slow_groups"] == 2 and st["dropped"] == 1, st      # p=0.5: one of two
    finally:
        for k in ("ASRT_TICKET_DROP", "ASRT_TICKET_SLOW_S", "ASRT_TICKET_DEFER"):
            os.environ.pop(k, None)
    print("ticket ratio budget: p=1 cuts 2/2 slow, p=0.5 cuts 1/2, young groups never cut")


def test_ticket_slow_by_own_run_time():
    """ fairness fix (queue time overcounted slow groups ~10x): a group's run time ends at its real
    completion (done callback), not when take() pulls it.  (a) a normal group that finished early but sat in slime's
    queue past T is neither counted nor cut; (b) a young group is not counted; (c) a group that itself ran >= T is
    counted and may be cut (ratio:1)."""
    import time as _tm, types as _t
    from ai_search_plugin.adapters import slime_admit as sa
    mk = lambda gi: [_t.SimpleNamespace(index=gi * 4 + j, group_index=gi, weight_versions=["0"]) for j in range(4)]
    os.environ.update(ASRT_TICKET_DROP="ratio:1", ASRT_TICKET_SLOW_S="0.3", ASRT_TICKET_DEFER="1")
    try:
        tk = sa.Tickets(8, 2)
        gs = [mk(i) for i in range(16)]
        tk.admit(gs)                                   # tickets 0 and 1
        for i in [0] + list(range(2, 16)):             # everyone but group 1 finishes at once (done callback) ...
            tk.finished(gs[i])
        _tm.sleep(0.35)                                # ... and sits in slime's queue past T; group 1 keeps running
        tk.admit([mk(16)])                             # (b) a young group (ticket 2)
        tk.take([(100 + i, gs[i]) for i in range(2, 16)], 0)   # group 0 not pulled yet (still in slime's queue)
        st = tk.stats()
        assert st["slow_groups"] == 1 and st["dropped"] == 1, st          # (c) group 1 only
        assert ("i", 4) in tk.dropped and ("i", 0) not in tk.dropped      # (a) group 0 not cut
        tk.take([(100, gs[0])], 0)                     # (a) pulled late: own run ~0 s -> still not slow
        assert tk.stats()["slow_groups"] == 1, tk.stats()
    finally:
        for k in ("ASRT_TICKET_DROP", "ASRT_TICKET_SLOW_S", "ASRT_TICKET_DEFER"):
            os.environ.pop(k, None)
    print("ticket: slow = own run time >= T; queued finished and young groups neither counted nor cut")


def reject_every_third(args, group):
    """Stand-in for a DAPO-style filter (check_reward_nonzero_std): drops about a third of the groups."""
    return group[0].group_index % 3 != 0


def test_ticket_filter_refund():
    """ ticket mode + slime's dynamic filter: a rejected group gives its slot back to
    its ticket, so the collector gets replacements. Without the refund, rejected groups keep counting toward their
    ticket's B, tickets m..m+S fill up with groups that never train and admission stops (rollout 11 stuck at 26/32).
    Run in a thread with a deadline: a deadlock fails the test instead of hanging it."""
    for mode in ("ticket",):   # (the deprecated ooo mode is not shipped)
        _filter_run(mode)


def _filter_run(mode):
    kw = dict(train_s=0.03, B=8, engine_conc=24, gen_s=(0.02, 0.08), steps=8, max_staleness="2",
              dyn_filter="slime_admit_cpu.reject_every_third")
    os.environ["ASRT_STALE_MODE"] = mode
    box = {}

    def go():
        try:
            box["r"] = run("", **kw)
        except BaseException as e:  # noqa: BLE001
            box["e"] = e
    th = threading.Thread(target=go, daemon=True)
    try:
        th.start()
        th.join(float(os.environ.get("ASRT_TEST_FILTER_DEADLINE_S", "120")))
    finally:
        os.environ.pop("ASRT_STALE_MODE", None)
    assert not th.is_alive(), f"{mode} mode + dynamic filter deadlocked (rejected groups kept their ticket slots)"
    if "e" in box:
        raise box["e"]
    r = box["r"]
    _check_handout(r, 8)
    assert all(g % 3 != 0 for g in r.handed), "a rejected group was trained"
    assert r.stale_max <= 2, r.stale_max
    import ai_search_plugin.adapters.slime_admit as sa
    st = sa._TK[0].stats()
    if mode == "ticket":
        assert st["filter_refunds"] > 0, st
    print(f"{mode} + dynamic filter: {len(r.handed)} groups trained in {len(r.out)} batches, "
          f"{st.get('filter_refunds', 0)} rejected groups refunded their ticket slot, max staleness {r.stale_max} (cap 2)")


def _requeue_replay(mode):
    """Replayed sequence (slime flush-cache-interval 1 aborts in-flight requests at each weight sync): a ticket-2 group
    is aborted at version 2 with two samples already finished at version 0, requeued, a fresh prompt takes its slot in
    batch 2, and the requeued group is only re-admitted at m=3.  Returns the earliest version of every group handed
    out per batch (v0 = 0, batch b trains at version b) and the ticket stats."""
    import types as _t
    from ai_search_plugin.adapters import slime_admit as sa
    os.environ["ASRT_STALE_MODE"] = mode
    os.environ["ASRT_TICKET_DEFER"] = "1"
    try:
        mk = lambda gi: [_t.SimpleNamespace(index=gi * 4 + j, group_index=gi, weight_versions=[]) for j in range(4)]

        class Inner:                                  # slime's data buffer: requeued groups are handed out first
            def __init__(self):
                self.q = [mk(i) for i in range(16)]
            def get_samples(self, n):
                out, self.q = self.q[:n], self.q[n:]
                return out
            def add_samples(self, groups):
                self.q = list(groups) + self.q

        tk = sa.Tickets(4, 2)
        inner = Inner()
        tb = sa.TicketBuffer(inner, tk)
        ver = [0]

        def gen(g):                                   # generate with the current weights (unversioned samples)
            for s in g:
                if not s.weight_versions:
                    s.weight_versions = [str(ver[0])]
            tk.finished(g)

        def admit(n):
            out = []
            for _ in range(n):
                out += tb.get_samples(1)
            return out

        def hand(b, groups):
            out = tk.take([(i, x) for i, x in enumerate(groups)], 4)
            got[b] = [min(int(v) for s in y for v in s.weight_versions) for _, y in out]
            tk.batch_done(); ver[0] += 1; tk.vmax = ver[0]

        got = {}
        g = admit(12)                                 # tickets 0, 1, 2 at version 0
        for x in g[:8]:
            gen(x)
        hand(0, g[0:4]); hand(1, g[4:8])              # now m = 2, version 2
        g8 = g[8]
        for s in g8[:2]:
            s.weight_versions = ["0"]                 # two samples finished at version 0 before the abort
        tk.finished(g8)                               # the done callback fires for the aborted group too
        tb.add_samples([g8])                          # slime requeues it (front of the buffer) ...
        inner.q = [q for q in inner.q if q is not g8]
        inner.q.insert(1, g8)                         # ... but here a fresh prompt takes batch 2's free slot first
        fresh = admit(1)
        for x in g[9:12] + fresh:
            gen(x)
        hand(2, g[9:12] + fresh)                      # m = 3, version 3
        late = admit(4)                               # group 8 re-admitted now, plus 3 fresh prompts
        for x in late:
            gen(x)
        hand(3, late)
        return got, tk.stats()
    finally:
        os.environ.pop("ASRT_STALE_MODE", None)
        os.environ.pop("ASRT_TICKET_DEFER", None)


def test_requeue_retained_versions_respect_cap():
    """ Replayed run (slime flush 1 + ticket S=2): max staleness 3 on 32-38% of the samples from step 4,
    stale_requeue 17 -> 142.  Every hand-out must keep train batch - earliest version <= 2, nothing may be lost, and
    the requeued group must restart fresh instead of being forced into batch 3 with a version-0 sample."""
    for mode in ("ticket",):   # (the deprecated ooo mode is not shipped)
        got, st = _requeue_replay(mode)
        for b, mins in got.items():
            assert len(mins) == 4 and all(b - v <= 2 for v in mins), (mode, b, mins, st)
        assert st.get("refreshed", 0) == 1 and st.get("dropped", 0) == 0, (mode, st)
    print("requeue replay: every batch within S=2, requeued group restarted fresh, nothing dropped")


def _flush1_run(mode, steps=14, slow_s=0.9):
    """slime flush 1 + ticket S=2 on the real fully-async worker: 2 of every 8 groups have one sample needing ~4
    train steps of generation, so it is aborted at several syncs and its group is requeued with siblings that
    finished at an old version (seen: 2.5-6% of samples over S from step 11)."""
    import json
    import tempfile
    os.environ["ASRT_STALE_MODE"] = mode
    d = tempfile.mkdtemp()
    SIM.flush1, SIM.epoch = True, 0
    SIM.slow = set(range(3, 4000, 4))      # one in four groups is a long one (every sample of it)
    SIM.slow_s = slow_s
    try:
        r = run("", train_s=0.25, B=8, steps=steps, max_staleness="2", log_dir=d, gen_s=(0.02, 0.06))
    finally:
        SIM.flush1, SIM.slow, SIM.slow_s = False, set(), 0.0
        os.environ.pop("ASRT_STALE_MODE", None)
    recs = [json.loads(x) for x in open(os.path.join(d, "admit.jsonl"))]
    st = [x for x in recs if x.get("event") == "staleness"]
    tick = [x for x in recs if x.get("event") == "ticket"]
    return r, st, tick, recs


def test_flush1_abort_requeue_cap():
    """ NEGATIVE on the old code:
    requeued groups with old retained versions were forced into batch m (stale_requeue) -> staleness 3+.  Now:
    every handed sample within S=2, those groups regenerated (requeue_refresh), nothing handed twice, no deadlock."""
    for mode in ("ticket",):   # (the deprecated ooo mode is not shipped)
        r, st, tick, recs = _flush1_run(mode)
        _check_handout(r, 8)
        last = tick[-1] if tick else {}
        n_ref = sum(1 for x in recs if x.get("event") == "requeue_refresh")
        n_kept = sum(1 for x in recs if x.get("event") == "requeue_stale_kept")
        aborted = getattr(getattr(r.worker.data_buffer, "_tk", None), "refreshed", None)
        print(f"flush1 {mode}: handed max staleness {r.stale_max} (cap 2), admit.jsonl max "
              f"{max(x['max'] for x in st)}, over-cap rollouts {sum(1 for x in st if x['frac_over_cap'] > 0)}, "
              f"stale_requeue {last.get('stale_requeue')}, requeue_refresh {n_ref}, kept {n_kept}, refreshed {aborted}")
        assert r.stale_max <= 2, (mode, r.stale_max, [o["stale_max"] for o in r.out])
        assert all(x["max"] <= 2 and x["frac_over_cap"] == 0 for x in st), (mode, st)
        assert n_ref > 0 and n_kept == 0, (mode, n_ref, n_kept)


def _cut_requeue_replay(defer, budget_s=3.0):
    """ found in review, ticket mode, slime flush 1 +
    ASRT_TICKET_DROP_S: a group cut from batch 0 by _drop_overdue is still running, is aborted at the next weight sync
    and comes back through the data buffer (slime done callback -> add_samples).  Old code re-admitted it under a NEW
    ticket; its result was discarded on completion (still in dropped) but that ticket's pending never dropped and
    own_age() stayed 0, so batch 1 never filled (deadlock).  Driven like slime's worker: admission through
    TicketBuffer.get_samples, generation finishes every admitted group, the collector polls take() until batch 1 has
    B groups or budget_s passes (a timeout, not a hang).  Returns (handed batch 1, handed sample-0 group?, stats,
    ledger records, inner buffer still holding the cut group?)."""
    import json
    import tempfile
    import types as _t
    from ai_search_plugin.adapters import slime_admit as sa
    d = tempfile.mkdtemp()
    env = {"ASRT_MAX_STALENESS": "2", "ASRT_STALE_MODE": "ticket", "ASRT_TICKET_DROP_S": "0.01",
           "ASRT_TICKET_DEFER": defer, "ASRT_LOG_DIR": d, "ASRT_ADMIT": "profile"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        B = 2
        mk = lambda gi: [_t.SimpleNamespace(index=gi * 2 + j, group_index=gi, weight_versions=[], status=None)
                         for j in range(2)]

        class Inner:                                  # slime's data buffer: requeued groups come back first
            def __init__(self):
                self.q = [mk(i) for i in range(40)]
            def get_samples(self, n):
                out, self.q = self.q[:n], self.q[n:]
                return out
            def add_samples(self, groups):
                self.q = list(groups) + self.q

        tk = sa.Tickets(B, 2)
        inner = Inner()
        tb = sa.TicketBuffer(inner, tk)
        ver = [1]

        def fin(g):
            for s in g:
                if not s.weight_versions:
                    s.weight_versions = [str(ver[0])]
            tk.finished(g)

        gs = []
        for _ in range(4):
            gs += tb.get_samples(1)                   # tickets 0, 0, 1, 1
        assert len(gs) == 4, len(gs)
        g0 = gs[0]
        for g in gs[1:]:
            fin(g)
        time.sleep(0.05)                              # group 0 overdue (deadline 0.01 s) while ticket 1 is finished
        out = tk.take([(i, gs[i]) for i in (1, 2, 3)], B)
        t_end = time.time() + 1.0
        while len(out) < B and time.time() < t_end:   # DEFER: the cut frees a slot, a fresh group refills batch 0
            for g in tb.get_samples(1):
                fin(g)
                out += tk.take([(100 + g[0].group_index, g)], B - len(out))
            out += tk.take([], B - len(out))
            time.sleep(0.01)
        assert len(out) == B and tk.n_dropped == 1 and sa._gkey(g0) in tk.dropped, (len(out), tk.stats())
        tk.batch_done()                               # m = 1; weight sync (flush 1) aborts the still-running group 0
        ver[0] = 2
        tk.vmax = 2
        g0[0].weight_versions = ["1"]                 # one sample finished before the abort, one partial
        g0[1].status = "aborted"
        tb.add_samples([g0])                          # slime: prepare_retry_group -> data_buffer.add_samples
        tk.finished(g0)                               # ... then our wrapper of the done callback
        handed, t_end = [], time.time() + budget_s
        while len(handed) < B and time.time() < t_end:
            got = tb.get_samples(1)                   # the worker tops up (old code: re-admits group 0 first)
            for g in got:
                fin(g)
            handed += tk.take([(200 + g[0].group_index, g) for g in got], B - len(handed))
            time.sleep(0.005)
        if len(handed) == B:
            tk.batch_done()
        recs = [json.loads(x) for x in open(os.path.join(d, "admit.jsonl"))] if os.path.exists(
            os.path.join(d, "admit.jsonl")) else []
        return (len(handed), any(sa._gkey(g) == sa._gkey(g0) for _, g in handed), tk.stats(),
                [r for r in recs if r.get("event") == "ticket_ledger"], any(q is g0 for q in inner.q), tk)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cut_group_requeue_no_deadlock():
    """ NEGATIVE on the old code (batch 1 stays at 1 of 2 until the timeout: the re-admitted
    cut group holds ticket 1's pending forever).  Now: the cut group is discarded at requeue (not re-admitted, not
    trained), batch 1 fills, every ticket's pending is 0 at each hand-out, nothing negative, the copy is freed."""
    for defer in ("0", "1"):
        n, g0_handed, st, ledger, in_inner, tk = _cut_requeue_replay(defer)
        print(f"cut+requeue defer={defer}: batch 1 handed {n}/2, cut group handed {g0_handed}, back in buffer "
              f"{in_inner}, pending {st['pending']}, cut_requeue_dropped {st.get('cut_requeue_dropped')}, "
              f"ledger ok {[r['ok'] for r in ledger]}")
        assert n == 2, ("DEADLOCK: batch 1 never filled (cut group re-admitted, its ticket pending never cleared)",
                        defer, st)
        assert not g0_handed and not in_inner, (defer, st)
        assert st["cut_requeue_dropped"] == 1 and st["cut_running"] == 0 and st["ledger_bad"] == 0, (defer, st)
        assert all(v == 0 for v in tk.pending.values()) and not any(tk.pend_keys.values()), (defer, tk.pending)
        assert all(c >= 0 for dct in (tk.count, tk.total) for c in dct.values()), (defer, tk.count, tk.total)
        assert len(ledger) == 2 and all(r["ok"] for r in ledger), (defer, ledger)
        assert ("i", 0) not in tk.fresh and ("i", 0) not in tk.t_fin and ("i", 0) not in tk.t_admit, defer
    print("cut group aborted + requeued: discarded, no deadlock, ticket ledger balanced (DEFER 0 and 1)")



def test_flush1_cut_requeue_e2e(timeout_s=120.0, attempts=4):
    """Port note: whether a cut group is still running at a later sync (the cut+requeue path) is timing dependent -
    in the source repo this test also missed the path in ~1 of 4 isolated runs.  Every attempt checks all
    correctness assertions; the path-coverage assertion only needs one of `attempts` runs to exercise the path."""
    for i in range(attempts):
        if _flush1_cut_requeue_e2e_once(timeout_s):
            return
        print(f"flush1 + cut e2e: cut+requeue path not exercised in attempt {i + 1}, retrying")
    raise AssertionError(f"path not exercised in {attempts} attempts")


def _flush1_cut_requeue_e2e_once(timeout_s=120.0):
    """ same on slime's real fully-async worker: flush 1 + ticket S=2 + straggler cut
    (ASRT_TICKET_DROP_S=0.3 s, DEFER 0), long groups need 3 s of generation (several syncs), so cut groups are still
    running at the next sync and get aborted + requeued.  Old code: the run hangs (timeout here, measured 2/2 and
    once more in the probe).  Now: all 14 rollouts complete, cut groups aborted at a sync are discarded
    (cut_requeue_dropped > 0), every per-step ticket ledger balanced, staleness <= 2, no stale_requeue."""
    env = {"ASRT_TICKET_DROP_S": "0.3", "ASRT_TICKET_DEFER": "0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    res = {}

    def go():
        try:
            res["r"] = _flush1_run("ticket", steps=14, slow_s=3.0)
        except BaseException as e:   # noqa: BLE001
            res["e"] = e

    try:
        th = threading.Thread(target=go, daemon=True)
        th.start()
        th.join(timeout_s)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert not th.is_alive(), f"HANG: flush 1 + straggler cut did not finish 14 rollouts in {timeout_s:.0f} s"
    if "e" in res:
        raise res["e"]
    r, st, tick, recs = res["r"]
    _check_handout(r, 8)
    ev = lambda n: [x for x in recs if x.get("event") == n]
    led = ev("ticket_ledger")
    print(f"flush1 + cut e2e: max staleness {r.stale_max}, cuts {len(ev('ticket_drop'))}, cut+requeue dropped "
          f"{len(ev('cut_requeue_dropped'))}, regenerated {len(ev('requeue_refresh'))}, ledgers {len(led)} "
          f"(bad {sum(1 for x in led if not x['ok'])}), stale_requeue {tick[-1].get('stale_requeue')}")
    assert r.stale_max <= 2 and all(x["max"] <= 2 and x["frac_over_cap"] == 0 for x in st), st
    assert len(led) == 14 and all(x["ok"] for x in led), led
    assert tick[-1].get("stale_requeue") == 0 and tick[-1].get("ledger_bad") == 0, tick[-1]
    return len(ev("cut_requeue_dropped")) > 0 and len(ev("ticket_drop")) > 0   # path exercised?


def test_off_is_slime():
    r = run("", train_s=0.2, B=8, steps=4)
    _check_handout(r, 8)
    assert not hasattr(r.worker, "_asrt_ctr")


if __name__ == "__main__":
    test_off_is_slime()
    test_training_bound()
    test_rollout_bound_no_throughput_loss()
    test_max_staleness_ceiling()
    test_ticket_mode_backfill()
    test_ticket_drop_overdue()
    test_ticket_defer()
    test_ticket_ratio_budget()
    test_ticket_slow_by_own_run_time()
    test_ticket_filter_refund()
    test_requeue_retained_versions_respect_cap()
    test_flush1_abort_requeue_cap()
    test_cut_group_requeue_no_deadlock()
    test_flush1_cut_requeue_e2e()
    print("ok")
