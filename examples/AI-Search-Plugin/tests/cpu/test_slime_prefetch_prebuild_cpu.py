"""CPU tests for slime_prefetch's manager-side prebuild (straw): the next batch is built during train(i) from the
same FIFO groups slime would hand out at update(i); admission accounting unchanged; BatchBuilder completes the
right batch although batch i+1 was begun first.  No slime / ray needed (fakes)."""
import collections
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.modules.setdefault("ray", types.SimpleNamespace(get=lambda x: x))  # BatchBuilder patch does ray.get(remote())

from ai_search_plugin.adapters import slime_prefetch as sp  # noqa: E402


class FakeD:  # the parts of slime DistributedRollout the prebuild touches
    def __init__(self, capacity=8):
        self.condition = threading.Condition()
        self.ready = collections.deque()
        self.demand = self.prefetch = 0
        self.capacity = capacity
        self.paused, self.error = False, None

    def limit(self):  # slime's collector pulls while pending + ready < demand + prefetch
        return self.demand + self.prefetch

    def pause(self, drain=True):
        was, self.paused = self.paused, True
        return was

    def generate(self, rid, B):  # slime: demand=B, prefetch=capacity, pop FIFO
        with self.condition:
            assert not self.demand
            self.demand, self.prefetch = B, self.capacity
            out = []
            while self.demand:
                out.append(self.ready.popleft())
                self.demand -= 1
            return out


sp.install_reservation(FakeD)


def make_mgr(num_rollout=6, B=2, **args):
    a = dict(num_rollout=num_rollout, rollout_batch_size=B, eval_interval=None, save_interval=None)
    a.update(args)
    m = types.SimpleNamespace(args=types.SimpleNamespace(**a), recovery=None,
                              batch_builder=types.SimpleNamespace(rollout_id=-1), _asrt_pb=None)
    m.data_source = types.SimpleNamespace(consumers={"fully_async": FakeD()})
    return m


def test_reservation_property_and_pause():
    d = FakeD(capacity=10)
    d.prefetch = 10
    assert d.prefetch == 10
    d.__dict__["_asrt_reserved"] = 4
    assert d.prefetch == 6 and d.limit() == 6
    d.pause()
    assert d.prefetch == 10  # cleared at the weight-update pause
    d.__dict__["_asrt_reserved"] = 99
    assert d.prefetch == 0


def test_reservation_keeps_pull_decision_identical():
    # baseline: ready R, idle (demand 0) -> pull iff R < P.  prebuild popped B of them early: ready R-B, limit P-B.
    for R in range(0, 12):
        base = FakeD(capacity=8); base.prefetch = 8; base.ready.extend(range(R))
        pre = FakeD(capacity=8); pre.prefetch = 8; pre.ready.extend(range(R))
        B = 2
        if R < B:
            continue
        pre.__dict__["_asrt_reserved"] = B
        for _ in range(B):
            pre.ready.popleft()
        assert (len(base.ready) < base.limit()) == (len(pre.ready) < pre.limit()), R


def run_inner(calls, posted):
    def inner(mgr, rid, t0):
        w = sp._consumer(mgr)
        calls.append((rid, threading.current_thread().name, w.__dict__.get("_asrt_reserved", 0)))
        return ("batch", rid, w.generate(rid, mgr.args.rollout_batch_size))
    return inner


def test_prebuild_hands_out_same_fifo_groups():
    calls, posted, logs = [], [], []
    m = make_mgr()
    w = sp._consumer(m)
    gen = sp.make_prebuild_generate(run_inner(calls, posted), FakeD, post=lambda r, o: posted.append((r, o)),
                                    log=logs.append)
    w.ready.extend(["g0", "g1"])
    assert gen(m, 0, 0.0) == ("batch", 0, ["g0", "g1"])        # nothing prebuilt for 0; schedules 1
    assert m._asrt_pb is not None and m._asrt_pb.rid == 1
    time.sleep(0.1)
    assert not m._asrt_pb.started                                # waits, never consumes partial batches
    w.ready.extend(["g2"])
    time.sleep(0.1)
    assert not m._asrt_pb.started
    w.ready.extend(["g3", "g4"])
    assert m._asrt_pb.done.wait(5)
    assert posted == [(1, ("batch", 1, ["g2", "g3"]))]          # posted to the board during train(0)
    assert calls[-1] == (1, "asrt-prebuild", 2)                  # built in the thread, B groups reserved
    assert w.__dict__["_asrt_reserved"] == 2
    out = gen(m, 1, 0.0)                                         # hand-out at update(0)
    assert out == ("batch", 1, ["g2", "g3"]) and list(w.ready) == ["g4"]
    assert w.__dict__["_asrt_reserved"] == 0
    assert logs[-1]["prebuilt"] is True and logs[0]["prebuilt"] is False


def test_handout_cancels_unstarted_prebuild():
    calls, logs = [], []
    m = make_mgr()
    w = sp._consumer(m)
    gen = sp.make_prebuild_generate(run_inner(calls, []), FakeD, post=None, log=logs.append)
    w.ready.extend(["a", "b"])
    gen(m, 0, 0.0)
    pb = m._asrt_pb
    w.ready.extend(["c"])                         # only 1 of 2 complete during train(0)
    time.sleep(0.1)
    assert not pb.started
    w.paused = True                               # update(0) began: the prebuild must not start any more
    w.ready.extend(["d"])
    time.sleep(0.1)
    assert not pb.started
    w.paused = False
    out = gen(m, 1, 0.0)                          # hand-out cancels it -> slime's normal path
    assert out == ("batch", 1, ["c", "d"]) and pb.cancel and pb.done.wait(5)
    assert logs[1]["prebuilt"] is False and calls[-1][1] != "asrt-prebuild"


def test_prebuild_blocked_reasons():
    m = make_mgr(num_rollout=3)
    assert sp.prebuild_blocked(m, 3, FakeD) == "last"
    assert sp.prebuild_blocked(m, 1, FakeD) is None
    assert sp.prebuild_blocked(make_mgr(eval_interval=5), 1, FakeD) == "eval"
    assert sp.prebuild_blocked(make_mgr(dynamic_sampling_filter_path="x.f"), 1, FakeD) == "filters"
    m2 = make_mgr(); m2.recovery = types.SimpleNamespace(batches={1: object()})
    assert sp.prebuild_blocked(m2, 1, FakeD) == "replay"
    m3 = make_mgr(); m3.data_source.consumers["fully_async"] = object()
    assert sp.prebuild_blocked(m3, 1, FakeD) == "worker"


class FakeCtl:
    def __init__(self):
        self.finished = []
        outer = self

        class _F:
            @staticmethod
            def remote(bid):
                outer.finished.append(bid)
        self.finish_batch = _F()


class FakeBB:
    def __init__(self):
        self.rollout_id, self.batch_id, self.controller = -1, None, FakeCtl()

    def begin(self, samples, *, replay=False):
        self.batch_id = f"batch:{samples}"

    def replay_converted(self, data, batch_id):
        self.batch_id = batch_id

    def training_completed(self, rollout_id):  # slime: one current batch only
        if self.batch_id is not None:
            if rollout_id != self.rollout_id:
                raise ValueError("Training completion does not match the current batch")
            self.controller.finish_batch.remote(self.batch_id)


def test_batch_builder_completes_the_right_batch():
    sp.install_batch_ids(FakeBB)
    bb = FakeBB()
    bb.rollout_id = 0; bb.begin("s0")
    bb.rollout_id = 1; bb.begin("s1")               # prebuild of 1 begun before training_completed(0)
    bb.training_completed(0)
    bb.training_completed(1)
    assert bb.controller.finished == ["batch:s0", "batch:s1"]
    bb.rollout_id = 2; bb.replay_converted(None, "batch:r2")
    bb.training_completed(2)
    assert bb.controller.finished[-1] == "batch:r2"
    bb2 = FakeBB(); bb2.rollout_id = 5; bb2.batch_id = "batch:x"   # nothing recorded -> slime's own check
    bb2.training_completed(5)
    assert bb2.controller.finished == ["batch:x"]


def test_prebuild_enabled_env(monkeypatch):
    a = types.SimpleNamespace(rollout_data_transport="straw")
    b = types.SimpleNamespace(rollout_data_transport="object-store")
    monkeypatch.delenv("ASRT_PREFETCH_PREBUILD", raising=False)
    assert sp.prebuild_enabled(a) and not sp.prebuild_enabled(b)
    monkeypatch.setenv("ASRT_PREFETCH_PREBUILD", "0")
    assert not sp.prebuild_enabled(a)
    monkeypatch.setenv("ASRT_PREFETCH_PREBUILD", "1")
    assert sp.prebuild_enabled(b)
