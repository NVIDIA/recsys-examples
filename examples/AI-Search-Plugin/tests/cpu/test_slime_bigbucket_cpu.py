"""slime_weightsync ``bigbucket`` (verl-style whole-bucket broadcast) on CPU.

* choose_bigbucket: AUTO is on only where parsync's per-engine fan-out is the problem (E >= fanout x PP).
* BigPacker: every tensor arrives byte-identical through SGLang's flattened_bucket layout (receiver views
  dtype at running offsets -> alignment is checked by torch), buckets <= cap, oversize tensors alone, at
  most nbuf buffers, deterministic layout, threaded double buffering.
* The patched slime class end to end against a fake slime/ray/NCCL: engine weights after the bigbucket
  sync == after slime's original per-tensor sync (bit for bit), one broadcast per bucket (not per engine),
  parsync per-engine groups not created under AUTO at 8 engines, kept at 2 engines; a sender error does
  not hang the producer.
"""
import queue
import sys
import threading
import time
import types
from contextlib import nullcontext

import pytest
import torch

from ai_search_plugin.adapters import slime_weightsync as ws
from ai_search_plugin.adapters.slime_weightsync import BigPacker, choose_bigbucket


@pytest.fixture(autouse=True)
def _real_torch(monkeypatch):
    """Some other test files leave a torch stub in sys.modules; the code under test imports torch lazily."""
    import torch.distributed as real_dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", real_dist)


# ---------------------------------------------------------------- decision
@pytest.mark.parametrize("E,pp,parsync,flatten,async_,mode,want", [
    (8, 2, True, True, False, "auto", True),     # mid-size: PP2 x 8 TP2 engines
    (4, 4, True, True, False, "auto", False),    # 4 PP stages, 4 engines -> parsync stays
    (2, 2, True, True, False, "auto", False),
    (8, 1, True, True, False, "auto", True),     # single source: one broadcast beats 8 sends
    (1, 1, True, True, False, "auto", False),
    (8, 2, False, True, False, "auto", False),   # nothing asked for -> nothing changes
    (8, 2, True, False, False, "auto", False),   # needs flatten
    (8, 2, True, True, True, "auto", False),     # not with async
    (2, 4, False, True, False, "1", True),       # forced
    (8, 2, True, True, False, "0", False),       # disabled
])
def test_choose(E, pp, parsync, flatten, async_, mode, want):
    on, why = choose_bigbucket(E, pp, parsync=parsync, flatten=flatten, async_=async_, mode=mode, fanout=2)
    assert on is want, why


def test_choose_env(monkeypatch):
    monkeypatch.setenv("ASRT_WS_FANOUT", "3")
    assert not choose_bigbucket(4, 2, parsync=True, flatten=True)[0]
    assert choose_bigbucket(6, 2, parsync=True, flatten=True)[0]
    monkeypatch.setenv("ASRT_WS_BIGBUCKET", "off")
    assert not choose_bigbucket(64, 1, parsync=True, flatten=True)[0]


# ---------------------------------------------------------------- packer
def receive(flat, names, dtypes, shapes):
    """SGLang flattened_bucket receive: back-to-back tensors, view(dtype) at running offsets."""
    out, off = {}, 0
    flat = flat.clone()  # the broadcast copies the bytes out before the buffer is released
    for n, dt, shp in zip(names, dtypes, shapes):
        nb = int(torch.Size(shp).numel()) * torch.empty(0, dtype=dt).element_size()
        out[n] = flat[off:off + nb].view(dt).reshape(shp)  # torch raises on a misaligned offset
        off += nb
    assert off == flat.numel()
    return out


def make_named(seed=0, n=60, big=None):
    g = torch.Generator().manual_seed(seed)
    named = []
    for i in range(n):
        k = i % 7
        shape = (int(torch.randint(1, 40, (1,), generator=g)), int(torch.randint(1, 33, (1,), generator=g)))
        if k == 0:
            t = torch.randn(shape, generator=g).float()
        elif k == 1:
            t = torch.randint(0, 255, (int(torch.randint(1, 30, (1,), generator=g)),), generator=g).to(torch.uint8)  # odd sizes
        elif k == 2:
            t = torch.randint(-9, 9, shape, generator=g).long()
        else:
            t = torch.randn(shape, generator=g).bfloat16()
        if i % 11 == 5:
            t = t.t()  # non-contiguous
        named.append((f"w{i}", t))
    if big:
        named.insert(17, ("huge", torch.randn(big, generator=g).bfloat16()))
    return named


def bytes_of(t):
    return t.contiguous().view(-1).view(torch.uint8)


def run_packer(named_chunks, cap, nbuf=2, hold=1):
    p = BigPacker(cap, nbuf=nbuf, device="cpu")
    got, buckets, held = {}, [], []

    def emit(flat, names, dtypes, shapes, release):
        assert flat.numel() <= cap or len(names) == 1
        got_b = receive(flat, names, dtypes, shapes)
        for k, v in got_b.items():
            assert k not in got
            got[k] = v
        buckets.append((tuple(names), flat.numel()))
        held.append(release)
        while len(held) > hold:  # keep `hold` buckets "in flight"
            held.pop(0)()

    for chunk in named_chunks:
        p.add(list(chunk), emit)
    p.finish(emit)
    for r in held:
        r()
    assert len(p.bufs) <= nbuf
    return got, buckets, p


@pytest.mark.parametrize("cap", [97, 256, 1000, 4096, 1 << 20])
def test_packer_bit_exact(cap):
    named = make_named(1, big=3000)
    chunks = [named[i:i + 9] for i in range(0, len(named), 9)]
    got, buckets, _ = run_packer(chunks, cap)
    assert set(got) == {n for n, _ in named}
    for n, t in named:
        assert got[n].dtype == t.dtype and tuple(got[n].shape) == tuple(t.shape)
        assert torch.equal(bytes_of(got[n]), bytes_of(t)), n
    if cap >= 1 << 20:
        assert len(buckets) <= 2  # whole buckets, not one per chunk (+ at most one of deferred tensors)


def test_packer_deterministic_and_oversize_alone():
    named = make_named(2, big=5000)
    a = run_packer([named[:30], named[30:]], 2048)[1]
    b = run_packer([named[:30], named[30:]], 2048)[1]
    assert a == b
    assert any(names == ("huge",) for names, _ in a)


def test_packer_threaded_double_buffer():
    """Producer fills while a consumer thread 'broadcasts' (slowly); never more than nbuf buffers."""
    named = make_named(3, n=300)
    p = BigPacker(3000, nbuf=2, device="cpu")
    q, got, maxbuf = queue.Queue(maxsize=2), {}, [0]

    def consumer():
        while True:
            it = q.get()
            if it is None:
                return
            flat, names, dtypes, shapes, release = it
            time.sleep(0.002)
            got.update(receive(flat, names, dtypes, shapes))
            release()

    th = threading.Thread(target=consumer)
    th.start()

    def emit(flat, names, dtypes, shapes, release):
        maxbuf[0] = max(maxbuf[0], len(p.bufs))
        q.put((flat, names, dtypes, shapes, release))

    for i in range(0, 300, 13):
        p.add(named[i:i + 13], emit)
    p.finish(emit)
    q.put(None)
    th.join()
    assert maxbuf[0] <= 2 and p.wait_s > 0  # the producer did wait for a free buffer
    for n, t in named:
        assert torch.equal(bytes_of(got[n]), bytes_of(t)), n


# ---------------------------------------------------------------- patched slime class, fake runtime
class _Ref:
    def __init__(self, fn):
        self.fn = fn


def _rget(x):
    if isinstance(x, list):
        return [_rget(y) for y in x]
    return x.fn() if isinstance(x, _Ref) else x


class _Remote:
    def __init__(self, fn, lazy=False):
        self.fn, self.lazy = fn, lazy

    def remote(self, *a, **k):
        return _Ref(lambda: self.fn(*a, **k)) if self.lazy else self.fn(*a, **k)


class FakeEngine:
    def __init__(self, i, stats):
        self.i, self.stats = i, stats
        # Ray actor handles carry _actor_id (the parsync path keys its per-engine groups by it).  Added in the port:
        # the original fixture lacked it, so these tests failed in the source repo too (stale fixture, not a code bug).
        self._actor_id = bytes([i]) * 16
        self.inbox = {}  # group -> list of received tensors
        self.state = {}
        self.update_weights_from_distributed = _Remote(self._update, lazy=True)
        self.pause_generation = _Remote(lambda **k: None)
        self.continue_generation = _Remote(lambda **k: None)
        self.flush_cache = _Remote(lambda **k: None)
        self.destroy_weights_update_group = _Remote(lambda g: None)

    def _update(self, names, dtypes, shapes, group_name, weight_version=None, load_format=None):
        box = self.inbox[group_name]
        if load_format == "flattened_bucket":
            flat = box.pop(0)
            self.state.update({k: bytes_of(v).clone() for k, v in receive(flat, names, dtypes, shapes).items()})
        else:
            for n in names:
                self.state[n] = bytes_of(box.pop(0)).clone()
        return {"success": True, "message": "ok"}


class FakeGroup:
    def __init__(self, name, engines):
        self.name, self.engines = name, engines


def build_fake(monkeypatch, n_engines, pp, chunks_fn, stats):
    uwd = types.ModuleType("fake_uwd")

    def connect(args, name, engines, engine_gpu_counts=None):
        stats["groups"].append(name)
        for e in engines:
            e.inbox[name] = []
        return FakeGroup(name, list(engines))

    def disconnect(name, grp, engines):
        pass

    def update_weights_from_distributed(group_name, group, weight_version, engines, named, load_format=None):
        refs = [e.update_weights_from_distributed.remote(
            names=[n for n, _ in named], dtypes=[t.dtype for _, t in named], shapes=[t.shape for _, t in named],
            group_name=group_name, weight_version=str(weight_version), load_format=load_format) for e in engines]
        for _, t in named:
            ws_dist.broadcast(t.data, 0, group=group, async_op=True)
        return refs

    class UWD:
        def __init__(self, args):
            self.args = args
            self.weight_version = 0
            self._model_update_groups = None
            self.update_weight_metrics = {}
            self.quantization_config = None

        def connect_rollout_engines(self, rollout_engines, rollout_engine_lock, engine_gpu_counts=None):
            self.rollout_engines, self.rollout_engine_lock = rollout_engines, rollout_engine_lock
            self._is_pp_src_rank = True
            self._group_name = "slime-pp_0"
            self._model_update_groups = connect(self.args, self._group_name, rollout_engines)

        def disconnect_rollout_engines(self):
            pass

        def update_weights(self):
            self.weight_version += 1
            self._send_weights(None)

        def _send_weights(self, pbar):
            for it in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
                for ch in it:
                    self._on_chunk(ch)
                    self._update_bucket_weights_from_distributed(ch, pbar)

        def _on_chunk(self, ch):
            pass

        def _iter_non_expert_chunks(self):
            return iter(chunks_fn()[0])

        def _iter_expert_chunks(self):
            return iter(chunks_fn()[1])

        def _update_bucket_weights_from_distributed(self, named, pbar=None, load_format=None):
            ray_mod.get(update_weights_from_distributed(self._group_name, self._model_update_groups,
                                                        self.weight_version, self.rollout_engines, named))
            named.clear()

    uwd.UpdateWeightFromDistributed = UWD
    uwd.connect_rollout_engines_from_distributed = connect
    uwd.disconnect_rollout_engines_from_distributed = disconnect
    uwd.update_weights_from_distributed = update_weights_from_distributed
    uwd.post_process_weights = lambda *a, **k: None

    # fake ray / slime utils / megatron mpu
    ray_mod = types.ModuleType("ray")
    ray_mod.get = _rget
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    for name in ("slime", "slime.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    du = types.ModuleType("slime.utils.distributed_utils")
    du.get_gloo_group = lambda: None
    monkeypatch.setitem(sys.modules, "slime.utils.distributed_utils", du)
    wsu = types.ModuleType("slime.utils.weight_sync")
    wsu.should_flush_cache = lambda *a, **k: False
    monkeypatch.setitem(sys.modules, "slime.utils.weight_sync", wsu)
    mc = types.ModuleType("megatron.core")
    mc.mpu = types.SimpleNamespace(get_pipeline_model_parallel_world_size=lambda: pp)
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", mc)

    # fake NCCL broadcast: every engine of the group gets a copy (taken now, like the wire)
    import torch.distributed as ws_dist

    class H:
        def wait(self):
            pass

    def broadcast(t, src, group=None, async_op=False):
        if isinstance(group, FakeGroup):
            stats["bcasts"] += 1
            stats["bcast_bytes"] += t.numel() * t.element_size() * len(group.engines)
            if stats.get("fail_after") is not None and stats["bcasts"] > stats["fail_after"]:
                raise RuntimeError("injected NCCL failure")
            for e in group.engines:
                e.inbox[group.name].append(t.clone())
        return H()

    monkeypatch.setattr(ws_dist, "broadcast", broadcast, raising=False)
    monkeypatch.setattr(ws_dist, "barrier", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ws_dist, "get_rank", lambda *a, **k: 0, raising=False)

    class Ev:
        def record(self, *a):
            pass

        def synchronize(self):
            pass

    class St:
        def synchronize(self):
            pass

        def wait_event(self, e):
            pass

    monkeypatch.setattr(torch.cuda, "Stream", St)
    monkeypatch.setattr(torch.cuda, "Event", Ev)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: St())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    return uwd


def model_chunks(seed):
    named = make_named(seed, n=120, big=4000)
    non = [named[i:i + 10] for i in range(0, 60, 10)]
    exp = [named[i:i + 7] for i in range(60, len(named), 7)]
    return non, exp


def sync_once(monkeypatch, opts, n_engines, pp, env, seed=7):
    for k in ("ASRT_WS_BIGBUCKET", "ASRT_WS_BUCKET_MB", "ASRT_WS_NBUF", "ASRT_WS_FANOUT", "ASRT_LOG_DIR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    stats = {"groups": [], "bcasts": 0, "bcast_bytes": 0}
    uwd = build_fake(monkeypatch, n_engines, pp, lambda: tuple([list(c) for c in part] for part in model_chunks(seed)), stats)
    if opts:  # no opts: the fake slime class unpatched = slime's original per-tensor path
        ws._patch(uwd, set(opts), 2)
    engines = [FakeEngine(i, stats) for i in range(n_engines)]
    lock = types.SimpleNamespace(acquire=_Remote(lambda: True), release=_Remote(lambda: None))
    obj = uwd.UpdateWeightFromDistributed(types.SimpleNamespace(rollout_num_gpus_per_engine=2,
                                                                pipeline_model_parallel_size=pp,
                                                                flush_cache_interval=0))
    obj.connect_rollout_engines(engines, lock)
    stats["bcasts"] = stats["bcast_bytes"] = 0
    obj.update_weights()
    return engines, stats, obj


@pytest.mark.parametrize("pipeline", [True, False])
def test_bigbucket_bit_exact_vs_stock(monkeypatch, pipeline):
    ref_eng, ref_stats, _ = sync_once(monkeypatch, [], 8, 2, {})  # slime's original per-tensor path
    _small_bucket(monkeypatch)
    opts = ["flatten", "parsync", "profile"] + (["pipeline"] if pipeline else [])
    eng, st, obj = sync_once(monkeypatch, opts, 8, 2, {})
    assert obj._asrt_big and not getattr(obj, "_asrt_par", [])
    assert st["groups"] == ["slime-pp_0"]  # AUTO at 8 engines: no per-engine groups
    for a, b in zip(ref_eng, eng):
        assert set(a.state) == set(b.state)
        for n in a.state:
            assert torch.equal(a.state[n], b.state[n]), n


def _small_bucket(monkeypatch):
    # BigPacker cap in bytes = MB << 20; use a tiny cap through the class default instead
    orig = ws.BigPacker.__init__

    def init(self, bucket_bytes, nbuf=2, device=None):
        orig(self, 20000, nbuf, device)
    monkeypatch.setattr(ws.BigPacker, "__init__", init)


def test_one_broadcast_per_bucket_not_per_engine(monkeypatch):
    _small_bucket(monkeypatch)
    eng, st, obj = sync_once(monkeypatch, ["flatten", "parsync", "pipeline", "profile"], 8, 2, {})
    assert obj._asrt_big
    assert st["bcast_bytes"] == 8 * sum(bytes_of(v).numel() for v in eng[0].state.values())
    n_chunks = sum(len(p) for p in model_chunks(7))
    assert 1 < st["bcasts"] < n_chunks  # whole big buckets, each broadcast once to all 8 engines
    ref_eng, ref_st, ref_obj = sync_once(monkeypatch, ["flatten", "parsync", "pipeline", "profile"], 8, 2,
                                         {"ASRT_WS_BIGBUCKET": "0"})
    assert len(ref_obj._asrt_par) == 8  # parsync kept when disabled: one send per engine per bucket
    assert ref_st["bcasts"] == 8 * n_chunks
    for a, b in zip(ref_eng, eng):
        assert {n: bytes(v.numpy()) for n, v in a.state.items()} == {n: bytes(v.numpy()) for n, v in b.state.items()}


def test_auto_keeps_parsync_with_few_engines(monkeypatch):
    eng, st, obj = sync_once(monkeypatch, ["flatten", "parsync", "pipeline", "profile"], 2, 2, {})
    assert not obj._asrt_big and len(obj._asrt_par) == 2


def test_flatten_without_parsync_unchanged_under_auto(monkeypatch):
    eng, st, obj = sync_once(monkeypatch, ["flatten", "pipeline", "profile"], 8, 2, {})
    assert not obj._asrt_big
    eng2, st2, obj2 = sync_once(monkeypatch, ["flatten", "pipeline", "profile"], 8, 2, {"ASRT_WS_BIGBUCKET": "1"})
    assert obj2._asrt_big
    for a, b in zip(eng, eng2):
        assert {n: bytes(v.numpy()) for n, v in a.state.items()} == {n: bytes(v.numpy()) for n, v in b.state.items()}


def test_sender_error_does_not_hang(monkeypatch):
    _small_bucket(monkeypatch)
    for k in ("ASRT_WS_BIGBUCKET", "ASRT_WS_BUCKET_MB"):
        monkeypatch.delenv(k, raising=False)
    stats = {"groups": [], "bcasts": 0, "bcast_bytes": 0}
    uwd = build_fake(monkeypatch, 8, 2, lambda: tuple([list(c) for c in part] for part in model_chunks(3)), stats)
    ws._patch(uwd, {"flatten", "parsync", "pipeline", "profile"}, 2)
    engines = [FakeEngine(i, stats) for i in range(8)]
    lock = types.SimpleNamespace(acquire=_Remote(lambda: True), release=_Remote(lambda: None))
    obj = uwd.UpdateWeightFromDistributed(types.SimpleNamespace(rollout_num_gpus_per_engine=2, flush_cache_interval=0))
    obj.connect_rollout_engines(engines, lock)
    stats["fail_after"] = 2
    res = {}

    def run():
        try:
            obj.update_weights()
        except RuntimeError as e:
            res["err"] = str(e)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(20)
    assert not th.is_alive(), "producer hung after a sender error"
    assert "injected" in res.get("err", "")
