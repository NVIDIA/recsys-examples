"""Take slime's training-side ``data_preprocess`` off the critical path.

slime 2026-10 loop (``train.py``), per step i:
    generate(i) -> train(i) [data_preprocess = ray.get of this DP rank's batch
    from the rollout manager's object store + H2D] -> update_weights(i) -> ...
With R3 (``--use-rollout-routing-replay``) the batch carries routed-expert
ids for every token: rows x layers x topk int32 (a few hundred ~40k-token
samples on a many-layer top-10 MoE: ~20 GB per step), and every training node pulls it through the Ray
object store.  That pull is what a ~20 s/step ``data_preprocess`` is.

Pieces (each optional, all monkey patches):
  * ``r3c`` (R3 compaction, manager side + trainer side): routed-expert ids
    are stored as int16 (uint8 if <=256 experts) before ``ray.put``; the
    trainer's replay buffer is still allocated as int32 so nothing
    downstream changes.  Halves (quarters) the bytes moved.
  * ``overlap`` (driver + trainer + manager): while the trainers run
    update_weights(i), the driver already asks the manager for batch i+1
    (the manager is made a threaded actor so the update's control calls are
    not queued behind it), posts the refs to a tiny board actor, and a
    background thread in every trainer runs slime's own
    ``_get_rollout_data`` on them.  ``train(i+1)`` then finds the batch ready.
    The batch content is unchanged: fully-async hands out completed groups
    FIFO either way; requests admitted during the update wait inside
    SGLang's paused scheduler exactly as before (in_place pause).
  * ``profile``: split ``data_preprocess`` into fetch vs. H2D and log it
    (``$ASRT_LOG_DIR/prefetch-rank*.jsonl``).

How to turn it on (ai_search_plugin.install(framework='slime') prints the same recipe):
  driver:   python3 -m ai_search_plugin.adapters.slime_train <path/to/slime>/train.py <slime args>
  trainer:  --custom-megatron-init-path ai_search_plugin.adapters.slime_weightsync.megatron_init
  manager:  --rollout-function-path ai_search_plugin.adapters.slime_prefetch.generate_rollout_fully_async
  env:      ASRT_PREFETCH=overlap,r3c,profile   (unset/empty = off)
``overlap`` works with the object-store and straw transports (with straw = slime's generation-side streaming
handoff, this is the trainer-side half: batch i+1 is pulled during update_weights(i)); non-colocated, no critic,
no offload; anything else falls back to slime.

``prebuild`` (manager side, part of ``overlap``; ASRT_PREFETCH_PREBUILD=auto|1|0, auto = on with straw):
with straw, the manager's generate(i+1) itself is slow at scale - it writes the raw collection, converts and
writes the converted pack to shared storage (~17 GB/step on a shared filesystem: ~16 s), so the batch reference only
exists ~12 s AFTER the 4 s weight update and the trainers' warm-up never had a window (prefetch-rank*.jsonl:
prefetched=false every step).  prebuild runs slime's own ``RolloutManager._generate(i+1)`` in a manager thread
during train(i) as soon as B completed groups are queued (FIFO pop of the same groups slime hands out at
update(i); never waits for generation), posts the refs to the board (trainers warm during train(i)), and
generate(i+1) at update(i) hands them out.  Same batch content; admission is unchanged (the popped groups stay
counted against the worker's prefetch until hand-out, so no group is admitted earlier than in slime).  Off for
eval, right before a checkpoint save, filters, replayed batches and the last rollout; DistributedRollout only.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

BOARD_NAME = "asrt_prefetch_board"


def opts() -> set:
    return {s.strip() for s in os.environ.get("ASRT_PREFETCH", "").split(",") if s.strip()}


def _log(kind: str, rec: dict):
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{kind}-rank{rec.get('rank', 0)}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def compact_dtype(num_experts: int):
    import torch

    if num_experts <= 256:
        return torch.uint8
    if num_experts <= 32767:
        return torch.int16
    return None


# --------------------------------------------------------------------------- manager side
_mgr_installed = False


def install_manager(num_experts: Optional[int] = None) -> bool:
    """R3 compaction before ``ray.put`` (runs in the RolloutManager process)."""
    global _mgr_installed
    if _mgr_installed or "r3c" not in opts():
        return _mgr_installed
    try:
        import torch
        from slime.data import batch_builder as bb
    except Exception:
        return False
    orig = bb.tensorize_rollout_data_for_training

    def tensorize(rollout_data):
        orig(rollout_data)
        vals = rollout_data.get("rollout_routed_experts")
        if not vals:
            return
        dt = compact_dtype(num_experts or int(os.environ.get("ASRT_NUM_EXPERTS", "65536")))
        if dt is None:
            return
        rollout_data["rollout_routed_experts"] = [
            v.to(dt) if isinstance(v, torch.Tensor) and v.dtype in (torch.int32, torch.int64) else v for v in vals
        ]

    bb.tensorize_rollout_data_for_training = tensorize
    _mgr_installed = True
    return True


_hm_installed = False


def _live_classes(name: str, method: str) -> list:
    """All live class objects called ``name`` that define ``method`` themselves (the module
    class and Ray's per-actor wrapper subclass, which copies methods).  Every probe is
    guarded: some lazy objects (absl flags) raise on attribute access."""
    import gc

    out = []
    for o in gc.get_objects():
        try:
            if type(o) is type and type.__getattribute__(o, "__name__") == name \
                    and method in type.__getattribute__(o, "__dict__"):
                out.append(o)
        except Exception:
            continue
    return out


class _SkipFinishRollout:
    """One-shot stand-in for the ServingCluster handle: ``finish_rollout.remote()`` returns the
    manager's current server snapshot without probing, then the real handle is restored."""

    def __init__(self, mgr, real):
        self._mgr, self._real = mgr, real
        outer = self

        class _M:
            @staticmethod
            def remote(*a, **kw):
                import ray

                outer._mgr.serving = outer._real
                return ray.put(outer._mgr.servers)

        self.finish_rollout = _M()

    def __getattr__(self, k):
        return getattr(self._real, k)


def install_manager_health_guard() -> bool:
    """With ``overlap`` the driver calls ``generate(i+1)`` while the engines are paused for
    update_weights(i).  ``RolloutManager.generate`` starts by resuming the serving health
    monitor, whose ``/health_generate`` probes then fail (503) on the paused engines, the
    engines get declared dead and the run hangs (seen in testing).  Guard: if rollout
    admission is paused (= we are inside the driver's pause_rollout_admission /
    resume_rollout_admission window, i.e. a weight update) skip that resume; the monitor then
    simply stays off for this one generate call (slime pauses it again when generate ends).
    Only installed with ``overlap``; in slime's own loop generate never runs in that window.
    (Patched on the class, not the Ray entry points: Ray binds remote methods at actor creation,
    but ``health_monitoring_resume`` is called internally via ``self``.)"""
    global _hm_installed
    if _hm_installed:
        return True
    # The manager's class is shipped to its Ray worker by value (the module attribute is the
    # ActorClass wrapper), so find the live class object; guard every probe because some
    # lazy objects (absl flags) raise on attribute access.
    classes = _live_classes("RolloutManager", "health_monitoring_resume")
    for C in classes:
        orig_hm = C.health_monitoring_resume
        orig_gen = C._generate

        def health_monitoring_resume(self, _hm=orig_hm):
            w = getattr(getattr(self, "data_source", None), "consumers", {}).get("fully_async")
            if w is not None and getattr(w, "paused", False):
                print("[asrt] generate during weight update: health monitor left paused for this call", flush=True)
                self._asrt_gen_in_update = True
                return None
            return _hm(self)

        def _generate(self, rollout_id, start_time, _g=orig_gen):
            out = _g(self, rollout_id, start_time)
            if getattr(self, "_asrt_gen_in_update", False) and getattr(self, "serving", None) is not None:
                # generate() next calls serving.finish_rollout(), which probes every engine once;
                # the engines are paused for the weight update -> probe fails -> engine declared
                # dead.  Skip that one probe (it only prunes failed engines early).
                self._asrt_gen_in_update = False
                self.serving = _SkipFinishRollout(self, self.serving)
            return out

        C.health_monitoring_resume = health_monitoring_resume
        C._generate = _generate
    _hm_installed = bool(classes)
    return _hm_installed


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Drop-in for ``--rollout-function-path``: installs manager-side patches
    then calls slime's fully-async rollout."""
    from . import slime_pool

    slime_pool.autosize(args)               # in-flight pool >= one batch (GRPO n=16: 1/4 batch otherwise)
    install_manager(getattr(args, "num_experts", None))
    from . import slime_r3check
    slime_r3check.install()                      # ASRT_R3CHECK=fast: copy-free R3 validation in the manager
    if os.environ.get("ASRT_R3PACK") == "1":
        from . import slime_r3pack

        slime_r3pack.install()
    if "overlap" in opts():
        if prebuild_enabled(args):  # before the health guard: the guard must wrap it (paused hand-out -> skip probe)
            install_manager_prebuild()
        install_manager_health_guard()
    from slime.rollout.fully_async_rollout import generate_rollout_fully_async as orig

    out = orig(args, rollout_id, data_buffer, evaluation=evaluation)
    d = float(os.environ.get("ASRT_TEST_ROLLOUT_DELAY_S", "0") or 0)
    if d > 0:  # TEST ONLY: emulate slow manager-side straw I/O (~16 s per generate on a shared filesystem)
        time.sleep(d)
    return out


# --------------------------------------------------------------------------- manager-side prebuild (straw)
def prebuild_enabled(args) -> bool:
    v = os.environ.get("ASRT_PREFETCH_PREBUILD", "auto").strip().lower()
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    return getattr(args, "rollout_data_transport", "object-store") == "straw"


def install_reservation(D) -> bool:
    """Make ``D.prefetch`` (slime DistributedRollout: the collector pulls results while
    ``pending + ready < demand + prefetch``) read ``prefetch - _asrt_reserved``.  A prebuild pops B groups from
    ``ready`` early; reserving B keeps that comparison exactly where it would be had they not been popped, so the
    workers admit nothing earlier than in slime.  Cleared at hand-out and at ``pause`` (weight update)."""
    if isinstance(D.__dict__.get("prefetch"), property):
        return True

    def _get(self):
        d = self.__dict__
        return max(0, d.get("_asrt_pf", d.get("prefetch", 0)) - d.get("_asrt_reserved", 0))

    def _set(self, v):
        self.__dict__["_asrt_pf"] = v

    D.prefetch = property(_get, _set)
    if "pause" in D.__dict__:
        orig_pause = D.__dict__["pause"]

        def pause(self, *a, **kw):
            self.__dict__["_asrt_reserved"] = 0
            return orig_pause(self, *a, **kw)

        D.pause = pause
    return True


def install_batch_ids(BB) -> bool:
    """slime BatchBuilder keeps ONE current batch (``training_completed(i)`` raises unless ``rollout_id == i``).
    With a prebuild, batch i+1 is begun before training_completed(i): remember each rollout's batch id."""
    if getattr(BB, "_asrt_batch_ids_patched", False):
        return True
    orig_begin, orig_replay, orig_tc = BB.begin, BB.replay_converted, BB.training_completed

    def _remember(self):
        ids = self.__dict__.setdefault("_asrt_batch_ids", {})
        ids[self.rollout_id] = self.batch_id
        for k in [k for k in ids if k < self.rollout_id - 4]:
            del ids[k]

    def begin(self, samples, *a, **kw):
        r = orig_begin(self, samples, *a, **kw)
        _remember(self)
        return r

    def replay_converted(self, data, batch_id):
        r = orig_replay(self, data, batch_id)
        _remember(self)
        return r

    def training_completed(self, rollout_id):
        ids = self.__dict__.get("_asrt_batch_ids", {})
        if rollout_id not in ids:
            return orig_tc(self, rollout_id)
        bid = ids.pop(rollout_id)
        if bid is not None:
            import ray

            ray.get(self.controller.finish_batch.remote(bid))

    BB.begin, BB.replay_converted, BB.training_completed = begin, replay_converted, training_completed
    BB._asrt_batch_ids_patched = True
    return True


class _Prebuild:
    def __init__(self, rid: int):
        self.rid = rid
        self.lock = threading.Lock()
        self.started = self.cancel = False
        self.done = threading.Event()
        self.out = self.err = None
        self.t_sched = time.time()
        self.t_start = self.t_end = None


def _consumer(mgr):
    return getattr(getattr(mgr, "data_source", None), "consumers", {}).get("fully_async")


def prebuild_blocked(mgr, rid: int, D=None) -> Optional[str]:
    """Why batch ``rid`` may NOT be prebuilt during train(rid-1) (None = allowed)."""
    a = mgr.args
    if rid >= getattr(a, "num_rollout", 1 << 30):
        return "last"
    if getattr(a, "dynamic_sampling_filter_path", None) or getattr(a, "rollout_sample_filter_path", None):
        return "filters"  # ready groups may be dropped -> cannot know B kept groups are queued
    if getattr(a, "eval_interval", None) is not None:
        return "eval"
    if getattr(a, "offload_rollout", False) or getattr(a, "colocate", False) or getattr(a, "debug_rollout_only", False) \
            or getattr(a, "load_debug_rollout_data", None):
        return "offload/colocate/debug"
    rec = getattr(mgr, "recovery", None)
    if rec is not None and rid in getattr(rec, "batches", {}):
        return "replay"
    try:  # checkpoint after train(rid-1) snapshots the queue/builder: do not pop before it
        from slime.utils.misc import should_run_periodic_action

        if should_run_periodic_action(rid - 1, getattr(a, "save_interval", None), None, getattr(a, "num_rollout", None)):
            return "save"
    except Exception:
        pass
    if D is None:
        try:
            from slime.rollout.fully_async_distributed import DistributedRollout as D
        except Exception:
            return "worker"
    if not isinstance(_consumer(mgr), D):
        return "worker"
    return None


def _prebuild_main(mgr, pb: _Prebuild, inner, post=None):
    """Wait (without consuming) until B completed groups are queued, then run slime's own _generate(rid)."""
    w = _consumer(mgr)
    need = int(mgr.args.rollout_batch_size)
    while True:
        with w.condition:
            ok = (not w.paused) and (not w.demand) and len(w.ready) >= need and w.error is None
        with pb.lock:
            if pb.cancel or w.error is not None:
                pb.done.set()
                return
            if ok:
                pb.started = True
                w.__dict__["_asrt_reserved"] = need   # the B popped groups keep counting until hand-out
                break
        time.sleep(0.02)
    pb.t_start = time.time()
    _tls.prebuild = True
    try:
        mgr.batch_builder.rollout_id = pb.rid
        pb.out = inner(mgr, pb.rid, pb.t_start)
        if post is not None:
            post(pb.rid, pb.out)
    except BaseException as e:  # surfaced by the generate call that takes it
        pb.err = e
    finally:
        _tls.prebuild = False
        pb.t_end = time.time()
        pb.done.set()


def make_prebuild_generate(inner, D=None, post=None, log=None):
    """Wrap ``RolloutManager._generate``: hand out a prebuilt batch if there is one, then schedule the next."""
    log = log or (lambda rec: _log("prefetch-mgr", rec))

    def _generate(self, rollout_id, start_time):
        if getattr(_tls, "prebuild", False):
            return inner(self, rollout_id, start_time)
        t_ask = time.time()
        rec = {"rollout": rollout_id, "t_ask": round(t_ask, 3)}
        pb = getattr(self, "_asrt_pb", None)
        self._asrt_pb = None
        out = None
        if pb is not None:
            with pb.lock:
                if not pb.started:
                    pb.cancel = True
            if pb.started:
                pb.done.wait()
                if pb.err is not None:
                    raise pb.err
                if pb.rid != rollout_id:  # never drop popped groups silently
                    raise RuntimeError(f"asrt prebuild built rollout {pb.rid} but rollout {rollout_id} was asked")
                out = pb.out
                rec.update(prebuilt=True, build_s=round(pb.t_end - pb.t_start, 3),
                           ready_before_ask_s=round(max(0.0, t_ask - pb.t_end), 3),
                           join_s=round(time.time() - t_ask, 3))
            else:
                rec.update(prebuilt=False, why="fewer than B completed groups before hand-out")
        else:
            rec.update(prebuilt=False, why="none scheduled")
        w = _consumer(self)
        if w is not None:
            w.__dict__["_asrt_reserved"] = 0
        if out is None:
            t0 = time.time()
            out = inner(self, rollout_id, start_time)
            rec["build_s"] = round(time.time() - t0, 3)
        rec["handout_s"] = round(time.time() - t_ask, 3)
        why = prebuild_blocked(self, rollout_id + 1, D)
        if why is None:
            nxt = _Prebuild(rollout_id + 1)
            self._asrt_pb = nxt
            threading.Thread(target=_prebuild_main, args=(self, nxt, inner, post), name="asrt-prebuild",
                             daemon=True).start()
        else:
            rec["next_prebuild_off"] = why
        log(rec)
        return out

    return _generate


def _post_to_board(rid, refs):
    import ray

    ray.get(get_or_create_board().post.remote(rid, refs))


_pb_installed = False


def install_manager_prebuild() -> bool:
    global _pb_installed
    if _pb_installed:
        return True
    try:
        from slime.data.batch_builder import BatchBuilder
        from slime.rollout.fully_async_distributed import DistributedRollout
    except Exception as e:
        print(f"[asrt] prebuild: import failed ({e!r}); off", flush=True)
        return False
    install_reservation(DistributedRollout)
    install_batch_ids(BatchBuilder)
    classes = _live_classes("RolloutManager", "_generate")
    for C in classes:
        C._generate = make_prebuild_generate(C.__dict__["_generate"], DistributedRollout, _post_to_board)
    _pb_installed = bool(classes)
    print(f"[asrt] slime_prefetch prebuild installed on {len(classes)} RolloutManager class(es)", flush=True)
    return _pb_installed


# --------------------------------------------------------------------------- board actor
def _board_cls():
    import ray

    @ray.remote(num_cpus=0)
    class PrefetchBoard:
        """Driver posts (rollout_id, refs); trainer threads long-poll."""

        def __init__(self):
            import asyncio

            self.items = {}
            self.cv = asyncio.Condition()

        async def post(self, rollout_id, refs):
            async with self.cv:
                self.items[rollout_id] = refs
                for k in [k for k in self.items if k < rollout_id - 2]:
                    del self.items[k]
                self.cv.notify_all()

        async def wait_after(self, last_id, timeout=30.0):
            import asyncio

            async with self.cv:
                try:
                    await asyncio.wait_for(self.cv.wait_for(lambda: any(k > last_id for k in self.items)), timeout)
                except asyncio.TimeoutError:
                    return None
                k = min(k for k in self.items if k > last_id)
                return k, self.items[k]

    return PrefetchBoard


def get_or_create_board():
    import ray

    try:
        return ray.get_actor(BOARD_NAME)
    except ValueError:
        return _board_cls().options(name=BOARD_NAME, get_if_exists=True, lifetime="detached").remote()



_straw_cache = {}


def _straw_key(ref):
    return (getattr(ref, "batch_id", None), getattr(ref, "rank", None), getattr(ref, "plan_digest", None))


def _install_straw_cache():
    try:
        import slime.data.transport as T
    except Exception:
        return
    if hasattr(T.unpack_rollout_payload, "__wrapped_orig__"):
        return
    orig = T.unpack_rollout_payload

    def unpack_rollout_payload(value):
        hit = _straw_cache.pop(_straw_key(value), None) if type(value).__name__ == "TrainBatchRef" else None
        return hit if hit is not None else orig(value)
    unpack_rollout_payload.__wrapped_orig__ = orig
    T.unpack_rollout_payload = unpack_rollout_payload


# --------------------------------------------------------------------------- trainer side
_actor_installed = False
_tls = threading.local()


def _key(rollout_data_ref):
    try:
        return tuple(b.inner.hex() for b in rollout_data_ref)
    except Exception:
        return None


def install_actor() -> bool:
    """Patch ``MegatronTrainRayActor._get_rollout_data`` (runs in each trainer)."""
    global _actor_installed
    o = opts()
    if _actor_installed or not o:
        return _actor_installed
    try:
        import torch
        import ray
        from slime.backends.megatron_utils import actor as act
        from slime.backends.megatron_utils import cp_utils
    except Exception as e:
        print(f"[asrt] slime_prefetch.install_actor: import failed: {e!r}", flush=True)
        return False

    if "r3c" in o:
        orig_buf = cp_utils._new_routed_experts_buffer

        def new_buf(rows, num_layers, topk, reference):
            buf = orig_buf(rows, num_layers, topk, reference)
            if buf.dtype in (torch.int16, torch.uint8):  # keep the replay contract: int32 ids
                return torch.empty_like(buf, dtype=torch.int32)
            return buf

        cp_utils._new_routed_experts_buffer = new_buf

    if "profile" in o and hasattr(act, "process_rollout_data"):
        orig_prd = act.process_rollout_data

        def process_rollout_data(*a, **kw):
            t0 = time.perf_counter()
            r = orig_prd(*a, **kw)
            _tls.ray_get_s = time.perf_counter() - t0
            return r

        act.process_rollout_data = process_rollout_data

    # Ray's actor wrapper subclass carries its own copies (tracing wrappers) of every method, so
    # patching the module class alone is bypassed; patch every live class that defines it.
    if "overlap" in o:
        _install_straw_cache()
    classes = _live_classes("MegatronTrainRayActor", "_get_rollout_data") or [act.MegatronTrainRayActor]
    if act.MegatronTrainRayActor not in classes:
        classes.append(act.MegatronTrainRayActor)
    orig_by_cls = {C: C.__dict__["_get_rollout_data"] for C in classes if "_get_rollout_data" in C.__dict__}
    orig_get = act.MegatronTrainRayActor.__dict__["_get_rollout_data"]
    overlap = "overlap" in o

    def _timed_get(self, ref):
        t0 = time.perf_counter()
        out = orig_get(self, ref)
        torch.cuda.current_stream().synchronize()
        dt = time.perf_counter() - t0
        return out, dt

    def _rg():
        v = getattr(_tls, "ray_get_s", None)
        _tls.ray_get_s = None
        return None if v is None else round(v, 4)

    # Trainer-side prefetch = WARM ONLY.  The background thread does the slow, CPU-only part for batch i+1 while
    # the trainers run update_weights(i): pull this DP rank's partition into the local object store (and, with
    # straw, unpack it from storage into a cache).  It never touches CUDA or slime state (process_rollout_data
    # sets global timers), and never works on a batch the main thread already started.  train(i+1) then runs
    # slime's own _get_rollout_data unchanged, which hits local memory.  The main thread waits for an in-flight
    # warm at most ASRT_PREFETCH_WAIT_S (default 120 s) and otherwise just proceeds - always safe.
    # (The previous version ran the whole _get_rollout_data, incl. H2D copies + stream syncs, in the side thread
    # and could race the main thread on the same batch; a 32-GPU real-model run hung in it.)
    wait_cap = float(os.environ.get("ASRT_PREFETCH_WAIT_S", "120"))

    def _dp_rank_size():
        from megatron.core import mpu
        return (mpu.get_data_parallel_rank(with_context_parallel=False),
                mpu.get_data_parallel_world_size(with_context_parallel=False))

    def _warm(refs, dp_rank):
        val = ray.get(refs[dp_rank].inner)        # object-store: bytes now local; straw: small batch ref
        held = [val]
        try:
            from slime.data.transport import TrainBatchRef, unpack_rollout_payload
            if isinstance(val, TrainBatchRef):
                held.append(unpack_rollout_payload.__wrapped_orig__(val)
                            if hasattr(unpack_rollout_payload, "__wrapped_orig__") else unpack_rollout_payload(val))
                _straw_cache[_straw_key(val)] = held[-1]
                while len(_straw_cache) > 2:      # a missed batch must not pin GBs (dicts keep insertion order)
                    _straw_cache.pop(next(iter(_straw_cache)))
        except Exception:
            pass
        return held

    def _bg_loop(self):
        last = -1
        board = None
        while True:
            try:
                if board is None:
                    board = get_or_create_board()
                got = ray.get(board.wait_after.remote(last, 30.0))
                if got is None:
                    continue
                rid, refs = got
                last = rid
                k = _key(refs)
                with self._asrt_lock:
                    if k in self._asrt_claimed:          # main thread is already fetching it itself
                        continue
                    for kk in [kk for kk, v in self._asrt_pref.items() if v["rid"] < rid - 1]:
                        del self._asrt_pref[kk]
                    ent = {"event": threading.Event(), "held": None, "err": None, "rid": rid}
                    self._asrt_pref[k] = ent
                t0 = time.perf_counter()
                try:  # only this local entry object is touched (train() may pop it from the dict meanwhile)
                    dp_rank, _ = _dp_rank_size()
                    ent["held"] = _warm(refs, dp_rank)
                except BaseException as e:
                    ent["err"] = repr(e)[:200]
                finally:
                    ent["warm_s"] = time.perf_counter() - t0
                    ent["event"].set()  # always: a waiter must never be left hanging
            except Exception:
                time.sleep(1.0)
                board = None

    def _get_rollout_data(self, rollout_data_ref):
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        if overlap and not getattr(self, "_asrt_bg", None):
            self._asrt_pref = {}
            self._asrt_claimed = set()
            self._asrt_lock = threading.Lock()
            self._asrt_bg = threading.Thread(target=_bg_loop, args=(self,), daemon=True, name="asrt-prefetch")
            self._asrt_bg.start()
        t0 = time.perf_counter()
        ent = None
        if overlap:
            k = _key(rollout_data_ref)
            with self._asrt_lock:
                ent = self._asrt_pref.pop(k, None)
                if ent is None:
                    self._asrt_claimed.add(k)            # bg thread will skip this batch
                    if len(self._asrt_claimed) > 8:
                        self._asrt_claimed = set(list(self._asrt_claimed)[-4:])
        waited = None
        if ent is not None:
            ok = ent["event"].wait(timeout=wait_cap)
            waited = round(time.perf_counter() - t0, 4)
            if not ok:
                ent = {"err": f"warm not done after {wait_cap:.0f} s"}
        out, dt = _timed_get(self, rollout_data_ref)    # slime's own path, main thread (hits warmed data)
        rec = {"rank": rank, "prefetched": bool(ent is not None and not ent.get("err")), "fetch_s": round(dt, 4),
               "ray_get_s": _rg()}
        if ent is not None:
            rec.update(rid=ent.get("rid"), warm_s=round(ent.get("warm_s", 0) or 0, 4), wait_s=waited, err=ent.get("err"))
        _log("prefetch", rec)
        return out

    for C in orig_by_cls:
        C._get_rollout_data = _get_rollout_data
    _actor_installed = True
    return True
