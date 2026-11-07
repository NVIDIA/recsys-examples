"""Pipelined trainer -> SGLang weight sync for slime (NCCL, non-colocated).

What slime does today (``UpdateWeightFromDistributed``, slime 2026-10):
  pause engines -> for every bucket (``--update-weight-buffer-size``, 512 MB):
  TP/EP all-gather + HF convert -> acquire the cross-PP Ray lock ->
  Ray RPC to every engine -> one NCCL broadcast *per HF tensor* -> wait until
  every engine has received and run ``model.load_weights`` -> release lock;
  all strictly serial, then continue engines.  With a 512-expert MoE that is
  ~1.5k HF tensors per layer (one per expert matrix): tens of thousands of
  small broadcasts and receiver-side ``torch.empty`` calls per sync.

What this module changes (all monkey patches, slime source untouched):
  * ``flatten``  - each bucket is sent as ONE contiguous uint8 buffer using
    SGLang's own ``load_format="flattened_bucket"`` receive path (already in
    the image's SGLang): one broadcast per bucket instead of one per tensor.
  * ``pipeline`` - a sender thread on the PP-source rank does lock/RPC/
    broadcast/engine-load of bucket k on a separate CUDA stream while the
    main thread gathers + converts bucket k+1 (bounded queue = double
    buffering; memory cost = ``depth`` extra buckets).
  * ``profile``  - per-phase timers (gather+convert, lock wait, broadcast +
    engine load, pause, continue) on every sync, written into slime's
    ``update_weight_metrics`` (appear on the step log as
    ``update_weights/asrt_*``) and appended to ``$ASRT_LOG_DIR/weightsync-rank*.jsonl``.
    Profile alone does not change behaviour - use it to explain a baseline.
  * ``async`` (opt-in) - the trainer only gathers/converts and stages the
    buckets (pinned host memory, reused across steps; ``stage_gpu`` keeps
    them on GPU) and returns; a background thread on the PP-source rank sends
    them while the next step trains.  Engines are paused per bucket (inside
    the cross-PP lock), not for the whole sync, so they keep generating in
    between: tokens produced during the window see a mix of old/new layers
    (their logprobs are still the true behaviour-policy logprobs, so TIS
    stays valid, but it IS a semantic change).  The engine version label
    flips on the last bucket; the next update_weights joins the previous
    transfer first; the first sync is blocking.
  * ``no_pause`` - do NOT use with SGLang: an un-paused update takes the
    tokenizer's model-update writer lock, which waits for every in-flight
    request to finish (= drains the engines).  Kept only for experiments.
  * ``parsync`` - the PP stages send in parallel.  slime gives every PP stage one NCCL group spanning
    ALL engines and serializes every bucket of every stage behind one Ray lock (otherwise two
    all-engine broadcasts could reach the engines in different orders and deadlock).  In one 4-stage
    layout that lock was ~2/3 of a 20 s sync (4 stages x ~62 GB, ~4 s of real transfer each).  parsync adds one group per
    (PP stage, engine) pair ("slime-pp_<p>-e<i>": source rank + that engine's GPUs) and sends each bucket
    to every engine over its own group without the global lock.  No deadlock: an engine executes one
    weight-update call at a time, and a sender only waits on the per-engine groups of its current
    bucket, so the waits form no cycle.  slime's own groups stay (used by ``verify``).  Cost: each
    stage pushes its bytes once per engine instead of one broadcast to all (more source-NIC traffic),
    and E x PP extra small communicators.  Not combined with ``async`` (per-bucket pause needs the lock).
  * ``inplace`` - engines keep generating during the sync (no pause, no drain).  Needs the engine-side
    hook ``ASRT_SGL_HOOKS=...,inplace`` (ai_search_plugin.adapters.slime_sglang_hooks; best with ``fastload``): buckets
    are sent un-paused as ``flattened_bucket`` with no version label; the engine posts each receive on a side
    stream and returns at once, applies complete buckets on its forward stream between two decode
    iterations, and a final empty update from rank 0 (the COMMIT) applies the rest and relabels the engine
    with the new version.  Engine stall per sync = the apply copies (~ms) instead of the whole sync.  Same
    semantic class as ``--flush-cache-interval 0`` / ``async``: requests spanning the sync see old, then
    mixed, then new weights (SGLang per-request version spans tag the mixed part "<old>~<new>").  The first
    sync, and any sync where slime's flush interval asks for abort+flush, use slime's paused path.  Works
    with flatten / pipeline / parsync / fastload; not with ``async`` (ignored there) or compressed-tensors.
  * ``bigbucket`` (AUTO, verl's NCCL checkpoint-engine scheme) - the PP-source rank packs the HF tensors into
    large preallocated GPU buckets (``ASRT_WS_BUCKET_MB``, default 4096; 2048 with ``inplace``) from a pool
    of ``ASRT_WS_NBUF`` (2) buffers - packing bucket k+1 overlaps the broadcast of bucket k - and broadcasts
    each whole bucket ONCE over slime's all-engine group (under slime's lock).  Why: ``parsync`` sends every
    bucket once per engine, so the source NIC carries E x the stage bytes; example PP2 x 8 TP2 engines:
    63 GB/stage x 8 = 505 GB per source NIC -> 12-13 s/sync, slower than stock slime's single broadcast
    (9.2 s) and verl (5.5 s).  AUTO (``ASRT_WS_BIGBUCKET=auto``): on when parsync is requested and
    engines >= ``ASRT_WS_FANOUT`` (2) x PP stages (parsync's per-engine groups are then not even created);
    otherwise every path stays exactly as before.  ``ASRT_WS_BIGBUCKET=1`` forces it for any flatten config,
    ``0`` disables it.  Bytes are copied verbatim, so engine weights are bit-identical.
  * ``verify`` (tests only) - after each sync, recompute SGLang's per-engine
    weight checksum, re-send the same weights with slime's original code
    path, and assert both checksums are identical.

Enable inside slime (training actors are separate Ray processes, so the
patch must run there):  ``--custom-megatron-init-path
ai_search_plugin.adapters.slime_weightsync.megatron_init`` (or the plugin entry
``ai_search_plugin.frameworks.slime.megatron_init``) with the package importable in every Ray process, and env ``ASRT_WEIGHTSYNC=pipeline,flatten,profile``
(default when unset).  To chain another init hook set
``ASRT_CHAIN_MEGATRON_INIT=module.func``.

Importing this module never imports slime/torch; ``install()`` does.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Dict, Optional

_DEFAULT = "pipeline,flatten,profile"
_installed: Dict[str, object] = {}


def _opts(spec: Optional[str]) -> set:
    spec = spec if spec is not None else os.environ.get("ASRT_WEIGHTSYNC", _DEFAULT)
    return {s.strip() for s in spec.split(",") if s.strip() and s.strip() != "off"}


class _Prof:
    """Accumulating wall-clock timers (seconds)."""

    def __init__(self):
        self.t: Dict[str, float] = {}
        self.n: Dict[str, float] = {}

    def add(self, k: str, dt: float, n: float = 0.0):
        self.t[k] = self.t.get(k, 0.0) + dt
        if n:
            self.n[k] = self.n.get(k, 0.0) + n


def _log(rec: dict):
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"weightsync-rank{rec.get('rank', 0)}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _engine_info(actor):
    """Runs inside slime's SGLangEngine actor (``__ray_call__``): where its HTTP server listens."""
    return (int(getattr(actor, "node_rank", 0) or 0), getattr(actor, "server_host", None),
            getattr(actor, "server_port", None))


class KVFreshError(RuntimeError):
    """ an engine's prefix cache could not be flushed on a kvfresh sync (stale KV would survive)."""


def _kv_http(url, method, endpoint, payload=None, timeout=60.0):
    """One control call to an engine's SGLang HTTP server -> (status_code, text). Network errors -> (-1, repr)."""
    import requests

    try:
        r = requests.request(method, f"{url}/{endpoint}", json=payload, timeout=timeout)
        return int(r.status_code), (r.text or "")[:300]
    except Exception as e:  # connection refused / timeout: counted as a failed attempt, retried
        return -1, repr(e)[:300]


def kv_flush_checked(urls, http=_kv_http, *, tries=None, base_s=None, cap_s=None, sleep=time.sleep, log=None):
    """ flush every engine's prefix (radix) cache while it is paused with mode="retract" and check it.

    SGLang's ``/flush_cache`` returns 200 only when ``Scheduler.flush_cache`` reset the tree (with the engine-side
    ``kvfresh`` hook: and the tree is empty afterwards); it returns 400 when the engine is not idle (a batch result
    still queued by the overlap scheduler, a chunked prefill, grammar queue, ...).  While paused the scheduler does not
    schedule, so a failed engine gets the retract-pause again (it retracts whatever running / chunked / last-batch
    work is left; a no-op on an engine that is already empty) and the flush is retried with bounded exponential backoff
    (``ASRT_WS_KV_FLUSH_TRIES`` attempts, default 8; delay ``ASRT_WS_KV_FLUSH_BASE_S`` 0.05 s doubling up to
    ``ASRT_WS_KV_FLUSH_CAP_S`` 2 s: worst case ~5 s).  Known non-drainable case (slime-cu130 SGLang): requests still
    in the grammar queue only move on during scheduling, which a paused engine never does -> all retries fail ->
    loud KVFreshError (structured-output workloads; none of ours).  ``urls``: per engine "http://host:port" or None (node_rank != 0,
    skipped like slime's actor).  Returns {"attempts", "retries", "failed": {engine index: last message}}."""
    tries = int(tries if tries is not None else os.environ.get("ASRT_WS_KV_FLUSH_TRIES", 8))
    delay = float(base_s if base_s is not None else os.environ.get("ASRT_WS_KV_FLUSH_BASE_S", 0.05))
    cap = float(cap_s if cap_s is not None else os.environ.get("ASRT_WS_KV_FLUSH_CAP_S", 2.0))
    pending = [i for i, u in enumerate(urls) if u]
    if not pending:  # nothing addressable = nothing verified: a failure, never a silent pass
        return {"attempts": 0, "retries": 0, "failed": {-1: "no engine HTTP URL (all None)"}}
    last: Dict[int, str] = {}
    attempts = retries = 0
    while pending and attempts < max(1, tries):
        attempts += 1
        if len(pending) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(pending)) as ex:
                res = dict(zip(pending, ex.map(lambda i: http(urls[i], "GET", "flush_cache"), pending)))
        else:
            res = {pending[0]: http(urls[pending[0]], "GET", "flush_cache")}
        failed = [i for i in pending if res[i][0] != 200]
        for i in failed:
            last[i] = f"HTTP {res[i][0]}: {res[i][1].strip()}"
        pending = failed
        if not pending or attempts >= tries:
            break
        retries += 1
        if log is not None:
            log(f"[asrt] kvfresh: flush_cache not done on engines {failed} (attempt {attempts}/{tries}: "
                f"{[last[i] for i in failed]}); re-retract and retry in {delay:.2f} s")
        sleep(delay)
        delay = min(delay * 2, cap)
        for i in failed:  # drain what kept the engine busy (overlap result, chunked prefill); paused engines only
            http(urls[i], "POST", "pause_generation", {"mode": "retract"})
    return {"attempts": attempts, "retries": retries, "failed": {i: last.get(i, "") for i in pending}}


def kv_flush_or_fail(urls, http=_kv_http, *, on_fail=None, log=print, err=None, **kw):
    """kv_flush_checked + loud failure: ERROR line on stderr (and ``logging``), and unless ``ASRT_WS_KV_ON_FAIL=continue``
    a KVFreshError (default: a sync that keeps stale KV must not be trained on silently)."""
    import sys

    r = kv_flush_checked(urls, http, log=log, **kw)
    if r["failed"]:
        on_fail = (on_fail if on_fail is not None else os.environ.get("ASRT_WS_KV_ON_FAIL", "raise")).strip().lower()
        msg = (f"[asrt] ERROR kvfresh: prefix cache NOT flushed on engines {sorted(r['failed'])} after "
               f"{r['attempts']} attempts: {r['failed']} -> old-weight KV survives this sync")
        (err or (lambda s: print(s, file=sys.stderr, flush=True)))(msg)
        try:
            import logging

            logging.getLogger("ai_search_plugin.kvfresh").error(msg)
        except Exception:
            pass
        if on_fail != "continue":
            raise KVFreshError(msg)
    return r


def _jsonable_dtypes(dtypes):
    return [str(d).replace("torch.", "") for d in dtypes]


def _jsonable_shapes(shapes):
    return [[int(x) for x in s] for s in shapes]


class DirectRPC:
    """ weight-sync control calls straight to each engine's SGLang HTTP server.

    slime routes every weight-sync call (pause, each bucket's ``update_weights_from_distributed``, commit,
    continue) through the engine's Ray actor: a synchronous, one-call-at-a-time actor living on the rollout
    node.  When that hop is late for ONE engine, every other engine's NCCL receive of the same all-engine
    broadcast waits for it: in one run 14 of 16 engine ranks sat 26.5 s in the receive of one bucket
    while the 2 ranks of a single engine started it 26.5 s late (engine-side scheduler idle, paused); another run
    had a 39 s commit whose engine-side handling took 0.2 s at the normal time.  This class posts
    the exact JSON slime's actor would post (``SGLangEngine._make_request``) from a thread pool in the trainer,
    so the only hops left are HTTP -> tokenizer -> scheduler.  Same endpoints, same payloads -> same engine
    behaviour; any setup failure leaves slime's Ray path in place.  Per-call reply times are kept so a late
    engine is named in the weight-sync record (``rpc_skew`` / ``rpc_slow``)."""

    def __init__(self, urls):
        from concurrent.futures import ThreadPoolExecutor

        self.urls = list(urls)  # per engine: "http://host:port" or None (node_rank != 0: slime's actor no-ops too)
        self.pool = ThreadPoolExecutor(max_workers=max(8, 2 * len(self.urls)), thread_name_prefix="asrt-ws-rpc")
        self._tl = threading.local()
        self.worst = (0.0, -1, "")

    @classmethod
    def build(cls, ray, engines, timeout: float = 60.0):
        infos = ray.get([e.__ray_call__.remote(_engine_info) for e in engines], timeout=timeout)
        urls = []
        for node_rank, host, port in infos:
            if node_rank != 0:
                urls.append(None)
            elif not host or not port:
                raise RuntimeError(f"engine without server address: {(node_rank, host, port)}")
            else:
                urls.append(f"http://{host}:{port}")
        return cls(urls)

    def _post(self, i, endpoint, payload):
        t0 = time.perf_counter()
        url = self.urls[i]
        if url is None:
            return None, t0, t0
        s = getattr(self._tl, "s", None)
        if s is None:
            import requests

            s = self._tl.s = requests.Session()
        r = s.post(f"{url}/{endpoint}", json=payload)
        r.raise_for_status()
        try:
            out = r.json()
        except ValueError:
            out = None
        return out, t0, time.perf_counter()

    def submit(self, idxs, endpoint, payload):
        """payload: dict (same for every engine) or callable(i) -> dict."""
        return [(i, self.pool.submit(self._post, i, endpoint, payload(i) if callable(payload) else payload))
                for i in idxs]

    def gather(self, futs, prof=None, what=""):
        outs, ends = [], []
        for i, f in futs:
            out, _, t1 = f.result()
            outs.append(out)
            if self.urls[i] is not None:
                ends.append((t1, i))
        if prof is not None and len(ends) > 1:
            ends.sort()
            skew = ends[-1][0] - ends[0][0]
            prof.add("rpc_skew", skew)
            if skew > self.worst[0]:
                self.worst = (skew, ends[-1][1], what)
        return outs


def choose_bigbucket(engines: int, pp: int, *, parsync: bool, flatten: bool, async_: bool = False,
                     mode: Optional[str] = None, fanout: Optional[float] = None):
    """Decide the bigbucket path.  Returns (on, reason).

    AUTO turns it on only where the measured problem exists: parsync was asked for and its per-engine
    fan-out (each PP source pushes its bytes once per engine) costs more than serialising the PP stages
    over one all-engine broadcast each.  Cost model (equal stages, B bytes each, source NIC bw N, one
    all-engine broadcast at bw C ~ N/2 measured): parsync ~ E*B/N, bigbucket ~ PP*B/C -> bigbucket wins
    when E >= (N/C) * PP = fanout * PP."""
    mode = (mode if mode is not None else os.environ.get("ASRT_WS_BIGBUCKET", "auto")).strip().lower()
    fanout = float(fanout if fanout is not None else os.environ.get("ASRT_WS_FANOUT", 2))
    if mode in ("0", "off", "false", "no"):
        return False, "off (ASRT_WS_BIGBUCKET=0)"
    if not flatten or async_:
        return False, "off (needs flatten, not with async)"
    if mode in ("1", "on", "true", "yes", "force"):
        return True, "on (forced)"
    if parsync and engines >= fanout * max(1, pp):
        return True, f"on (auto: {engines} engines >= {fanout:g} x {pp} PP stages: one broadcast per bucket " \
                     f"instead of {engines} per-engine sends)"
    return False, f"off (auto: parsync={parsync}, {engines} engines, {pp} PP stages)"


class BigPacker:
    """Packs (name, tensor) streams into whole buckets of a pool of preallocated uint8 buffers.

    ``emit(flat, names, dtypes, shapes, release)`` gets each finished bucket; ``release()`` must be called
    once the bucket's bytes are no longer needed (after its broadcast) - the buffer then goes back to the
    pool.  ``get_free()`` blocks while every buffer is in flight (double buffering).  Layout = SGLang's
    ``flattened_bucket`` (tensors back to back, no padding, receiver ``.view(dtype)`` at running offsets),
    so every tensor must start at an offset divisible by its element size: a tensor that would not is
    deferred to the start of the next bucket (offset 0).  Tensors larger than a bucket are emitted alone
    as a view of themselves (no copy).  Deterministic: the same input order gives the same buckets (the
    engine-side fastload plans are keyed on the layout)."""

    def __init__(self, bucket_bytes: int, nbuf: int = 2, device=None):
        import torch

        self.torch = torch
        self.cap = int(bucket_bytes)
        self.nbuf = max(1, int(nbuf))
        self.device = device
        self.bufs = []
        self.free: "queue.Queue" = queue.Queue()
        self.cur = None  # index of the buffer being filled
        self.off = 0
        self.meta = []  # (name, dtype, shape)
        self.deferred = []  # (name, tensor)
        self.wait_s = 0.0
        self.copy_bytes = 0

    def _take(self):
        if self.cur is not None:
            return
        if len(self.bufs) < self.nbuf and self.free.empty():
            self.bufs.append(self.torch.empty(self.cap, dtype=self.torch.uint8, device=self.device))
            self.cur = len(self.bufs) - 1
        else:
            t0 = time.perf_counter()
            self.cur = self.free.get()
            self.wait_s += time.perf_counter() - t0
        self.off = 0
        self.meta = []

    def _release_fn(self, idx):
        done = []

        def release():
            if not done:
                done.append(1)
                self.free.put(idx)
        return release

    def add(self, named, emit):
        """Pack a list of (name, tensor); calls ``emit`` for every bucket that fills up."""
        # largest element size first (as _flat does) so a chunk rarely breaks alignment
        for name, t in sorted(named, key=lambda nt: -nt[1].element_size()):
            self._add_one(name, t, emit)

    def _add_one(self, name, t, emit):
        t = t.detach()
        nb = t.numel() * t.element_size()
        if nb > self.cap:  # oversize: its own bucket, no copy
            self.flush(emit)
            v = t.contiguous().view(-1).view(self.torch.uint8)
            emit(v, [name], [t.dtype], [t.shape], lambda: None)
            return
        self._take()
        while self.off + nb > self.cap:  # (the deferred tensors that open the next bucket may fill it again)
            self.flush(emit)
            self._take()
        if self.off % t.element_size():
            self.deferred.append((name, t))
            return
        self.bufs[self.cur][self.off:self.off + nb].copy_(t.contiguous().view(-1).view(self.torch.uint8))
        self.off += nb
        self.copy_bytes += nb
        self.meta.append((name, t.dtype, t.shape))
        if self.deferred:  # the offset may be aligned again: place what fits now instead of opening a bucket for it
            keep = []
            for dn, dt in self.deferred:
                dnb = dt.numel() * dt.element_size()
                if self.off % dt.element_size() == 0 and self.off + dnb <= self.cap:
                    self.bufs[self.cur][self.off:self.off + dnb].copy_(dt.contiguous().view(-1).view(self.torch.uint8))
                    self.off += dnb
                    self.copy_bytes += dnb
                    self.meta.append((dn, dt.dtype, dt.shape))
                else:
                    keep.append((dn, dt))
            self.deferred = keep

    def flush(self, emit):
        """Emit the bucket being filled (if any), then start the next one with the deferred tensors."""
        if self.cur is not None and self.meta:
            idx, n = self.cur, self.off
            self.cur = None
            emit(self.bufs[idx][:n], [m[0] for m in self.meta], [m[1] for m in self.meta],
                 [m[2] for m in self.meta], self._release_fn(idx))
            self.meta = []
        while self.deferred:  # at offset 0 everything is aligned again
            d, self.deferred = self.deferred, []
            for name, t in sorted(d, key=lambda nt: -nt[1].element_size()):
                self._add_one(name, t, emit)

    def finish(self, emit):
        self.flush(emit)
        if self.cur is not None and self.meta:  # deferred tensors refilled a bucket
            self.flush(emit)
        if self.cur is not None:  # an empty taken buffer goes back
            self.free.put(self.cur)
            self.cur = None


def _big_mb(inplace: bool = False) -> int:
    v = os.environ.get("ASRT_WS_BUCKET_MB")
    return int(v) if v else (2048 if inplace else 4096)  # inplace engines hold in-flight buckets: smaller default


def _big_nbuf() -> int:
    return max(1, int(os.environ.get("ASRT_WS_NBUF", 2)))


def install(spec: Optional[str] = None, depth: Optional[int] = None) -> list:
    """Patch slime's ``UpdateWeightFromDistributed``. Returns the list of active
    options (empty if slime is not importable or spec is 'off')."""
    opts = _opts(spec)
    if not opts:
        return []
    try:
        import torch  # noqa: F401
        from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as uwd
    except Exception:
        return []
    if _installed.get("cls") is uwd.UpdateWeightFromDistributed:
        return sorted(_installed["opts"])
    depth = int(depth or os.environ.get("ASRT_WEIGHTSYNC_DEPTH", 2))
    _patch(uwd, opts, depth)
    _installed.update(cls=uwd.UpdateWeightFromDistributed, opts=opts)
    return sorted(opts)


def megatron_init(args):
    """Entry for ``--custom-megatron-init-path``; also chains another hook."""
    try:  # weight version continues across a --load resume (see slime_resume; no-op on a fresh run)
        from . import slime_resume

        slime_resume.install(args)
    except Exception as e:  # noqa: BLE001
        print(f"[asrt] slime_resume not installed: {e!r}", flush=True)
    active = install()
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"[asrt] slime_weightsync installed: {active}", flush=True)
    except Exception:
        pass
    if os.environ.get("ASRT_JITFIX"):  # stop torch.compile recompile storms (Megatron @jit_fuser)
        try:
            import torch.distributed as dist

            from . import slime_jitfix

            rep = slime_jitfix.install(args=args)
            if not dist.is_initialized() or dist.get_rank() == 0:
                if rep.get("skipped"):
                    print(f"[asrt] jitfix: off ({rep['skipped']}; ASRT_JITFIX_FORCE=1 to apply)", flush=True)
                else:
                    print(f"[asrt] jitfix: mode={rep.get('mode')} limit={rep.get('limit')} model={rep.get('model')} "
                          f"replaced {len(rep.get('replaced', []))}: {rep.get('replaced', [])[:40]}", flush=True)
        except Exception as e:
            print(f"[asrt] jitfix not installed: {e}", flush=True)
    # other asrt control-plane patches that live in the training actor
    if os.environ.get("ASRT_PREFETCH", ""):
        from . import slime_prefetch

        ok = slime_prefetch.install_actor()
        try:
            import torch.distributed as dist

            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"[asrt] slime_prefetch actor patches installed: {ok} ({sorted(slime_prefetch.opts())})", flush=True)
        except Exception:
            pass
    if os.environ.get("ASRT_R3CHECK"):  # copy-free R3 validation in the trainer process too
        from . import slime_r3check
        slime_r3check.install()
    chain = os.environ.get("ASRT_CHAIN_MEGATRON_INIT")
    if chain:  # comma list of further init hooks, called in order
        from slime.utils.misc import load_function

        for fn in [c.strip() for c in chain.split(",") if c.strip()]:
            r = load_function(fn)(args)
            try:
                import torch.distributed as dist

                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"[asrt] chained init {fn} -> {str(r)[:300]}", flush=True)
            except Exception:
                pass


def _patch(uwd, opts: set, depth: int):
    import torch
    import torch.distributed as dist
    import ray
    from slime.utils.distributed_utils import get_gloo_group
    from slime.utils.weight_sync import should_flush_cache

    Cls = uwd.UpdateWeightFromDistributed
    orig_update_weights = Cls.update_weights
    orig_send = Cls._send_weights
    orig_bucket = Cls._update_bucket_weights_from_distributed
    Cls._asrt_orig = (orig_update_weights, orig_send, orig_bucket)
    flatten = "flatten" in opts
    pipeline = "pipeline" in opts
    async_ = "async" in opts
    no_pause = "no_pause" in opts or async_
    flatten = flatten or async_
    stage_gpu = "stage_gpu" in opts
    verify = "verify" in opts
    parsync = "parsync" in opts and not async_
    inplace = "inplace" in opts and not async_
    flatten = flatten or inplace
    # weight-sync control calls bypass the engines' Ray actors (DirectRPC); ASRT_WS_DIRECT=0 -> slime's path.
    # Only with our sync paths (flatten); profile-only stages keep slime's behaviour untouched.
    direct_on = flatten and os.environ.get("ASRT_WS_DIRECT", "1").strip().lower() not in ("0", "off", "false", "no")

    # keep in-flight requests on a sync slime would flush: slime's --flush-cache-interval (default 1 = EVERY
    # sync) pauses with mode="abort" + flush_cache, which throws away every running request; SGLang's abort-pause then
    # waits until all of them have left the tokenizer (one 8-GPU engine x 127 running long-context requests with
    # R3: pause 71.6-77.7 s on every sync, sync 16.5 -> 80 s, and the aborted partial rollouts are regenerated).  With
    # keep_inflight the engines pause in place (KV kept), take the weights, continue: the same semantics as slime's own
    # --flush-cache-interval 0 (the 'best slime' baseline knob; tokens before/after the sync are both true behaviour-policy
    # tokens, TIS stays valid), and the in-place path (inplace) is used where enabled.  ASRT_WS_KEEP_INFLIGHT=0 restores
    # slime's abort+flush.  Only with our sync paths (flatten).
    keep_inflight = flatten and os.environ.get("ASRT_WS_KEEP_INFLIGHT", "1").strip().lower() not in ("0", "off", "false", "no")

    # KV freshness after a sync that keeps in-flight requests (keep_inflight, slime's own
    # --flush-cache-interval 0, inplace): SGLang's prefix (radix) cache and the paused requests' KV were computed with
    # the OLD weights and are reused under the new ones - the next turn of a multi-turn agent hits ~70% of its prompt in
    # that cache, so the engine samples from a mix of versions that no trainer weight reproduces (multi-turn
    # search agent, 7B dense, real weights: train/rollout logp diff 0.017 -> 0.12-0.33, entropy 1.2 -> 0.2, policy collapse; AReaL,
    # flushing every sync, fine).  ASRT_WS_KV=fresh (default): pause with mode="retract" (running requests go back to
    # the waiting queue WITH their generated tokens, their KV is freed) and flush the prefix cache before continuing,
    # so every KV entry after the sync is rebuilt under the new weights; no token is thrown away (unlike abort), the
    # cost is re-prefilling the in-flight contexts.  Opt-in (ASRT_WS_KV=fresh) while the default is decided by the
    # A/B: the engine hook kvepoch (ASRT_SGL_HOOKS=...,kvepoch) already stops NEW requests from matching old-weight prefix
    # KV without interrupting anyone; fresh additionally rebuilds the in-flight requests' KV.
    kv_fresh = os.environ.get("ASRT_WS_KV", "keep").strip().lower() == "fresh"

    def _kv_urls(self):
        """Engine HTTP base URLs for the checked flush (DirectRPC's when on, else asked from the actors once per
        engine set); None -> slime's actor flush (it retries 60 x 1 s on HTTP != 200 and raises TimeoutError)."""
        d = getattr(self, "_asrt_direct", None)
        if d is not None:
            return d.urls
        try:
            key = tuple(e._actor_id.hex() for e in self.rollout_engines)
        except Exception:
            key = None
        cached = getattr(self, "_asrt_kv_urls", None)
        if cached is not None and key is not None and cached[0] == key:
            return cached[1]
        try:
            urls = DirectRPC.build(ray, list(self.rollout_engines)).urls
        except Exception as e:
            print(f"[asrt] kvfresh: engine URLs unavailable ({e!r}); using slime's flush_cache (raises after 60 s)",
                  flush=True)
            return None
        self._asrt_kv_urls = (key, urls)
        return urls

    def _kv_refresh(self, prof, already_paused=False):
        """rank 0: retract-pause (unless the caller paused with retract), flush every engine's prefix cache, check
        the flush (retry with re-retract + bounded backoff, ERROR + KVFreshError if it still fails), continue.
        Order matters: the retract (running / chunked / just-prefilled requests back to the waiting queue WITH their
        tokens, their KV freed, not inserted) happens BEFORE the flush, so no in-flight request keeps old-weight KV and
        the flush empties the tree of everything older; requests tokenized before the sync but not prefilled yet hold no KV and find an empty tree afterwards."""
        t0 = time.perf_counter()
        if not already_paused:
            _rpc_get(self, _rpc(self, self.rollout_engines, "pause_generation", {"mode": "retract"}), prof, "kv_pause")
        urls = _kv_urls(self)
        if urls is None:
            ray.get([e.flush_cache.remote() for e in self.rollout_engines])
        else:
            try:
                r = kv_flush_or_fail(urls)
            except KVFreshError:
                self._asrt_kv_flush_fail = getattr(self, "_asrt_kv_flush_fail", 0) + 1
                _log({"rank": dist.get_rank(), "op": "kv_flush_fail", "version": self.weight_version,
                      "fails": self._asrt_kv_flush_fail})
                raise
            if r["failed"]:  # ASRT_WS_KV_ON_FAIL=continue: counted, ERROR already printed
                self._asrt_kv_flush_fail = getattr(self, "_asrt_kv_flush_fail", 0) + 1
                _log({"rank": dist.get_rank(), "op": "kv_flush_fail", "version": self.weight_version,
                      "fails": self._asrt_kv_flush_fail, "failed": {str(k): v for k, v in r["failed"].items()}})
            if r["retries"]:
                prof.add("kv_flush_retry", 0.0, r["retries"])
                self._asrt_kv_flush_retries = getattr(self, "_asrt_kv_flush_retries", 0) + r["retries"]
        prof.add("kv_flush", time.perf_counter() - t0)
        if not already_paused:
            _rpc_get(self, _rpc(self, self.rollout_engines, "continue_generation", {}), prof, "kv_continue")
        self._asrt_kv_refreshed = getattr(self, "_asrt_kv_refreshed", 0) + 1

    def _flush_wanted(self, version):
        want = should_flush_cache(self.args.flush_cache_interval, version, getattr(self.args, "update_weight_start_version", 0))
        if want and keep_inflight:
            self._asrt_flush_skipped = True
            return False
        return want

    def _eidx(self, engines):
        ids = getattr(self, "_asrt_eidx", {})
        try:
            return [ids[e._actor_id.hex()] for e in engines]
        except Exception:
            return None

    def _rpc(self, engines, method, kwargs):
        """Start ``method`` on every engine (slime actor kwargs; ``kwargs`` may be callable(engine))."""
        d = getattr(self, "_asrt_direct", None)
        idx = _eidx(self, engines) if d is not None else None
        if idx is None:
            return ("ray", [getattr(e, method).remote(**(kwargs(e) if callable(kwargs) else kwargs)) for e in engines])
        if method == "update_weights_from_distributed":
            def payload(i, _e=dict(zip(idx, engines))):
                k = kwargs(_e[i]) if callable(kwargs) else kwargs
                p = {"names": list(k["names"]), "dtypes": _jsonable_dtypes(k["dtypes"]),
                     "shapes": _jsonable_shapes(k["shapes"]), "group_name": k["group_name"],
                     "flush_cache": bool(k.get("flush_cache", False))}
                if k.get("weight_version") is not None:
                    p["weight_version"] = k["weight_version"]
                if k.get("load_format") is not None:
                    p["load_format"] = k["load_format"]
                return p
        elif method == "pause_generation":
            payload = {"mode": kwargs.get("mode", "abort")}
        elif method == "continue_generation":
            payload = {}
        else:
            raise ValueError(method)
        return ("http", d.submit(idx, method, payload))

    def _rpc_get(self, h, prof=None, what=""):
        kind, xs = h
        if kind == "ray":
            return ray.get(xs)
        return self._asrt_direct.gather(xs, prof, what)

    def _setup_direct(self, engines):
        self._asrt_direct = None
        if not direct_on or not engines:
            return
        try:
            d = DirectRPC.build(ray, engines)
            self._asrt_eidx = {e._actor_id.hex(): i for i, e in enumerate(engines)}
            self._asrt_direct = d
            if dist.get_rank() == 0 or getattr(self, "_is_pp_src_rank", False):
                print(f"[asrt] ws direct: on ({sum(u is not None for u in d.urls)}/{len(d.urls)} engine HTTP servers; "
                      f"weight-sync calls skip the engine Ray actors)", flush=True)
        except Exception as e:  # keep slime's path
            print(f"[asrt] ws direct: off ({e!r})", flush=True)

    def _ver(self, version):
        if version != "cur":
            return version
        return None if getattr(self, "_asrt_inplace_now", False) else str(self.weight_version)

    def _check_inplace(self, res):
        if getattr(self, "_asrt_inplace_now", False):
            for r in res:
                if isinstance(r, dict) and "asrt-inplace" not in str(r.get("message", "")):
                    raise RuntimeError("asrt inplace weight sync: the engine did not take the in-place path "
                                       f"(ASRT_SGL_HOOKS must include 'inplace' in the SGLang processes): {r}")

    # ---- parsync: one NCCL group per (PP source, engine) ----
    orig_connect = Cls.connect_rollout_engines
    orig_disconnect = Cls.disconnect_rollout_engines

    def _drop_par(self):
        for name, grp, eng in getattr(self, "_asrt_par", []) or []:
            try:
                uwd.disconnect_rollout_engines_from_distributed(name, grp, [eng])
            except Exception as e:  # pragma: no cover - best effort on teardown
                print(f"[asrt] parsync: disconnect {name} failed: {e!r}", flush=True)
        self._asrt_par = []

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock, engine_gpu_counts=None, *a, **k):
        orig_connect(self, rollout_engines, rollout_engine_lock, engine_gpu_counts, *a, **k)
        self._asrt_big = False
        if getattr(self, "_is_pp_src_rank", False) or dist.get_rank() == 0:
            _setup_direct(self, list(rollout_engines))
        if not getattr(self, "_is_pp_src_rank", False):
            return
        try:
            from megatron.core import mpu
            pp = mpu.get_pipeline_model_parallel_world_size()
        except Exception:
            pp = int(getattr(self.args, "pipeline_model_parallel_size", 1) or 1)
        # every PP source computes the same decision from the same inputs (all stages must take the same path:
        # a lock-path stage and a lock-free parsync stage could reach the engines in different orders)
        big, why = choose_bigbucket(len(rollout_engines), pp, parsync=parsync, flatten=flatten, async_=async_)
        self._asrt_big = big
        print(f"[asrt] bigbucket {self._group_name}: {why}"
              + (f"; buckets of {_big_mb(inplace)} MB x {_big_nbuf()} buffers" if big else ""), flush=True)
        if not parsync:
            return
        _drop_par(self)
        if big:
            print(f"[asrt] parsync: off ({self._group_name}: bigbucket sends one all-engine broadcast per bucket)", flush=True)
            return
        # auto-off: with a single PP stage there is one sender and nothing to parallelize - per-engine groups would
        # only make it send the bytes once per engine instead of one broadcast (measured slower)
        if pp <= 1:
            print(f"[asrt] parsync: off (single PP stage, nothing to parallelize)", flush=True)
            return
        counts = list(engine_gpu_counts) if engine_gpu_counts is not None else \
            [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        par = []
        for i, eng in enumerate(rollout_engines):
            name = f"{self._group_name}-e{i}"
            # group creation is one-off; take slime's lock so engines join one group at a time
            while not ray.get(rollout_engine_lock.acquire.remote()):
                time.sleep(0.05)
            try:
                grp = uwd.connect_rollout_engines_from_distributed(self.args, name, [eng], engine_gpu_counts=[counts[i]])
            finally:
                ray.get(rollout_engine_lock.release.remote())
            par.append((name, grp, eng))
        self._asrt_par = par
        print(f"[asrt] parsync: {self._group_name} -> {len(par)} per-engine groups", flush=True)

    def disconnect_rollout_engines(self, *a, **k):
        if parsync:
            _drop_par(self)
        self._asrt_direct = None  # engines may be replaced (recovery): rebuilt on the next connect
        return orig_disconnect(self, *a, **k)

    if parsync or flatten:
        Cls.connect_rollout_engines = connect_rollout_engines
        Cls.disconnect_rollout_engines = disconnect_rollout_engines

    def _par_transfer(self, named, prof, stream, ready_event, version):
        """parsync: no global lock; per-engine RPC + broadcast on that engine's own group."""
        t0 = time.perf_counter()
        ver = _ver(self, version)
        ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
        if flatten:
            flat, names, dtypes, shapes = named
            nbytes = flat.numel()
            gname = {e._actor_id.hex(): name for name, _, e in self._asrt_par}
            refs = _rpc(self, [eng for _, _, eng in self._asrt_par], "update_weights_from_distributed",
                        lambda e: dict(names=names, dtypes=dtypes, shapes=shapes, group_name=gname[e._actor_id.hex()],
                                       weight_version=ver, load_format="flattened_bucket"))
            with ctx:
                if ready_event is not None:
                    torch.cuda.current_stream().wait_event(ready_event)
                hs = [dist.broadcast(flat, 0, group=grp, async_op=True) for _, grp, _ in self._asrt_par]
                for h in hs:
                    h.wait()
        else:
            nbytes = sum(t.numel() * t.element_size() for _, t in named)
            gname = {e._actor_id.hex(): name for name, _, e in self._asrt_par}
            refs = _rpc(self, [eng for _, _, eng in self._asrt_par], "update_weights_from_distributed",
                        lambda e: dict(names=[n for n, _ in named], dtypes=[t.dtype for _, t in named],
                                       shapes=[t.shape for _, t in named], group_name=gname[e._actor_id.hex()],
                                       weight_version=ver, load_format=None))
            with ctx:
                if ready_event is not None:
                    torch.cuda.current_stream().wait_event(ready_event)
                hs = [dist.broadcast(t.data, 0, group=grp, async_op=True) for _, t in named for _, grp, _ in self._asrt_par]
                for h in hs:
                    h.wait()
        t1 = time.perf_counter()
        res = _rpc_get(self, refs, prof, "bucket")
        t2 = time.perf_counter()
        for r in res:
            if isinstance(r, dict) and r.get("success") is False:
                raise RuntimeError(f"engine weight load failed: {r}")
        _check_inplace(self, res)
        if stream is not None:
            stream.synchronize()
        elif getattr(self, "_asrt_inplace_now", False):  # engine returned before the bytes arrived
            torch.cuda.current_stream().synchronize()
        prof.add("lock_wait", 0.0)
        prof.add("bcast_issue", t1 - t0, nbytes)
        prof.add("engine_load_wait", t2 - t1)
        prof.add("transfer_total", time.perf_counter() - t0, 1)

    def _flat(named):
        """[(name, t)] -> (uint8 flat tensor, names, dtypes, shapes) matching
        SGLang FlattenedTensorBucket layout (concat of uint8 views, in order).
        The receiver rebuilds each tensor with ``.view(dtype)`` at its byte
        offset, which needs offset % element_size == 0 -> order by element
        size, largest first (stable), so every offset stays aligned."""
        named = sorted(named, key=lambda nt: -nt[1].element_size())
        flat = torch.cat([t.detach().contiguous().view(-1).view(torch.uint8) for _, t in named])
        return flat, [n for n, _ in named], [t.dtype for _, t in named], [t.shape for _, t in named]

    def _transfer(self, named, prof: _Prof, stream=None, ready_event=None, version="cur", pause_each=False):
        """Lock -> RPC -> broadcast -> wait engines -> unlock (one bucket).
        pause_each: pause the engines (in_place) just for this bucket, inside the lock.  SGLang
        only skips its model-update writer lock when paused, and that lock waits for every
        in-flight request to finish, so an un-paused update would drain the engines."""
        if parsync and not pause_each and getattr(self, "_asrt_par", None):
            return _par_transfer(self, named, prof, stream, ready_event, version)
        t0 = time.perf_counter()
        poll = 0.002 if getattr(self, "_asrt_big", False) else 0.01  # bigbucket: PP stages hand the lock over often
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(poll)
        t1 = time.perf_counter()
        nbytes = 0
        paused = False
        try:
            if pause_each:
                _rpc_get(self, _rpc(self, self.rollout_engines, "pause_generation", {"mode": "in_place"}))
                paused = True
                prof.add("bucket_pause", time.perf_counter() - t1)
            if flatten:
                flat, names, dtypes, shapes = named
                nbytes = flat.numel()
                refs = _rpc(self, self.rollout_engines, "update_weights_from_distributed",
                            dict(names=names, dtypes=dtypes, shapes=shapes, group_name=self._group_name,
                                 weight_version=_ver(self, version), load_format="flattened_bucket"))
                ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
                with ctx:
                    if ready_event is not None:
                        torch.cuda.current_stream().wait_event(ready_event)
                    h = dist.broadcast(flat, 0, group=self._model_update_groups, async_op=True)
                    h.wait()
            else:
                nbytes = sum(t.numel() * t.element_size() for _, t in named)
                ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
                with ctx:
                    if ready_event is not None:
                        torch.cuda.current_stream().wait_event(ready_event)
                    refs = ("ray", uwd.update_weights_from_distributed(
                        self._group_name, self._model_update_groups, self.weight_version,
                        self.rollout_engines, named,
                    ))
            t2 = time.perf_counter()
            res = _rpc_get(self, refs, prof, "bucket")
            t3 = time.perf_counter()
            for r in res:  # SGLang returns (success, message) through slime's engine wrapper
                if isinstance(r, dict) and r.get("success") is False:
                    raise RuntimeError(f"engine weight load failed: {r}")
            _check_inplace(self, res)
            if stream is not None:
                stream.synchronize()  # buffers may be freed/reused by the producer's stream after this
            elif getattr(self, "_asrt_inplace_now", False):  # engine returned before the bytes arrived
                torch.cuda.current_stream().synchronize()
        finally:
            if paused:
                tc = time.perf_counter()
                _rpc_get(self, _rpc(self, self.rollout_engines, "continue_generation", {}))
                prof.add("bucket_continue", time.perf_counter() - tc)
            ray.get(self.rollout_engine_lock.release.remote())
        t4 = time.perf_counter()
        prof.add("lock_wait", t1 - t0)
        prof.add("bcast_issue", t2 - t1, nbytes)
        prof.add("engine_load_wait", t3 - t2)
        prof.add("unlock", t4 - t3)
        prof.add("transfer_total", t4 - t0, 1)

    def _send_weights(self, pbar):
        prof = self._asrt_prof
        src = self._is_pp_src_rank
        stream = torch.cuda.Stream() if (pipeline and src) else None
        q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        err = []

        def sender():
            while True:
                item = q.get()
                if item is None:
                    return
                payload, ev, release = item
                try:
                    if err:
                        continue
                    _transfer(self, payload, prof, stream, ev)
                    if pbar is not None:
                        pbar.update(1)
                except BaseException as e:  # surface on main thread
                    err.append(e)
                finally:
                    if release is not None:  # bigbucket: buffer back to the pool (also after an error: no hang)
                        release()

        th = None
        if pipeline and src:
            th = threading.Thread(target=sender, name="asrt-weight-sender", daemon=True)
            th.start()

        def submit(payload, release=None):
            if th is not None:
                ev = torch.cuda.Event()
                ev.record()
                t1 = time.perf_counter()
                q.put((payload, ev, release))
                prof.add("producer_blocked", time.perf_counter() - t1)
                if err:
                    raise err[0]
            else:
                try:
                    _transfer(self, payload, prof)
                finally:
                    if release is not None:
                        release()
                if pbar is not None:
                    pbar.update(1)

        big = flatten and src and getattr(self, "_asrt_big", False) and getattr(self, "_asrt_stage", None) is None
        packer = BigPacker(_big_mb(inplace) << 20, _big_nbuf(), device=torch.cuda.current_device()) if big else None

        def emit(flat, names, dtypes, shapes, release):
            submit((flat, names, dtypes, shapes), release)

        try:
            for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
                it = iter(chunk_iter)
                while True:
                    t0 = time.perf_counter()
                    try:
                        hf_chunk = next(it)
                    except StopIteration:
                        prof.add("gather_convert", time.perf_counter() - t0)
                        break
                    self._on_chunk(hf_chunk)
                    prof.add("gather_convert", time.perf_counter() - t0)
                    if packer is not None:
                        t1 = time.perf_counter()
                        w0 = packer.wait_s
                        packer.add(hf_chunk, emit)  # may emit full buckets (and wait for a free buffer)
                        hf_chunk.clear()  # the buffers own the bytes now
                        prof.add("buf_wait", packer.wait_s - w0)
                        prof.add("pack", time.perf_counter() - t1)
                        continue
                    t0 = time.perf_counter()
                    payload = _flat(hf_chunk) if flatten else list(hf_chunk)
                    if flatten:
                        hf_chunk.clear()  # flat copy owns the bytes now
                    prof.add("gather_convert", time.perf_counter() - t0)
                    if getattr(self, "_asrt_stage", None) is not None:
                        t1 = time.perf_counter()
                        self._asrt_stage.append(_stage(self, payload, len(self._asrt_stage)))
                        prof.add("stage_copy", time.perf_counter() - t1)
                        continue
                    submit(payload)
                t2 = time.perf_counter()
                dist.barrier(group=get_gloo_group())
                prof.add("barrier", time.perf_counter() - t2)
            if packer is not None:
                t1 = time.perf_counter()
                packer.finish(emit)
                prof.add("pack", time.perf_counter() - t1)
        finally:
            if th is not None:
                t3 = time.perf_counter()
                q.put(None)
                th.join()
                prof.add("drain", time.perf_counter() - t3)
        if err:
            raise err[0]

    def _stage(self, payload, idx):
        """Copy one flat bucket out of the trainer's way: pinned host (default, reused across
        steps) or a GPU clone (stage_gpu).  Returns (buffer, names, dtypes, shapes, ready_event)."""
        flat, names, dtypes, shapes = payload
        if stage_gpu:
            buf = flat  # already a fresh tensor owned by us
        else:
            pool = self.__dict__.setdefault("_asrt_pinned", [])
            if idx < len(pool) and pool[idx].numel() == flat.numel():
                buf = pool[idx]
            else:
                buf = torch.empty(flat.numel(), dtype=torch.uint8, pin_memory=True)
                if idx < len(pool):
                    pool[idx] = buf
                else:
                    pool.append(buf)
            buf.copy_(flat, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        return buf, names, dtypes, shapes, ev

    def _bg_send(self, stage, prof):
        """Background (src rank): H2D each staged bucket on a side stream, broadcast, wait
        for the engines; engines keep generating (no pause); the version label flips on the
        last bucket only."""
        try:
            torch.cuda.set_device(self._asrt_device)
            stream = torch.cuda.Stream()
            t0 = time.perf_counter()
            for i, (buf, names, dtypes, shapes, ev) in enumerate(stage):
                ev.synchronize()
                with torch.cuda.stream(stream):
                    flat = buf if buf.is_cuda else buf.to(torch.cuda.current_device(), non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record()
                last = i == len(stage) - 1
                _transfer(self, (flat, names, dtypes, shapes), prof, stream, ready,
                          version=str(self._asrt_bg_version) if last else None, pause_each=True)
                del flat
                stage[i] = None
            if kv_fresh and dist.get_rank() == 0:
                # (PP > 1: another stage's last bucket may land after this refresh; async is opt-in and not KV-exact
                # there)
                _kv_refresh(self, prof)
            prof.add("bg_total", time.perf_counter() - t0)
            _log({"rank": dist.get_rank(), "bg_version": self._asrt_bg_version,
                  **{k: round(v, 4) for k, v in prof.t.items()}})
        except BaseException as e:
            self._asrt_bg_err = e

    def _join_bg(self):
        th = getattr(self, "_asrt_bg", None)
        if th is not None:
            t0 = time.perf_counter()
            th.join()
            self._asrt_bg = None
            self._asrt_prof.add("join_prev", time.perf_counter() - t0)
        err = getattr(self, "_asrt_bg_err", None)
        if err is not None:
            self._asrt_bg_err = None
            raise err

    @torch.no_grad()
    def update_weights(self):
        self._asrt_prof = prof = _Prof()
        t_start = time.perf_counter()
        rank0 = dist.get_rank() == 0
        if async_:
            # trainer only gathers/converts/stages (~gather time); the transfer runs in the
            # background while the next step trains.  First sync stays blocking (startup
            # weight check expects it).
            _join_bg(self)
            dist.barrier(group=get_gloo_group())
            self.weight_version += 1
            first = not getattr(self, "_asrt_synced_once", False)
            self._asrt_stage = []
            try:
                self._send_weights(None)
                stage = self._asrt_stage
            finally:
                self._asrt_stage = None
            if self._is_pp_src_rank and stage:
                self._asrt_device = torch.cuda.current_device()
                self._asrt_bg_version = self.weight_version
                self._asrt_bg = threading.Thread(target=_bg_send, args=(self, stage, _Prof()),
                                                 daemon=True, name="asrt-weight-bg")
                self._asrt_bg.start()
            if first or verify or os.environ.get("ASRT_WEIGHTSYNC_BLOCKING") == "1":
                _join_bg(self)
            dist.barrier(group=get_gloo_group())
            self._asrt_synced_once = True
        elif inplace and getattr(self, "_asrt_synced_once", False) and not (
                self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]) \
                and not _flush_wanted(self, self.weight_version + 1):
            # engines keep generating: buckets go out un-paused and unlabelled, rank 0 commits at the end
            self.weight_version += 1
            t1 = time.perf_counter()
            dist.barrier(group=get_gloo_group())
            prof.add("barrier", time.perf_counter() - t1)
            self._asrt_inplace_now = True
            try:
                self._send_weights(None)
            finally:
                self._asrt_inplace_now = False
            t2 = time.perf_counter()
            dist.barrier(group=get_gloo_group())
            prof.add("barrier", time.perf_counter() - t2)
            if rank0:
                t3 = time.perf_counter()
                gname = getattr(self, "_group_name", None) or "asrt-commit"
                res = _rpc_get(self, _rpc(self, self.rollout_engines, "update_weights_from_distributed",
                                          dict(names=[], dtypes=[], shapes=[], group_name=gname,
                                               weight_version=str(self.weight_version), load_format="flattened_bucket")),
                               prof, "commit")
                for r in res:
                    if isinstance(r, dict) and (r.get("success") is False or "asrt-inplace" not in str(r.get("message", ""))):
                        raise RuntimeError(f"asrt inplace commit failed: {r}")
                prof.add("commit", time.perf_counter() - t3)
                if kv_fresh:   # requests that ran through the window hold KV of both versions: rebuild
                    _kv_refresh(self, prof)
            t4 = time.perf_counter()
            dist.barrier(group=get_gloo_group())
            prof.add("barrier", time.perf_counter() - t4)
        elif no_pause:
            # same as slime but without pause/continue; flush is not possible without a pause
            self.weight_version += 1
            dist.barrier(group=get_gloo_group())
            self._send_weights(None)
            dist.barrier(group=get_gloo_group())
            if kv_fresh and rank0:
                _kv_refresh(self, prof)
            dist.barrier(group=get_gloo_group())
        else:
            # slime's body with timers around pause / continue (behaviour unchanged)
            self.weight_version += 1
            flush_cache = _flush_wanted(self, self.weight_version)
            t0 = time.perf_counter()
            if rank0:
                _rpc_get(self, _rpc(self, self.rollout_engines, "pause_generation",
                                    {"mode": "abort" if flush_cache else ("retract" if kv_fresh else "in_place")}),
                         prof, "pause")
                if flush_cache:
                    ray.get([e.flush_cache.remote() for e in self.rollout_engines])
                if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                    uwd.post_process_weights(True, False, self.rollout_engines)
            prof.add("pause", time.perf_counter() - t0)
            t1 = time.perf_counter()
            dist.barrier(group=get_gloo_group())
            prof.add("barrier", time.perf_counter() - t1)
            self._send_weights(None)
            # every PP source must be done before the engines resume: parsync / bigbucket send the PP stages
            # concurrently, and a bucket that reaches an un-paused engine waits on SGLang's model-update writer
            # lock until every in-flight request has finished (1 engine, long answers: rank 0
            # resumed first and the other stage's last bucket waited 570 s per sync)
            t1b = time.perf_counter()
            dist.barrier(group=get_gloo_group())
            prof.add("sources_done", time.perf_counter() - t1b)
            t2 = time.perf_counter()
            if rank0:
                if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                    uwd.post_process_weights(False, True, self.rollout_engines)
                if kv_fresh and not flush_cache:
                    _kv_refresh(self, prof, already_paused=True)
                _rpc_get(self, _rpc(self, self.rollout_engines, "continue_generation", {}), prof, "continue")
            dist.barrier(group=get_gloo_group())
            prof.add("continue", time.perf_counter() - t2)
        total = time.perf_counter() - t_start
        self._asrt_synced_once = True
        if verify and (async_ or not no_pause):
            _verify(self)
        rec = {"rank": dist.get_rank(), "src": bool(self._is_pp_src_rank), "version": self.weight_version,
               "opts": sorted(opts), "total_s": round(total, 4)}
        rec.update({k: round(v, 4) for k, v in prof.t.items()})
        rec["bytes"] = int(prof.n.get("bcast_issue", 0))
        rec["buckets"] = int(prof.n.get("transfer_total", 0))
        if getattr(self, "_asrt_big", False):
            rec["bigbucket_mb"] = _big_mb(inplace)
        rec["kv"] = "fresh" if kv_fresh else "keep"
        if getattr(self, "_asrt_kv_refreshed", 0):
            rec["kv_refreshed"] = self._asrt_kv_refreshed
        if kv_fresh:  # cumulative: flushes that needed a retry / that failed after all retries
            rec["kv_flush_retries"] = getattr(self, "_asrt_kv_flush_retries", 0)
            rec["kv_flush_fail"] = getattr(self, "_asrt_kv_flush_fail", 0)
        if getattr(self, "_asrt_flush_skipped", False):  # slime asked for abort+flush; kept in-flight requests instead
            rec["flush_skipped"] = True
            self._asrt_flush_skipped = False
        d = getattr(self, "_asrt_direct", None)
        if d is not None:
            rec["direct"] = True
            if d.worst[0] > 0:  # slowest engine reply of this sync (seconds behind the fastest, which engine, which call)
                rec["rpc_slow"] = [round(d.worst[0], 3), d.worst[1], d.worst[2]]
            d.worst = (0.0, -1, "")
        if self._is_pp_src_rank or rank0:
            _log(rec)
            if rank0:
                print(f"[asrt] weight sync v{self.weight_version}: {json.dumps(rec)}", flush=True)
        for k, v in rec.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("rank", "version"):
                self.update_weight_metrics[f"update_weights/asrt_{k}"] = float(v)

    def _verify(self):
        """Checksum(engine after asrt sync) == checksum(engine after slime's original sync)."""
        rank0 = dist.get_rank() == 0

        def checksum():
            if not rank0:
                return None
            out = ray.get([e.check_weights.remote(action="checksum") for e in self.rollout_engines])
            return json.dumps(out, sort_keys=True, default=str)

        a = checksum()
        # re-send the same weights with slime's original code path
        saved = self.weight_version
        self.weight_version -= 1
        self._asrt_in_orig = True
        try:
            orig_update_weights(self)
        finally:
            self._asrt_in_orig = False
        self.weight_version = saved
        b = checksum()
        if rank0:
            ok = a == b
            diff = []
            if not ok:  # name the differing tensors
                try:
                    def flat(js):
                        out = {}
                        for ei, eng in enumerate(json.loads(js)):
                            for ri, rk in enumerate((eng or {}).get("ranks", []) if isinstance(eng, dict) else []):
                                for n, c in (rk.get("checksums") or {}).items():
                                    out[f"e{ei}r{ri}:{n}"] = c
                        return out
                    fa, fb = flat(a), flat(b)
                    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
                except Exception as e:
                    diff = [f"(could not parse: {e!r})"]
            print(f"[asrt] verify v{self.weight_version}: asrt-vs-slime engine checksums equal = {ok}"
                  + (f"; {len(diff)} differ: {diff[:12]}" if diff else ""), flush=True)
            _log({"rank": 0, "verify_version": self.weight_version, "equal": ok,
                  "checksum_head": (a or "")[:300], "differ": diff[:200]})
            if not ok:
                raise RuntimeError("asrt weight sync verification failed: engine checksums differ")

    def _send_dispatch(self, pbar):
        if getattr(self, "_asrt_in_orig", False):
            return orig_send(self, pbar)
        return _send_weights(self, pbar)

    Cls._send_weights = _send_dispatch
    Cls.update_weights = update_weights


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
