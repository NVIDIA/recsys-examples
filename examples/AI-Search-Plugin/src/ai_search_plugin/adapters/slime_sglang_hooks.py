"""Engine-side (SGLang process) hooks, loaded through ai_search_plugin/adapters/sgl_boot/sitecustomize.py.

``ASRT_SGL_HOOKS`` is a comma list:

* ``profile``  - time every weight-update receive (NCCL broadcast incl. sync) and
  model.load_weights call; append to $ASRT_LOG_DIR/sgl-weightload-<pid>.jsonl.
  Used to split slime's per-bucket ``engine_load_wait`` into transport vs. load.
* ``fastload`` - record-and-replay loading of ``flattened_bucket`` updates (default off).
* ``fastload_check`` - with fastload: on the first replay of each bucket layout, also run the original
  loader on the same bucket and compare a fingerprint of every parameter/buffer; any difference
  disables replay for that layout (logged as op=fastload_check).  Costs one original load per layout, once.
* ``inplace`` - weight updates while the engine keeps generating (no pause, no drain); pair with
  ``slime_weightsync`` option ``inplace``.  See "inplace" below.  Default off.  ``inplace_trace`` (test only)
  logs every apply with the TP rank and scheduler iteration to sgl-inplace-trace-<pid>.jsonl.
* ``nosync`` - keep SGLang's overlap scheduler effective with return_logprob.  The slime image's SGLang sampler
  checks ``bool(sampling_info.return_top_p_log_probs.any().item())`` on EVERY sampling step that returns logprobs
  (score-centering support, layers/sampler.py:321): a device->host sync on the batch being launched, so run_batch()
  blocks until that forward has finished and the CPU work of the step (result processing, scheduling, launch) no longer
  overlaps the GPU -- overlap on/off was measured equal.  With ``nosync`` the per-request flag is
  decided on the CPU when the batch's SamplingBatchInfo is built: if no request asks for top-p logprobs (slime only
  does with --use-score-centering) the tensor is set to None, so the sampler skips the check; merging with a batch
  that does have it materialises zeros, so behaviour is unchanged for requests that do ask.  Default off.
* tensor path (verl): ``update_weights_from_tensor`` with plain named tensors (load_format None; each
  tensor its own CUDA-IPC storage, HF names, fused experts) gets its own ``fastload`` variant (copy source recorded as
  input index + offset; the per-expert copies of one fused expert tensor coalesce), ``profile`` (op "tload") and
  ``inplace`` (an un-paused update skips the tokenizer's writer lock and the cache flush and is loaded on the forward
  stream between two iterations; the IPC sources stay referenced until those copies ran).  flattened_bucket / direct /
  custom loaders on that path are untouched.  Works on SGLang 0.5.20 (verl's pin, tensor path only) and 0.5.21.
* ``kvepoch`` - tag every new request's radix-cache key with a weight-update epoch, so no request reuses
  prefix KV computed with older weights (see patch_kvepoch).  Needed whenever the engine keeps its prefix cache across
  weight updates (slime flush 0 / keep_inflight / inplace, verl keepkv / inplace).
* ``kvfresh`` - (also implied by ASRT_WS_KV=fresh) the weight-sync prefix-cache flush must leave an
  empty tree or it reports failure; flush results and retract counts go to sgl-kvfresh-<pid>.jsonl (see _patch_kvfresh).
* ``tokstat`` - scheduler-side timeline: every 0.5 s bin of output tokens / prefill tokens / forward
  iterations, plus pause / continue / weight-update events, to $ASRT_LOG_DIR/sgl-tokstat-<pid>.jsonl
  (TP rank 0 only).  Measures how many tokens a weight sync costs the engine.

``inplace`` (why and how).  SGLang already runs a weight-update request on the scheduler thread between
two forward iterations, so the only things that stop generation during an un-paused update are
(1) the tokenizer's model-update WRITER lock (taken when not paused; it waits until every in-flight request
has finished = drains the engine) and (2) the scheduler thread / forward stream sitting in the NCCL receive
of each bucket (network time) and the Python loader.  With ``inplace``:
  * the tokenizer skips the writer lock for ``flattened_bucket`` updates (requests keep running);
  * the scheduler handler only *posts* the NCCL receive of bucket k into a staging buffer on a side
    stream (returns at once; the sender's broadcast completes when the bytes have arrived), and applies
    the earlier buckets of the same NCCL group, which are complete by then (one source sends bucket k only
    after bucket k-1's broadcast completed).  "Apply" = fastload replay (or the original loader) issued on
    the engine's FORWARD stream after waiting on the receive event: CUDA stream order puts it after the
    forward already in flight and before the next one, so no forward ever reads a half-copied tensor, and
    because every TP rank executes the same request at the same scheduler iteration (requests are
    broadcast to all TP ranks), all TP shards of a tensor switch at the same iteration;
  * an empty ``flattened_bucket`` update (names == []) is a COMMIT: apply everything pending, make the
    schedule stream wait for it (later checksum / control ops see the new weights), relabel the engine
    with ``weight_version``.  Tokens generated between the first applied bucket and the commit are tagged
    with SGLang's own per-request version spans as version "<old>~<new>" (meta_info["weight_versions"]),
    and counted in sgl-inplace-<pid>.jsonl.
  Memory: a staging buffer per bucket in flight (<= one per sending group + one), not a model shadow.
  Semantics: like slime ``--flush-cache-interval 0`` / asrt ``async``: a request that spans the sync has
  tokens from old, mixed (some layers new) and new weights; every token's logprob is still the true
  behaviour-policy logprob (TIS stays valid).  While paused (first sync, slime's own path) nothing changes.

Why ``fastload``: SGLang's ``model.load_weights`` is a per-HF-tensor Python loop.  For a
512-expert MoE that is ~1.5k tensors per layer (one per expert matrix), and every expert
tensor goes through a linear scan of the 1.5k-entry ``expert_params_mapping`` plus the
generic ``FusedMoE.weight_loader`` -> ~100-150 us of Python per tensor, ~75k tensors per
sync for a ~48-layer model = 7-10 s per engine GPU, all in the scheduler's single thread.
Once ``parsync`` lets the PP stages send in parallel, every engine has to run that loop for
all stages back to back, so it becomes the sync's critical path (measured engine wait
2.8 -> 8.8 s).  The bytes themselves are trivial for HBM (a 250 GB model is < 0.1 s of copy).

How: the first time a bucket with a given layout (names, dtypes, shapes) arrives, it is
received into ONE uint8 buffer (no per-tensor ``torch.empty`` + ``torch.cat`` as in
SGLang's own flattened receive) and loaded with the ORIGINAL ``model.load_weights``, under a
``TorchFunctionMode`` that records every ``copy_`` whose destination aliases a model
parameter/buffer and whose source aliases the receive buffer: (dst view, src offset/
shape/strides).  Any other in-place write to a parameter (``fill_``, ``__setitem__``,
copies from a derived temporary, ...) marks that input tensor "not replayable".
Later buckets with the same layout skip ``load_weights`` entirely: the recorded copies are
re-issued against the new buffer, coalesced into strided batches (all 512 experts of a
layer's gate/up/down -> one ``copy_`` each), and only the not-replayable input tensors go
through the original ``load_weights``.  Replay issues the same element-wise copies as the
original loader, so the result is bit-identical (checked end to end by slime_weightsync
``verify``).  A plan is dropped and re-recorded if any destination parameter's storage moved.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

TARGET = "sglang.srt.model_executor.model_runner_components.weight_updater"
SCHED_UPD = "sglang.srt.managers.scheduler_components.weight_updater"
TOK_MIXIN = "sglang.srt.managers.tokenizer_control_mixin"
SCHED = "sglang.srt.managers.scheduler"
SCHED_BRP = "sglang.srt.managers.scheduler_components.batch_result_processor"
SAMPLING = "sglang.srt.sampling.sampling_batch_info"


def _opts():
    return {s.strip() for s in os.environ.get("ASRT_SGL_HOOKS", "").split(",") if s.strip()}


def _log(rec, kind="sgl-weightload"):
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{kind}-{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------------------
# fastload: record / replay
# --------------------------------------------------------------------------------------
_MUTATING_DUNDER = {"__setitem__", "__set__", "__iadd__", "__isub__", "__imul__", "__itruediv__",
                    "__ifloordiv__", "__imod__", "__ipow__", "__iand__", "__ior__", "__ixor__",
                    "__ilshift__", "__irshift__"}


_PLANS = {}  # id(WeightUpdater) -> {bucket layout key: _Plan | None}
_CHECKED = {}  # id(WeightUpdater) -> {tensor kind} covered by a passed fastload self-check


def _layout_kinds(key):
    """ the tensor kinds of a bucket layout: (name with every digit run replaced, dtype, shape) per tensor.
    ``fastload_check`` only runs on a bucket that holds a kind no passed check has covered yet, so it tests a few
    buckets per model instead of every bucket: the per-layout check ran the original loader on every bucket of the
    first replay sync (+10-20 s, e.g. 60 buckets x 0.33 s, on the critical path); ~12.6k checks in
    testing, 0 failures.  Kinds, not whole layouts: big buckets cut layers at arbitrary tensors,
    so whole-layout signatures never repeat.  ``ASRT_FASTLOAD_CHECK=all`` restores the per-layout check."""
    import re

    def norm(n):
        return re.sub(r"\d+", "#", n)
    if len(key) == 3 and all(isinstance(k, tuple) for k in key) and key[0] and isinstance(key[0][0], str):
        names, dtypes, shapes = key  # flattened_bucket key: (names, dtypes, shapes)
        return {(norm(n), str(d), tuple(sh)) for n, d, sh in zip(names, dtypes, shapes)}
    return {(norm(e[0]),) + tuple(e[1:3]) for e in key}  # tensor path key: ((name, dtype, shape, stride), ...)


def _check_needed(updater_id, key, plan):
    """False -> mark the plan checked without running the original loader (all its tensor kinds already passed)."""
    if os.environ.get("ASRT_FASTLOAD_CHECK", "kind").strip().lower() == "all":
        return True
    if _layout_kinds(key) <= _CHECKED.get(updater_id, set()):
        plan.checked = True
        return False
    return True


def _check_passed(updater_id, key):
    _CHECKED.setdefault(updater_id, set()).update(_layout_kinds(key))
_FASTFN = []  # [fastload fn (or None), original load_weights_from_distributed] once the updater is patched


class _FlatBucket(list):
    """List of (name, view) like SGLang's receive returns, plus the receive buffer."""

    def __init__(self, flat, names, dtypes, shapes, key):
        super().__init__()
        self.flat, self.key = flat, key
        self.ranges = []  # byte range of each input tensor inside ``flat``
        off = 0
        for name, dt, shape in zip(names, dtypes, shapes):
            n = 1
            for s in shape:
                n *= int(s)
            nb = n * dt.itemsize
            self.append((name, flat[off:off + nb].view(dt).reshape(tuple(shape))))
            self.ranges.append((off, off + nb))
            off += nb


def _span(size, stride):
    return 1 + sum((s - 1) * st for s, st in zip(size, stride)) if all(s > 0 for s in size) else 0


class _Plan:
    __slots__ = ("ops", "fallback", "skip", "params", "nbytes", "ncopies", "checked")

    def valid(self, flat):
        if flat.numel() != self.nbytes:
            return False
        for p, ptr in self.params:
            if p.untyped_storage().data_ptr() != ptr:
                return False
        return True


def _build_plan(copies, fallback, n_inputs, params, nbytes):
    """copies: list of (dst_view, src_dtype, src_off, src_size, src_stride) in record order."""
    plan = _Plan()
    plan.fallback = sorted(fallback)
    plan.params = params
    plan.nbytes = nbytes
    plan.ncopies = len(copies)
    plan.checked = False
    recs = []
    for dst, sdt, soff, ssize, sstride in copies:
        recs.append((dst.untyped_storage().data_ptr(), dst.dtype, dst.storage_offset(), tuple(dst.shape),
                     tuple(dst.stride()), sdt, soff, ssize, sstride, dst))
    # Coalescing reorders copies, which is only safe if no two copies write overlapping bytes.
    by_store = {}
    for r in recs:
        es = r[1].itemsize
        by_store.setdefault(r[0], []).append((r[2] * es, (r[2] + _span(r[3], r[4])) * es))
    overlap = False
    for iv in by_store.values():
        iv.sort()
        for (a0, a1), (b0, b1) in zip(iv, iv[1:]):
            if b0 < a1:
                overlap = True
                break
        if overlap:
            break
    ops = []
    if overlap:
        for r in recs:
            ops.append((r[9], r[3], r[4], r[2], r[5], r[7], r[8], r[6]))
    else:
        groups = {}
        for r in recs:
            k = (r[0], r[1], r[3], r[4], r[5], r[7], r[8]) if r[3] == r[7] else ("single", len(groups))
            groups.setdefault(k, []).append(r)
        for lst in groups.values():
            # Partition into arithmetic progressions in (dst offset, src offset).  Writes are
            # disjoint (checked above), so any partition is exact; pick, for each start, the
            # longest run among a few candidate steps (e.g. gate/up of one expert interleave in
            # w13, but gate of expert e and e+1 form the long run).
            lst.sort(key=lambda r: r[2])
            at = {r[2]: k for k, r in enumerate(lst)}
            used = [False] * len(lst)
            for i, r0 in enumerate(lst):
                if used[i]:
                    continue
                used[i] = True
                best, bd, bs = [], 0, 0
                cand = [k for k in range(i + 1, min(len(lst), i + 9)) if not used[k]][:4]
                for k in cand:
                    d, s = lst[k][2] - r0[2], lst[k][6] - r0[6]
                    if d <= 0 or s < 0:
                        continue
                    run, last = [], r0
                    while True:
                        n = at.get(last[2] + d)
                        if n is None or used[n] or lst[n][6] - last[6] != s:
                            break
                        run.append(n)
                        last = lst[n]
                    if len(run) > len(best):
                        best, bd, bs = run, d, s
                if best:
                    for n in best:
                        used[n] = True
                    n = len(best) + 1
                    ops.append((r0[9], (n,) + r0[3], (bd,) + r0[4], r0[2], r0[5], (n,) + r0[7], (bs,) + r0[8], r0[6]))
                else:
                    ops.append((r0[9], r0[3], r0[4], r0[2], r0[5], r0[7], r0[8], r0[6]))
    plan.ops = ops
    plan.skip = n_inputs - len(plan.fallback)
    return plan


def _fingerprint(model):
    """{name: int} cheap content fingerprint of every parameter and buffer (one host sync)."""
    import torch

    names, sums = [], []
    for n, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if t.numel() == 0 or not t.is_contiguous():
            continue
        v = t.detach().view(-1)
        v = v.view(torch.int16) if (v.element_size() == 2 or (v.element_size() > 2 and v.numel() * v.element_size() % 2 == 0)) \
            and v.dtype != torch.bool else v.view(torch.uint8)
        names.append(n)
        sums.append(torch.sum(v, dtype=torch.int64))
    return dict(zip(names, torch.stack(sums).tolist())) if sums else {}


def _replay(plan, flat):
    import torch

    bases = {}
    for dst, dsize, dstride, doff, sdt, ssize, sstride, soff in plan.ops:
        b = bases.get(sdt)
        if b is None:
            es = sdt.itemsize
            b = torch.empty((0,), dtype=sdt, device=flat.device).set_(
                flat.untyped_storage(), 0, (flat.numel() // es,), (1,))
            bases[sdt] = b
        torch.as_strided(dst, dsize, dstride, doff).copy_(torch.as_strided(b, ssize, sstride, soff))


_REC_CLS = []


def _recorder_mode():
    if not _REC_CLS:
        from torch.overrides import TorchFunctionMode

        class _Rec(TorchFunctionMode):
            def __init__(self, handler):
                super().__init__()
                self.handler = handler

            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if self.handler is not None:
                    self.handler(func, args, kwargs)
                return func(*args, **kwargs)

        _REC_CLS.append(_Rec)
    return _REC_CLS[0]


def _record(model, bucket, load_fn):
    """Run the original loader on ``bucket`` while recording parameter writes."""
    import torch

    pstore = {}
    for _, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        pstore.setdefault(t.untyped_storage().data_ptr(), t)
    flat = bucket.flat
    fptr = flat.untyped_storage().data_ptr()
    starts = [a for a, _ in bucket.ranges]
    import bisect

    state = {"cur": -1, "mismatch": False}
    copies, bad, touched = [], set(), {}

    def owner(t):
        """index of the input tensor whose bytes ``t`` reads (by storage offset)."""
        b = t.storage_offset() * t.element_size()
        i = bisect.bisect_right(starts, b) - 1
        if i < 0 or b + _span(t.shape, t.stride()) * t.element_size() > bucket.ranges[i][1] and t.numel() > 0:
            return -1
        return i

    def aliases(t):
        try:
            return isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() in pstore
        except Exception:
            return False

    def handler(func, args, kwargs):
        name = getattr(func, "__name__", "")
        cur = state["cur"]
        inplace = (name.endswith("_") and not name.startswith("__")) or name in _MUTATING_DUNDER
        if args and inplace and aliases(args[0]):
            dst = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            ok = (name == "copy_" and isinstance(src, torch.Tensor) and src.device == dst.device
                  and src.untyped_storage().data_ptr() == fptr and len(args) <= 3)
            if ok:
                i = owner(src)
                if i < 0:
                    ok = False
                else:
                    if i != cur:
                        state["mismatch"] = True
                    copies.append((i, dst, src.dtype, src.storage_offset(), tuple(src.shape), tuple(src.stride())))
                    dptr = dst.untyped_storage().data_ptr()
                    touched[dptr] = pstore[dptr]
            if not ok:
                bad.add(cur)
        # other ways to write a parameter/buffer: ``out=`` (e.g. GemmaRMSNorm's
        # torch.add(w, 1, out=self.gemma_weight)) and custom ops (torch.ops.*) that get one
        out = kwargs.get("out")
        outs = out if isinstance(out, (tuple, list)) else (out,)
        if any(aliases(o) for o in outs):
            bad.add(cur)
        elif type(func).__name__ in ("OpOverload", "OpOverloadPacket") and name not in ("copy_", "copy"):
            if any(aliases(x) for x in list(args) + list(kwargs.values())):
                bad.add(cur)

    def gen():
        for i, nt in enumerate(bucket):
            state["cur"] = i
            yield nt
        state["cur"] = -1

    # The mode class lives at module level and drops its handler afterwards: a class defined in here
    # would sit in a reference cycle holding ``bucket`` (2 GB) until the next gc, and SGLang freezes gc.
    mode = _recorder_mode()(handler)
    try:
        with mode:
            load_fn(gen())
    finally:
        mode.handler = None
    if -1 in bad or (bad and state["mismatch"]):
        return None  # writes we cannot attribute: never replay this layout
    keep = [(dst, sdt, soff, ssize, sstride) for i, dst, sdt, soff, ssize, sstride in copies if i not in bad]
    params = [(t, t.untyped_storage().data_ptr()) for t in touched.values()]
    return _build_plan(keep, bad, len(bucket), params, flat.numel())


def _derived_cache_error(mod, model):
    """SGLang 0.5.21 takes the model, 0.5.20 (verl's pin) takes no argument."""
    fn = getattr(mod, "_unsupported_derived_weight_cache_error", None)
    if fn is None:
        return None
    try:
        return fn(model)
    except TypeError:
        return fn()


# --------------------------------------------------------------------------------------
# fastload on the tensor path (verl, and anything else that calls update_weights_from_tensor with
# plain named tensors): every input tensor is its own CUDA-IPC storage, so a copy's source is
# recorded as (input index, element offset relative to that input's first byte).  Copies of one
# input are still coalesced (all 512 experts of a fused ``experts.gate_up_proj`` -> one strided copy
# per destination half).
# --------------------------------------------------------------------------------------
class _TBucket(list):
    """(name, tensor) list of one update_weights_from_tensor call + its layout key."""

    def __init__(self, named):
        super().__init__(named)
        self.key = tuple((n, str(t.dtype), tuple(t.shape), tuple(t.stride())) for n, t in named)
        self.spans = []  # (storage ptr, first byte, end byte) per input
        for _, t in named:
            b = t.storage_offset() * t.element_size()
            self.spans.append((t.untyped_storage().data_ptr(), b, b + _span(t.shape, t.stride()) * t.element_size()))


class _TPlan:
    __slots__ = ("by_src", "fallback", "skip", "params", "ncopies", "nops", "checked")

    def valid(self, bucket):
        for p, ptr in self.params:
            if p.untyped_storage().data_ptr() != ptr:
                return False
        return True


def _record_tensors(model, bucket, load_fn):
    """Original loader on ``bucket`` (a _TBucket) under the recorder; returns a _TPlan or None."""
    import torch

    pstore = {}
    for _, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        pstore.setdefault(t.untyped_storage().data_ptr(), t)
    by_ptr = {}
    for i, (ptr, b0, b1) in enumerate(bucket.spans):
        by_ptr.setdefault(ptr, []).append((b0, b1, i))
    state = {"cur": -1, "mismatch": False}
    copies, bad, touched = [], set(), {}

    def owner(t):
        lst = by_ptr.get(t.untyped_storage().data_ptr())
        if not lst:
            return -1, 0
        b = t.storage_offset() * t.element_size()
        e = b + _span(t.shape, t.stride()) * t.element_size()
        for b0, b1, i in lst:
            if b0 <= b and (e <= b1 or t.numel() == 0):
                if (b - b0) % t.element_size():
                    return -1, 0
                return i, (b - b0) // t.element_size()
        return -1, 0

    def aliases(t):
        try:
            return isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() in pstore
        except Exception:
            return False

    def handler(func, args, kwargs):
        name = getattr(func, "__name__", "")
        cur = state["cur"]
        inplace = (name.endswith("_") and not name.startswith("__")) or name in _MUTATING_DUNDER
        if args and inplace and aliases(args[0]):
            dst = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            ok = name == "copy_" and isinstance(src, torch.Tensor) and src.device == dst.device and len(args) <= 3
            if ok:
                i, rel = owner(src)
                if i < 0:
                    ok = False
                else:
                    if i != cur:
                        state["mismatch"] = True
                    copies.append((i, dst, src.dtype, rel, tuple(src.shape), tuple(src.stride())))
                    dptr = dst.untyped_storage().data_ptr()
                    touched[dptr] = pstore[dptr]
            if not ok:
                bad.add(cur)
        out = kwargs.get("out")
        outs = out if isinstance(out, (tuple, list)) else (out,)
        if any(aliases(o) for o in outs):
            bad.add(cur)
        elif type(func).__name__ in ("OpOverload", "OpOverloadPacket") and name not in ("copy_", "copy"):
            if any(aliases(x) for x in list(args) + list(kwargs.values())):
                bad.add(cur)

    def gen():
        for i, nt in enumerate(bucket):
            state["cur"] = i
            yield nt
        state["cur"] = -1

    mode = _recorder_mode()(handler)
    try:
        with mode:
            load_fn(gen())
    finally:
        mode.handler = None
    if -1 in bad or (bad and state["mismatch"]):
        return None
    keep = [c for c in copies if c[0] not in bad]
    plan = _TPlan()
    plan.fallback = sorted(bad)
    plan.params = [(t, t.untyped_storage().data_ptr()) for t in touched.values()]
    plan.ncopies = len(keep)
    plan.checked = False
    plan.skip = len(bucket) - len(plan.fallback)
    # writes of different inputs never coalesce; if any two copies overlap, keep record order (one op each)
    allr = [(dst.untyped_storage().data_ptr(), dst.dtype.itemsize, dst.storage_offset(), tuple(dst.shape),
             tuple(dst.stride())) for _, dst, *_ in keep]
    by_store = {}
    for ptr, es, off, size, stride in allr:
        by_store.setdefault(ptr, []).append((off * es, (off + _span(size, stride)) * es))
    overlap = False
    for iv in by_store.values():
        iv.sort()
        if any(b0 < a1 for (a0, a1), (b0, b1) in zip(iv, iv[1:])):
            overlap = True
            break
    if overlap:
        plan.by_src = [(i, [(dst, tuple(dst.shape), tuple(dst.stride()), dst.storage_offset(), sdt, ssize, sstride, rel)])
                       for i, dst, sdt, rel, ssize, sstride in keep]
    else:
        groups = {}
        for i, dst, sdt, rel, ssize, sstride in keep:
            groups.setdefault(i, []).append((dst, sdt, rel, ssize, sstride))
        plan.by_src = []
        for i, lst in groups.items():
            sub = _build_plan(lst, (), 0, [], 0)  # coalescing only; the offsets are relative to input i
            plan.by_src.append((i, sub.ops))
    plan.nops = sum(len(ops) for _, ops in plan.by_src)
    return plan


def _replay_tensors(plan, bucket):
    import torch

    for i, ops in plan.by_src:
        t = bucket[i][1]
        st = t.untyped_storage()
        b0 = bucket.spans[i][1]
        bases = {}
        for dst, dsize, dstride, doff, sdt, ssize, sstride, soff in ops:
            es = sdt.itemsize
            b = bases.get(sdt)
            if b is None:
                b = torch.empty((0,), dtype=sdt, device=t.device).set_(st, 0, (st.nbytes() // es,), (1,))
                bases[sdt] = b
            torch.as_strided(dst, dsize, dstride, doff).copy_(torch.as_strided(b, ssize, sstride, b0 // es + soff))


_TPLANS = {}  # id(WeightUpdater) -> {layout key: _TPlan | None}
_KEEP = []  # inplace: [(cuda event, tensors)] kept alive until the forward-stream copies reading them are done
_TLS_INPL = [False]  # set by the scheduler hook while it runs an in-place tensor update


def _prune_keep():
    while _KEEP and _KEEP[0][0].query():
        _KEEP.pop(0)


def _patch_tensor_path(mod, C, profile, fastload, check):
    import torch

    orig = C.update_weights_from_tensor

    def _load_tensors(self, named):
        """fastload for plain named tensors (load_format None); returns (ok, msg, info)."""
        model = self.get_model()
        bucket = _TBucket(named)
        cache = _TPLANS.setdefault(id(self), {})
        plan = cache.get(bucket.key, False)
        if plan is False or (plan is not None and not plan.valid(bucket)):
            plan = _record_tensors(model, bucket, model.load_weights)
            cache[bucket.key] = plan
            return True, "Success", ("record", plan)
        if plan is None:
            model.load_weights(list(bucket))
            return True, "Success", ("orig", None)
        _replay_tensors(plan, bucket)
        if plan.fallback:
            model.load_weights([bucket[i] for i in plan.fallback])
        if check and not plan.checked and _check_needed(id(self), bucket.key, plan):
            fp0 = _fingerprint(model)
            model.load_weights(list(bucket))
            fp1 = _fingerprint(model)
            diff = [n for n in fp1 if fp0.get(n) != fp1[n]]
            plan.checked = True
            if diff:
                cache[bucket.key] = None
                print(f"[asrt] fastload (tensor path) self-check FAILED for a {len(bucket)}-tensor bucket, disabled "
                      f"for this layout; differing: {diff[:8]}", file=sys.stderr, flush=True)
                _log({"t": time.time(), "op": "fastload_check", "path": "tensor", "ok": False, "diff": diff[:50]})
                return True, "Success", ("check_fail", None)
            _check_passed(id(self), bucket.key)
            _log({"t": time.time(), "op": "fastload_check", "path": "tensor", "ok": True, "tensors": len(bucket)})
            return True, "Success", ("replay_checked", plan)
        return True, "Success", ("replay", plan)

    def update_weights_from_tensor(self, named_tensors, load_format=None):
        if load_format is not None:  # flattened_bucket (slime colocated), direct, custom loaders: untouched
            return orig(self, named_tensors, load_format)
        t0 = time.perf_counter()
        info = None
        inpl = _TLS_INPL[0]
        if fastload or inpl:
            error = _derived_cache_error(mod, self.get_model())
            if error is not None:
                return False, error
            mod.monkey_patch_torch_reductions()
            self._assert_weight_cache_inactive("update_weights_from_tensor")
            dev = torch.get_device_module(self.device).current_device()
            named = [(n, mod._unwrap_tensor(t, tp_rank=self.tp_rank, device=dev)) for n, t in named_tensors]
            try:
                if fastload:
                    ok, msg, info = _load_tensors(self, named)
                else:
                    self.get_model().load_weights(named)
                    ok, msg = True, "Success"
            except Exception as e:
                _TPLANS.get(id(self), {}).clear()
                msg = (f"Failed to update parameter online: {e}. The full weights of the ModelRunner are "
                       f"partially updated. Please discard the whole weights.")
                mod.logger.error(msg)
                return False, msg
            if inpl:  # the copies are queued on the forward stream: keep their sources alive until they ran
                ev = torch.cuda.Event()
                ev.record()
                _KEEP.append((ev, [t for _, t in named]))
            out = (ok, msg)
        else:
            out = orig(self, named_tensors, load_format)
        if profile:
            torch.cuda.synchronize()
            rec = {"t": time.time(), "op": "tload", "s": round(time.perf_counter() - t0, 5),
                   "tensors": len(named_tensors), "inplace": inpl}
            if info is not None:
                rec["mode"] = info[0]
                if info[1] is not None:
                    rec.update(copies=info[1].ncopies, ops=info[1].nops, fallback=len(info[1].fallback))
            _log(rec)
        return out

    C.update_weights_from_tensor = update_weights_from_tensor


def patch_weight_updater(mod):
    import torch

    C = getattr(mod, "WeightUpdater", None)
    if C is None or getattr(C, "_asrt_patched", False):
        return
    opts = _opts()
    profile, fastload, check = "profile" in opts, "fastload" in opts, "fastload_check" in opts
    if not hasattr(C, "receive_weights_from_distributed"):  # SGLang 0.5.20 (verl's pin): tensor path only
        if hasattr(C, "update_weights_from_tensor") and ({"profile", "fastload", "inplace"} & opts):
            _patch_tensor_path(mod, C, profile, fastload, check)
        C._asrt_patched = True
        print(f"[asrt] sgl weight-updater hooks installed (tensor path only): {sorted(opts)} (pid {os.getpid()})",
              file=sys.stderr, flush=True)
        return
    recv, load = C.receive_weights_from_distributed, C.load_weights_from_distributed

    def receive_weights_from_distributed(self, **kw):
        t0 = time.perf_counter()
        if fastload and kw.get("load_format") == "flattened_bucket":
            group_name = kw["group_name"]
            assert group_name in self._model_update_group, f"Group {group_name} not initialized"
            names, shapes = list(kw["names"]), [tuple(int(x) for x in s) for s in kw["shapes"]]
            dtypes = [d if isinstance(d, torch.dtype) else getattr(torch, str(d).replace("torch.", "")) for d in kw["dtypes"]]
            nbytes = 0
            for dt, shape in zip(dtypes, shapes):
                n = 1
                for s in shape:
                    n *= s
                nbytes += n * dt.itemsize
            flat = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            torch.distributed.broadcast(flat, src=0, group=self._model_update_group[group_name])
            out = _FlatBucket(flat, names, dtypes, shapes, (tuple(names), tuple(map(str, dtypes)), tuple(shapes)))
        else:
            out = recv(self, **kw)
        if profile:
            torch.cuda.synchronize()
            n = sum(t.numel() * t.element_size() for _, t in out)
            _log({"t": time.time(), "op": "recv", "s": round(time.perf_counter() - t0, 5), "bytes": n,
                  "tensors": len(out), "fmt": kw.get("load_format"), "group": kw.get("group_name")})
        return out

    def _fast(self, bucket):
        self._assert_weight_cache_inactive("update_weights_from_distributed")
        error = _derived_cache_error(mod, self.get_model())
        if error is not None:
            return False, error, None
        model = self.get_model()
        cache = _PLANS.setdefault(id(self), {})  # WeightUpdater is a slots dataclass: no attributes of ours
        plan = cache.get(bucket.key, False)
        try:
            if plan is False or (plan is not None and not plan.valid(bucket.flat)):
                plan = _record(model, bucket, model.load_weights)
                cache[bucket.key] = plan
                return True, "Succeeded to update parameter online.", ("record", plan)
            if plan is None:  # layout known to be not replayable
                model.load_weights(bucket)
                return True, "Succeeded to update parameter online.", ("orig", None)
            _replay(plan, bucket.flat)
            if plan.fallback:
                model.load_weights([bucket[i] for i in plan.fallback])
            if check and not plan.checked and _check_needed(id(self), bucket.key, plan):
                # self-test on the first replay of each layout (only buckets with an unchecked tensor kind, _layout_kinds): the original loader
                # on the same bucket must not change any parameter/buffer; otherwise never replay this layout again.
                fp0 = _fingerprint(model)
                model.load_weights(bucket)
                fp1 = _fingerprint(model)
                diff = [n for n in fp1 if fp0.get(n) != fp1[n]]
                plan.checked = True
                if diff:
                    cache[bucket.key] = None
                    print(f"[asrt] fastload self-check FAILED for a {len(bucket)}-tensor bucket, disabled for this "
                          f"layout; differing: {diff[:8]}", file=sys.stderr, flush=True)
                    _log({"t": time.time(), "op": "fastload_check", "ok": False, "diff": diff[:50]})
                    return True, "Succeeded to update parameter online.", ("check_fail", None)
                _check_passed(id(self), bucket.key)
                _log({"t": time.time(), "op": "fastload_check", "ok": True, "tensors": len(bucket)})
                return True, "Succeeded to update parameter online.", ("replay_checked", plan)
            return True, "Succeeded to update parameter online.", ("replay", plan)
        except Exception as e:
            cache.pop(bucket.key, None)
            msg = (f"Failed to update parameter online: {e}. The full weights of the ModelRunner are "
                   f"partially updated. Please discard the whole weights.")
            mod.logger.error(msg)
            return False, msg, None

    def load_weights_from_distributed(self, named_tensors):
        t0 = time.perf_counter()
        info = None
        if fastload and isinstance(named_tensors, _FlatBucket):
            ok, msg, info = _fast(self, named_tensors)
            out = (ok, msg)
        else:
            out = load(self, named_tensors)
        if profile:
            torch.cuda.synchronize()
            rec = {"t": time.time(), "op": "load", "s": round(time.perf_counter() - t0, 5), "tensors": len(named_tensors)}
            if info is not None:
                mode, plan = info
                rec["mode"] = mode
                if plan is not None:
                    rec.update(copies=plan.ncopies, ops=len(plan.ops), fallback=len(plan.fallback))
            _log(rec)
        return out

    C.receive_weights_from_distributed = receive_weights_from_distributed
    C.load_weights_from_distributed = load_weights_from_distributed
    if hasattr(C, "update_weights_from_tensor") and ({"profile", "fastload", "inplace"} & opts):
        _patch_tensor_path(mod, C, profile, fastload, check)
    C._asrt_patched = True
    _FASTFN[:] = [(_fast if fastload else None), load]
    print(f"[asrt] sgl weight-updater hooks installed: {sorted(opts)} (pid {os.getpid()})", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# inplace: weight updates without pausing / draining the engine; tokstat: token timeline
# --------------------------------------------------------------------------------------
_CNT = {"out": 0, "pre": 0, "it": 0}  # this scheduler process: output tokens / prefill tokens / iterations


def _is_rank0():
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


class _TokStat:
    """0.5 s bins of scheduler output; written by TP rank 0 only."""

    BIN = 0.5

    def __init__(self):
        self.on = None
        self.t_bin = None
        self.last = dict(_CNT)
        self.buf = []

    def enabled(self):
        if self.on is None:
            self.on = "tokstat" in _opts() and _is_rank0()
        return self.on

    def tick(self, force=False):
        if not self.enabled():
            return
        # called BEFORE a batch's counts are added: a bin holds the batches processed inside it; a gap
        # with no batch at all (paused / idle) has no bin = zero tokens
        now = time.time()
        if self.t_bin is None:
            self.t_bin = now
            self.last = dict(_CNT)
            return
        if now < self.t_bin + self.BIN and not force:
            return
        d = {k: _CNT[k] - self.last[k] for k in _CNT}
        self.buf.append({"t": round(self.t_bin, 3), "dt": round(min(now, self.t_bin + self.BIN) - self.t_bin, 3), **d})
        self.last = dict(_CNT)
        self.t_bin = now
        if len(self.buf) >= 40 or force:
            self.flush()

    def event(self, ev, **kw):
        if not self.enabled():
            return
        self.tick(force=True)
        self.buf.append({"ev": ev, "t": round(time.time(), 4), **kw})
        self.flush()

    def flush(self):
        d = os.environ.get("ASRT_LOG_DIR")
        if d and self.buf:
            try:
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"sgl-tokstat-{os.getpid()}.jsonl"), "a") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in self.buf))
            except OSError:
                pass
        self.buf = []


_TS = _TokStat()


_KVF = {"flush": 0, "flush_fail": 0, "retract_calls": 0, "retracted": 0}


def _kvfresh_on():
    return "kvfresh" in _opts() or os.environ.get("ASRT_WS_KV", "").strip().lower() == "fresh"


def _tree_tokens(tc):
    """Tokens held by the prefix tree (evictable + protected), or None when the cache exposes neither counter
    (unknown cache type: the emptiness check cannot be made - reported as unchecked, never as a silent pass)."""
    n, seen = 0, False
    for f in ("evictable_size", "protected_size"):
        try:
            v = getattr(tc, f)()
            n += int(sum(v) if isinstance(v, (tuple, list)) else v)
            seen = True
        except Exception:
            pass
    return n if seen else None


def _tree_children(tc):
    try:
        ch = tc.root_node.children
        return len(ch)
    except Exception:
        return 0


def _sched_busy(sch):
    out = {}
    for k, f in (("running", lambda: len(sch.running_batch.reqs)), ("waiting", lambda: len(sch.waiting_queue)),
                 ("chunked", lambda: int(sch.chunked_req is not None)),
                 ("last_batch", lambda: int(sch.last_batch is not None and not sch.last_batch.is_empty())),
                 ("result_queue", lambda: len(sch.result_queue) if getattr(sch, "enable_overlap", False) else 0),
                 ("grammar", lambda: len(sch.grammar_manager.grammar_queue)),
                 ("paused", lambda: int(bool(sch._engine_paused)))):
        try:
            out[k] = f()
        except Exception:
            pass
    return out


def _patch_kvfresh(S):
    """ ``kvfresh`` (also on with ASRT_WS_KV=fresh): make the weight-sync flush verifiable.

    * ``flush_cache``: after SGLang's own reset, the prefix tree must hold 0 tokens (evictable + protected) and the root
      no children; otherwise the call reports failure (HTTP 400 -> the trainer re-retracts and retries).  Every call
      (TP rank 0; failures on every rank) goes to sgl-kvfresh-<pid>.jsonl with the tree size before/after and, on
      failure, what kept the engine busy; failures also print an ERROR line.
    * ``pause_generation(mode="retract")``: counts the requests sent back to the waiting queue (in-flight requests whose
      KV is rebuilt under the new weights)."""
    if getattr(S, "_asrt_kvfresh", False) or not _kvfresh_on():
        return
    orig_flush, orig_pause = S.flush_cache, S.pause_generation

    def flush_cache(self, *a, **k):
        tc = getattr(self, "tree_cache", None)
        before, busy = _tree_tokens(tc), _sched_busy(self)
        ok = bool(orig_flush(self, *a, **k))
        after, kids = _tree_tokens(tc), _tree_children(tc)
        why = ""
        if after is None:  # cannot verify this cache type: loud, counted, the flush result itself still stands
            _KVF["unchecked"] = _KVF.get("unchecked", 0) + 1
            if _KVF["unchecked"] == 1:
                print(f"[asrt] ERROR kvfresh: cannot check tree emptiness for {type(tc).__name__} "
                      f"(no evictable_size/protected_size); flush results are UNCHECKED (pid {os.getpid()})",
                      file=sys.stderr, flush=True)
        if ok and ((after or 0) or kids):
            ok, why = False, f"tree not empty after reset: {after} tokens, {kids} root children"
        elif not ok:
            why = f"engine not idle: {busy}"
        _KVF["flush"] += 1
        if not ok:
            _KVF["flush_fail"] += 1
            print(f"[asrt] ERROR kvfresh flush_cache failed (pid {os.getpid()}): {why}", file=sys.stderr, flush=True)
        if not ok or _is_rank0():
            _log({"t": time.time(), "op": "flush", "ok": ok, "checked": after is not None, "tree_before": before,
                  "tree_after": after,
                  "root_children": kids, "busy": busy, "why": why, **_KVF}, kind="sgl-kvfresh")
        return ok

    def pause_generation(self, recv_req):
        mode = getattr(recv_req, "mode", None)
        try:
            w0 = len(self.waiting_queue)
        except Exception:
            w0 = None
        out = orig_pause(self, recv_req)
        if mode == "retract" and w0 is not None:
            try:
                n = len(self.waiting_queue) - w0
            except Exception:
                n = 0
            _KVF["retract_calls"] += 1
            _KVF["retracted"] += max(0, n)
            if _is_rank0():
                _log({"t": time.time(), "op": "retract", "n": n, "waiting": w0 + n,
                      "tree": _tree_tokens(getattr(self, "tree_cache", None)), **_KVF}, kind="sgl-kvfresh")
        return out

    S.flush_cache = flush_cache
    S.pause_generation = pause_generation
    S._asrt_kvfresh = True
    print(f"[asrt] sgl kvfresh installed: checked flush_cache + retract count (pid {os.getpid()})",
          file=sys.stderr, flush=True)


def _r3keep_on():
    return _kvfresh_on() and os.environ.get("ASRT_KVFRESH_R3KEEP", "1").strip().lower() not in ("0", "off", "no", "false")


_R3K = {"reqs": 0, "rows": 0, "rows_flipped": 0, "spliced": 0}


def _r3_stash(reqs, req_to_token_pool):
    """ before a retract frees a request's KV: keep the routed-expert rows R3 already holds for it.

    SGLang captures routed experts per KV SLOT (state_capturer get_topk reads host_cache.buffer[req_to_token[...]]) and
    gathers them only when the request finishes.  A retract frees the slots; the re-prefill after the sync recomputes the
    routing of every token of the call so far UNDER THE NEW WEIGHTS and writes it into new slots, so the rows returned
    for the tokens generated (or prefilled) before the retract are no longer the routes those tokens were computed with
    - R3 then replays routes that do not match their rollout logprobs (measured: train/rollout logp diff rises under
    kvfresh on a retract-heavy MoE run; flat without retracts).  Rows [routed_experts_start_len, current length - 1)
    are kept here and spliced back at collection (first computation wins, like an in-place pause)."""
    try:
        from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
        cap = get_global_experts_capturer()
    except Exception:
        return
    if cap is None:
        return
    for req in reqs:
        try:
            if not getattr(req, "return_routed_experts", False):
                continue
            idx = req.kv.req_pool_idx
            if idx is None:
                continue
            cur = len(req.origin_input_ids) + len(req.output_ids)
            start = int(getattr(req, "routed_experts_start_len", 0) or 0)
            # only rows whose KV was actually computed: a request retracted in the middle of a chunked prefill has
            # [0, kv_committed_len) filled; req_to_token beyond it still holds a previous occupant's slots (review)
            hi = cur - 1
            kc = getattr(getattr(req, "kv", None), "kv_committed_len", None)
            if kc is not None:
                hi = min(hi, int(kc))
            if hi <= start:
                continue
            rows = cap.get_topk(req_pool_idx=idx, seqlen=hi + 1, req_to_token_pool=req_to_token_pool, start_len=start)
            prev = getattr(req, "_asrt_r3_keep", None)
            if prev is not None:   # retracted again: the oldest computation of each row wins
                if len(prev) >= len(rows):   # e.g. retracted again mid chunked RE-prefill: nothing new to add
                    rows = prev
                else:
                    import torch
                    rows = torch.cat([prev, rows[len(prev):]], dim=0)
            req._asrt_r3_keep = rows.clone()
            _R3K["reqs"] += 1
        except Exception as e:  # never break a retract
            print(f"[asrt] kvfresh r3keep stash failed: {e!r}", file=sys.stderr, flush=True)


def patch_brp(mod):
    """ splice the rows kept at retract into the routed experts gathered at request finish."""
    C = getattr(mod, "BatchResultProcessor", None) or next(
        (v for v in vars(mod).values() if isinstance(v, type) and hasattr(v, "_maybe_collect_routed_experts")), None)
    if C is None or getattr(C, "_asrt_r3keep", False) or not _r3keep_on():
        return
    orig = C._maybe_collect_routed_experts

    def _maybe_collect_routed_experts(self, req):
        out = orig(self, req)
        keep = getattr(req, "_asrt_r3_keep", None)
        if keep is not None and getattr(req, "routed_experts", None) is not None:
            try:
                re_ = req.routed_experts
                k = min(len(keep), re_.shape[0])
                kk = keep[:k].to(dtype=re_.dtype)
                # route SETS per (token, layer) (top-k order ignored): how many the re-prefill would have changed
                a = kk.reshape(k, -1, kk.shape[-1]).sort(dim=-1).values
                b = re_[:k].reshape(k, -1, re_.shape[-1]).sort(dim=-1).values
                diff_tl = (a != b).any(dim=-1)                       # [rows, layers]
                # how many of the top-k experts differ in a changed (token, layer) set: 1-of-k = a near-tie flip
                # (drift / numerics), most of k = a different token's row (misalignment) - review of 063
                inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(dim=-1).sum(dim=-1)   # [rows, layers]
                ndiff = (a.shape[-1] - inter)[diff_tl]
                hist = [int((ndiff == j).sum()) for j in range(1, a.shape[-1] + 1)]
                _R3K["ndiff_hist"] = [x + y for x, y in zip(_R3K.get("ndiff_hist", [0] * len(hist)), hist)]
                zero_rows = int((kk.reshape(k, -1) == 0).all(dim=1).sum())
                zero_final = int((re_[:k].reshape(k, -1) == 0).all(dim=1).sum())   # before the splice overwrites them
                flipped = int(diff_tl.any(dim=1).sum())
                re_[:k] = kk
                _R3K["rows"] += k
                _R3K["rows_flipped"] += flipped
                _R3K["spliced"] += 1
                _R3K["tl"] = _R3K.get("tl", 0) + int(diff_tl.numel())
                _R3K["tl_flipped"] = _R3K.get("tl_flipped", 0) + int(diff_tl.sum())
                if _is_rank0():
                    _log({"t": time.time(), "op": "r3keep", "req_rows": k, "req_rows_route_changed": flipped,
                          "req_token_layers": int(diff_tl.numel()), "req_token_layers_changed": int(diff_tl.sum()),
                          "req_ndiff_hist": hist, "req_zero_rows_kept": zero_rows,
                          "req_zero_rows_final": zero_final,
                          "total_rows": int(re_.shape[0]), **{"cum_" + x: v for x, v in _R3K.items()}},
                         kind="sgl-kvfresh")
            except Exception as e:
                print(f"[asrt] kvfresh r3keep splice failed: {e!r}", file=sys.stderr, flush=True)
            req._asrt_r3_keep = None
        return out

    C._maybe_collect_routed_experts = _maybe_collect_routed_experts
    C._asrt_r3keep = True
    print(f"[asrt] sgl kvfresh r3keep installed: retracted requests keep their routed-expert rows (pid {os.getpid()})",
          file=sys.stderr, flush=True)


def patch_scheduler(mod):
    """Count output / prefill tokens per forward iteration (needed by inplace's window stats and tokstat)."""
    opts = _opts()
    S = getattr(mod, "Scheduler", None)
    if S is not None:
        _patch_kvfresh(S)
    ra = getattr(mod, "retract_all", None)
    if ra is not None and _r3keep_on() and not getattr(ra, "_asrt_r3keep", False):
        def retract_all(*a, **k):   # scheduler.pause_generation(retract) -> retract_all(reqs=..., req_to_token_pool=...)
            reqs = k.get("reqs", a[0] if a else None)
            pool = k.get("req_to_token_pool", a[1] if len(a) > 1 else None)
            if reqs and pool is not None:
                _r3_stash(reqs, pool)
            return ra(*a, **k)
        retract_all._asrt_r3keep = True
        mod.retract_all = retract_all
    if S is None or getattr(S, "_asrt_patched", False) or not ({"inplace", "tokstat"} & opts):
        return
    orig_pbr, orig_pause, orig_cont = S.process_batch_result, S.pause_generation, S.continue_generation

    def process_batch_result(self, batch, result):
        _TS.tick()
        if _KEEP:
            _prune_keep()
        try:
            n = len(batch.reqs)
            _CNT["it"] += 1
            fm = batch.forward_mode
            if fm.is_decode():
                _CNT["out"] += n
            elif fm.is_extend():
                _CNT["out"] += n  # each request whose prefill (chunk) ran; ~1 sampled token each
                _CNT["pre"] += int(getattr(batch, "extend_num_tokens", 0) or 0)
        except Exception:
            pass
        return orig_pbr(self, batch, result)

    def pause_generation(self, recv_req):
        _TS.event("pause", mode=getattr(recv_req, "mode", None))
        return orig_pause(self, recv_req)

    def continue_generation(self, recv_req):
        _TS.event("continue")
        return orig_cont(self, recv_req)

    S.process_batch_result = process_batch_result
    S.pause_generation = pause_generation
    S.continue_generation = continue_generation
    S._asrt_patched = True
    print(f"[asrt] sgl scheduler hooks installed: {sorted(opts & {'inplace', 'tokstat'})} (pid {os.getpid()})",
          file=sys.stderr, flush=True)


class _Inplace:
    __slots__ = ("pending", "recv_stream", "win_t0", "snap", "old", "tok0", "it0", "stats")

    def __init__(self):
        self.pending = []  # [(group, bucket, recv_event, t_posted)] in arrival order
        self.recv_stream = None
        self.win_t0 = None
        self.snap = None
        self.old = None
        self.tok0 = self.it0 = 0
        self.stats = {}


_INPL_FMT = "asrt_inplace"  # load_format the tokenizer gives un-paused flattened_bucket updates
_INPLACE = {}  # id(SchedulerWeightUpdaterManager) -> _Inplace (the manager is a slots dataclass)


def _live_reqs(sch):
    live = set()
    try:
        live |= set(sch.collect_inflight_reqs())
    except Exception:
        pass
    try:
        live |= set(sch.waiting_queue)
    except Exception:
        pass
    if getattr(sch, "chunked_req", None) is not None:
        live.add(sch.chunked_req)
    return live


def patch_sched_updater(mod):
    import torch

    C = getattr(mod, "SchedulerWeightUpdaterManager", None)
    opts = _opts()
    if C is None or getattr(C, "_asrt_patched", False) or not ({"inplace", "tokstat"} & opts):
        return
    inplace = "inplace" in opts
    trace = "inplace_trace" in opts
    orig = C.update_weights_from_distributed
    Out = mod.UpdateWeightsFromDistributedReqOutput
    maxpend = int(os.environ.get("ASRT_SGL_INPLACE_PEND", "6"))

    def _fwd(self):
        sch = self.scheduler
        s = getattr(sch, "forward_stream", None) if sch is not None else None
        if s is None:
            s = getattr(getattr(self.tp_worker, "model_runner", None), "forward_stream", None)
        return s if s is not None else torch.cuda.current_stream()

    def _open_window(self, st):
        from sglang.srt.runtime_context import get_serving
        from sglang.srt.utils.weight_versions import record_weight_version_events

        st.win_t0 = time.time()
        st.tok0, st.it0 = _CNT["out"], _CNT["it"]
        st.stats = {"buckets": 0, "bytes": 0, "host_apply_s": 0.0, "forced": 0, "post_s": 0.0}
        sch = self.scheduler
        live = _live_reqs(sch) if sch is not None else set()
        st.snap = {id(r) for r in live if getattr(r, "output_ids", None)}
        st.old = get_serving().weight_version
        record_weight_version_events(live, old_version=st.old)  # closes the pure-old segment of live requests
        _TS.event("inplace_open", live=len(live))

    def _apply(self, st, item):
        """Load one received bucket on the forward stream (after its receive event)."""
        group, bucket, ev, _ = item
        fast, load = (_FASTFN + [None, None])[:2]
        t0 = time.perf_counter()
        if st.win_t0 is None:
            _open_window(self, st)
        fwd = _fwd(self)
        ok, msg = True, "Succeeded to update parameter online."
        with torch.cuda.stream(fwd):
            fwd.wait_event(ev)
            for _, runner in self._select_runners("all"):
                wu = runner.weight_updater
                if fast is not None:
                    ok, msg, _ = fast(wu, bucket)
                elif load is not None:
                    ok, msg = load(wu, bucket)
                else:
                    ok, msg = wu.load_weights_from_distributed(list(bucket))
                if not ok:
                    break
            bucket.flat.record_stream(fwd)
        st.stats["host_apply_s"] = st.stats.get("host_apply_s", 0.0) + time.perf_counter() - t0
        if trace:  # TP consistency check: every TP rank must apply each bucket at the same iteration
            import torch.distributed as dist

            _log({"op": "apply", "rank": dist.get_rank() if dist.is_initialized() else 0, "it": _CNT["it"],
                  "first": bucket[0][0] if len(bucket) else "", "n": len(bucket), "group": group},
                 kind="sgl-inplace-trace")
        if not ok:
            raise RuntimeError(msg)

    def _commit(self, st, version):
        from sglang.srt.utils.weight_versions import record_weight_version_events

        while st.pending:
            _apply(self, st, st.pending.pop(0))
        torch.cuda.current_stream().wait_stream(_fwd(self))  # control ops after this see the new weights
        sch = self.scheduler
        if version is None or sch is None:
            return
        rec = {"t": time.time(), "op": "commit", "version": version}
        if st.win_t0 is not None:
            live = _live_reqs(sch)
            record_weight_version_events(live, old_version=f"{st.old}~{version}")
            rec.update(old=st.old, window_s=round(time.time() - st.win_t0, 4),
                       tokens_in_window=_CNT["out"] - st.tok0, iters_in_window=_CNT["it"] - st.it0,
                       live_reqs=len(live), spanning_reqs=sum(1 for r in live if id(r) in st.snap),
                       **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.stats.items()})
            st.win_t0 = None
        sch.record_weight_version_change(new_version=version)
        if _is_rank0():
            _log(rec, kind="sgl-inplace")
        _TS.event("commit", version=version, window_s=rec.get("window_s"))

    def update_weights_from_distributed(self, recv_req):
        # Only requests the tokenizer tagged "asrt_inplace" (= sent while the tokenizer was NOT paused) take
        # the in-place path.  The scheduler's own _engine_paused is not enough: slime's abort-mode pause
        # (flush syncs) never reaches the scheduler.
        sch = self.scheduler
        mine = inplace and getattr(recv_req, "load_format", None) == _INPL_FMT
        paused = sch is None or getattr(sch, "_engine_paused", False)
        if not mine:
            st = _INPLACE.get(id(self))
            if st is not None and st.pending:  # keep order: nothing of ours may land after a later update
                _commit(self, st, None)
            t0 = time.time()
            out = orig(self, recv_req)
            _TS.event("upd", dur=round(time.time() - t0, 4), n=len(recv_req.names), paused=bool(paused))
            return out
        st = _INPLACE.setdefault(id(self), _Inplace())
        try:
            if not recv_req.names:  # commit
                _commit(self, st, recv_req.weight_version)
                return Out(success=True, message="asrt-inplace commit")
            t0 = time.perf_counter()
            group = recv_req.group_name
            keep = []
            for it in st.pending:  # the same sender only sends bucket k after k-1 has fully arrived
                if it[0] == group:
                    _apply(self, st, it)
                else:
                    keep.append(it)
            st.pending = keep
            while len(st.pending) >= maxpend:  # bound staging memory (GPU then waits for that receive)
                _apply(self, st, st.pending.pop(0))
                st.stats["forced"] = st.stats.get("forced", 0) + 1
            target = self.tp_worker.model_runner.weight_updater
            assert group in target._model_update_group, f"Group {group} not initialized"
            names, shapes = list(recv_req.names), [tuple(int(x) for x in sh) for sh in recv_req.shapes]
            dtypes = [d if isinstance(d, torch.dtype) else getattr(torch, str(d).replace("torch.", ""))
                      for d in recv_req.dtypes]
            nbytes = 0
            for dt, shape in zip(dtypes, shapes):
                n = 1
                for x in shape:
                    n *= x
                nbytes += n * dt.itemsize
            if st.recv_stream is None:
                st.recv_stream = torch.cuda.Stream(device=target.device)
            with torch.cuda.stream(st.recv_stream):
                flat = torch.empty(nbytes, dtype=torch.uint8, device=target.device)
                work = torch.distributed.broadcast(flat, src=0, group=target._model_update_group[group], async_op=True)
                work.wait()  # recv stream waits for the NCCL stream; the host does not
                ev = torch.cuda.Event()
                ev.record()
            key = (tuple(names), tuple(map(str, dtypes)), tuple(shapes))
            st.pending.append((group, _FlatBucket(flat, names, dtypes, shapes, key), ev, time.time()))
            if st.win_t0 is None:
                _open_window(self, st)
            st.stats["buckets"] = st.stats.get("buckets", 0) + 1
            st.stats["bytes"] = st.stats.get("bytes", 0) + nbytes
            st.stats["post_s"] = st.stats.get("post_s", 0.0) + time.perf_counter() - t0
            return Out(success=True, message="asrt-inplace queued")
        except Exception as e:
            mod.logger.error(f"asrt inplace weight update failed: {e!r}")
            return Out(success=False, message=f"asrt-inplace failed: {e!r}")

    C.update_weights_from_distributed = update_weights_from_distributed
    orig_t = getattr(C, "update_weights_from_tensor", None)
    twait = os.environ.get("ASRT_SGL_INPLACE_WAIT", "1") != "0"
    if orig_t is not None:
        def update_weights_from_tensor(self, recv_req):
            # verl path: plain named tensors over CUDA IPC.  An update the tokenizer tagged "asrt_inplace" (sent while
            # not paused) is loaded on the FORWARD stream: after the forward in flight, before the next one (same TP
            # iteration on every rank), no pause, no drain; the IPC sources stay referenced until those copies ran.
            mine = inplace and getattr(recv_req, "load_format", None) == _INPL_FMT
            t0 = time.time()
            if not mine:
                out = orig_t(self, recv_req)
                _TS.event("upd", dur=round(time.time() - t0, 4), n=len(recv_req.serialized_named_tensors),
                          path="tensor", paused=bool(getattr(self.scheduler, "_engine_paused", False)))
                return out
            recv_req.load_format = None
            recv_req.flush_cache = False
            fwd = _fwd(self)
            fwd.wait_stream(torch.cuda.current_stream())
            _TLS_INPL[0] = True
            try:
                with torch.cuda.stream(fwd):
                    out = orig_t(self, recv_req)
            finally:
                _TLS_INPL[0] = False
            if twait:
                # wait for this bucket's copies (= at most the forward in flight + the copies) so the CUDA-IPC sources are
                # released before the reply: the sender's clones of the next bucket then never coexist with this one's
                # (verl at mem 0.85 ran out of memory on the rollout GPU without it).  ASRT_SGL_INPLACE_WAIT=0: release
                # them asynchronously at the next scheduler iteration instead (one bucket more of sender memory).
                ev = torch.cuda.Event()
                ev.record(fwd)
                ev.synchronize()
            _prune_keep()
            _TS.event("upd", dur=round(time.time() - t0, 4), path="tensor", inplace=True)
            return out

        C.update_weights_from_tensor = update_weights_from_tensor
    C._asrt_patched = True
    print(f"[asrt] sgl scheduler weight-update hooks installed: inplace={inplace} maxpend={maxpend} "
          f"(pid {os.getpid()})", file=sys.stderr, flush=True)


def patch_tokenizer(mod):
    """inplace: un-paused ``flattened_bucket`` updates skip the model-update writer lock (no drain)."""
    if "inplace" not in _opts():
        return
    M = getattr(mod, "TokenizerControlMixin", None)
    if M is None or getattr(M, "_asrt_patched", False):
        return
    orig = M.update_weights_from_distributed

    async def update_weights_from_distributed(self, obj, request=None):
        if getattr(obj, "load_format", None) != "flattened_bucket":
            return await orig(self, obj, request)
        async with self.is_pause_cond:
            paused = self.is_pause
        if paused:
            return await orig(self, obj, request)
        self.auto_create_handle_loop()
        obj.load_format = _INPL_FMT  # tells the scheduler hook: in place (not paused, no writer lock)
        results = await self.update_weights_from_distributed_communicator(obj)
        success, message = mod.FanOutCommunicator.merge_results(results)
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
        return success, message

    M.update_weights_from_distributed = update_weights_from_distributed
    orig_t = getattr(M, "update_weights_from_tensor", None)
    if orig_t is not None:
        async def update_weights_from_tensor(self, obj, request=None):
            # plain named-tensor updates (load_format None, e.g. verl) sent while NOT paused: no writer lock (it waits
            # for every running request = drains the engine), no cache flush (it asserts an idle engine).
            if getattr(obj, "load_format", None) is not None or getattr(obj, "abort_all_requests", False):
                return await orig_t(self, obj, request)
            async with self.is_pause_cond:
                paused = self.is_pause
            if paused:
                return await orig_t(self, obj, request)
            self.auto_create_handle_loop()
            obj.serialized_named_tensors = mod.normalize_serialized_named_tensor_payloads(obj.serialized_named_tensors) \
                if hasattr(mod, "normalize_serialized_named_tensor_payloads") else obj.serialized_named_tensors
            obj.load_format = _INPL_FMT
            obj.flush_cache = False
            results = await self.update_weights_from_tensor_communicator(obj)
            success, message = mod.FanOutCommunicator.merge_results(results)
            if success and obj.weight_version is not None:
                self._update_weight_version_if_provided(obj.weight_version)
            return success, message

        M.update_weights_from_tensor = update_weights_from_tensor
    M._asrt_patched = True
    print(f"[asrt] sgl tokenizer inplace hook installed (pid {os.getpid()})", file=sys.stderr, flush=True)


def _wants_top_p_logprobs(req):
    cp = getattr(req.sampling_params, "custom_params", None)
    return isinstance(cp, dict) and bool(cp.get("return_top_p_log_probs", False))


def patch_sampling_info(mod):
    """``nosync``: drop the all-False return_top_p_log_probs mask (decided on the CPU) so the sampler does not sync."""
    if "nosync" not in _opts():
        return
    C = mod.SamplingBatchInfo
    if getattr(C, "_asrt_nosync", False):
        return
    import dataclasses
    fields = {f.name for f in dataclasses.fields(C)} if dataclasses.is_dataclass(C) else set(vars(C))
    if "return_top_p_log_probs" not in fields:
        # stock SGLang (e.g. 0.5.20 pinned by verl): no top-p logprob mask, so no per-step sync to remove
        print(f"[asrt] sgl nosync: not needed on this SGLang (no return_top_p_log_probs) (pid {os.getpid()})",
              file=sys.stderr, flush=True)
        C._asrt_nosync = True
        return
    import torch
    orig_adjust = C.adjusted_from_schedule_batch
    orig_merge = C.merge_batch

    def adjusted_from_schedule_batch(self, batch, vocab_size):
        orig_adjust(self, batch, vocab_size)
        if self.return_top_p_log_probs is not None and not any(_wants_top_p_logprobs(r) for r in batch.reqs):
            self.return_top_p_log_probs = None

    def merge_batch(self, other):
        a, b = self.return_top_p_log_probs, other.return_top_p_log_probs
        if a is None and b is not None:
            self.return_top_p_log_probs = torch.zeros(len(self), dtype=torch.bool, device=b.device)
        elif b is None and a is not None:
            other.return_top_p_log_probs = torch.zeros(len(other), dtype=torch.bool, device=a.device)
        return orig_merge(self, other)

    C.adjusted_from_schedule_batch = adjusted_from_schedule_batch
    C.merge_batch = merge_batch
    C._asrt_nosync = True
    print(f"[asrt] sgl nosync hook installed (pid {os.getpid()})", file=sys.stderr, flush=True)


TOK_MGR = "sglang.srt.managers.tokenizer_manager"
_KVE = {"epoch": 0, "tagged": 0, "updates": 0}


def patch_kvepoch(mod):
    """ ``kvepoch``: no request may reuse prefix-cache KV computed with older weights.

    SGLang keeps its radix (prefix) cache across weight updates unless the update flushes it (slime only flushes at
    ``--flush-cache-interval`` syncs and pauses with abort; with flush 0 / asrt keep_inflight / inplace / verl keepkv it
    never does).  A multi-turn agent's next turn then reuses the previous turns' KV that the OLD weights computed (hit rate
    ~0.7), while the trainer recomputes the whole sequence with the new weights: rollout/train logp diff grows step by step
    (measured: multi-turn search agent 0.017 -> 0.12-0.33 with entropy collapse; a 4B dense run 0.016 -> 0.054
    over 30 steps).
    Fix without interrupting anyone: the tokenizer stamps every new request's radix ``extra_key`` with a cache epoch
    that advances on every weight-update call (before and after it).  RadixKey matching requires equal extra_key, so a
    request started after an update can only hit KV inserted by requests started after it; old-epoch KV is never matched
    again and is evicted by LRU.  In-flight requests keep their own KV (their tokens are already accounted to the earliest
    version).  Cost: one extra prefill of each trajectory's context after each update (measured per run)."""
    if "kvepoch" not in _opts():
        return
    C = getattr(mod, "TokenizerManager", None)
    if C is None or getattr(C, "_asrt_kvepoch", False):
        return
    import dataclasses

    orig_create = getattr(C, "_create_tokenized_object", None)
    if orig_create is None:
        print(f"[asrt] sgl kvepoch NOT installed: no TokenizerManager._create_tokenized_object (pid {os.getpid()})",
              file=sys.stderr, flush=True)
        return

    def _create_tokenized_object(self, obj, *a, **k):
        out = orig_create(self, obj, *a, **k)
        tag = f"@wv{_KVE['epoch']}"
        try:
            if hasattr(out, "extra_key"):
                cur = getattr(out, "extra_key", None) or ""
                try:
                    out.extra_key = cur + tag
                except (dataclasses.FrozenInstanceError, AttributeError):
                    out = dataclasses.replace(out, extra_key=cur + tag)
                _KVE["tagged"] += 1
        except Exception as e:  # never break a request for the cache tag
            print(f"[asrt] sgl kvepoch tag failed: {e!r}", file=sys.stderr, flush=True)
        return out

    C._create_tokenized_object = _create_tokenized_object
    names = [n for n in ("update_weights_from_distributed", "update_weights_from_tensor", "update_weights_from_disk",
                         "update_weights_from_ipc") if hasattr(C, n)]

    def make(name):
        async def upd(self, *a, **k):
            _KVE["epoch"] += 1          # requests created during the update window: their own epoch
            _KVE["updates"] += 1
            try:
                return await getattr(super(C, self), name)(*a, **k)   # resolved per call: keeps later mixin patches
            finally:
                _KVE["epoch"] += 1      # requests after the update: never share KV with anything older
                if _KVE["updates"] % int(os.environ.get("ASRT_KVEPOCH_LOG_EVERY", 50)) == 1:
                    _log({"t": time.time(), "op": "kvepoch", **_KVE}, kind="sgl-kvepoch")
        upd.__name__ = name
        return upd

    for n in names:
        if n in C.__dict__:  # defined on TokenizerManager itself (not the mixin): wrap that function directly
            inner = C.__dict__[n]

            def make_direct(inner, n=n):
                async def upd(self, *a, **k):
                    _KVE["epoch"] += 1
                    _KVE["updates"] += 1
                    try:
                        return await inner(self, *a, **k)
                    finally:
                        _KVE["epoch"] += 1
                return upd
            setattr(C, n, make_direct(inner))
        else:
            setattr(C, n, make(n))
    C._asrt_kvepoch = True
    print(f"[asrt] sgl kvepoch installed: radix extra_key += @wv<epoch>, epoch advances on {names} (pid {os.getpid()})",
          file=sys.stderr, flush=True)


def _patchers():
    return {
        SAMPLING: patch_sampling_info,
        TARGET: patch_weight_updater,
        SCHED_UPD: patch_sched_updater,
        TOK_MIXIN: patch_tokenizer,
        TOK_MGR: patch_kvepoch,
        SCHED: patch_scheduler,
        SCHED_BRP: patch_brp,
    }


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fn = _patchers().get(name)
        if fn is None:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                fn(module)
            except Exception as e:
                print(f"[asrt] sgl patch of {name} failed: {e!r}", file=sys.stderr)

        spec.loader.exec_module = exec_module
        return spec


def register_import_hooks():
    pending = False
    for name, fn in _patchers().items():
        if name in sys.modules:
            try:
                fn(sys.modules[name])
            except Exception as e:
                print(f"[asrt] sgl patch of {name} failed: {e!r}", file=sys.stderr)
        else:
            pending = True
    if pending and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
