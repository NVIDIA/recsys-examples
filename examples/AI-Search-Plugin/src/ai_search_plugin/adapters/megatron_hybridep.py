# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HybridEP MoE dispatch + N-chunk compute/communication overlap for Megatron-Core MoE layers.

Two parts:

1. **HybridEP dispatcher** = Megatron's own ``flex`` token dispatcher with the ``hybridep`` backend (DeepEP's
   HybridEP kernels: NVLink all-to-all with the expert permute fused in).  Pure configuration:
   ``moe_token_dispatcher_type='flex'``, ``moe_flex_dispatcher_backend='hybridep'`` and
   ``moe_hybridep_pad_uneven_dispatch_inputs=True`` (ranks hold different token counts in RL).  See
   :func:`megatron_args` / :func:`config_overrides`.  The HybridEP buffer is pre-sized once
   (:func:`presize_buffer`): Megatron sizes it from the first dispatch and HybridEP reallocates when a later call is
   larger, which invalidates handles saved between a micro-batch's forward and its backward (PP/VPP interleaving,
   activation recompute -> "buffer generation" error).

2. **Chunked overlap** (:func:`install`): patches ``MoELayer.forward`` (training path only).  The layer's tokens are
   split into N row chunks; all HybridEP kernels of a layer run, in order, on one high-priority side stream, so
   chunk i+1's dispatch and chunk i's combine overlap chunk i / i+1's expert GEMMs::

       forward   comm:    d0 | d1 (|| E0) | c0 (|| E1) | d2 ... | c_{N-1}
                 compute:      E0         | E1          | ...
       backward: mirrored by the autograd engine (each node's backward runs on its forward stream; the consumer waits
                 when the grad is handed over), so chunk i's combine-backward / dispatch-backward overlap the
                 neighbouring chunks' expert backward.

   Each chunk dispatches into a fixed number of slots (``ceil(max_tokens / N)`` rounded up to 64), so no
   all-reduce of token counts / host sync is needed for uneven ranks.  The router, shared experts, the experts
   module and activation checkpointing are Megatron's own code; everything that is not the plain training path
   (inference, FP8/FP4, latent MoE, capacity factor, shared-expert overlap, expert tensor parallel, CUDA-graph
   partial capture, ...) falls back to the original forward (:func:`fallback_reason`).

Numerics: in the source project the chunked path was bitwise identical to Megatron's flex path in a single-layer test
(output, input grad and prob grad; uneven ranks, real expert GEMMs).  In full training it is not bitwise identical
because the expert GEMMs run on smaller M (different cuBLAS kernels / accumulation order) and the expert wgrads are
accumulated per chunk; differences measured there were at bf16 noise level (loss relative difference <= 2e-4).

Requirements (checked by :func:`check_environment`; nothing here is imported at module import time):

* Megatron-Core with the flex dispatcher and the HybridEP backend.  The overlap patch was developed and validated on
  Megatron-Core **0.19**; other versions are refused unless ``ASRT_HYBRIDEP_ALLOW_UNTESTED=1`` (``MoELayer.forward``
  changed after 0.19: CUDA-graph execution maps, index routing maps, ...).  The dispatcher part alone is Megatron's
  code and is not version gated.
* DeepEP with ``HybridEPBuffer`` (DeepEP ``hybrid-ep`` branch; validated at commit 4e890fd, built for the GPU
  architecture in use - intranode / NVLink only was used, NVSHMEM off).  Stock DeepEP wheels and many container
  images ship ``deep_ep`` without ``HybridEPBuffer``; then Megatron reports ``HAVE_HYBRIDEP=False`` and the
  component is not usable (``auto`` mode switches itself off, ``on`` mode raises).
* ``$CUDA_HOME/bin/nvcc`` at run time (HybridEP JIT-compiles its kernels per process into ``HYBRID_EP_CACHE_DIR``).
* The EP group inside one NVLink domain; expert tensor parallel size 1 for the chunked path.

Not ported from the source (separate items there, off by default or measured without gain): the DeepEP (non-hybrid)
chunked path (DeepEP was slower than NCCL all-to-all on the tested GPUs), re-forward reuse of the HybridEP handle
under recompute, delayed expert wgrad under the dispatch backward, dispatch-ahead ordering, the dummy-wgrad
de-duplication, the host-sync-free bounded expert path, HybridEP custom all-gather and SM-budget knobs.

Env (read by :func:`parse_spec`): ``ASRT_HYBRIDEP`` = auto | on | off, ``ASRT_HYBRIDEP_CHUNKS`` (default 2; 1 =
HybridEP dispatcher without the overlap patch), ``ASRT_HYBRIDEP_MAX_TOKENS`` (default 32768, multiple of 64: tokens
per rank per micro-batch the buffer is sized for; a larger micro-batch raises), ``ASRT_HYBRIDEP_SMS`` (optional,
``moe_flex_dispatcher_num_sms``), ``ASRT_HYBRIDEP_ALLOW_UNTESTED`` (1 = allow the patch on other Megatron versions).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import inspect
import os
from typing import Dict, List, Optional

ALIGN = 64                       # HybridEP's token alignment for its public APIs
DEFAULT_CHUNKS = 2
DEFAULT_MAX_TOKENS = 32768
TESTED_MEGATRON = ((0, 19),)     # (major, minor) versions the overlap patch was validated on
MODES = ("auto", "on", "off")

# TransformerConfig fields used by the dispatcher configuration / the chunked path.
DISPATCHER_FIELDS = ("moe_token_dispatcher_type", "moe_flex_dispatcher_backend",
                     "moe_hybridep_pad_uneven_dispatch_inputs")
OVERLAP_FIELDS = ("moe_flex_dispatcher_num_sms", "moe_permute_fusion_into_hybridep", "moe_hybridep_num_blocks_permute",
                  "moe_hybridep_num_blocks_unpermute", "moe_hybridep_num_sms_preprocessing")
DISPATCH_PARAMS = ("routing_map", "probs", "num_of_experts_per_rank", "num_permuted_tokens", "non_blocking",
                   "num_of_tokens_per_rank", "handle")
COMBINE_PARAMS = ("hidden", "probs", "handle")

_FUSED_A2A = "megatron.core.transformer.moe.fused_a2a"
_MOE_LAYER = "megatron.core.transformer.moe.moe_layer"
_TCONFIG = "megatron.core.transformer.transformer_config"


# ---------------------------------------------------------------------------------------------------- pure helpers
@dataclasses.dataclass(frozen=True)
class Spec:
    mode: str = "auto"
    chunks: int = DEFAULT_CHUNKS
    max_tokens: int = DEFAULT_MAX_TOKENS
    sms: Optional[int] = None
    allow_untested: bool = False


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def parse_spec(env=None) -> Spec:
    """Read the ``ASRT_HYBRIDEP*`` switches; raises ``ValueError`` on malformed values."""
    env = os.environ if env is None else env
    mode = (env.get("ASRT_HYBRIDEP") or "auto").strip().lower()
    mode = {"1": "on", "true": "on", "yes": "on", "0": "off", "false": "off", "no": "off"}.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"ASRT_HYBRIDEP must be one of {MODES}, got {mode!r}")
    chunks = int(env.get("ASRT_HYBRIDEP_CHUNKS") or DEFAULT_CHUNKS)
    if chunks < 1:
        raise ValueError(f"ASRT_HYBRIDEP_CHUNKS must be >= 1, got {chunks}")
    cap = int(env.get("ASRT_HYBRIDEP_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    if cap <= 0 or cap % ALIGN:
        raise ValueError(f"ASRT_HYBRIDEP_MAX_TOKENS must be a positive multiple of {ALIGN}, got {cap}")
    sms = env.get("ASRT_HYBRIDEP_SMS")
    sms = int(sms) if sms not in (None, "") else None
    if sms is not None and sms <= 0:
        raise ValueError(f"ASRT_HYBRIDEP_SMS must be > 0, got {sms}")
    return Spec(mode, chunks, cap, sms, _truthy(env.get("ASRT_HYBRIDEP_ALLOW_UNTESTED")))


def split_sizes(num_tokens: int, n: int) -> List[int]:
    """Row sizes of the N chunks (first ``num_tokens % n`` chunks one row longer).  Always N entries - every EP rank
    must issue the same number of dispatch/combine calls, so a rank with fewer than N tokens still sends empty
    chunks."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [num_tokens // n + (1 if i < num_tokens % n else 0) for i in range(n)]


def chunk_slots(max_tokens: int, n: int) -> int:
    """Per-chunk receive slots per rank: ``ceil(max_tokens / n)`` rounded up to the HybridEP alignment.  With
    ``max_tokens`` a multiple of the alignment this never exceeds the buffer size, and it bounds every chunk of a
    micro-batch with at most ``max_tokens`` rows."""
    s = -(-max_tokens // n)
    return s + (-s % ALIGN)


def megatron_args(sms: Optional[int] = None) -> List[str]:
    """Megatron command-line arguments that select the HybridEP flex dispatcher (the rest of the configuration is
    applied to the parsed args by the framework hook, see :func:`config_overrides`)."""
    a = ["--moe-token-dispatcher-type", "flex", "--moe-flex-dispatcher-backend", "hybridep"]
    if sms is not None:
        a += ["--moe-flex-dispatcher-num-sms", str(int(sms))]
    return a


def config_overrides(sms: Optional[int] = None, available_fields=None) -> Dict[str, object]:
    """TransformerConfig / args fields for the HybridEP flex dispatcher.

    ``available_fields``: names the installed Megatron's TransformerConfig has (fields it lacks are left out, except
    the two that select the dispatcher).  Constraints (Megatron 0.19 source): per-rank token counts differ in RL, so
    ``moe_hybridep_pad_uneven_dispatch_inputs``; permute is *not* fused into HybridEP (the fused variant was slower,
    and the chunked path needs the unfused API); with the newer index routing-map mode the router must produce the
    bool map the chunked path consumes."""
    o: Dict[str, object] = dict(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep",
                                moe_hybridep_pad_uneven_dispatch_inputs=True, moe_permute_fusion_into_hybridep=False,
                                moe_hybridep_routing_map_mode="bool")
    if sms is not None:
        o["moe_flex_dispatcher_num_sms"] = int(sms)
    if available_fields is not None:
        keep = set(available_fields) | {"moe_token_dispatcher_type", "moe_flex_dispatcher_backend"}
        o = {k: v for k, v in o.items() if k in keep}
    return o


def fallback_reason(layer, intermediate_tensors=None, extra_kwargs=None) -> Optional[str]:
    """Why this ``MoELayer.forward`` call must take Megatron's original path (None = chunked path applies).  Only
    depends on configuration and the training flag, so all ranks of an EP group decide the same way."""
    c = layer.config
    g = lambda name, default=None: getattr(c, name, default)  # noqa: E731
    if extra_kwargs:
        return "extra forward kwargs"
    if intermediate_tensors is not None:
        return "intermediate tensors (partial CUDA-graph capture)"
    if not getattr(layer, "training", False):
        return "not training"
    if g("moe_token_dispatcher_type") != "flex" or g("moe_flex_dispatcher_backend") != "hybridep":
        return "dispatcher is not flex/hybridep"
    if getattr(layer, "shared_expert_overlap", False):
        return "shared-expert overlap"
    if g("moe_permute_fusion_into_hybridep"):
        return "permute fused into HybridEP"
    if g("moe_expert_capacity_factor") is not None or g("moe_pad_expert_input_to_capacity"):
        return "expert capacity"
    if g("moe_expert_rank_capacity_factor") is not None:
        return "expert rank capacity"
    if g("moe_latent_size"):
        return "latent MoE"
    if g("fp8") or g("fp4"):
        return "fp8/fp4"
    if g("moe_hybridep_routing_map_mode", "bool") != "bool":
        return "index routing map"
    if g("overlap_dispatch_backward_with_experts_wgrad"):
        return "delayed expert wgrad (not supported by the chunked path)"
    if getattr(layer, "_inference_token_dispatcher", None) is not None:
        return "inference dispatcher present"
    fem = getattr(layer, "fwd_execution_map", None)
    if fem is not None and not all(k in fem for k in ("route", "expert_compute", "postprocess")):
        return "partial forward execution map"
    if getattr(getattr(layer, "token_dispatcher", None), "tp_size", 1) != 1:
        return "expert tensor parallel > 1"
    return None


# ----------------------------------------------------------------------------------------- environment detection
def megatron_version():
    """(major, minor, patch) of the importable Megatron-Core, or None."""
    try:
        pi = importlib.import_module("megatron.core.package_info")
    except Exception:  # noqa: BLE001
        return None
    try:
        return int(pi.MAJOR), int(pi.MINOR), int(pi.PATCH)
    except Exception:  # noqa: BLE001
        v = str(getattr(pi, "__version__", ""))
        parts = [int(x) for x in v.split(".")[:3] if x.isdigit()]
        return tuple(parts + [0] * (3 - len(parts))) if parts else None


def _params(fn):
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def check_environment(allow_untested: bool = False, env=None) -> dict:
    """Probe Megatron / DeepEP without touching the GPU.

    Returns ``{"version": (M, m, p) | None, "hybridep": [problems], "overlap": [problems]}``: no ``hybridep`` problems
    = Megatron's HybridEP flex dispatcher is usable; no ``overlap`` problems either = the chunked overlap patch can be
    installed as well."""
    env = os.environ if env is None else env
    hyb: List[str] = []
    ovl: List[str] = []
    ver = megatron_version()
    if ver is None:
        hyb.append("megatron.core is not importable")
    elif ver[:2] not in TESTED_MEGATRON and not allow_untested:
        ovl.append(f"Megatron-Core {'.'.join(map(str, ver))} is not a validated version for the overlap patch "
                   f"(validated: {', '.join('%d.%d' % v for v in TESTED_MEGATRON)}; ASRT_HYBRIDEP_ALLOW_UNTESTED=1 "
                   "to try)")
    fa = None
    try:
        fa = importlib.import_module(_FUSED_A2A)
    except Exception as e:  # noqa: BLE001
        hyb.append(f"{_FUSED_A2A} not importable ({type(e).__name__}: {e})")
    if fa is not None:
        if not getattr(fa, "HAVE_HYBRIDEP", False):
            hyb.append("Megatron reports HAVE_HYBRIDEP=False (deep_ep without HybridEPBuffer; needs a DeepEP build "
                       "from the hybrid-ep branch)")
        if not callable(getattr(fa, "init_hybrid_ep_buffer", None)) or not hasattr(fa, "_hybrid_ep_buffer"):
            ovl.append("fused_a2a has no init_hybrid_ep_buffer/_hybrid_ep_buffer (unsupported Megatron layout)")
    try:
        tc = importlib.import_module(_TCONFIG)
        names = {f.name for f in dataclasses.fields(tc.TransformerConfig)}
        miss = [f for f in DISPATCHER_FIELDS if f not in names]
        if miss:
            hyb.append(f"TransformerConfig lacks {miss}")
        miss = [f for f in OVERLAP_FIELDS if f not in names]
        if miss:
            ovl.append(f"TransformerConfig lacks {miss}")
    except Exception as e:  # noqa: BLE001
        hyb.append(f"TransformerConfig not inspectable ({type(e).__name__}: {e})")
    try:
        de = importlib.import_module("deep_ep")
        B = getattr(de, "HybridEPBuffer", None)
        if B is None:
            hyb.append("deep_ep has no HybridEPBuffer")
        else:
            miss = [p for p in DISPATCH_PARAMS if p not in _params(getattr(B, "dispatch_with_permute", None))]
            miss += [p for p in COMBINE_PARAMS if p not in _params(getattr(B, "combine_with_unpermute", None))]
            if miss:
                ovl.append(f"HybridEPBuffer API differs from the validated one (missing parameters {miss})")
    except Exception as e:  # noqa: BLE001
        hyb.append(f"deep_ep not importable ({type(e).__name__}: {e})")
    try:
        ml = importlib.import_module(_MOE_LAYER)
        miss = [m for m in ("forward", "route", "shared_experts_compute", "postprocess")
                if not callable(getattr(ml.MoELayer, m, None))]
        if miss:
            ovl.append(f"MoELayer lacks {miss}")
    except Exception as e:  # noqa: BLE001
        ovl.append(f"MoELayer not importable ({type(e).__name__}: {e})")
    nvcc = os.path.join(env.get("CUDA_HOME") or "/usr/local/cuda", "bin", "nvcc")
    if not os.path.exists(nvcc):
        hyb.append(f"{nvcc} not found (HybridEP JIT-compiles its kernels at run time; set CUDA_HOME)")
    return {"version": ver, "hybridep": hyb, "overlap": ovl}


def setup_runtime_env(env=None) -> None:
    """Defaults HybridEP needs at run time: CUDA_HOME (nvcc for the JIT) and a writable kernel cache directory."""
    env = os.environ if env is None else env
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("HYBRID_EP_CACHE_DIR", env.get("TMPDIR") or "/tmp")


# --------------------------------------------------------------------------------------------- buffer handling
def presize_buffer(max_tokens: int = DEFAULT_MAX_TOKENS) -> bool:
    """Make Megatron allocate the HybridEP buffer for at least ``max_tokens`` tokens per rank on first use, so it is
    never reallocated mid-step (that would invalidate saved dispatch handles).  Idempotent; returns True when the
    wrapper is in place."""
    fa = importlib.import_module(_FUSED_A2A)
    cur = getattr(fa, "init_hybrid_ep_buffer", None)
    if cur is None:
        return False
    if getattr(cur, "_asrt_presized", None) is not None:
        return True
    orig = cur

    def init_hybrid_ep_buffer(group, hidden_dim, num_tokens, *args, **kwargs):
        return orig(group, hidden_dim, max(int(num_tokens), max_tokens), *args, **kwargs)

    init_hybrid_ep_buffer._asrt_presized = max_tokens
    init_hybrid_ep_buffer._asrt_orig = orig
    fa.init_hybrid_ep_buffer = init_hybrid_ep_buffer
    return True


def _hbuffer(group, hidden_dim, num_local_experts, config, max_tokens):
    fa = importlib.import_module(_FUSED_A2A)
    if fa._hybrid_ep_buffer is None:
        sms = getattr(config, "moe_flex_dispatcher_num_sms", None)
        fa.init_hybrid_ep_buffer(group, hidden_dim, max_tokens, num_local_experts, sms, sms,
                                 getattr(config, "moe_hybridep_num_blocks_permute", None),
                                 getattr(config, "moe_hybridep_num_blocks_unpermute", None), False,
                                 getattr(config, "moe_hybridep_num_sms_preprocessing", None))
    return fa._hybrid_ep_buffer


# ------------------------------------------------------------------------------------------------ stream control
_comm_streams: Dict[int, object] = {}


class _CudaStreams:
    """Compute stream = current stream; comm stream = one high-priority side stream per device (all HybridEP calls of
    a layer go there, in order, so the shared HybridEP buffer is never used concurrently)."""

    def __init__(self, torch):
        self.t = torch
        self.cur = torch.cuda.current_stream()
        d = torch.cuda.current_device()
        if d not in _comm_streams:
            _comm_streams[d] = torch.cuda.Stream(priority=-1)
        self.comm = _comm_streams[d]

    def on_comm(self):
        return self.t.cuda.stream(self.comm)

    def comm_wait_cur(self):
        self.comm.wait_stream(self.cur)

    def cur_wait_comm(self):
        self.cur.wait_stream(self.comm)

    def comm_event(self):
        return self.comm.record_event()

    def cur_event(self):
        return self.cur.record_event()

    def cur_wait(self, ev):
        self.cur.wait_event(ev)

    def comm_wait(self, ev):
        self.comm.wait_event(ev)

    def keep(self, t, on_comm: bool):
        t.record_stream(self.comm if on_comm else self.cur)


class _NoStreams:
    """CPU tensors (unit tests): same call sequence, no streams."""

    def on_comm(self):
        return contextlib.nullcontext()

    def comm_event(self):
        return None

    cur_event = comm_event

    def __getattr__(self, name):  # comm_wait_cur, cur_wait_comm, cur_wait, comm_wait, keep
        return lambda *a, **k: None


# ---------------------------------------------------------------------------------------- autograd ops (lazy)
_OPS = None


def _ops():
    """Build the autograd functions on first use (keeps torch out of module import)."""
    global _OPS
    if _OPS is not None:
        return _OPS
    import torch

    class HDispatch(torch.autograd.Function):
        """HybridEP dispatch (+ permute) of one chunk into fixed slots; backward = combine (+ unpermute) of grads."""

        @staticmethod
        def forward(ctx, x, rmap, probs, buf, nle, slots, st):
            out, dprobs, _, tpe, handle = buf.dispatch_with_permute(
                hidden=x, routing_map=rmap, probs=probs, scaling_factor=None, num_of_experts_per_rank=nle,
                pad_multiple=None, num_permuted_tokens=None, non_blocking=False, num_of_tokens_per_rank=slots)
            tpe = tpe.to(torch.int64)
            # host read of the permuted row count (one sync per chunk, as in Megatron's blocking HybridEP path)
            st.update(handle=handle, tpe=tpe, nperm=int(tpe.sum()))
            ctx.buf, ctx.st = buf, st
            return out, dprobs

        @staticmethod
        def backward(ctx, gx, gp):
            hx, hp = ctx.buf.combine_with_unpermute(hidden=gx, probs=gp, handle=ctx.st["handle"])
            return hx, None, hp, None, None, None, None

    class HCombine(torch.autograd.Function):
        """HybridEP (unpermute +) combine of one chunk's expert output; backward = dispatch of grads with the saved
        handle."""

        @staticmethod
        def forward(ctx, y, buf, st):
            out, _ = buf.combine_with_unpermute(hidden=y, handle=st["handle"])
            ctx.buf, ctx.st = buf, st
            return out

        @staticmethod
        def backward(ctx, g):
            st = ctx.st
            d, _, _, _, _ = ctx.buf.dispatch_with_permute(hidden=g, scaling_factor=None, handle=st["handle"],
                                                          pad_multiple=None, num_permuted_tokens=st["nperm"])
            return d, None, None

    class _O:
        pass

    _O.torch, _O.HDispatch, _O.HCombine = torch, HDispatch, HCombine
    _OPS = _O
    return _OPS


def chunked_hybridep(layer, hidden, routing_map, probs, n, max_tokens=DEFAULT_MAX_TOKENS, buffer=None):
    """Routed-expert part of a MoE layer with N-chunk dispatch/combine overlap.

    hidden [T, H] (bf16), routing_map [T, E] bool, probs [T, E] (router output; E = all experts, expert TP 1).
    Returns the combined expert output [T, H].  ``buffer``: HybridEP buffer (default: Megatron's global one, created
    for ``max_tokens`` tokens per rank)."""
    ops = _ops()
    torch = ops.torch
    disp = layer.token_dispatcher
    if getattr(disp, "tp_size", 1) != 1:
        raise RuntimeError("hybridep_overlap: expert tensor parallel > 1 is not supported by the chunked path")
    T, H = hidden.shape
    if T > max_tokens:
        raise RuntimeError(f"hybridep_overlap: {T} tokens on this rank > ASRT_HYBRIDEP_MAX_TOKENS={max_tokens}; raise "
                           "it (multiple of 64) so the HybridEP buffer is never reallocated mid-step")
    slots = chunk_slots(max_tokens, n)
    probs = probs.reshape(T, -1).float()
    routing_map = routing_map.reshape(T, -1)
    if routing_map.dtype != torch.bool or routing_map.shape != probs.shape:
        raise RuntimeError(f"hybridep_overlap: expected a bool routing map shaped like probs {tuple(probs.shape)}, got "
                           f"{routing_map.dtype} {tuple(routing_map.shape)}")
    nle = disp.num_local_experts
    buf = buffer if buffer is not None else _hbuffer(disp.tp_ep_group, H, nle, layer.config, max_tokens)
    sizes = split_sizes(T, n)
    s = _CudaStreams(torch) if hidden.is_cuda else _NoStreams()
    s.comm_wait_cur()
    for t in (hidden, routing_map, probs):
        s.keep(t, True)
    with s.on_comm():   # the split (and the final concat) live on the comm stream so dispatch-backward results do
        xs, ps, rms = hidden.split(sizes), probs.split(sizes), routing_map.split(sizes)   # not stall compute early
    sts = [dict() for _ in range(n)]

    def dispatch(i):
        with s.on_comm():
            d, dp = ops.HDispatch.apply(xs[i], rms[i], ps[i], buf, nle, slots, sts[i])
            return d, dp, s.comm_event()

    def experts(i, d, dp, ev_d):
        s.cur_wait(ev_d)
        s.keep(d, False)
        s.keep(dp, False)
        y, bias = layer.experts(d, sts[i]["tpe"], dp)
        if bias is not None:
            raise RuntimeError("hybridep_overlap: expert bias is not supported")
        s.keep(y, True)
        return y, s.cur_event()

    def combine(y, ev_e, j):  # inside the comm stream context
        s.comm_wait(ev_e)
        return ops.HCombine.apply(y, buf, sts[j])

    # Creation order matters for the backward overlap: chunk i's expert chain is created before chunk i+1's
    # dispatch and chunk i's combine after it, so the autograd engine runs chunk i+1's expert backward first.
    outs, pending = [], None
    for i in range(n):
        d, dp, ev_d = dispatch(i)
        if pending is not None:   # previous chunk's combine, queued after this dispatch on the comm stream
            with s.on_comm():
                outs.append(combine(*pending))
        y, ev_e = experts(i, d, dp, ev_d)
        pending = (y, ev_e, i)
    with s.on_comm():
        outs.append(combine(*pending))
        out = torch.cat(outs)
    s.cur_wait_comm()
    s.keep(out, False)
    return out


# --------------------------------------------------------------------------------------------- MoELayer patch
_state: Dict[str, object] = {"installed": False, "calls": 0, "fallback": 0, "fallback_why": {}}


def install(num_chunks: int = DEFAULT_CHUNKS, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Patch ``MoELayer.forward`` so the training forward of flex/hybridep layers runs :func:`chunked_hybridep`.
    Idempotent for the same parameters; raises if called again with different ones."""
    if num_chunks < 2:
        raise ValueError("the overlap patch needs num_chunks >= 2 (1 = plain HybridEP dispatcher, no patch)")
    if max_tokens <= 0 or max_tokens % ALIGN:
        raise ValueError(f"max_tokens must be a positive multiple of {ALIGN}")
    if _state["installed"]:
        if (_state["num_chunks"], _state["max_tokens"]) != (num_chunks, max_tokens):
            raise RuntimeError(f"hybridep_overlap already installed with num_chunks={_state['num_chunks']} "
                               f"max_tokens={_state['max_tokens']}")
        return snapshot()
    ml = importlib.import_module(_MOE_LAYER)
    MoELayer = ml.MoELayer
    orig = MoELayer.forward

    def forward(self, hidden_states, intermediate_tensors=None, padding_mask=None, **kwargs):
        why = fallback_reason(self, intermediate_tensors, kwargs)
        if why is not None:
            _state["fallback"] += 1
            fw = _state["fallback_why"]
            fw[why] = fw.get(why, 0) + 1
            return orig(self, hidden_states, intermediate_tensors, padding_mask, **kwargs)
        if padding_mask is not None:  # [b, s] -> [s, b], as Megatron's forward does
            padding_mask = padding_mask.transpose(0, 1).bool()

        def custom_forward(hidden_states, padding_mask=None):
            shared = self.shared_experts_compute(hidden_states)
            probs, routing_map = self.route(hidden_states, padding_mask)
            # Megatron's hybridep preprocess all-reduces the token count (host sync); the chunked path uses fixed slots
            # and only needs the shape for combine_postprocess.
            self.token_dispatcher.hidden_shape = hidden_states.shape
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            out = chunked_hybridep(self, flat, routing_map, probs, num_chunks, max_tokens)
            _state["calls"] += 1
            return self.postprocess(out, shared), None

        if getattr(self, "moe_layer_recompute", False):
            tp = importlib.import_module("megatron.core.tensor_parallel")
            return tp.checkpoint(custom_forward, False, hidden_states, padding_mask)
        return custom_forward(hidden_states, padding_mask)

    forward._asrt_hybridep = True
    MoELayer.forward = forward
    _state.update(installed=True, num_chunks=num_chunks, max_tokens=max_tokens, orig=orig, cls=MoELayer)
    return snapshot()


def uninstall() -> None:
    """Restore the original ``MoELayer.forward`` (tests)."""
    if _state.get("installed"):
        _state["cls"].forward = _state["orig"]
    _state.clear()
    _state.update(installed=False, calls=0, fallback=0, fallback_why={})


def snapshot() -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _state.items() if k not in ("orig", "cls")}
