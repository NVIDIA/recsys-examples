#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""host_sync_probe.py -- census of host syncs per training step in DynamicEmb.

A "host sync" is any point where the CPU thread blocks on the GPU or reads a
GPU-computed value: torch.cuda.synchronize / Stream.synchronize /
Event.synchronize, cudaStreamSynchronize / cudaDeviceSynchronize /
cudaEventSynchronize / synchronous cudaMemcpy, cudaMemcpyAsync D2H into
pageable memory, .item(), .cpu(), .tolist(), int()/bool() of a CUDA tensor,
nonzero, boolean-mask indexing, masked_select, unique, repeat_interleave
without output_size, H2D copies from pageable memory with non_blocking=False,
and cudaMalloc / cudaFree / cudaHostAlloc on the hot path.

The tool builds BatchedDynamicEmbeddingTablesV2 modules in several storage
layouts, drives them through scenarios (all-hit, miss, miss+evict, eval,
growth), and counts the host syncs of every measured step.  It uses only
public APIs of the module that existed before the sync removal, so the same
file measures a tree before and after it.  test_sync_ratchet.py holds the
census of that "before" tree as budgets.json and lets it only go down;
test_no_host_syncs.py is the acceptance test that every in-tree configuration
reads zero.

How it counts (three instruments, merged):
  (a) torch.cuda.set_sync_debug_mode("warn") with a warnings hook: every
      PyTorch-visible sync raised from Python becomes a warning; the hook walks
      the Python stack, so each sync gets a file:line:function call site.
  (b) torch.profiler (CPU+CUDA): CUDA runtime events cudaStreamSynchronize,
      cudaDeviceSynchronize, cudaEventSynchronize, cudaMemcpy*, cudaMalloc,
      cudaFree, cudaHostAlloc, and GPU-side "Memcpy DtoH (Device -> Pageable)"
      events, counted inside per-iteration annotation windows on every thread
      (the backward runs on an autograd worker thread).  This is the authority
      for the totals, and the only instrument that sees raw C++ syncs.
  (c) Python-level interposition of torch.cuda.synchronize, Stream.synchronize
      and Event.synchronize (sync-debug mode does not flag torch.cuda.synchronize
      or Event.synchronize on this torch).
  To give profiler-only events a call site, every function of
  dynamicemb_extensions is wrapped in a record_function range that carries the
  Python call site.  A sync inside such a range is "cpp": [aten] if an ATen op
  (a .cpu()/.item() issued from C++) encloses it, [raw] if it is a direct CUDA
  runtime call.  ATen-visible syncs inside a pybind function are NOT delivered
  as Python warnings (they go to C++ stderr), which is why (a) alone misses them.

Usage (from corelib/dynamicemb, with the built dynamicemb importable):
  python test/unit_tests/no_host_sync/host_sync_probe.py --selftest
  python test/unit_tests/no_host_sync/host_sync_probe.py [--configs a,b]
      [--scenarios x,y] [--iters N] [--json out.json] [--assert-zero] [--list]
  from host_sync_probe import measure, check_zero
  report = measure(configs=["cache_storage"], scenarios=["all_hit"], iters=8)
  assert not check_zero(report)

The external-PS configurations import PyDictStorage from
test/test_batched_dynamic_embedding_tables_v2.py; that directory is added to
sys.path on demand.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import contextlib
import dataclasses
import fnmatch
import functools
import gc
import json
import linecache
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# What is ALLOWED to sync, with the reason; --assert-zero / check_zero() skip it.
# This is the list of "inherent syncs that remain" of the sync-removal design
# (section 5), cut down to what the measured step of a probe configuration can
# contain:
#   I1  an external parameter server (cache_ps*): the Storage interface takes
#       request-sized host tensors, so DynamicEmb reads the dedup count and the
#       miss counts in front of every PS call (every step, hits included), and
#       the PS stand-in's own host code adds more.
#   I3  DEMB_DETERMINISM_MODE (*_determ) in the training scenarios: the wave
#       loop of a deterministic insert needs host counts and the miss path is
#       gated by one blocking read in that mode.
#   I4  table growth events (*_growth, *_noevict in the miss scenario, and the
#       growth scenario): new buffers are sized on the host and
#       ExtendableBuffer.extend / HostVMMTensor.extend synchronize the device.
# Section 5 also lists what no probe step contains, so it needs no entry here:
# public APIs that return host data (the cache_metrics getter, pops, dumps,
# checkpoint IO), construction and teardown, user-defined admitters and
# storages, the TorchRec boundary and allocator growth (the 'al' kind is bounded
# by test_sync_ratchet.py).
# Keys are (config glob, scenario glob): the whole case is excused, or
# (config glob, scenario glob, call-site file glob): only the syncs whose call
# site is in a matching file are excused (the rest of the case must be zero).
# A scenario matches both modes (plain / prefetch).  Edit this table when a sync
# is declared inherent.
# --------------------------------------------------------------------------
PS_STANDIN = "*test_batched_dynamic_embedding_tables_v2.py"
_GROWTH_WHY = (
    "table growth is an event: the new tables are sized on the host and the "
    "device is synchronized while values move (ExtendableBuffer.extend, "
    "HostVMMTensor.extend)"
)
_PS_WHY = (
    "external parameter server: its Storage interface takes request-sized host "
    "tensors (DynamicEmb reads the dedup and miss counts before each PS call) and "
    "the stand-in itself reads CUDA tensors on the host (PyDictStorage.find/insert)"
)
_DETERM_WHY = (
    "DEMB_DETERMINISM_MODE: the wave count of a deterministic insert is a host "
    "loop bound, and the miss path is gated by one blocking read in this mode"
)
INHERENT: Dict[Tuple[str, ...], str] = {
    ("*", "growth"): _GROWTH_WHY,
    ("*_growth", "miss"): _GROWTH_WHY,
    ("*_noevict", "miss"): _GROWTH_WHY,
    ("cache_ps*", "*"): _PS_WHY,
}
for _scenario in ("all_hit", "miss", "miss_evict", "tier2_hit"):
    INHERENT[("*_determ", _scenario)] = _DETERM_WHY
del _scenario

KINDS = ("ss", "ds", "es", "ms", "pd", "al")
KIND_DOC = {
    "ss": "cudaStreamSynchronize",
    "ds": "cudaDeviceSynchronize",
    "es": "cudaEventSynchronize",
    "ms": "synchronous cudaMemcpy*",
    "pd": "D2H memcpy into pageable memory with no stream sync after it",
    "al": "cudaMalloc/cudaFree/cudaHostAlloc/cudaMallocHost/cudaFreeHost",
}

# CUDA runtime / driver API names -> kind.  Verified against the profiler of
# torch 2.13.0a0+nv26.07 (cudaStreamSynchronize etc. are exposed as cuda_runtime
# events); driver names are listed in case CUPTI reports them.
SYNC_KIND = {
    "cudaStreamSynchronize": "ss",
    "cuStreamSynchronize": "ss",
    "cudaDeviceSynchronize": "ds",
    "cuCtxSynchronize": "ds",
    "cudaEventSynchronize": "es",
    "cuEventSynchronize": "es",
    "cudaMemcpy": "ms",
    "cudaMemcpy2D": "ms",
    "cudaMemcpy3D": "ms",
    "cudaMemcpyToSymbol": "ms",
    "cudaMemcpyFromSymbol": "ms",
    "cuMemcpyDtoH_v2": "ms",
    "cuMemcpyHtoD_v2": "ms",
    "cuMemcpyDtoD_v2": "ms",
    "cuMemcpy": "ms",
    "cudaMalloc": "al",
    "cudaMalloc3D": "al",
    "cudaMallocPitch": "al",
    "cudaMallocManaged": "al",
    "cudaMallocHost": "al",
    "cudaHostAlloc": "al",
    "cudaFree": "al",
    "cudaFreeHost": "al",
    "cuMemAlloc_v2": "al",
    "cuMemFree_v2": "al",
    "cuMemHostAlloc": "al",
    "cuMemFreeHost": "al",
    "cuMemCreate": "al",
    "cuMemRelease": "al",
    "cuMemMap": "al",
    "cuMemUnmap": "al",
    "cuMemAddressReserve": "al",
    "cuMemAddressFree": "al",
}
BLOCKING_KINDS = ("ss", "ds", "es", "ms")  # what PyTorch's sync debug mode can flag
# runtime calls that may sit between a pageable D2H memcpy and its stream sync
NOISE_RT = {
    "cudaStreamIsCapturing",
    "cudaPeekAtLastError",
    "cudaGetLastError",
    "cudaGetDevice",
    "cudaSetDevice",
    "cudaDeviceGetAttribute",
    "cudaEventRecordWithFlags",
    "cudaEventRecord",
    "cudaStreamGetCaptureInfo_v2",
    "cudaStreamGetCaptureInfo",
}
SYNC_AFTER_COPY = {
    "cudaStreamSynchronize",
    "cuStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
}

_THIS_FILE = os.path.abspath(__file__)
WARN_TEXT = "called a synchronizing CUDA operation"


# --------------------------------------------------------------------------
# Configurations and scenarios
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Cfg:
    name: str
    layout: str  # cache | cache_ps | hbm | hybrid | host
    pooling: str = "none"  # none | sum
    optimizer: str = "sgd"  # sgd | adam
    admit: bool = False  # FrequencyAdmissionStrategy(threshold=2)
    retain: bool = False  # evicted_item_mode=RETAIN_KEY
    lfu: bool = False  # score_strategy=LFU, frequency counters passed in
    growth: bool = False  # init_capacity < max_capacity
    metrics: bool = False  # set_record_cache_metrics(True)
    determ: bool = False  # DEMB_DETERMINISM_MODE=ON
    noevict: bool = (
        False  # score_strategy=NO_EVICTION (tables expand instead of evicting)
    )
    doc: str = ""


def _build_configs() -> "collections.OrderedDict[str, Cfg]":
    out: "collections.OrderedDict[str, Cfg]" = collections.OrderedDict()

    def add(name, layout, doc, **kw):
        out[name] = Cfg(name=name, layout=layout, doc=doc, **kw)

    base = (
        ("cache", "cache_storage", "HBM cache + DynamicEmbStorage (host values)"),
        ("cache_ps", "cache_ps", "HBM cache + storage + external PS (PyDictStorage)"),
        ("hbm", "hbm_only", "all values in HBM, no cache"),
        ("hybrid", "hybrid", "HybridStorage: HBM tier + host tier"),
        (
            "host",
            "host_only",
            "DynamicEmbStorage with host values (local_hbm_for_values=0)",
        ),
    )
    for layout, name, doc in base:
        add(name, layout, doc)
        add(name + "_sum", layout, doc + ", SUM pooling", pooling="sum")
    for layout, name, doc in base:
        if layout == "cache_ps":
            continue
        add(name + "_admit", layout, doc + " + FrequencyAdmissionStrategy", admit=True)
        add(name + "_retain", layout, doc + " + RETAIN_KEY", retain=True)
        add(name + "_lfu", layout, doc + " + LFU score strategy", lfu=True)
    add(
        "cache_storage_adam",
        "cache",
        "cache layout with Adam (SGD is the default)",
        optimizer="adam",
    )
    add("hbm_only_adam", "hbm", "HBM-only layout with Adam", optimizer="adam")
    add(
        "cache_storage_metrics",
        "cache",
        "cache layout + set_record_cache_metrics(True)",
        metrics=True,
    )
    add(
        "cache_storage_growth",
        "cache",
        "cache layout, storage init_capacity < max_capacity",
        growth=True,
    )
    add("hbm_only_growth", "hbm", "HBM-only, init_capacity < max_capacity", growth=True)
    add(
        "host_only_growth",
        "host",
        "host-only, init_capacity < max_capacity",
        growth=True,
    )
    add(
        "cache_storage_noevict",
        "cache",
        "cache layout, storage NO_EVICTION, init_capacity < max_capacity",
        growth=True,
        noevict=True,
    )
    add(
        "hbm_only_noevict",
        "hbm",
        "HBM-only NO_EVICTION, init_capacity < max_capacity",
        growth=True,
        noevict=True,
    )
    add(
        "cache_storage_determ",
        "cache",
        "cache layout with DEMB_DETERMINISM_MODE=ON",
        determ=True,
    )
    add("hbm_only_determ", "hbm", "HBM-only with DEMB_DETERMINISM_MODE=ON", determ=True)
    return out


CONFIGS = _build_configs()

SCENARIO_DOC = collections.OrderedDict(
    [
        ("all_hit", "same batch repeated after warm-up: every key resident"),
        ("miss", "fresh NEW keys every step, tiers not yet full"),
        (
            "miss_evict",
            "fresh keys every step from a key space >> capacity: evictions every step",
        ),
        (
            "tier2_hit",
            "keys live only in the 2nd tier (storage under a cache, host tier of a hybrid, PS): "
            "every step misses the top tier and hits the 2nd",
        ),
        ("eval", "module.eval() + no_grad, keys resident"),
        ("eval_miss", "module.eval() + no_grad, fresh unknown keys every step"),
        (
            "growth",
            "fresh keys push a table past its load factor: expand_if_need fires",
        ),
    ]
)
EVAL_SCENARIOS = ("eval", "eval_miss")
MODES = ("plain", "prefetch")

# batch geometry: 2 tables, 3 features [0, 0, 1], B samples, KPB keys per bag
FEATURE_TABLE_MAP = [0, 0, 1]
BATCH = 96
KPB = 2
DIM = 32
ROW_N = {0: 2 * BATCH * KPB, 1: 1 * BATCH * KPB}  # key slots per table per step
UNIQ_FRAC = 0.9
UNIQ = {t: int(UNIQ_FRAC * n) for t, n in ROW_N.items()}  # 345, 172
ADMIT_THRESHOLD = 2


def scenarios_of(cfg: Cfg) -> List[str]:
    if cfg.growth:
        return ["all_hit", "miss", "growth"]
    out = ["all_hit", "miss", "miss_evict", "eval", "eval_miss"]
    if cfg.layout in ("cache", "cache_ps", "hybrid") and not cfg.admit:
        out.insert(
            3, "tier2_hit"
        )  # (with admission the old keys would have to be re-admitted first)
    if cfg.layout == "cache_ps":
        # the module refuses eval() with an external storage (see forward())
        out = [x for x in out if x not in EVAL_SCENARIOS]
    return out


def plan_of(cfg: Cfg, scenario: str) -> Dict[str, Any]:
    """Capacities (rows per table) for one (config, scenario).

    The cache / HBM tier must hold the working set for all_hit and miss (no
    evictions) and be much smaller than the stream of fresh keys for miss_evict.
    """
    evict = scenario in ("miss_evict", "tier2_hit")
    p: Dict[str, Any] = {"max_capacity": 65536, "hbm_rows": None, "init_capacity": None}
    if cfg.growth:
        p["max_capacity"] = 16384
        p["init_capacity"] = 1024
        if cfg.layout == "cache":
            p["hbm_rows"] = 8192  # the cache must not fill while the storage grows
        return p
    if cfg.layout == "cache":
        p["max_capacity"], p["hbm_rows"] = (16384, 2048) if evict else (65536, 8192)
    elif cfg.layout == "cache_ps":
        p["max_capacity"], p["hbm_rows"] = (4096, 1024) if evict else (65536, 8192)
    elif cfg.layout in ("hbm", "host"):
        p["max_capacity"] = 4096 if evict else 65536
    elif cfg.layout == "hybrid":
        p["max_capacity"], p["hbm_rows"] = (16384, 2048) if evict else (65536, 8192)
    return p


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
_SHORT_CACHE: Dict[str, str] = {}


def _short(path: str) -> str:
    s = _SHORT_CACHE.get(path)
    if s is not None:
        return s
    p = path.replace("\\", "/")
    out = None
    for marker in ("site-packages/", "dist-packages/"):
        i = p.rfind(marker)
        if i >= 0:
            out = p[i + len(marker) :]
            break
    if out is None:
        out = "/".join(p.split("/")[-2:])
    _SHORT_CACHE[path] = out
    return out


def _is_skip_frame(filename: str) -> bool:
    """Frames the call site must not stop at: the probe, the warnings machinery
    and torch's own Python (the user code is what called into it)."""
    if filename == _THIS_FILE or filename.endswith("host_sync_probe.py"):
        return True
    if filename.endswith("/warnings.py") or filename.startswith("<frozen"):
        return True
    p = filename.replace("\\", "/")
    if (
        "site-packages/torch/" in p or "dist-packages/torch/" in p
    ) and "torchrec" not in p:
        return True
    return False


def _resolve(
    frame, chain_len: int = 3
) -> Tuple[Tuple[str, int, str], List[Tuple[str, int, str]]]:
    """(site, chain) for the innermost non-skipped frame and its callers."""
    site = None
    chain: List[Tuple[str, int, str]] = []
    f = frame
    guard = 0
    while f is not None and guard < 200:
        guard += 1
        fn = f.f_code.co_filename
        if not _is_skip_frame(fn):
            ent = (fn, f.f_lineno, f.f_code.co_name)
            if site is None:
                site = ent
            else:
                chain.append(ent)
                if len(chain) >= chain_len:
                    break
        f = f.f_back
    if site is None:
        site = ("<unknown>", 0, "<unknown>")
    return site, chain


def _fmt_site(site: Tuple[str, int, str]) -> str:
    return "%s:%d:%s" % (_short(site[0]), site[1], site[2])


def _snippet(site: Tuple[str, int, str]) -> str:
    try:
        line = linecache.getline(site[0], site[1]).strip()
    except Exception:
        return ""
    return line[:110]


def _mix64(x):
    """splitmix64 finalizer on a numpy uint64 array (a bijection on 64 bits)."""
    import numpy as np

    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(
            0xFFFFFFFFFFFFFFFF
        )
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(
            0xFFFFFFFFFFFFFFFF
        )
        x = x ^ (x >> np.uint64(31))
    return x


# --------------------------------------------------------------------------
# Instrument (a)+(c): warnings hook and Python-level interposition
# --------------------------------------------------------------------------
class _Capture:
    def __init__(self):
        self.cur_iter = -1
        self.tls = threading.local()
        self.warns: List[Dict[str, Any]] = []
        self.explicit: List[Dict[str, Any]] = []
        self.other_warnings: "collections.Counter[str]" = collections.Counter()
        self.stderr_lines: Dict[int, int] = collections.Counter()
        self.chains: Dict[str, Tuple[str, int, List[Tuple[str, int, str]]]] = {}
        self.use_markers = False


_CAP: Optional[_Capture] = None


def _showwarning_hook(message, category, filename, lineno, file=None, line=None):
    cap = _CAP
    if cap is None:
        return
    text = str(message)
    if WARN_TEXT not in text:
        if cap.cur_iter >= 0:
            cap.other_warnings[text[:100]] += 1
        return
    if cap.cur_iter < 0 or getattr(cap.tls, "suppress", 0):
        return
    site, chain = _resolve(sys._getframe(1))
    idx = len(cap.warns)
    cap.warns.append(
        {
            "idx": idx,
            "iter": cap.cur_iter,
            "tid": threading.get_native_id(),
            "site": site,
            "chain": chain,
        }
    )
    if cap.use_markers:
        # A zero-length range stamps the delivery time into the profiler trace, so
        # the warning can be matched with the sync event that raised it.
        from torch.profiler import record_function

        with record_function("PROBE_WARN|%d" % idx):
            pass


@contextlib.contextmanager
def _annot(name: str):
    from torch.profiler import record_function

    with record_function(name):
        yield


def _install_interposers(cap: _Capture, use_annot: bool):
    """Patch torch.cuda.synchronize / Stream.synchronize / Event.synchronize."""
    import torch

    undo: List[Any] = []

    def make(api: str, fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            if cap.cur_iter < 0:
                return fn(*a, **k)
            site, chain = _resolve(sys._getframe(1))
            cap.explicit.append(
                {
                    "iter": cap.cur_iter,
                    "tid": threading.get_native_id(),
                    "api": api,
                    "site": site,
                    "chain": chain,
                }
            )
            cap.tls.suppress = getattr(cap.tls, "suppress", 0) + 1
            try:
                if use_annot:
                    with _annot("PROBE_PYSYNC|%s|%s" % (api, _fmt_site(site))):
                        return fn(*a, **k)
                return fn(*a, **k)
            finally:
                cap.tls.suppress -= 1

        return wrapper

    for owner, attr, api in (
        (torch.cuda.Stream, "synchronize", "Stream.synchronize"),
        (torch.cuda.Event, "synchronize", "Event.synchronize"),
        (torch.cuda, "synchronize", "torch.cuda.synchronize"),
    ):
        had_own = attr in getattr(owner, "__dict__", {})
        orig = getattr(owner, attr)
        setattr(owner, attr, make(api, orig))
        undo.append((owner, attr, orig, had_own))

    def restore():
        for owner, attr, orig, had_own in reversed(undo):
            if had_own or isinstance(owner, type(torch.cuda)):
                setattr(owner, attr, orig)
            else:
                try:
                    delattr(owner, attr)
                except Exception:
                    setattr(owner, attr, orig)

    return restore


def _install_ext_wrappers(cap: _Capture):
    """Wrap every dynamicemb_extensions function in a record_function range that
    carries the Python call site, so a profiler-only sync gets a call site."""
    import dynamicemb_extensions as ext
    from torch.profiler import record_function

    wrappers: Dict[int, Tuple[str, Any, Any]] = {}

    def make(label: str, fn):
        def wrapper(*a, **k):
            if cap.cur_iter < 0:
                return fn(*a, **k)
            f = sys._getframe(1)
            co = f.f_code
            ss = "%s:%d:%s" % (_short(co.co_filename), f.f_lineno, co.co_name)
            if ss not in cap.chains:
                cap.chains[ss] = (co.co_filename, f.f_lineno, _resolve(f)[1])
            with record_function("PROBE_EXT|%s|%s" % (label, ss)):
                return fn(*a, **k)

        try:
            wrapper = functools.wraps(fn)(wrapper)
        except Exception:
            pass
        return wrapper

    for n in dir(ext):
        if n.startswith("__"):
            continue
        o = getattr(ext, n)
        if type(o).__name__ == "builtin_function_or_method":
            wrappers[id(o)] = (n, o, make(n, o))

    patched: List[Tuple[dict, str, Any]] = []
    mods = [ext] + [
        m
        for name, m in list(sys.modules.items())
        if m is not None and (name == "dynamicemb" or name.startswith("dynamicemb."))
    ]
    for mod in mods:
        d = vars(mod)
        for an, av in list(d.items()):
            w = wrappers.get(id(av))
            if w is not None:
                d[an] = w[2]
                patched.append((d, an, av))

    # methods of the pybind classes (VMMTensor, HostVMMTensor, CurandStateContext)
    cls_patched: List[Tuple[type, str, Any]] = []
    for cn in ("VMMTensor", "HostVMMTensor", "CurandStateContext"):
        cls = getattr(ext, cn, None)
        if cls is None:
            continue
        for mn, mv in list(vars(cls).items()):
            if mn.startswith("__") or type(mv).__name__ != "instancemethod":
                continue
            try:
                setattr(
                    cls, mn, _wrap_method(cap, "%s.%s" % (cn, mn), mv, record_function)
                )
                cls_patched.append((cls, mn, mv))
            except Exception:
                pass

    def restore():
        for d, an, av in reversed(patched):
            d[an] = av
        for cls, mn, mv in reversed(cls_patched):
            try:
                setattr(cls, mn, mv)
            except Exception:
                pass

    return restore, len(patched) + len(cls_patched)


def _wrap_method(cap, label, fn, record_function):
    def wrapper(self, *a, **k):
        if cap.cur_iter < 0:
            return fn(self, *a, **k)
        f = sys._getframe(1)
        co = f.f_code
        ss = "%s:%d:%s" % (_short(co.co_filename), f.f_lineno, co.co_name)
        if ss not in cap.chains:
            cap.chains[ss] = (co.co_filename, f.f_lineno, _resolve(f)[1])
        with record_function("PROBE_EXT|%s|%s" % (label, ss)):
            return fn(self, *a, **k)

    return wrapper


class _StderrTap:
    """Route fd 2 into a temp file for the measured loop: C++ warnings raised
    inside pybind functions (sync-debug mode) go to stderr, not to Python.  They
    are counted per iteration as an independent cross-check of (b)."""

    def __init__(self, cap: _Capture):
        self.cap = cap
        self.saved = None
        self.tmp = None
        self.stderr_ok = False

    def start(self):
        try:
            sys.stderr.flush()
            self.tmp = tempfile.TemporaryFile(mode="w+b")
            self.saved = os.dup(2)
            os.dup2(self.tmp.fileno(), 2)
            self.stderr_ok = True
        except Exception:
            self.saved = None

    def mark(self, i: int):
        if self.saved is not None:
            try:
                os.write(2, ("@@ITER %d\n" % i).encode())
            except Exception:
                pass

    def stop(self) -> str:
        if self.saved is None:
            return ""
        try:
            os.dup2(self.saved, 2)
            os.close(self.saved)
        finally:
            self.saved = None
        self.tmp.seek(0)
        data = self.tmp.read().decode("utf-8", "replace")
        self.tmp.close()
        cur = -1
        others: List[str] = []
        for ln in data.splitlines():
            if ln.startswith("@@ITER "):
                try:
                    cur = int(ln.split()[1])
                except Exception:
                    cur = -1
            elif WARN_TEXT in ln:
                if cur >= 0:
                    self.cap.stderr_lines[cur] += 1
            elif ln.strip():
                others.append(ln)
        return "\n".join(others[-20:])


class Probe:
    """Context manager around the measured loop; see module docstring."""

    def __init__(
        self,
        use_profiler: bool = True,
        stacks: bool = False,
        keep_trace_dir: Optional[str] = None,
        sync_debug: bool = True,
    ):
        self.use_profiler = use_profiler
        self.stacks = stacks
        self.keep_trace_dir = keep_trace_dir
        self.sync_debug = sync_debug
        self.cap = _Capture()
        self.trace_events: Optional[List[dict]] = None
        self.trace_path: Optional[str] = None
        self.stderr_other = ""
        self.n_wrapped = 0
        self._restores: List[Any] = []
        self._wctx = None
        self._prof = None
        self._tap: Optional[_StderrTap] = None
        self._label = "case"

    def __enter__(self):
        global _CAP

        _CAP = self.cap
        self.cap.use_markers = self.use_profiler
        self._wctx = warnings.catch_warnings()
        self._wctx.__enter__()
        warnings.simplefilter("always")
        self._old_show = warnings.showwarning
        warnings.showwarning = _showwarning_hook
        self._restores.append(_install_interposers(self.cap, self.use_profiler))
        if self.use_profiler:
            r, n = _install_ext_wrappers(self.cap)
            self._restores.append(r)
            self.n_wrapped = n
        self._tap = _StderrTap(self.cap)
        self._tap.start()
        if self.use_profiler:
            from torch.profiler import ProfilerActivity, profile

            self._prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=self.stacks,
            )
            self._prof.start()
        return self

    @contextlib.contextmanager
    def iteration(self, i: int):
        import torch

        cap = self.cap
        cap.cur_iter = i
        self._tap.mark(i)
        if self.sync_debug:
            torch.cuda.set_sync_debug_mode("warn")
        try:
            if self.use_profiler:
                with _annot("PROBE_ITER|%d" % i):
                    yield
            else:
                yield
        finally:
            if self.sync_debug:
                torch.cuda.set_sync_debug_mode("default")
            cap.cur_iter = -1

    def __exit__(self, et, ev, tb):
        global _CAP
        import torch

        try:
            if self._prof is not None:
                try:
                    torch.cuda.synchronize()
                finally:
                    self._prof.stop()
                d = self.keep_trace_dir or tempfile.mkdtemp(prefix="hsp_")
                os.makedirs(d, exist_ok=True)
                self.trace_path = os.path.join(
                    d, "trace_%s_%d.json" % (self._label, os.getpid())
                )
                self._prof.export_chrome_trace(self.trace_path)
                with open(self.trace_path) as f:
                    self.trace_events = json.load(f)["traceEvents"]
                if not self.keep_trace_dir:
                    try:
                        os.remove(self.trace_path)
                        os.rmdir(d)
                    except Exception:
                        pass
        finally:
            if self._tap is not None:
                self.stderr_other = self._tap.stop()
            for r in reversed(self._restores):
                try:
                    r()
                except Exception:
                    pass
            warnings.showwarning = self._old_show
            self._wctx.__exit__(None, None, None)
            _CAP = None
        return False


# --------------------------------------------------------------------------
# Trace analysis (instrument b)
# --------------------------------------------------------------------------
def analyze_trace(events: List[dict], n_iters: int) -> Dict[str, Any]:
    """Per-iteration sync events from a chrome trace of the measured loop."""
    wins: Dict[int, Tuple[float, float]] = {}
    warn_ts: Dict[int, float] = {}
    containers: Dict[Any, List[dict]] = collections.defaultdict(list)
    rt: Dict[Any, List[dict]] = collections.defaultdict(list)
    gpu_by_corr: Dict[Any, dict] = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat == "user_annotation":
            nm = e.get("name", "")
            if nm.startswith("PROBE_ITER|"):
                wins[int(nm.split("|")[1])] = (e["ts"], e["ts"] + e["dur"])
            elif nm.startswith("PROBE_WARN|"):
                warn_ts[int(nm.split("|")[1])] = e["ts"]
            elif nm.startswith("PROBE_EXT|"):
                _, label, site = (nm.split("|", 2) + ["", ""])[:3]
                containers[e["tid"]].append(
                    {
                        "ts": e["ts"],
                        "end": e["ts"] + e["dur"],
                        "typ": "ext",
                        "name": label,
                        "site": site,
                    }
                )
            elif nm.startswith("PROBE_PYSYNC|"):
                _, label, site = (nm.split("|", 2) + ["", ""])[:3]
                containers[e["tid"]].append(
                    {
                        "ts": e["ts"],
                        "end": e["ts"] + e["dur"],
                        "typ": "pysync",
                        "name": label,
                        "site": site,
                    }
                )
        elif cat == "cpu_op":
            nm = e.get("name", "")
            if nm.startswith("aten::"):
                containers[e["tid"]].append(
                    {
                        "ts": e["ts"],
                        "end": e["ts"] + e["dur"],
                        "typ": "aten",
                        "name": nm,
                        "site": "",
                    }
                )
        elif cat in ("cuda_runtime", "cuda_driver"):
            rt[e["tid"]].append(e)
        elif cat == "gpu_memcpy":
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                gpu_by_corr[corr] = e
    win_list = sorted((v[0], v[1], k) for k, v in wins.items())
    starts = [w[0] for w in win_list]

    def window_of(t: float) -> int:
        j = bisect.bisect_right(starts, t) - 1
        if j >= 0 and t <= win_list[j][1] + 1e-3:
            return win_list[j][2]
        return -1

    per_iter: Dict[int, List[dict]] = collections.defaultdict(list)
    info = collections.Counter()
    for tid, rts in rt.items():
        rts.sort(key=lambda e: e["ts"])
        conts = containers.get(tid, [])
        conts.sort(key=lambda c: (c["ts"], -(c["end"] - c["ts"])))
        stack: List[dict] = []
        ci = 0
        n = len(rts)
        for idx, e in enumerate(rts):
            nm = e["name"]
            kind = SYNC_KIND.get(nm)
            if kind is None and nm == "cudaMemcpyAsync":
                g = gpu_by_corr.get((e.get("args") or {}).get("correlation"))
                if g is not None:
                    gname = g.get("name", "")
                    if "DtoH" in gname and "Pageable" in gname:
                        j = idx + 1
                        while j < n and rts[j]["name"] in NOISE_RT:
                            j += 1
                        nxt = rts[j]["name"] if j < n else None
                        same_win = j < n and window_of(rts[j]["ts"]) == window_of(
                            e["ts"]
                        )
                        if nxt not in SYNC_AFTER_COPY or not same_win:
                            kind = "pd"
                    elif "HtoD" in gname and "Pageable" in gname:
                        w = window_of(e["ts"])
                        if w >= 0:
                            info["h2d_pageable_async"] += 1
            if kind is None:
                continue
            w = window_of(e["ts"])
            if w < 0:
                continue
            t = e["ts"]
            while ci < len(conts) and conts[ci]["ts"] <= t:
                c = conts[ci]
                while stack and stack[-1]["end"] <= c["ts"]:
                    stack.pop()  # an ended range must not hide below a newer one
                stack.append(c)
                ci += 1
            while stack and stack[-1]["end"] < t:
                stack.pop()
            owner = None
            aten = None
            for c in reversed(stack):
                if owner is None and c["typ"] in ("ext", "pysync"):
                    owner = c
                if aten is None and c["typ"] == "aten":
                    aten = c
                if owner is not None and aten is not None:
                    break
            per_iter[w].append(
                {
                    "kind": kind,
                    "api": nm,
                    "tid": tid,
                    "ts": t,
                    "owner_typ": owner["typ"] if owner else None,
                    "owner": owner["name"] if owner else None,
                    "site": owner["site"] if owner else None,
                    "aten": aten["name"] if aten else None,
                }
            )
    return {
        "per_iter": per_iter,
        "windows": len(wins),
        "info": dict(info),
        "warn_ts": warn_ts,
    }


# --------------------------------------------------------------------------
# Building modules and batches
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Batch:
    indices: Any
    offsets: Any
    freq: Any = None


class BatchMaker:
    def __init__(self, cfg: Cfg, device, seed: int):
        import numpy as np

        self.np = np
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.counter = [0, 0]

    def fresh(self, t: int, n: int):
        np = self.np
        ids = np.arange(self.counter[t], self.counter[t] + n, dtype=np.uint64)
        self.counter[t] += n
        keys = _mix64(ids + np.uint64(t) * np.uint64(1 << 40)) & np.uint64(
            (1 << 62) - 1
        )
        return keys.astype(np.int64)

    def build(self, uniq: Dict[int, Any]) -> Batch:
        import torch

        np = self.np
        feats: List[Any] = [None] * len(FEATURE_TABLE_MAP)
        for t in (0, 1):
            fs = [f for f, tt in enumerate(FEATURE_TABLE_MAP) if tt == t]
            n_slots = len(fs) * BATCH * KPB
            u = uniq[t]
            if len(u) > n_slots:
                u = u[:n_slots]
            extra = (
                self.rng.choice(u, size=n_slots - len(u), replace=True)
                if n_slots > len(u)
                else u[:0]
            )
            slots = np.concatenate([u, extra])
            self.rng.shuffle(slots)
            for j, f in enumerate(fs):
                feats[f] = slots[j * BATCH * KPB : (j + 1) * BATCH * KPB]
        idx = np.concatenate(feats).astype(np.int64)
        off = np.arange(0, len(idx) + 1, KPB, dtype=np.int64)
        indices = torch.from_numpy(idx).to(self.device)
        offsets = torch.from_numpy(off).to(self.device)
        freq = (
            torch.ones(len(idx), dtype=torch.int64, device=self.device)
            if self.cfg.lfu
            else None
        )
        return Batch(indices, offsets, freq)

    def fresh_batch(self, scale: float = 1.0) -> Batch:
        return self.build(
            {t: self.fresh(t, max(1, int(UNIQ[t] * scale))) for t in (0, 1)}
        )

    def overlap_stream(self, n: int) -> List[Batch]:
        """Each batch = 45% fresh keys + 45% of the previous batch's keys, so an
        admission strategy with threshold 2 both refuses and admits keys."""
        out: List[Batch] = []
        prev: Dict[int, Any] = {t: self.fresh(t, 0) for t in (0, 1)}
        for _ in range(n):
            cur = {}
            for t in (0, 1):
                nf = int(0.45 * ROW_N[t])
                new = self.fresh(t, nf)
                carry = prev[t][: int(0.45 * ROW_N[t])]
                cur[t] = self.np.concatenate([new, carry])
                prev[t] = new
            out.append(self.build(cur))
        return out


def _load_ps_class():
    # The PS stand-in lives in <dynamicemb root>/test; this file is three levels
    # below that directory, and the runner only puts the test file's own
    # directory on sys.path.
    test_dir = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
    if test_dir not in sys.path and os.path.isfile(
        os.path.join(test_dir, "test_batched_dynamic_embedding_tables_v2.py")
    ):
        sys.path.append(test_dir)
    from test_batched_dynamic_embedding_tables_v2 import PyDictStorage

    return PyDictStorage


def build_module(cfg: Cfg, plan: Dict[str, Any], device_index: int):
    import torch
    from dynamicemb import (
        DynamicEmbPoolingMode,
        DynamicEmbScoreStrategy,
        DynamicEmbTableOptions,
        EmbOptimType,
        EvictedItemMode,
    )
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
    from dynamicemb.optimizer import get_optimizer_state_dim

    opt_type = EmbOptimType.ADAM if cfg.optimizer == "adam" else EmbOptimType.SGD
    opt_params: Dict[str, Any] = {"learning_rate": 0.3}
    if cfg.optimizer == "adam":
        opt_params.update(weight_decay=0.06, eps=3e-5, beta1=0.8, beta2=0.888)
    row_bytes = 4 * (DIM + get_optimizer_state_dim(opt_type, DIM, torch.float32))
    max_cap = plan["max_capacity"]
    if cfg.layout in ("cache", "cache_ps", "hybrid"):
        local_hbm = (
            plan["hbm_rows"] * row_bytes
        )  # exact power-of-two fraction of the table
    elif cfg.layout == "hbm":
        local_hbm = 1024**3
    else:
        local_hbm = 0
    ps = _load_ps_class() if cfg.layout == "cache_ps" else None
    admit = None
    if cfg.admit:
        from dynamicemb.embedding_admission import FrequencyAdmissionStrategy, KVCounter

        admit = FrequencyAdmissionStrategy(
            threshold=ADMIT_THRESHOLD, counter=KVCounter(capacity=min(max_cap, 16384))
        )
    kw: Dict[str, Any] = {}
    if cfg.retain:
        kw["evicted_item_mode"] = EvictedItemMode.RETAIN_KEY
    if plan["init_capacity"] is not None:
        kw["init_capacity"] = plan["init_capacity"]
        kw["bucket_capacity"] = 128
    options = [
        DynamicEmbTableOptions(
            dim=DIM,
            max_capacity=max_cap,
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device_index,
            score_strategy=(
                DynamicEmbScoreStrategy.LFU
                if cfg.lfu
                else DynamicEmbScoreStrategy.NO_EVICTION
                if cfg.noevict
                else DynamicEmbScoreStrategy.TIMESTAMP
            ),
            caching=cfg.layout in ("cache", "cache_ps"),
            local_hbm_for_values=local_hbm,
            external_storage=ps,
            admit_strategy=admit,
            **kw,
        )
        for _ in range(2)
    ]
    mod = BatchedDynamicEmbeddingTablesV2(
        table_names=["t0", "t1"],
        table_options=options,
        feature_table_map=list(FEATURE_TABLE_MAP),
        pooling_mode=DynamicEmbPoolingMode.SUM
        if cfg.pooling == "sum"
        else DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )
    if cfg.metrics:
        mod.set_record_cache_metrics(True)
    return mod


def tier_sizes(mod) -> Dict[str, Dict[str, List[int]]]:
    """Occupancy and capacity per tier and table (host reads: call outside windows)."""
    out: Dict[str, Dict[str, List[int]]] = {}

    def of_km(km, n):
        capf = getattr(km, "main_capacity", None) or km.capacity
        return {
            "size": [int(km.size(t)) for t in range(n)],
            "cap": [int(capf(t)) for t in range(n)],
        }

    try:
        cache = getattr(mod, "_cache", None)
        if cache is not None:
            out["cache"] = of_km(cache.key_index_map, 2)
        st = getattr(mod, "_storage", None)
        if st is not None:
            tabs = getattr(st, "tables", None)
            if hasattr(st, "key_index_map"):
                out["storage"] = of_km(st.key_index_map, 2)
            elif isinstance(tabs, list) and len(tabs) == 2:
                out["hbm"] = of_km(tabs[0].key_index_map, 2)
                out["host"] = of_km(tabs[1].key_index_map, 2)
        ps = getattr(mod, "_external_storage", None)
        if ps is not None and hasattr(ps, "dict"):
            out["ps"] = {"size": [len(ps.dict)], "cap": [0]}
    except Exception as e:  # diagnostics must never break a run
        out["error"] = {"size": [], "cap": [], "msg": repr(e)}  # type: ignore
    return out


def top_tier(cfg: Cfg) -> str:
    return {"cache": "cache", "cache_ps": "cache", "hybrid": "hbm"}.get(
        cfg.layout, "storage"
    )


def regime_check(cfg: Cfg, scenario: str, before, after, k: int) -> Tuple[bool, str]:
    """Did the scenario really happen?  Judged from tier occupancy before/after
    the measured window."""
    if scenario == "tier2_hit":
        tiers = (
            ["storage"]
            if cfg.layout == "cache"
            else (["hbm", "host"] if cfg.layout == "hybrid" else [])
        )
        if not tiers or any(t not in after or t not in before for t in tiers):
            return True, "no backing-tier occupancy check for layout %s" % cfg.layout
        d = sum(
            after[t]["size"][i] - before[t]["size"][i] for t in tiers for i in range(2)
        )
        return (
            d == 0,
            "backing tiers %s size delta=%d (expected 0: every key is already stored)"
            % (tiers, d),
        )
    tier = top_tier(cfg)
    b, a = before.get(tier), after.get(tier)
    if b is None or a is None:
        return True, "no occupancy info for tier %s" % tier
    new = [k * (int(0.45 * ROW_N[t]) if cfg.admit else UNIQ[t]) for t in (0, 1)]
    dsz = [a["size"][t] - b["size"][t] for t in range(2)]
    load = [a["size"][t] / max(1, a["cap"][t]) for t in range(2)]
    if scenario in ("all_hit", "eval"):
        ok = all(d == 0 for d in dsz)
        return ok, "tier %s size delta=%s (expected 0)" % (tier, dsz)
    if scenario == "eval_miss":
        ok = all(d == 0 for d in dsz)
        return ok, "tier %s size delta=%s (eval inserts nothing)" % (tier, dsz)
    if scenario == "miss":
        ok = all(dsz[t] >= 0.5 * new[t] for t in range(2)) and max(load) < 0.85
        return ok, "tier %s size delta=%s of ~%s new keys, load=%s" % (
            tier,
            dsz,
            new,
            ["%.2f" % x for x in load],
        )
    if scenario == "miss_evict":
        ok = load[0] >= 0.8 and dsz[0] < 0.7 * new[0]
        return (
            ok,
            "tier %s load=%s size delta=%s of ~%s new keys (net growth << new keys = evicting)"
            % (tier, ["%.2f" % x for x in load], dsz, new),
        )
    if scenario == "growth":
        grew = False
        for tname in after:
            if tname in before and any(
                after[tname]["cap"][t] > before[tname]["cap"][t] for t in range(2)
            ):
                grew = True
        return grew, "capacity before=%s after=%s" % (
            {n: v["cap"] for n, v in before.items()},
            {n: v["cap"] for n, v in after.items()},
        )
    return True, ""


# --------------------------------------------------------------------------
# Running one (config, scenario, mode)
# --------------------------------------------------------------------------
def _make_step(mod, cfg: Cfg, mode: str, train: bool):
    import torch

    def kw(b: Batch) -> Dict[str, Any]:
        return {"frequency_counters": b.freq} if cfg.lfu else {}

    if not train:

        def step(b: Batch):
            with torch.no_grad():
                mod(b.indices, b.offsets, **kw(b))

        return step
    if mode == "plain":

        def step(b: Batch):
            out = mod(b.indices, b.offsets, **kw(b))
            out.backward(torch.ones_like(out))

        return step
    ps = torch.cuda.Stream()
    fs = torch.cuda.Stream()

    def step(b: Batch):
        ps.wait_stream(fs)
        with torch.cuda.stream(ps):
            mod.prefetch(b.indices, b.offsets, fs, **kw(b))
        fs.wait_stream(ps)
        with torch.cuda.stream(fs):
            out = mod(b.indices, b.offsets, **kw(b))
            out.backward(torch.ones_like(out))

    return step


def _fill_steps(cfg: Cfg, scenario: str, plan: Dict[str, Any]) -> int:
    """Steps of fresh keys that bring the top tier of the scarcest table past 1.7x
    its capacity, so that every measured step evicts."""
    if scenario not in ("miss_evict", "tier2_hit"):
        return 0
    rows = plan["hbm_rows"] if plan["hbm_rows"] else plan["max_capacity"]
    return int(math.ceil(1.7 * rows / max(1, UNIQ[1])))


def run_case(
    cfg: Cfg,
    scenario: str,
    mode: str,
    iters: int = 8,
    warmup: int = 3,
    device_index: int = 0,
    seed: int = 1234,
    use_profiler: bool = True,
    stacks: bool = False,
    keep_trace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    import torch

    res: Dict[str, Any] = {
        "config": cfg.name,
        "scenario": scenario,
        "mode": mode,
        "layout": cfg.layout,
        "status": "ok",
        "iters": iters,
        "warmup": warmup,
    }
    t0 = time.time()
    mod = None
    probe = None
    old_det = os.environ.get("DEMB_DETERMINISM_MODE")
    if cfg.determ:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"
    try:
        torch.manual_seed(seed)
        plan = plan_of(cfg, scenario)
        res["plan"] = plan
        mod = build_module(cfg, plan, device_index)
        dev = torch.device("cuda", device_index)
        bm = BatchMaker(cfg, dev, seed)
        train = scenario not in EVAL_SCENARIOS
        stream_overlap = cfg.admit and scenario in ("miss", "miss_evict", "growth")
        n_fill = _fill_steps(cfg, scenario, plan)
        warm = max(warmup, 4) if cfg.admit else warmup
        if scenario in ("all_hit", "eval", "eval_miss"):
            fixed = bm.fresh_batch()
            warm_train = [fixed] * max(warm, 4 if cfg.admit else 3)
            if scenario == "all_hit":
                pre = warm_train
                measured = [fixed] * iters
                warm_eval: List[Batch] = []
            elif scenario == "eval":
                pre = warm_train
                warm_eval = [fixed] * warm
                measured = [fixed] * iters
            else:
                pre = warm_train
                warm_eval = [bm.fresh_batch() for _ in range(warm)]
                measured = [bm.fresh_batch() for _ in range(iters)]
            fill: List[Batch] = []
        elif scenario == "growth":
            fill = []
            pre = [bm.fresh_batch(scale=0.1)] * 1
            warm_eval = []
            n = warm - 1 if warm > 1 else 0
            pre += [bm.fresh_batch(scale=0.1) for _ in range(n)]
            measured = (
                bm.overlap_stream(iters)
                if stream_overlap
                else [bm.fresh_batch() for _ in range(iters)]
            )
        elif scenario == "tier2_hit":
            # a pool of batches is inserted, fresh keys then push it out of the top tier
            # (into the storage / the host tier / the PS), and the pool is read again
            pool = [bm.fresh_batch() for _ in range(warm + iters)]
            fill = pool + [bm.fresh_batch() for _ in range(n_fill)]
            pre = pool[:warm]
            measured = pool[warm:]
            warm_eval = []
        else:  # miss / miss_evict
            total = n_fill + warm + iters
            allb = (
                bm.overlap_stream(total)
                if stream_overlap
                else [bm.fresh_batch() for _ in range(total)]
            )
            fill, pre, measured = (
                allb[:n_fill],
                allb[n_fill : n_fill + warm],
                allb[n_fill + warm :],
            )
            warm_eval = []
        res["n_fill"] = len(fill)
        step_train = _make_step(mod, cfg, mode, True)
        step_eval = _make_step(mod, cfg, mode, False)
        # ---- unmeasured phases
        mod.train()
        for b in fill + pre:
            step_train(b)
        if not train:
            mod.eval()
            for b in warm_eval:
                step_eval(b)
        torch.cuda.synchronize()
        step = step_train if train else step_eval
        before = tier_sizes(mod)
        # ---- measured loop
        probe = Probe(
            use_profiler=use_profiler, stacks=stacks, keep_trace_dir=keep_trace_dir
        )
        probe._label = "%s_%s_%s" % (cfg.name, scenario, mode)
        with probe:
            for i, b in enumerate(measured):
                with probe.iteration(i):
                    step(b)
                torch.cuda.synchronize()
        after = tier_sizes(mod)
        ok, note = regime_check(cfg, scenario, before, after, iters)
        res["regime"] = {"ok": ok, "note": note}
        res["tiers_before"] = before
        res["tiers_after"] = after
        res.update(
            _summarize(
                probe, iters, [g for g, _ in inherent_site_globs(cfg.name, scenario)]
            )
        )
    except Exception as e:
        res["status"] = "error"
        res["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        res["traceback"] = traceback.format_exc()[-2500:]
        if probe is not None and probe.stderr_other:
            res["stderr_tail"] = probe.stderr_other[-1500:]
    finally:
        if cfg.determ:
            if old_det is None:
                os.environ.pop("DEMB_DETERMINISM_MODE", None)
            else:
                os.environ["DEMB_DETERMINISM_MODE"] = old_det
        try:
            del mod
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
    res["seconds"] = round(time.time() - t0, 2)
    res["inherent"] = inherent_reason(cfg.name, scenario)
    sg = inherent_site_globs(cfg.name, scenario)
    res["inherent_sites"] = sg[0][1] if sg else None
    return res


def _summarize(
    probe: Probe, iters: int, site_globs: Sequence[str] = ()
) -> Dict[str, Any]:
    cap = probe.cap
    out: Dict[str, Any] = {}
    site_excused = [
        0
    ] * iters  # syncs per iteration whose call site is in an INHERENT file

    def excused(file_short: str) -> bool:
        return any(fnmatch.fnmatch(file_short, g) for g in site_globs)

    py_a = [0] * iters
    py_c = [0] * iters
    for w in cap.warns:
        py_a[w["iter"]] += 1
    for w in cap.explicit:
        py_c[w["iter"]] += 1
    warn_ops: Dict[int, List[Tuple[str, str]]] = collections.defaultdict(list)
    per_kind = {k: [0] * iters for k in KINDS}
    origin = {
        o: [0] * iters
        for o in ("explicit", "cpp_aten", "cpp_raw", "direct_aten", "direct_raw")
    }
    sites: Dict[Tuple, Dict[str, Any]] = {}
    info: Dict[str, Any] = {}
    notes: List[str] = []
    prof_total = [0] * iters
    if probe.trace_events is not None:
        tr = analyze_trace(probe.trace_events, iters)
        info = tr["info"]
        if tr["windows"] != iters:
            notes.append(
                "profiler saw %d/%d iteration windows" % (tr["windows"], iters)
            )
        for i in range(iters):
            evs = sorted(tr["per_iter"].get(i, []), key=lambda e: e["ts"])
            for ev in evs:
                k = ev["kind"]
                per_kind[k][i] += 1
                prof_total[i] += 1
                owner_kind = ev["owner_typ"]
                if owner_kind == "ext":
                    o = (
                        "cpp_aten"
                        if (ev["aten"] and k in BLOCKING_KINDS)
                        else "cpp_raw"
                    )
                    if k in ("al", "pd"):
                        o = "cpp_raw"
                    origin[o][i] += 1
                    key = ("cpp", ev["owner"], ev["site"], o, k)
                elif owner_kind == "pysync":
                    origin["explicit"][i] += 1
                    key = ("explicit", ev["owner"], ev["site"], "explicit", k)
                elif ev["aten"] and k in BLOCKING_KINDS:
                    origin["direct_aten"][i] += 1
                    warn_ops[i].append((k, ev["aten"]))
                    key = ("direct_aten", ev["aten"], "", "py_warn", k)
                else:
                    origin["direct_raw"][i] += 1
                    key = ("direct_raw", ev["aten"] or "-", "", "raw", k)
                s = sites.setdefault(key, {"count": 0, "iters": set()})
                s["count"] += 1
                s["iters"].add(i)
                if (
                    owner_kind == "ext"
                    and ev["site"]
                    and excused(ev["site"].split(":")[0])
                ):
                    site_excused[i] += 1
        # Match every python warning with the ATen-visible sync event that raised it:
        # the latest unmatched one before the warning's delivery marker.  A warning
        # matched to an event inside an extension call is a duplicate: the sync-debug
        # warning of a C++-internal .cpu() is buffered by an enclosing handler
        # (Function.apply, the autograd engine) and delivered late, at the caller of
        # that handler, so its Python site says nothing about where the sync is.
        warn_ts = tr["warn_ts"]
        dup_per_iter = [0] * iters
        unmatched_w = 0
        unmatched_d = 0
        for i in range(iters):
            evs = sorted(tr["per_iter"].get(i, []), key=lambda e: e["ts"])
            cand = [
                e
                for e in evs
                if e["kind"] in BLOCKING_KINDS
                and e["aten"]
                and e["owner_typ"] != "pysync"
            ]
            used = set()
            ws = sorted(
                (w for w in cap.warns if w["iter"] == i and w["idx"] in warn_ts),
                key=lambda w: warn_ts[w["idx"]],
            )
            for w in ws:
                tsw = warn_ts[w["idx"]]
                j = len(cand) - 1
                while j >= 0 and (j in used or cand[j]["ts"] > tsw):
                    j -= 1
                if j < 0:
                    w["match"] = None
                    unmatched_w += 1
                    continue
                used.add(j)
                w["match"] = cand[j]
                if cand[j]["owner_typ"] == "ext":
                    w["dup"] = True
                    dup_per_iter[i] += 1
            unmatched_d += sum(
                1
                for j, e in enumerate(cand)
                if j not in used and e["owner_typ"] != "ext"
            )
        ex = sum(origin["explicit"])
        if ex != sum(py_c):
            notes.append(
                "profiler counts %d explicit syncs, interposers saw %d"
                % (ex, sum(py_c))
            )
        if unmatched_w:
            notes.append(
                "%d python warning(s) matched no profiler sync event" % unmatched_w
            )
        if unmatched_d:
            notes.append(
                "%d direct ATen sync(s) raised no python warning" % unmatched_d
            )
        out["dup_warnings"] = sum(dup_per_iter) / float(iters)
    # ---- python-delivered (a) sites, with call chain
    py_sites: Dict[Tuple, Dict[str, Any]] = {}
    py_direct = [0] * iters
    for w in cap.warns:
        if w.get("dup"):
            continue
        key = ("py_warn", w["site"])
        s = py_sites.setdefault(
            key,
            {
                "count": 0,
                "iters": set(),
                "chain": w["chain"],
                "kinds": collections.Counter(),
                "ops": collections.Counter(),
            },
        )
        s["count"] += 1
        s["iters"].add(w["iter"])
        py_direct[w["iter"]] += 1
        if excused(_short(w["site"][0])):
            site_excused[w["iter"]] += 1
        m = w.get("match")
        if m is not None:
            s["kinds"][m["kind"]] += 1
            s["ops"][m["aten"]] += 1
    for w in cap.explicit:
        key = ("explicit", w["site"], w["api"])
        s = py_sites.setdefault(
            key,
            {
                "count": 0,
                "iters": set(),
                "chain": w["chain"],
                "kinds": collections.Counter(),
                "ops": collections.Counter(),
            },
        )
        s["count"] += 1
        s["iters"].add(w["iter"])
        if excused(_short(w["site"][0])):
            site_excused[w["iter"]] += 1
        s["kinds"][
            {
                "Stream.synchronize": "ss",
                "Event.synchronize": "es",
                "torch.cuda.synchronize": "ds",
            }.get(w["api"], "ss")
        ] += 1
    site_rows: List[Dict[str, Any]] = []
    for key, s in py_sites.items():
        site = key[1]
        site_rows.append(
            {
                "source": key[0],
                "api": key[2] if key[0] == "explicit" else "",
                "site": _fmt_site(site),
                "file": _short(site[0]),
                "line": site[1],
                "func": site[2],
                "snippet": _snippet(site),
                "chain": [_fmt_site(c) for c in s["chain"]],
                "per_iter": s["count"] / float(iters),
                "total": s["count"],
                "iters_hit": len(s["iters"]),
                "kinds": dict(s["kinds"]),
                "ops": dict(s["ops"]),
            }
        )
    for key, s in sites.items():
        typ, owner, site_s, org, k = key
        if typ in ("direct_aten", "explicit"):
            continue  # reported through the python-visible rows above
        ch = cap.chains.get(site_s)
        row = {
            "source": typ,
            "api": owner,
            "site": site_s or "(inside %s)" % owner,
            "file": site_s.split(":")[0] if site_s else "",
            "line": int(site_s.split(":")[1]) if site_s.count(":") >= 2 else 0,
            "func": site_s.split(":")[2] if site_s.count(":") >= 2 else "",
            "snippet": _snippet((ch[0], ch[1], "")) if ch else "",
            "chain": [_fmt_site(c) for c in ch[2]] if ch else [],
            "per_iter": s["count"] / float(iters),
            "total": s["count"],
            "iters_hit": len(s["iters"]),
            "kinds": {k: s["count"]},
            "origin": org,
        }
        site_rows.append(row)
    site_rows.sort(key=lambda r: (-r["per_iter"], r["site"]))
    if probe.trace_events is not None:
        py_a = py_direct  # warnings that name the line of a direct python-level sync
    py_vis = [py_a[i] + py_c[i] for i in range(iters)]
    if probe.trace_events is not None:
        totals = [max(prof_total[i], py_vis[i]) for i in range(iters)]
        src = "profiler+python"
    else:
        totals = py_vis
        src = "python-only (no profiler: syncs inside C++ extension calls are NOT seen)"
    if (
        probe.trace_events is not None
        and probe._tap is not None
        and probe._tap.stderr_ok
    ):
        se = sum(cap.stderr_lines.values())
        ca = sum(origin["cpp_aten"])
        dp = int(round(out.get("dup_warnings", 0.0) * iters))
        if se + dp != ca:
            notes.append(
                "C++-internal ATen syncs: profiler=%d, sync-debug stderr lines=%d + late python "
                "warnings=%d" % (ca, se, dp)
            )
    out["per_iter_total"] = totals
    out["inherent_site_per_iter"] = site_excused
    out["inherent_site_mean"] = sum(site_excused) / float(iters)
    out["per_iter_kind"] = per_kind
    out["mean"] = {k: sum(v) / float(iters) for k, v in per_kind.items()}
    out["mean"]["total"] = sum(totals) / float(iters)
    out["mean"]["blocking"] = sum(
        sum(per_kind[k][i] for k in BLOCKING_KINDS + ("pd",)) for i in range(iters)
    ) / float(iters)
    out["total_source"] = src
    out["python_visible"] = {
        "warn": sum(py_a) / float(iters),
        "explicit": sum(py_c) / float(iters),
        "dup_late_warn": out.get("dup_warnings", 0.0),
    }
    out["origin"] = {k: sum(v) / float(iters) for k, v in origin.items()}
    out["sites"] = site_rows
    out["info"] = info
    out["notes"] = notes
    out["other_warnings"] = dict(cap.other_warnings.most_common(5))
    out["stderr_sync_warnings_per_iter"] = (
        sum(cap.stderr_lines.values()) / float(iters)
        if probe._tap is not None
        else None
    )
    out["stderr_other"] = probe.stderr_other
    out["wrapped_ext_callables"] = probe.n_wrapped
    return out


# --------------------------------------------------------------------------
# measure(), inherent table, zero check
# --------------------------------------------------------------------------
def inherent_reason(config: str, scenario: str) -> Optional[str]:
    """Reason the whole (config, scenario) is allowed to sync, else None."""
    for key, why in INHERENT.items():
        if (
            len(key) == 2
            and fnmatch.fnmatch(config, key[0])
            and fnmatch.fnmatch(scenario, key[1])
        ):
            return why
    return None


def inherent_site_globs(config: str, scenario: str) -> List[Tuple[str, str]]:
    """(call-site file glob, reason) for the syncs excused at call-site level."""
    out = []
    for key, why in INHERENT.items():
        if (
            len(key) == 3
            and fnmatch.fnmatch(config, key[0])
            and fnmatch.fnmatch(scenario, key[1])
        ):
            out.append((key[2], why))
    return out


def _git_info(root: str) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", root, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _meta(iters: int, warmup: int, label: str = "") -> Dict[str, Any]:
    import torch

    meta: Dict[str, Any] = {
        "tool": "host_sync_probe.py",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "label": label,
        "iters": iters,
        "warmup": warmup,
        "batch": {
            "B": BATCH,
            "keys_per_bag": KPB,
            "dim": DIM,
            "features": FEATURE_TABLE_MAP,
            "unique_keys_per_step": UNIQ,
        },
    }
    try:
        import dynamicemb

        meta["dynamicemb"] = os.path.dirname(os.path.abspath(dynamicemb.__file__))
        root = os.path.dirname(meta["dynamicemb"])
        meta["git"] = _git_info(root)
    except Exception as e:
        meta["dynamicemb"] = "import failed: %r" % (e,)
    return meta


def _warm_profiler():
    """CUPTI initialization costs seconds the first time; pay it up front."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    x = torch.ones(8, device="cuda")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        (x + 1).sum().item()
    torch.cuda.synchronize()


def measure(
    configs: Optional[Sequence[str]] = None,
    scenarios: Optional[Sequence[str]] = None,
    modes: Sequence[str] = MODES,
    iters: int = 8,
    warmup: int = 3,
    device: int = 0,
    seed: int = 1234,
    profiler: bool = True,
    stacks: bool = False,
    keep_trace_dir: Optional[str] = None,
    verbose: bool = False,
    label: str = "",
) -> Dict[str, Any]:
    """Run the census in this process and return a structured report.

    report["results"] has one dict per (config, scenario, mode) with
    mean syncs per iteration by kind (res["mean"]), per-iteration totals,
    ranked call sites (res["sites"]) and diagnostics.  Use check_zero(report)
    to list non-inherent cases that still sync.
    """
    import torch

    torch.cuda.set_device(device)
    cfgs = (
        list(CONFIGS.values())
        if not configs or list(configs) == ["all"]
        else [CONFIGS[c] for c in configs]
    )
    want = None if not scenarios or list(scenarios) == ["all"] else list(scenarios)
    if profiler:
        _warm_profiler()
    results: List[Dict[str, Any]] = []
    for cfg in cfgs:
        avail = scenarios_of(cfg)
        for sc in want if want is not None else avail:
            if sc not in SCENARIO_DOC:
                raise ValueError(
                    "unknown scenario %r; have %s" % (sc, list(SCENARIO_DOC))
                )
            for mode in modes:
                if sc in EVAL_SCENARIOS and mode != "plain":
                    continue  # prefetch is a no-op in eval
                if sc not in avail:
                    results.append(
                        {
                            "config": cfg.name,
                            "scenario": sc,
                            "mode": mode,
                            "layout": cfg.layout,
                            "status": "skipped",
                            "error": "scenario not applicable to this config",
                            "inherent": inherent_reason(cfg.name, sc),
                        }
                    )
                    continue
                if verbose:
                    print(
                        "[probe] %s / %s / %s ..." % (cfg.name, sc, mode),
                        file=sys.stderr,
                        flush=True,
                    )
                r = run_case(
                    cfg,
                    sc,
                    mode,
                    iters=iters,
                    warmup=warmup,
                    device_index=device,
                    seed=seed,
                    use_profiler=profiler,
                    stacks=stacks,
                    keep_trace_dir=keep_trace_dir,
                )
                if verbose:
                    m = r.get("mean", {})
                    print(
                        "[probe]   -> %s total/iter=%s (%.1fs) %s"
                        % (
                            r["status"],
                            "%.2f" % m["total"] if "total" in m else "-",
                            r.get("seconds", 0),
                            r.get("error", ""),
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                results.append(r)
    return {
        "meta": _meta(iters, warmup, label),
        "inherent": {"|".join(k): v for k, v in INHERENT.items()},
        "results": results,
    }


def check_zero(report: Dict[str, Any], allow_inherent: bool = True) -> List[str]:
    """Violations: measured (config, scenario, mode) that sync, are not marked
    INHERENT, or errored.  Empty list == zero host syncs."""
    bad: List[str] = []
    for r in report["results"]:
        tag = "%s/%s/%s" % (r["config"], r["scenario"], r["mode"])
        if r["status"] == "skipped":
            continue
        if r["status"] != "ok":
            bad.append("%s: ERROR %s" % (tag, r.get("error")))
            continue
        if allow_inherent and r.get("inherent"):
            continue
        site_excused = r.get("inherent_site_per_iter") or [0] * len(r["per_iter_total"])
        eff = [
            t - (h if allow_inherent else 0)
            for t, h in zip(r["per_iter_total"], site_excused)
        ]
        if sum(eff) > 0:
            globs = (
                [g for g, _ in inherent_site_globs(r["config"], r["scenario"])]
                if allow_inherent
                else []
            )
            top = next(
                (
                    x["site"]
                    for x in r.get("sites", [])
                    if not any(fnmatch.fnmatch(x.get("file", ""), g) for g in globs)
                ),
                "?",
            )
            bad.append(
                "%s: %.2f syncs/iter (max %d) outside inherent sites, top site %s"
                % (tag, sum(eff) / float(len(eff)), max(eff), top)
            )
    return bad


def measure_callable(
    fn, iters: int = 4, warmup: int = 2, profiler: bool = True
) -> Dict[str, Any]:
    """The census of one callable: fn(i) runs `warmup` unmeasured times (with
    i = -1) and then once per measured window i = 0 .. iters-1, with a device
    barrier between windows.

    Returns the per-case summary that measure() keeps in report["results"]:
    "mean" (syncs per iteration by kind plus "total" and "blocking"),
    "per_iter_total", ranked "sites", "origin", "python_visible" and "notes".
    """
    import torch

    if profiler:
        _warm_profiler()
    for _ in range(warmup):
        fn(-1)
    torch.cuda.synchronize()
    probe = Probe(use_profiler=profiler)
    probe._label = "callable"
    with probe:
        for i in range(iters):
            with probe.iteration(i):
                fn(i)
            torch.cuda.synchronize()
    return _summarize(probe, iters)


# --------------------------------------------------------------------------
# Self test: synthetic workloads with a known number of syncs
# --------------------------------------------------------------------------
def selftest(
    device: int = 0, iters: int = 4, verbose: bool = True
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Check the instruments against workloads whose sync count is known, plus a
    negative control that must read zero.  Returns (all_ok, rows)."""
    import dynamicemb  # noqa: F401  (loads dynamicemb_extensions)
    import dynamicemb_extensions as ext
    import torch

    torch.cuda.set_device(device)
    dev = torch.device("cuda", device)
    x = torch.arange(4096, device=dev, dtype=torch.int64)
    mask = (x % 3) == 0
    pinned = torch.empty(8, dtype=torch.int64, pin_memory=True)
    pageable = torch.empty(8, dtype=torch.int64)
    ev = torch.cuda.Event()
    grow = [0]
    emb_in = torch.randn(64, 32, device=dev)
    emb_out = torch.empty(128, 32, device=dev)
    gidx = torch.randint(0, 64, (128,), device=dev)

    def alloc(i):
        grow[0] += 1  # every call asks for a block bigger than anything cached
        t = torch.empty(grow[0] * (96 << 20), dtype=torch.uint8, device=dev)
        del t

    def event_sync(i):
        ev.record()
        ev.synchronize()

    cases = [
        ("negative control: pure GPU work", lambda i: ((x + 1) * 2).sum(), {}, {}),
        (
            "negative control: pinned D2H non_blocking",
            lambda i: pinned.copy_(x[:8], non_blocking=True),
            {},
            {},
        ),
        (
            "negative control: pageable H2D non_blocking",
            lambda i: x[:8].copy_(pageable, non_blocking=True),
            {},
            {},
        ),
        ("Tensor.item()", lambda i: x[0].item(), {"ss": 1}, {"py_warn": 1}),
        ("Tensor.cpu()", lambda i: x[:8].cpu(), {"ss": 1}, {"py_warn": 1}),
        ("Tensor.tolist()", lambda i: x[:4].tolist(), {"ss": 1}, {"py_warn": 1}),
        ("torch.nonzero", lambda i: torch.nonzero(mask), {"ss": 1}, {"py_warn": 1}),
        ("boolean-mask indexing", lambda i: x[mask], {"ss": 1}, {"py_warn": 1}),
        ("torch.unique", lambda i: torch.unique(x % 7), {"ss": 1}, {"py_warn": 1}),
        ("bool(tensor.any())", lambda i: bool(mask.any()), {"ss": 1}, {"py_warn": 1}),
        (
            "H2D pageable blocking (torch.tensor(device=))",
            lambda i: torch.tensor([1, 2, 3], device=dev),
            {"ss": 1},
            {"py_warn": 1},
        ),
        (
            "torch.cuda.synchronize()",
            lambda i: torch.cuda.synchronize(),
            {"ds": 1},
            {"explicit": 1},
        ),
        (
            "Stream.synchronize()",
            lambda i: torch.cuda.current_stream().synchronize(),
            {"ss": 1},
            {"explicit": 1},
        ),
        ("Event.synchronize()", event_sync, {"es": 1}, {"explicit": 1}),
        (
            "D2H pageable non_blocking (no sync call)",
            lambda i: pageable.copy_(x[:8], non_blocking=True),
            {"pd": 1},
            {},
        ),
        ("cudaMalloc on the hot path", alloc, {"al": 1}, {}),
    ]
    # cases that go through dynamicemb_extensions: only where the function exists on this tree
    if hasattr(ext, "gather_embedding"):
        cases.insert(
            3,
            (
                "negative control: ext call without a host read",
                lambda i: ext.gather_embedding(emb_in, emb_out, gidx),
                {},
                {},
            ),
        )
    if hasattr(ext, "device_timestamp"):
        cases.append(
            (
                "ext raw cudaStreamSynchronize (device_timestamp)",
                lambda i: ext.device_timestamp(),
                {"ss": 1},
                {"cpp_raw": 1},
            )
        )
    if hasattr(ext, "flagged_compact"):
        cases.append(
            (
                "ext ATen .cpu() inside C++ (flagged_compact)",
                lambda i: ext.flagged_compact(mask, [x]),
                {"ss": 1},
                {"cpp_aten": 1},
            )
        )
    # Informational: idioms that look sync-free.  Measured and printed, never part of pass/fail.
    idx = torch.tensor([1, 5, 9], device=dev)
    neg1 = torch.tensor(-1, device=dev)
    ybuf = torch.zeros(64, dtype=torch.int64, device=dev)
    info_cases = [
        (
            "y[idx_tensor] = -1   (python scalar into index_put)",
            lambda i: ybuf.__setitem__(idx, -1),
        ),
        ("y[idx_tensor] = cuda 0-dim tensor", lambda i: ybuf.__setitem__(idx, neg1)),
        ("y.index_fill_(0, idx, -1)", lambda i: ybuf.index_fill_(0, idx, -1)),
        (
            "y[bool_mask] = -1    (masked_fill_)",
            lambda i: ybuf.__setitem__(ybuf > 100, -1),
        ),
        ("y[idx_tensor] = tensor_of_same_shape", lambda i: ybuf.__setitem__(idx, idx)),
        ("torch.where(mask, a, b)", lambda i: torch.where(mask, x, x + 1)),
        ("x.sum() (a CUDA 0-dim result, not read)", lambda i: x.sum()),
    ]
    rows: List[Dict[str, Any]] = []
    ok_all = True
    for name, fn, exp_kinds, exp_org in cases:
        try:
            fn(-1)  # warm-up (allocator, lazy init)
            torch.cuda.synchronize()
            probe = Probe(use_profiler=True)
            probe._label = "selftest"
            with probe:
                for i in range(iters):
                    with probe.iteration(i):
                        fn(i)
                    torch.cuda.synchronize()
            r = _summarize(probe, iters)
        except Exception as e:  # noqa: BLE001
            ok_all = False
            rows.append(
                {
                    "case": name,
                    "expected": exp_kinds,
                    "got": {},
                    "origin": {},
                    "python_visible": {},
                    "notes": ["raised %s: %s" % (type(e).__name__, str(e)[:120])],
                    "ok": False,
                }
            )
            if verbose:
                print(
                    "  %-4s %-52s raised %s: %s"
                    % ("FAIL", name, type(e).__name__, str(e)[:100]),
                    flush=True,
                )
            continue
        got = {k: r["mean"][k] for k in KINDS if r["mean"][k]}
        want = {k: float(v) for k, v in exp_kinds.items()}
        good = got == want
        for k, v in exp_org.items():
            ov = r["origin"].get(k, 0.0)
            pv = r["python_visible"].get(k, ov) if k in ("explicit",) else ov
            if k == "py_warn":
                pv = r["python_visible"]["warn"]
            if k == "explicit":
                pv = r["python_visible"]["explicit"]
            good = good and abs(pv - v) < 1e-9
        if r["notes"]:
            good = False
        ok_all = ok_all and good
        rows.append(
            {
                "case": name,
                "expected": want,
                "got": got,
                "origin": r["origin"],
                "python_visible": r["python_visible"],
                "notes": r["notes"],
                "ok": good,
            }
        )
        if verbose:
            print(
                "  %-4s %-52s expected=%-12s got=%-12s %s"
                % (
                    "PASS" if good else "FAIL",
                    name,
                    want or "{}",
                    got or "{}",
                    ("notes=%s" % r["notes"]) if r["notes"] else "",
                ),
                flush=True,
            )
        torch.cuda.empty_cache()
    if verbose:
        print(
            "  informational (measured, not part of the verdict): syncs per iteration of common idioms"
        )
    for name, fn in info_cases:
        try:
            fn(-1)
            torch.cuda.synchronize()
            probe = Probe(use_profiler=True)
            probe._label = "selftest_info"
            with probe:
                for i in range(iters):
                    with probe.iteration(i):
                        fn(i)
                    torch.cuda.synchronize()
            r = _summarize(probe, iters)
            got = {k: r["mean"][k] for k in KINDS if r["mean"][k]}
        except Exception as e:  # noqa: BLE001
            got = {"error": "%s: %s" % (type(e).__name__, str(e)[:80])}
        rows.append({"case": name, "info": True, "got": got, "ok": True})
        if verbose:
            print("  INFO %-52s syncs/iter=%s" % (name, got or "{}"), flush=True)
    return ok_all, rows


# --------------------------------------------------------------------------
# Before / after comparison of two reports (CPU only)
# --------------------------------------------------------------------------
def _effective_total(r: Dict[str, Any]) -> float:
    """Mean syncs per iteration outside INHERENT call sites (0 for a whole-case INHERENT)."""
    site_excused = r.get("inherent_site_per_iter") or [0] * len(r["per_iter_total"])
    eff = [t - h for t, h in zip(r["per_iter_total"], site_excused)]
    return sum(eff) / float(len(eff))


def compare_reports(
    before: Dict[str, Any], after: Dict[str, Any], tol: float = 1e-9
) -> Dict[str, Any]:
    """Per (config, scenario, mode): mean syncs/iter before and after.  "regressed"
    = more syncs after than before; "remaining" = after-cases that still sync
    outside INHERENT; "missing" = cases of `before` that `after` did not measure."""
    key = lambda r: (r["config"], r["scenario"], r["mode"])  # noqa: E731
    b = collections.OrderedDict(
        (key(r), r) for r in before["results"] if r["status"] == "ok"
    )
    a = {key(r): r for r in after["results"] if r["status"] == "ok"}
    rows: List[Dict[str, Any]] = []
    missing: List[Tuple[str, str, str]] = []
    for k, rb in b.items():
        ra = a.get(k)
        if ra is None:
            missing.append(k)
            continue
        tb, ta = rb["mean"]["total"], ra["mean"]["total"]
        rows.append(
            {
                "case": k,
                "before": tb,
                "after": ta,
                "after_effective": _effective_total(ra),
                "inherent": bool(ra.get("inherent")),
                "delta": ta - tb,
            }
        )
    return {
        "rows": rows,
        "improved": [r for r in rows if r["delta"] < -tol],
        "regressed": [r for r in rows if r["delta"] > tol],
        "unchanged": [r for r in rows if abs(r["delta"]) <= tol],
        "remaining": [
            r for r in rows if not r["inherent"] and r["after_effective"] > tol
        ],
        "missing": missing,
    }


def format_compare(cmp: Dict[str, Any], only_changed: bool = False) -> str:
    L = [
        "%-24s %-10s %-8s %8s %8s %8s  %s"
        % ("config", "scenario", "mode", "before", "after", "delta", "")
    ]
    for r in cmp["rows"]:
        if only_changed and abs(r["delta"]) < 1e-9:
            continue
        tag = (
            "INHERENT"
            if r["inherent"]
            else ("STILL SYNCS" if r["after_effective"] > 1e-9 else "zero")
        )
        L.append(
            "%-24s %-10s %-8s %8s %8s %+8.3g  %s"
            % (
                r["case"][0],
                r["case"][1],
                r["case"][2],
                "%.3g" % r["before"],
                "%.3g" % r["after"],
                r["delta"],
                tag,
            )
        )
    L.append("")
    L.append(
        "%d cases: %d improved, %d unchanged, %d regressed, %d still sync outside INHERENT, %d missing "
        "from the second report"
        % (
            len(cmp["rows"]),
            len(cmp["improved"]),
            len(cmp["unchanged"]),
            len(cmp["regressed"]),
            len(cmp["remaining"]),
            len(cmp["missing"]),
        )
    )
    return "\n".join(L)


# --------------------------------------------------------------------------
# Static inventory of sync patterns (cross-check for what the dynamic run saw)
# --------------------------------------------------------------------------
PY_PAT = re.compile(
    r"\.item\(\)|\.cpu\(\)|\.tolist\(\)|synchronize\(|\.nonzero\(|nonzero\(|masked_select|"
    r"torch\.unique|repeat_interleave|\.any\(\)|\.all\(\)|non_blocking=False|"
    r"\bbool\(|\bint\([^)]*\.(sum|max|min)\(|torch\.cuda\.synchronize"
)
CPP_PAT = re.compile(
    r"cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|cudaMemcpy\(|"
    r"\.cpu\(\)|\.item<|\.item\(\)|cudaMalloc\(|cudaFree\(|cudaHostAlloc|cudaMallocHost|"
    r"\.sync\(|synchronize"
)


def static_scan(tree: str) -> List[Tuple[str, int, str]]:
    rows: List[Tuple[str, int, str]] = []
    for sub, pat, exts in (
        ("dynamicemb", PY_PAT, (".py",)),
        ("src", CPP_PAT, (".cu", ".cuh", ".h", ".cpp")),
    ):
        for root, _, files in os.walk(os.path.join(tree, sub)):
            for fn in sorted(files):
                if not fn.endswith(exts):
                    continue
                p = os.path.join(root, fn)
                try:
                    with open(p, errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            s = line.strip()
                            if s.startswith("#") or s.startswith("//"):
                                continue
                            if pat.search(s):
                                rows.append((os.path.relpath(p, tree), i, s[:120]))
                except Exception:
                    pass
    return rows


# --------------------------------------------------------------------------
# Text report
# --------------------------------------------------------------------------
def _f(x: float) -> str:
    return "%.2f" % x if abs(x - round(x)) > 1e-9 else "%d" % int(round(x))


def format_pivot(report: Dict[str, Any]) -> str:
    """config x scenario matrix of mean syncs per iteration, one block per mode."""
    L: List[str] = []
    scs = [k for k in SCENARIO_DOC]
    for mode in MODES:
        rows = collections.OrderedDict()
        for r in report["results"]:
            if r.get("mode") != mode and not (
                mode == "plain" and r.get("mode") in ("*",)
            ):
                continue
            rows.setdefault(r["config"], {})[r["scenario"]] = r
        if not rows:
            continue
        L.append(
            "PIVOT mode=%s (syncs per iteration; '*' = case marked INHERENT, '(-n)' = n of them are in "
            "INHERENT call sites, '[a-b]' = per-iteration range, 'E' = error, '-' = n/a)"
            % mode
        )
        table: List[List[str]] = []
        for cfg, d in rows.items():
            cells = []
            for sc in scs:
                r = d.get(sc)
                if r is None or r["status"] == "skipped":
                    cells.append("-")
                elif r["status"] != "ok":
                    cells.append("E")
                else:
                    pt = r["per_iter_total"]
                    c = "%.3g" % r["mean"]["total"]
                    if min(pt) != max(pt):
                        c += "[%d-%d]" % (min(pt), max(pt))
                    if r.get("inherent_site_mean"):
                        c += "(-%.3g)" % r["inherent_site_mean"]
                    if r.get("inherent"):
                        c += "*"
                    cells.append(c)
            table.append([cfg] + cells)
        widths = [
            max(len(scs[j][:11]), *(len(row[j + 1]) for row in table)) + 2
            for j in range(len(scs))
        ]
        L.append(
            "%-24s" % "config"
            + "".join(("%" + str(w) + "s") % sc[:11] for w, sc in zip(widths, scs))
        )
        for row in table:
            L.append(
                "%-24s" % row[0]
                + "".join(("%" + str(w) + "s") % c for w, c in zip(widths, row[1:]))
            )
        L.append("")
    return "\n".join(L)


def format_top_sites(report: Dict[str, Any], mode: str = "plain", n: int = 30) -> str:
    """Call sites ranked by the number of cases they sync in (one mode)."""
    agg: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    ncases = 0
    for r in report["results"]:
        if r["status"] != "ok" or r["mode"] != mode:
            continue
        ncases += 1
        for s in r.get("sites", []):
            key = (s["source"], s["site"], s.get("api", ""))
            a = agg.setdefault(
                key,
                {"cases": 0, "max": 0.0, "ops": set(), "snippet": s.get("snippet", "")},
            )
            a["cases"] += 1
            a["max"] = max(a["max"], s["per_iter"])
            a["ops"].update(s.get("ops", {}) or [s.get("api", "")])
    rows = sorted(
        agg.items(), key=lambda kv: (-kv[1]["cases"], -kv[1]["max"], kv[0][1])
    )
    L = [
        "TOP CALL SITES across the %d measured cases of mode=%s (cases = in how many it syncs, "
        "max/it = the largest count per iteration in any case)" % (ncases, mode)
    ]
    L.append("%6s %7s  %-9s %s" % ("cases", "max/it", "source", "site (api / op)"))
    for (src, site, api), a in rows[:n]:
        what = api or ",".join(sorted(x for x in a["ops"] if x))
        L.append("%6d %7s  %-9s %s  [%s]" % (a["cases"], _f(a["max"]), src, site, what))
        if a["snippet"]:
            L.append("%17s| %s" % ("", a["snippet"]))
    return "\n".join(L)


def format_report(report: Dict[str, Any], sites: int = 6, detail: str = "all") -> str:
    L: List[str] = []
    m = report["meta"]
    L.append(
        "host_sync_probe: %s | torch %s cuda %s | %s | dynamicemb %s (git %s) %s"
        % (
            m.get("time"),
            m.get("torch"),
            m.get("cuda"),
            m.get("device"),
            m.get("dynamicemb"),
            m.get("git"),
            ("| " + m["label"]) if m.get("label") else "",
        )
    )
    L.append(
        "method: one measured step = prefetch (mode=prefetch) or the forward's internal prefetch (mode=plain) "
        "+ forward + backward (train) or no_grad forward (eval).  Counted per iteration, on all threads "
        "(the backward runs on an autograd worker): CUDA runtime syncs/allocs from the torch profiler "
        "(authority), python-level syncs from sync-debug warnings (matched to profiler events) and "
        "interposed torch.cuda sync APIs; every dynamicemb_extensions call is wrapped so a sync inside C++ "
        "gets the Python line that called it.  Warm-up and barriers are outside the windows."
    )
    L.append(
        "iters=%s warmup=%s batch=%s"
        % (m.get("iters"), m.get("warmup"), m.get("batch"))
    )
    L.append("kinds: " + "; ".join("%s=%s" % (k, KIND_DOC[k]) for k in KINDS))
    L.append(
        "columns: tot=syncs/iter (profiler authority, merged with python-visible); "
        "py=python-delivered sync warnings+explicit sync calls; "
        "cppA=sync inside a C++ ext call under an ATen op (.cpu()/.item() issued from C++); "
        "cppR=raw CUDA runtime sync/alloc inside a C++ ext call or outside any ATen op; "
        "[min-max]=per-iteration range of tot"
    )
    L.append("")
    L.append(format_pivot(report))
    L.append(format_top_sites(report))
    L.append("")
    hdr = "%-24s %-10s %-8s %6s | %5s %5s %5s %5s %5s %5s | %5s %5s %5s %5s | %s" % (
        "config",
        "scenario",
        "mode",
        "tot",
        "ss",
        "ds",
        "es",
        "ms",
        "pd",
        "al",
        "py",
        "cppA",
        "cppR",
        "exc",
        "flags",
    )
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in report["results"]:
        if r["status"] != "ok":
            L.append(
                "%-24s %-10s %-8s %6s | %s"
                % (
                    r["config"],
                    r["scenario"],
                    r["mode"],
                    r["status"].upper(),
                    (r.get("error") or "")[:110],
                )
            )
            continue
        mk = r["mean"]
        pt = r["per_iter_total"]
        flags = []
        if r.get("inherent"):
            flags.append("INHERENT")
        elif r.get("inherent_site_mean"):
            flags.append("excused-sites")
        if not r["regime"]["ok"]:
            flags.append("REGIME?")
        if r.get("notes"):
            flags.append("notes")
        rng = "" if min(pt) == max(pt) else " [%d-%d]" % (min(pt), max(pt))
        org = r["origin"]
        py = r["python_visible"]["warn"] + r["python_visible"]["explicit"]
        cppr = org["cpp_raw"] + org["direct_raw"]
        L.append(
            "%-24s %-10s %-8s %6s | %5s %5s %5s %5s %5s %5s | %5s %5s %5s %5s | %s%s"
            % (
                r["config"],
                r["scenario"],
                r["mode"],
                _f(mk["total"]),
                _f(mk["ss"]),
                _f(mk["ds"]),
                _f(mk["es"]),
                _f(mk["ms"]),
                _f(mk["pd"]),
                _f(mk["al"]),
                _f(py),
                _f(org["cpp_aten"]),
                _f(cppr),
                _f(r.get("inherent_site_mean", 0.0)),
                " ".join(flags),
                rng,
            )
        )
    if detail == "none":
        return "\n".join(L)
    L.append("")
    L.append("=" * 100)
    L.append(
        "CALL SITES (syncs per iteration, ranked).  src: py=python warning, explicit=torch.cuda sync API,"
    )
    L.append(
        "cpp=inside a C++ extension call (call site is the Python line that made the call)"
    )
    L.append("=" * 100)
    for r in report["results"]:
        if r["status"] != "ok":
            if r["status"] == "error":
                L.append("")
                L.append(
                    "### %s / %s / %s: ERROR %s"
                    % (r["config"], r["scenario"], r["mode"], r.get("error"))
                )
                tb = r.get("traceback", "")
                if tb:
                    L.extend("    " + ln for ln in tb.splitlines()[-8:])
            continue
        if detail == "plain" and r["mode"] != "plain":
            continue
        L.append("")
        L.append(
            "### %s / %s / %s  tot=%s/iter  regime=%s (%s)%s"
            % (
                r["config"],
                r["scenario"],
                r["mode"],
                _f(r["mean"]["total"]),
                "ok" if r["regime"]["ok"] else "SUSPECT",
                r["regime"]["note"],
                "  INHERENT: " + r["inherent"] if r.get("inherent") else "",
            )
        )
        for n in r.get("notes", []):
            L.append("    note: " + n)
        if not r["sites"]:
            L.append("    (no host sync)")
        for s in r["sites"][: sites if detail != "full" else 999]:
            extra = ""
            if s["source"] in ("cpp", "direct_raw", "direct_aten"):
                extra = " %s %s" % (s["api"], s.get("origin", ""))
                extra += " " + ",".join("%s" % k for k in s["kinds"])
            elif s["source"] == "explicit":
                extra = " " + s["api"]
            elif s["source"] == "py_warn" and s.get("ops"):
                extra = (
                    " "
                    + ",".join(sorted(s["ops"]))
                    + " "
                    + ",".join(sorted(s["kinds"]))
                )
            L.append(
                "    %6s/it  %-9s %s%s"
                % (_f(s["per_iter"]), s["source"], s["site"], extra)
            )
            if s["snippet"]:
                L.append("                      | " + s["snippet"])
            if s["chain"] and detail == "full":
                L.append("                      <- " + " <- ".join(s["chain"]))
        extra_sites = len(r["sites"]) - sites
        if extra_sites > 0 and detail != "full":
            L.append("    ... %d more sites" % extra_sites)
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _run_isolated(args, cfg_names: List[str]) -> Dict[str, Any]:
    """One subprocess per config, so a crash (sticky CUDA error) loses one config."""
    allres: List[Dict[str, Any]] = []
    meta = None
    for name in cfg_names:
        fd, tmp = tempfile.mkstemp(prefix="hsp_", suffix=".json")
        os.close(fd)
        cmd = [
            sys.executable,
            _THIS_FILE,
            "--worker",
            "--configs",
            name,
            "--json",
            tmp,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--modes",
            args.modes,
            "--label",
            args.label,
            "--quiet-text",
        ]
        if args.scenarios:
            cmd += ["--scenarios", args.scenarios]
        if args.no_profiler:
            cmd += ["--no-profiler"]
        if args.stacks:
            cmd += ["--stacks"]
        if args.keep_trace:
            cmd += ["--keep-trace", args.keep_trace]
        t0 = time.time()
        print("[probe] config %s ..." % name, file=sys.stderr, flush=True)
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.config_timeout,
            )
        except subprocess.TimeoutExpired:
            print(
                "[probe]   TIMEOUT in config %s after %ds"
                % (name, args.config_timeout),
                file=sys.stderr,
                flush=True,
            )
            allres.append(
                {
                    "config": name,
                    "scenario": "*",
                    "mode": "*",
                    "layout": CONFIGS[name].layout,
                    "status": "error",
                    "error": "worker timed out after %ds" % args.config_timeout,
                    "inherent": None,
                }
            )
            try:
                os.remove(tmp)
            except Exception:
                pass
            continue
        try:
            with open(tmp) as f:
                rep = json.load(f)
            meta = meta or rep["meta"]
            allres.extend(rep["results"])
            bad = [r for r in rep["results"] if r["status"] == "error"]
            print(
                "[probe]   done %s in %.0fs: %d cases, %d errors"
                % (name, time.time() - t0, len(rep["results"]), len(bad)),
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            tail = (p.stderr or "")[-600:]
            print(
                "[probe]   CRASH in config %s (exit %s): %s"
                % (name, p.returncode, tail[-200:]),
                file=sys.stderr,
                flush=True,
            )
            allres.append(
                {
                    "config": name,
                    "scenario": "*",
                    "mode": "*",
                    "layout": CONFIGS[name].layout,
                    "status": "error",
                    "error": "worker crashed, exit=%s: %s" % (p.returncode, tail),
                    "inherent": None,
                }
            )
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
    return {
        "meta": meta or {},
        "inherent": {"|".join(k): v for k, v in INHERENT.items()},
        "results": allres,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples (GPU runs need the built dynamicemb on PYTHONPATH; run --selftest first):\n"
            "  python host_sync_probe.py --selftest\n"
            "  python host_sync_probe.py --json after.json --text after.txt --compare before.json --assert-zero\n"
            "  python host_sync_probe.py --configs hbm_only,hybrid --scenarios all_hit,miss --modes plain --iters 4\n"
            "  python host_sync_probe.py --diff before.json after.json     # CPU only\n"
            "  python host_sync_probe.py --render after.json               # CPU only\n"
            "  python host_sync_probe.py --static --tree corelib/dynamicemb  # CPU only\n"
            "from python: report = measure(configs=[...]); assert not check_zero(report)\n"
            "budgets.json is regenerated from a --json report with sync_budgets.py --from-report"
        ),
    )
    ap.add_argument("--configs", default="all", help="comma list or 'all' (see --list)")
    ap.add_argument(
        "--scenarios",
        default="",
        help="comma list; default = the scenarios of each config",
    )
    ap.add_argument(
        "--modes",
        default="plain,prefetch",
        help="plain (forward runs the prefetch) and/or prefetch",
    )
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--label",
        default="",
        help="free text recorded in the report (e.g. the commit measured)",
    )
    ap.add_argument("--json", default="", help="write the structured report here")
    ap.add_argument("--text", default="", help="write the text report here too")
    ap.add_argument(
        "--detail",
        default="all",
        choices=["none", "plain", "all", "full"],
        help="how many call-site sections to print (full: all sites + call chains)",
    )
    ap.add_argument(
        "--sites", type=int, default=6, help="call sites per case in the text report"
    )
    ap.add_argument(
        "--assert-zero",
        action="store_true",
        help="exit 1 if a case not in INHERENT syncs (2 if a case errored)",
    )
    ap.add_argument(
        "--list", action="store_true", help="list configs and scenarios, then exit"
    )
    ap.add_argument(
        "--no-isolate", action="store_true", help="run all configs in this process"
    )
    ap.add_argument(
        "--no-profiler",
        action="store_true",
        help="python-only counting (blind to syncs inside C++)",
    )
    ap.add_argument(
        "--stacks", action="store_true", help="profiler with_stack (bigger traces)"
    )
    ap.add_argument(
        "--keep-trace", default="", help="directory to keep the chrome traces in"
    )
    ap.add_argument(
        "--static",
        action="store_true",
        help="print the static inventory of sync patterns and exit",
    )
    ap.add_argument(
        "--tree",
        default="",
        help="with --static: the corelib/dynamicemb directory to scan",
    )
    ap.add_argument(
        "--diff",
        nargs=2,
        metavar=("BEFORE.json", "AFTER.json"),
        help="CPU only: compare two saved reports case by case, then exit "
        "(exit 1 if AFTER regressed anywhere or still syncs outside INHERENT)",
    )
    ap.add_argument(
        "--render",
        default="",
        metavar="REPORT.json",
        help="CPU only: print the text report of a saved JSON report, then exit",
    )
    ap.add_argument(
        "--compare",
        default="",
        metavar="BEFORE.json",
        help="after the run, also print the comparison of the new report with BEFORE.json",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="check the instruments against synthetic workloads with known sync counts",
    )
    ap.add_argument("--config-timeout", type=int, default=1500)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--quiet-text", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.list:
        print("configs:")
        for c in CONFIGS.values():
            print(
                "  %-24s layout=%-8s scenarios=%s  %s"
                % (c.name, c.layout, ",".join(scenarios_of(c)), c.doc)
            )
        print("scenarios:")
        for k, v in SCENARIO_DOC.items():
            print("  %-11s %s" % (k, v))
        print("inherent (allowed to sync):")
        for k, v in INHERENT.items():
            print(
                "  %s / %s%s : %s"
                % (k[0], k[1], (" / sites " + k[2]) if len(k) == 3 else "", v)
            )
        return 0
    if args.render:
        with open(args.render) as f:
            print(format_report(json.load(f), sites=args.sites, detail=args.detail))
        return 0
    if args.diff:
        with open(args.diff[0]) as f:
            before = json.load(f)
        with open(args.diff[1]) as f:
            after = json.load(f)
        cmp = compare_reports(before, after)
        print(format_compare(cmp))
        return 1 if (cmp["regressed"] or cmp["remaining"]) else 0
    if args.selftest:
        print(
            "host_sync_probe self test (known syncs per iteration; negative controls must read 0):"
        )
        ok, _ = selftest(args.device, iters=args.iters if args.iters != 8 else 4)
        print("SELFTEST: %s" % ("ok" if ok else "FAILED"))
        return 0 if ok else 1
    if args.static:
        # CPU only: scan the tree given by --tree (default: the one dynamicemb is imported from)
        tree = args.tree
        if not tree:
            import dynamicemb

            tree = os.path.dirname(
                os.path.dirname(os.path.abspath(dynamicemb.__file__))
            )
        for f, i, s in static_scan(tree):
            print("%s:%d: %s" % (f, i, s))
        return 0

    cfg_names = (
        list(CONFIGS)
        if args.configs == "all"
        else [c for c in args.configs.split(",") if c]
    )
    for c in cfg_names:
        if c not in CONFIGS:
            print("unknown config %r; see --list" % c, file=sys.stderr)
            return 2
    scen = [s for s in args.scenarios.split(",") if s] or None
    modes = [m for m in args.modes.split(",") if m]
    if args.worker or args.no_isolate or len(cfg_names) == 1 and not args.worker:
        report = measure(
            cfg_names,
            scen,
            modes,
            iters=args.iters,
            warmup=args.warmup,
            device=args.device,
            seed=args.seed,
            profiler=not args.no_profiler,
            stacks=args.stacks,
            keep_trace_dir=args.keep_trace or None,
            verbose=not args.worker,
            label=args.label,
        )
    else:
        report = _run_isolated(args, cfg_names)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
    if args.worker and args.quiet_text:
        return 0
    text = format_report(report, sites=args.sites, detail=args.detail)
    print(text)
    if args.text:
        with open(args.text, "w") as f:
            f.write(text + "\n")
    if args.compare:
        with open(args.compare) as f:
            cmp = compare_reports(json.load(f), report)
        print("")
        print("COMPARISON with %s" % args.compare)
        print(format_compare(cmp))
    if args.assert_zero:
        bad = check_zero(report)
        errs = [b for b in bad if ": ERROR" in b]
        print("")
        if bad:
            print("ASSERT-ZERO: %d violation(s)" % len(bad))
            for b in bad:
                print("  " + b)
            return 2 if errs and len(errs) == len(bad) else 1
        print("ASSERT-ZERO: ok (every non-inherent case has zero host syncs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
