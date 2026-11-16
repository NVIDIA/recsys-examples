# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct-parquet per-stream GPU timeline with source attribution.

For a time window, slice each CUDA stream into ``n_buckets`` equal segments
and, per bucket, report:
  * how busy the stream was — GPU execution time, idle, utilisation,
  * how many kernels / H2D / D2H / D2D copies ran,
  * where the busy time came from — the source line of the launching
    ``torch.*`` call (``file:line``), recovered by correlating each CUDA
    event to the innermost ``talos::`` NVTX range open on its launching thread.

A kernel that spans a bucket boundary is **clipped**: each bucket gets only
the slice of execution that falls inside its window (so utilisation is always
≤ 100% and a long-running kernel shows as busy across every bucket it covers).
A kernel is *counted* in the bucket its start falls in (left-closed,
right-open). Both forward and backward kernels attribute to a ``file:line``:
stats.py copies the forward call's label verbatim onto each backward op, so a
backward kernel shows the same source line as its forward.
"""

from __future__ import annotations

import bisect
import os
import re
from collections import defaultdict, namedtuple

import pyarrow.parquet as pq

# One talos:: function instance's identity. ``path``/``call_site`` are None
# for a free call (no module / no resolved source).
_FuncMeta = namedtuple("_FuncMeta", "path func call_site")

# talos::<func>[#<path>@<Class>][!!<file>:<line>]. Forward and synthesized
# backward share the label (the synth copies it verbatim); the timeline only
# needs the source line, never the direction, so direction isn't parsed here.
_TALOS_RE = re.compile(r"^talos::([^#!]+)(?:#(.+)@([^!@]+))?(?:!!(.+))?$")
# The observer wraps every backward node in this range; used to drop the
# loss.backward() driver range.
_AUTOGRAD = "autograd::engine::evaluate_function"


# ─── parquet loading ────────────────────────────────────────────────


def _read(parquetdir: str, table: str, columns: list[str] | None = None) -> list[dict]:
    path = os.path.join(parquetdir, table + ".parquet")
    if not os.path.exists(path):
        return []
    return pq.read_table(path, columns=columns).to_pylist()


def _string_ids(parquetdir: str) -> dict[int, str]:
    return {
        r["id"]: r.get("value")
        for r in _read(parquetdir, "StringIds")
        if r.get("id") is not None
    }


def _copy_kind_map(parquetdir: str) -> dict[int, str]:
    """copyKind id → 'h2d' | 'd2h' | 'd2d' | 'other' (from ENUM_CUDA_MEMCPY_OPER)."""
    out: dict[int, str] = {}
    for r in _read(parquetdir, "ENUM_CUDA_MEMCPY_OPER"):
        name = (r.get("name") or "").lower()
        if "host-to-device" in name or "htod" in name:
            out[r["id"]] = "h2d"
        elif "device-to-host" in name or "dtoh" in name:
            out[r["id"]] = "d2h"
        elif "device-to-device" in name or "dtod" in name:
            out[r["id"]] = "d2d"
        else:
            out[r["id"]] = "other"
    return out


def _parse_ta(text: str) -> tuple[str, str | None, str | None, str | None] | None:
    """``(func, path, cls, call_site)`` or ``None``."""
    m = _TALOS_RE.match(text)
    if m is None:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def _source_of(m: _FuncMeta) -> str:
    """Readable source key: the ``file:line`` call site when known, else the
    module path / function name."""
    return m.call_site or m.path or m.func or "?"


# ─── talos:: function ranges, per launching thread ────────────────────


class _FuncIndex:
    def __init__(self):
        self._by_tid: dict[int, list[tuple[int, int, int]]] = defaultdict(
            list
        )  # tid → [(start,end,fid)]
        self.meta: list[_FuncMeta] = []
        self._ag: list[tuple[int, int]] = []  # autograd intervals (any thread)
        self._ag_tids: set[
            int
        ] = set()  # threads that run the autograd engine (backward)

    def add_range(self, tid, start, end, m: _FuncMeta):
        self._by_tid[tid].append((start, end, len(self.meta)))
        self.meta.append(m)

    def add_autograd(self, tid, start, end):
        self._ag.append((start, end))
        self._ag_tids.add(tid)

    def seal_autograd(self):
        self._ag.sort()
        self._ag_starts = [iv[0] for iv in self._ag]

    def finalize(self):
        for ranges in self._by_tid.values():
            ranges.sort()
        self._starts = {tid: [r[0] for r in rs] for tid, rs in self._by_tid.items()}

    def lookup(self, tid, t) -> int | None:
        """Innermost function range on ``tid`` containing time ``t``."""
        ranges = self._by_tid.get(tid)
        if not ranges:
            return None
        starts = self._starts[tid]
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0:
            _s, e, fid = ranges[i]
            if e >= t:
                return fid
            i -= 1
        return None

    def encloses_autograd(self, start, end) -> bool:
        """``[start, end]`` contains the start of an autograd interval — i.e.
        the ``loss.backward()`` driver call, not a compute op."""
        i = bisect.bisect_left(self._ag_starts, start)
        return i < len(self._ag_starts) and self._ag_starts[i] <= end


def _build_func_index(
    parquetdir: str, sid: dict[int, str], s_ns: int, e_ns: int
) -> _FuncIndex:
    ta_ranges: list[tuple] = []  # (tid, start, end, _FuncMeta)
    idx = _FuncIndex()
    for r in _read(
        parquetdir, "NVTX_EVENTS", ["start", "end", "text", "textId", "globalTid"]
    ):
        start, end, tid = r.get("start"), r.get("end"), r.get("globalTid")
        if start is None or end is None or tid is None or end < s_ns or start > e_ns:
            continue
        text = r.get("text") or (
            sid.get(r["textId"]) if r.get("textId") is not None else None
        )
        if not text:
            continue
        if text.startswith("talos::"):
            parsed = _parse_ta(text)
            if parsed is not None:
                func, path, _cls, call_site = parsed
                ta_ranges.append((tid, start, end, _FuncMeta(path, func, call_site)))
        elif text.startswith(_AUTOGRAD):
            idx.add_autograd(tid, start, end)

    idx.seal_autograd()
    for tid, start, end, m in ta_ranges:
        # Drop the backward-driver range (loss.backward()): it encloses the
        # autograd engine and would shadow the real per-op ranges. A genuine
        # leaf op (forward or synthesized backward) never encloses one.
        if idx.encloses_autograd(start, end):
            continue
        idx.add_range(tid, start, end, m)
    idx.finalize()
    return idx


# ─── per-stream, per-bucket timeline ───────────────────────────────


def _gpu_events(parquetdir: str, copy_kinds: dict, s_ns: int, e_ns: int):
    """Yield ``(start, end, stream, kind, correlationId)`` for every kernel /
    memcpy / memset that executes within ``[s_ns, e_ns)``. ``kind`` ∈
    {'kernel','h2d','d2h','d2d','other','memset'}."""
    for r in _read(
        parquetdir,
        "CUPTI_ACTIVITY_KIND_KERNEL",
        ["start", "end", "streamId", "correlationId"],
    ):
        if r["end"] > s_ns and r["start"] < e_ns:
            yield r["start"], r["end"], r.get("streamId"), "kernel", r.get(
                "correlationId"
            )
    for r in _read(
        parquetdir,
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        ["start", "end", "streamId", "correlationId", "copyKind"],
    ):
        if r["end"] > s_ns and r["start"] < e_ns:
            yield (
                r["start"],
                r["end"],
                r.get("streamId"),
                copy_kinds.get(r.get("copyKind"), "other"),
                r.get("correlationId"),
            )
    for r in _read(
        parquetdir,
        "CUPTI_ACTIVITY_KIND_MEMSET",
        ["start", "end", "streamId", "correlationId"],
    ):
        if r["end"] > s_ns and r["start"] < e_ns:
            yield r["start"], r["end"], r.get("streamId"), "memset", r.get(
                "correlationId"
            )


def _new_bucket() -> dict:
    return {
        "kernel": 0,
        "h2d": 0,
        "d2h": 0,
        "d2d": 0,
        "memset": 0,
        "kernel_ns": 0,
        "memcpy_ns": 0,
        "memset_ns": 0,
        "src": defaultdict(lambda: [0, 0]),
    }  # source → [gpu_ns, events]


def _ms(ns) -> float:
    return round(ns / 1e6, 4)


def build_timeline(
    parquetdir: str, start_ns: int, end_ns: int, n_buckets: int = 100
) -> dict:
    """Per-stream, per-bucket GPU timeline with source-line attribution for
    ``[start_ns, end_ns)``. Returns ``{window, streams: {id: {...}}}``;
    ``streams`` is empty when the parquet tables are missing/empty."""
    sid = _string_ids(parquetdir)
    copy_kinds = _copy_kind_map(parquetdir)
    funcs = _build_func_index(parquetdir, sid, start_ns, end_ns)

    launch: dict[
        int, tuple[int, int]
    ] = {}  # correlationId → (launch_tid, launch_start)
    rt_cover: dict[
        int, int
    ] = {}  # globalTid → CUDA-API coverage ns (clipped to window)
    for r in _read(
        parquetdir,
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        ["start", "end", "globalTid", "correlationId"],
    ):
        cid, tid, st, en = (
            r.get("correlationId"),
            r.get("globalTid"),
            r.get("start"),
            r.get("end"),
        )
        if cid is not None and tid is not None and st is not None:
            launch[cid] = (tid, st)
        if (
            tid is not None
            and st is not None
            and en is not None
            and en > start_ns
            and st < end_ns
        ):
            rt_cover[tid] = rt_cover.get(tid, 0) + (min(en, end_ns) - max(st, start_ns))

    span_ns = end_ns - start_ns
    bucket_ns = max(span_ns // n_buckets, 1)

    def bucket_of(t):
        return min(max((t - start_ns) // bucket_ns, 0), n_buckets - 1)

    def bucket_end(bi):
        return end_ns if bi == n_buckets - 1 else start_ns + (bi + 1) * bucket_ns

    def source_of(cid):
        info = launch.get(cid)
        if info is None:
            return "(unattributed)"
        fid = funcs.lookup(*info)
        return _source_of(funcs.meta[fid]) if fid is not None else "(unattributed)"

    # stream id → {buckets: [...], src_total: source → [gpu_ns, events]}
    def _new_stream():
        return {
            "buckets": [_new_bucket() for _ in range(n_buckets)],
            "src_total": defaultdict(lambda: [0, 0]),
        }

    streams: dict[int, dict] = defaultdict(_new_stream)

    for start, end, stream, kind, cid in _gpu_events(
        parquetdir, copy_kinds, start_ns, end_ns
    ):
        es, ee = max(start, start_ns), min(end, end_ns)
        if ee <= es:
            continue
        sdata = streams[stream]
        buckets = sdata["buckets"]
        # count the event once, in the bucket its start falls in
        b0 = buckets[bucket_of(es)]
        b0[kind if kind in ("kernel", "h2d", "d2h", "d2d", "memset") else "kernel"] += 1
        src = source_of(cid)
        # whole-stream attribution: each event's clipped time, counted once
        st = sdata["src_total"][src]
        st[0] += ee - es
        st[1] += 1
        # per-bucket: distribute execution time (clipped) across spanned buckets
        time_key = (
            "kernel_ns"
            if kind == "kernel"
            else ("memset_ns" if kind == "memset" else "memcpy_ns")
        )
        first, last = bucket_of(es), bucket_of(ee - 1)
        for bi in range(first, last + 1):
            bs, be = start_ns + bi * bucket_ns, bucket_end(bi)
            ov = min(ee, be) - max(es, bs)
            if ov <= 0:
                continue
            b = buckets[bi]
            b[time_key] += ov
            sd = b["src"][src]
            sd[0] += ov
            sd[1] += 1

    return {
        "window": {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ms": _ms(span_ns),
            "bucket_ns": bucket_ns,
            "n_buckets": n_buckets,
        },
        "streams": {
            str(stream): _emit_stream(
                stream, sdata, start_ns, bucket_ns, end_ns, n_buckets, span_ns
            )
            for stream, sdata in sorted(streams.items())
        },
        "threads": _cpu_threads(
            funcs, rt_cover, start_ns, end_ns, n_buckets, bucket_ns
        ),
    }


# A thread with no talos:: ranges is listed only if its CUDA-API activity covers at
# least this fraction of the window — drops near-idle helper/comm threads.
_THREAD_ACTIVITY_FRAC = 0.01


def _cpu_threads(funcs, rt_cover, start_ns, end_ns, n_buckets, bucket_ns) -> list[dict]:
    """Per active CPU thread, the **torch-function utilisation** timeline: each
    of the ``n_buckets`` segments reports the fraction of that segment the
    thread spent inside a ``talos::`` range (a torch op's CPU span). High util ⇒
    the thread issues torch ops back-to-back (low glue overhead); low util ⇒
    the gaps are framework glue — Python on the forward/main thread, the C++
    autograd engine on the backward worker (the synthesized backward ranges
    cover the backward ops).

    Which threads: those that ran ≥1 ``talos::`` range (the torch threads —
    ``forward`` / ``backward``), **plus** any thread whose CUDA-API activity
    covers ≥ ``_THREAD_ACTIVITY_FRAC`` of the window — these have no torch ops
    (``role: other``; e.g. a comm / copy / CUDA-graph thread) so their
    ``util`` is ~0, but ``cuda_api_cover_ms`` shows they are busy. Near-idle
    helper threads are dropped."""
    span_ns = end_ns - start_ns
    min_cover = span_ns * _THREAD_ACTIVITY_FRAC

    def bend(bi):
        return end_ns if bi == n_buckets - 1 else start_ns + (bi + 1) * bucket_ns

    out = []
    for tid in set(funcs._by_tid) | set(rt_cover):
        ranges = funcs._by_tid.get(tid, [])
        rtc = rt_cover.get(tid, 0)
        if not ranges and rtc < min_cover:  # no torch ops + near-idle ⇒ drop
            continue
        cov = [0] * n_buckets
        for start, end, _fid in ranges:
            es, ee = max(start, start_ns), min(end, end_ns)
            if ee <= es:
                continue
            first = min(max((es - start_ns) // bucket_ns, 0), n_buckets - 1)
            last = min(max((ee - 1 - start_ns) // bucket_ns, 0), n_buckets - 1)
            for bi in range(first, last + 1):
                bs, be = start_ns + bi * bucket_ns, bend(bi)
                ov = min(ee, be) - max(es, bs)
                if ov > 0:
                    cov[bi] += ov
        util = [
            min(round(cov[bi] / (bend(bi) - (start_ns + bi * bucket_ns)), 4), 1.0)
            if bend(bi) > start_ns + bi * bucket_ns
            else 0.0
            for bi in range(n_buckets)
        ]
        role = (
            "backward" if tid in funcs._ag_tids else ("forward" if ranges else "other")
        )
        out.append(
            {
                "global_tid": tid,
                "role": role,
                "cover_ms": _ms(sum(cov)),  # torch-function (talos::) coverage
                "cuda_api_cover_ms": _ms(
                    rtc
                ),  # CUDA-API (kernel-launch) coverage — busyness
                "util": util,
            }
        )
    out.sort(key=lambda t: -(t["cover_ms"] + t["cuda_api_cover_ms"]))
    return out


def _emit_stream(
    stream, sdata, start_ns, bucket_ns, end_ns, n_buckets, span_ns
) -> dict:
    out_buckets = []
    tot = {"kernel": 0, "h2d": 0, "d2h": 0, "d2d": 0, "memset": 0, "busy_ns": 0}
    for i, b in enumerate(sdata["buckets"]):
        bs, be = start_ns + i * bucket_ns, (
            end_ns if i == n_buckets - 1 else start_ns + (i + 1) * bucket_ns
        )
        win = be - bs
        busy = b["kernel_ns"] + b["memcpy_ns"] + b["memset_ns"]
        by_source = sorted(
            (
                {"source": s, "gpu_ms": _ms(ns), "count": c}
                for s, (ns, c) in b["src"].items()
            ),
            key=lambda r: -r["gpu_ms"],
        )
        out_buckets.append(
            {
                "i": i,
                "start_ns": bs,
                "end_ns": be,
                "kernel_count": b["kernel"],
                "h2d_count": b["h2d"],
                "d2h_count": b["d2h"],
                "d2d_count": b["d2d"],
                "memset_count": b["memset"],
                "busy_ms": _ms(busy),
                "kernel_ms": _ms(b["kernel_ns"]),
                "memcpy_ms": _ms(b["memcpy_ns"]),
                "idle_ms": _ms(max(0, win - busy)),
                "util": round(busy / win, 4) if win > 0 else 0.0,
                "by_source": by_source,
            }
        )
        for k in ("kernel", "h2d", "d2h", "d2d", "memset"):
            tot[k] += b[k]
        tot["busy_ns"] += busy
    # whole-stream source attribution, largest first
    by_source = sorted(
        (
            {
                "source": s,
                "gpu_ms": _ms(ns),
                "count": c,
                "pct_of_step": round(ns / span_ns * 100, 2) if span_ns else 0.0,
            }
            for s, (ns, c) in sdata["src_total"].items()
        ),
        key=lambda r: -r["gpu_ms"],
    )
    return {
        "stream": stream,
        "totals": {
            "kernel_count": tot["kernel"],
            "h2d_count": tot["h2d"],
            "d2h_count": tot["d2h"],
            "d2d_count": tot["d2d"],
            "memset_count": tot["memset"],
            "busy_ms": _ms(tot["busy_ns"]),
            "mean_util": round(tot["busy_ns"] / span_ns, 4) if span_ns > 0 else 0.0,
        },
        "buckets": out_buckets,
        "by_source": by_source,
    }


# ─── kernel-kind classification + fusion candidates (tech4) ─────────

# Ordered rules: the first match wins, so compute-bound / non-fusible
# families (gemm / conv / attention / comm / sort) are matched *before* the
# memory-bound elementwise & reduction families. Matched against the kernel's
# demangled / short name. Names are matched case-insensitively.
_KIND_RULES = [
    (
        "gemm",
        re.compile(r"gemm|xmma|cutlass|cublas|sgemm|hgemm|dgemm|wgrad|dgrad", re.I),
    ),
    ("conv", re.compile(r"conv|cudnn|winograd|implicit_gemm", re.I)),
    ("attention", re.compile(r"attention|flash|fmha|\bmha\b", re.I)),
    (
        "comm",
        re.compile(
            r"nccl|allreduce|all_reduce|allgather|all_gather|reduce_scatter|sendrecv",
            re.I,
        ),
    ),
    (
        "sort",
        re.compile(
            r"sort|radix|\bscan\b|devicescan|deviceradix|devicereduce|cub::", re.I
        ),
    ),
    (
        "reduction",
        re.compile(
            r"reduce|layer_norm|layernorm|batch_norm|group_norm|rms_?norm|softmax|\bnorm_",
            re.I,
        ),
    ),
    (
        "elementwise",
        re.compile(
            r"elementwise|pointwise|vectorized|unrolled|activation|gather|scatter|index|\bcopy|\bfill|\bcat_|where|cast|gelu|relu|silu|sigmoid|tanh|dropout",
            re.I,
        ),
    ),
]

# Kinds whose GPU time is memory-bound and therefore a CUDA-fusion (tech4)
# target. Compute-bound (gemm/conv/attention) and non-fusible (comm/sort)
# kinds are excluded from the candidate ranking.
_FUSIBLE = frozenset({"elementwise", "reduction"})


def classify_kernel(name: str) -> str:
    """Bucket a kernel's demangled/short name into a coarse kind. Returns
    ``other`` when nothing matches (unrecognized / ambiguous, e.g. a bare
    ``kernel`` / ``Kernel2`` cublas-internal name)."""
    if not name:
        return "other"
    for kind, rx in _KIND_RULES:
        if rx.search(name):
            return kind
    return "other"


def build_fusion_candidates(
    parquetdir: str, start_ns: int, end_ns: int, top: int = 40
) -> list[dict]:
    """Rank source-line regions by **fusible** (memory-bound elementwise /
    reduction) GPU time over ``[start_ns, end_ns)`` — the tech4 candidate list.

    Each kernel is attributed to its launching ``torch.*`` source line (same
    correlation as ``build_timeline``) and classified by name. Compute-bound
    families (gemm / conv / attention) and non-fusible ones (comm / sort) are
    excluded. CUPTI carries no per-kernel byte count, so ``gpu_ms`` is used as
    the memory-traffic proxy: among memory-bound kernels, time ≈ traffic / BW,
    which catches both the launch-bound (many tiny kernels) and bandwidth-bound
    (few large elementwise kernels at high util) regimes."""
    sid = _string_ids(parquetdir)
    funcs = _build_func_index(parquetdir, sid, start_ns, end_ns)

    launch: dict[int, tuple[int, int]] = {}
    for r in _read(
        parquetdir,
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        ["start", "globalTid", "correlationId"],
    ):
        cid, tid, st = r.get("correlationId"), r.get("globalTid"), r.get("start")
        if cid is not None and tid is not None and st is not None:
            launch[cid] = (tid, st)

    def source_of(cid):
        info = launch.get(cid)
        if info is None:
            return "(unattributed)"
        fid = funcs.lookup(*info)
        return _source_of(funcs.meta[fid]) if fid is not None else "(unattributed)"

    span_ns = end_ns - start_ns
    regions: dict[str, dict] = defaultdict(
        lambda: {
            "gpu_ns": 0,
            "count": 0,
            "fusible_ns": 0,
            "fusible_count": 0,
            "by_kind": defaultdict(lambda: [0, 0]),
        }
    )  # kind → [gpu_ns, count]

    for r in _read(
        parquetdir,
        "CUPTI_ACTIVITY_KIND_KERNEL",
        ["start", "end", "correlationId", "shortName", "demangledName"],
    ):
        s, e = r.get("start"), r.get("end")
        if s is None or e is None or e <= start_ns or s >= end_ns:
            continue
        es, ee = max(s, start_ns), min(e, end_ns)
        dur = ee - es
        if dur <= 0:
            continue
        name = sid.get(r.get("shortName")) or sid.get(r.get("demangledName")) or "?"
        kind = classify_kernel(name)
        reg = regions[source_of(r.get("correlationId"))]
        reg["gpu_ns"] += dur
        reg["count"] += 1
        bk = reg["by_kind"][kind]
        bk[0] += dur
        bk[1] += 1
        if kind in _FUSIBLE:
            reg["fusible_ns"] += dur
            reg["fusible_count"] += 1

    out = []
    for src, reg in regions.items():
        if reg["fusible_ns"] <= 0:
            continue
        out.append(
            {
                "source": src,
                "fusible_gpu_ms": _ms(reg["fusible_ns"]),
                "fusible_kernel_count": reg["fusible_count"],
                "total_gpu_ms": _ms(reg["gpu_ns"]),
                "total_kernel_count": reg["count"],
                "pct_of_step": round(reg["fusible_ns"] / span_ns * 100, 2)
                if span_ns
                else 0.0,
                "by_kind": {
                    k: {"gpu_ms": _ms(v[0]), "count": v[1]}
                    for k, v in sorted(reg["by_kind"].items(), key=lambda x: -x[1][0])
                },
            }
        )
    out.sort(key=lambda r: -r["fusible_gpu_ms"])
    return out[:top]


# ─── exposed-idle attribution (which line/region starves the GPU) ────
#
# Ranks by *GPU idle a line/region is responsible for* rather than by the op's
# own time. The device is idle when no kernel / memcpy / memset is executing;
# each idle gap is charged to the launch that *ends* it (the kernel the GPU sat
# waiting to be issued), and, separately, the slice of idle that overlaps a host
# ``*Synchronize`` call is charged to that sync's source. This surfaces
# launch-bound / CPU-starvation idle — invisible to a self-time ranking, where a
# big ``.item()`` host wait (GPU still draining underneath it) looks like the top
# hotspot but removing it moves nothing.

# NVTX push/pop ranges that name a *region* (a ``nvtx_range(...)`` marker or a
# ``[fwd] xxx`` phase), used to group same-cause work that is split across
# several source lines. We exclude the talos:: op ranges (that is the per-line
# view), the aten/autograd/backward-seq machinery, and the step marker itself.
_STEP_RE = re.compile(r"^(ta:)?step", re.I)


class _RegionIndex:
    """Deepest enclosing named NVTX region per launching thread."""

    def __init__(self):
        self._by_tid: dict[int, list[tuple[int, int, str]]] = defaultdict(list)

    def add(self, tid, start, end, name):
        self._by_tid[tid].append((start, end, name))

    def finalize(self):
        for rs in self._by_tid.values():
            rs.sort()
        self._starts = {tid: [r[0] for r in rs] for tid, rs in self._by_tid.items()}

    def lookup(self, tid, t) -> str | None:
        ranges = self._by_tid.get(tid)
        if not ranges:
            return None
        starts = self._starts[tid]
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0:
            _s, e, name = ranges[i]
            if e >= t:
                return name
            i -= 1
        return None


def _build_region_index(
    parquetdir: str, sid: dict[int, str], s_ns: int, e_ns: int
) -> _RegionIndex:
    idx = _RegionIndex()
    for r in _read(
        parquetdir, "NVTX_EVENTS", ["start", "end", "text", "textId", "globalTid"]
    ):
        start, end, tid = r.get("start"), r.get("end"), r.get("globalTid")
        if start is None or end is None or tid is None or end < s_ns or start > e_ns:
            continue
        text = r.get("text") or (
            sid.get(r["textId"]) if r.get("textId") is not None else None
        )
        if not text:
            continue
        if (
            text.startswith("talos::")
            or text.startswith("aten::")
            or text.startswith(_AUTOGRAD)
        ):
            continue
        # Bare backward-node ranges ("MulBackward0", seq in payload) and
        # "talos_seq:" seq marks are machinery, not regions.
        if text.startswith("talos_seq:") or re.match(r"^\w+Backward\d*$", text):
            continue
        if _STEP_RE.match(text):
            continue
        idx.add(tid, start, end, text.split("!!", 1)[0][:80])
    idx.finalize()
    return idx


def _file_of(source: str) -> str:
    """Group key when there is no region marker: the file (drop ``:line``)."""
    return source.rsplit(":", 1)[0] if ":" in source else source


def _idle_overlap(idle_sorted, idle_starts, a: int, b: int) -> int:
    """Total overlap of ``[a, b)`` with the disjoint, sorted ``idle_sorted``."""
    if b <= a or not idle_sorted:
        return 0
    tot = 0
    i = max(bisect.bisect_right(idle_starts, a) - 1, 0)
    while i < len(idle_sorted) and idle_sorted[i][0] < b:
        s, e = idle_sorted[i]
        ov = min(b, e) - max(a, s)
        if ov > 0:
            tot += ov
        i += 1
    return tot


def build_exposed_idle(
    parquetdir: str, start_ns: int, end_ns: int, top: int = 25
) -> dict:
    """Attribute GPU idle over ``[start_ns, end_ns)`` to the line / region that
    starves the device. Returns the ``exposed_idle.json`` payload (minus the
    ``step`` key, which the caller adds)."""
    sid = _string_ids(parquetdir)
    copy_kinds = _copy_kind_map(parquetdir)
    funcs = _build_func_index(parquetdir, sid, start_ns, end_ns)
    regions = _build_region_index(parquetdir, sid, start_ns, end_ns)

    launch: dict[int, tuple[int, int]] = {}  # correlationId → (tid, launch_start)
    sync_calls: list[
        tuple[int, int, int]
    ] = []  # (start, end, tid) for host *Synchronize*
    for r in _read(
        parquetdir,
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        ["start", "end", "globalTid", "correlationId", "nameId"],
    ):
        cid, tid, st, en = (
            r.get("correlationId"),
            r.get("globalTid"),
            r.get("start"),
            r.get("end"),
        )
        if cid is not None and tid is not None and st is not None:
            launch[cid] = (tid, st)
        nm = sid.get(r.get("nameId")) if r.get("nameId") is not None else None
        if (
            nm
            and "Synchronize" in nm
            and tid is not None
            and st is not None
            and en is not None
            and en > start_ns
            and st < end_ns
        ):
            sync_calls.append((max(st, start_ns), min(en, end_ns), tid))

    # GPU events (kernel / memcpy / memset), clipped to the window
    events: list[tuple[int, int, object]] = []
    for s, e, _stream, _kind, cid in _gpu_events(
        parquetdir, copy_kinds, start_ns, end_ns
    ):
        es, ee = max(s, start_ns), min(e, end_ns)
        if ee > es:
            events.append((es, ee, cid))
    events.sort()

    # device busy union → idle gaps
    merged: list[list[int]] = []
    for s, e, _cid in events:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    busy_ns = sum(e - s for s, e in merged)
    idle: list[tuple[int, int]] = []
    cur = start_ns
    for s, e in merged:
        if s > cur:
            idle.append((cur, s))
        cur = max(cur, e)
    if cur < end_ns:
        idle.append((cur, end_ns))
    idle_ns = sum(e - s for s, e in idle)
    span_ns = end_ns - start_ns

    def attribute(cid):
        """(source, op, region) for a launch correlation id."""
        info = launch.get(cid)
        if info is None:
            return "(unattributed)", None, "(unattributed)"
        tid, lst = info
        fid = funcs.lookup(tid, lst)
        if fid is None:
            source, op = "(unattributed)", None
        else:
            m = funcs.meta[fid]
            source, op = _source_of(m), m.func
        region = regions.lookup(tid, lst) or (
            _file_of(source) if source != "(unattributed)" else "(unattributed)"
        )
        return source, op, region

    # launch counts + region per source (all events in the window)
    src_launches: dict[str, int] = defaultdict(int)
    src_op: dict[str, str] = {}
    src_region: dict[str, str] = {}
    for _s, _e, cid in events:
        source, op, region = attribute(cid)
        src_launches[source] += 1
        src_region[source] = region
        if op and source not in src_op:
            src_op[source] = op

    # busy lever: GPU-busy and *fusible* GPU time per source / region — the
    # bandwidth/compute side that runs at high util (no idle), invisible to the
    # idle attribution above. Lets each hotspot carry both levers so the lever
    # is chosen per region, not globally.
    src_busy: dict[str, int] = defaultdict(int)
    src_fus: dict[str, int] = defaultdict(int)
    reg_busy: dict[str, int] = defaultdict(int)
    reg_fus: dict[str, int] = defaultdict(int)
    for r in _read(
        parquetdir,
        "CUPTI_ACTIVITY_KIND_KERNEL",
        ["start", "end", "correlationId", "shortName", "demangledName"],
    ):
        ks, ke = r.get("start"), r.get("end")
        if ks is None or ke is None or ke <= start_ns or ks >= end_ns:
            continue
        dur = min(ke, end_ns) - max(ks, start_ns)
        if dur <= 0:
            continue
        source, _op, region = attribute(r.get("correlationId"))
        src_busy[source] += dur
        reg_busy[region] += dur
        name = sid.get(r.get("shortName")) or sid.get(r.get("demangledName")) or "?"
        if classify_kernel(name) in _FUSIBLE:
            src_fus[source] += dur
            reg_fus[region] += dur

    # charge each idle gap to the launch that ends it (next GPU event)
    estarts = [e[0] for e in events]
    src_idle: dict[str, float] = defaultdict(float)
    src_gap: dict[str, int] = defaultdict(int)
    reg_idle: dict[str, float] = defaultdict(float)
    reg_gap: dict[str, int] = defaultdict(int)
    for gs, ge in idle:
        i = bisect.bisect_left(estarts, ge)
        if i >= len(events):
            continue
        source, _op, region = attribute(events[i][2])
        d = ge - gs
        src_idle[source] += d
        src_gap[source] += 1
        src_region.setdefault(source, region)
        reg_idle[region] += d
        reg_gap[region] += 1

    # host-sync-exposed slice: idle overlapped by a *Synchronize* call
    idle_starts = [iv[0] for iv in idle]
    sync_src_idle: dict[str, float] = defaultdict(float)
    sync_src_gap: dict[str, int] = defaultdict(int)
    sync_op: dict[str, str] = {}
    sync_iv: list[tuple[int, int]] = []
    for ss, se, tid in sync_calls:
        ov = _idle_overlap(idle, idle_starts, ss, se)
        sync_iv.append((ss, se))
        if ov <= 0:
            continue
        fid = funcs.lookup(tid, ss)
        if fid is None:
            source, op = "(unattributed)", None
        else:
            m = funcs.meta[fid]
            source, op = _source_of(m), m.func
        sync_src_idle[source] += ov
        sync_src_gap[source] += 1
        if op and source not in sync_op:
            sync_op[source] = op

    # union of host-sync coverage over idle (no double count) → the split
    sync_iv.sort()
    sync_union: list[list[int]] = []
    for s, e in sync_iv:
        if sync_union and s <= sync_union[-1][1]:
            sync_union[-1][1] = max(sync_union[-1][1], e)
        else:
            sync_union.append([s, e])
    j = 0
    host_sync_ns = 0
    for gs, ge in idle:
        while j < len(sync_union) and sync_union[j][1] <= gs:
            j += 1
        k = j
        while k < len(sync_union) and sync_union[k][0] < ge:
            host_sync_ns += min(ge, sync_union[k][1]) - max(gs, sync_union[k][0])
            k += 1

    def pct(ns):
        return round(ns / span_ns * 100, 2) if span_ns else 0.0

    def lever(idle_ns, fus_ns):
        # per-hotspot lever: cut idle vs cut (fusible) busy time — whichever
        # carries the larger step-time potential for THIS region/line.
        return "idle" if idle_ns >= fus_ns else "busy"

    # by_source / by_region carry BOTH levers so the choice is per-hotspot:
    # exposed_idle_ms (starvation) and fusible_ms (bandwidth/compute reducible
    # by fusion). A region can be idle-bound while another is busy-bound.
    src_keys = set(src_idle) | set(src_fus)
    by_source = sorted(
        (
            {
                "source": s,
                "op": src_op.get(s),
                "launch_count": src_launches.get(s, 0),
                "region": src_region.get(s),
                "exposed_idle_ms": _ms(src_idle.get(s, 0)),
                "gap_count": src_gap.get(s, 0),
                "busy_ms": _ms(src_busy.get(s, 0)),
                "fusible_ms": _ms(src_fus.get(s, 0)),
                "lever": lever(src_idle.get(s, 0), src_fus.get(s, 0)),
                "pct_of_step": pct(max(src_idle.get(s, 0), src_fus.get(s, 0))),
            }
            for s in src_keys
        ),
        key=lambda r: -max(r["exposed_idle_ms"], r["fusible_ms"]),
    )[:top]
    reg_keys = set(reg_idle) | set(reg_fus)
    by_region = sorted(
        (
            {
                "region": r,
                "exposed_idle_ms": _ms(reg_idle.get(r, 0)),
                "gap_count": reg_gap.get(r, 0),
                "busy_ms": _ms(reg_busy.get(r, 0)),
                "fusible_ms": _ms(reg_fus.get(r, 0)),
                "lever": lever(reg_idle.get(r, 0), reg_fus.get(r, 0)),
                "pct_of_step": pct(max(reg_idle.get(r, 0), reg_fus.get(r, 0))),
            }
            for r in reg_keys
        ),
        key=lambda r: -max(r["exposed_idle_ms"], r["fusible_ms"]),
    )[:top]
    by_sync_source = sorted(
        (
            {
                "source": s,
                "op": sync_op.get(s),
                "exposed_idle_ms": _ms(ms),
                "pct_of_step": pct(ms),
                "gap_count": sync_src_gap[s],
            }
            for s, ms in sync_src_idle.items()
        ),
        key=lambda r: -r["exposed_idle_ms"],
    )[:top]

    return {
        "window": {"start_ns": start_ns, "end_ns": end_ns, "duration_ms": _ms(span_ns)},
        "gpu": {
            "busy_ms": _ms(busy_ns),
            "idle_ms": _ms(idle_ns),
            "idle_pct": round(idle_ns / span_ns * 100, 2) if span_ns else 0.0,
            "gap_count": len(idle),
        },
        "idle_split": {
            "host_sync_ms": _ms(host_sync_ns),
            "launch_bound_ms": _ms(max(0, idle_ns - host_sync_ns)),
            "note": "host_sync_ms = idle overlapped by a host *Synchronize* (.item()/.cpu() waits); "
            "launch_bound_ms = the rest — GPU starved between kernel launches (CPU/dispatch/Python).",
        },
        "by_region": by_region,
        "by_source": by_source,
        "by_sync_source": by_sync_source,
        "method": "device idle = window minus union(kernel,memcpy,memset); each gap charged to the "
        "launch that ends it (kernel→RUNTIME→enclosing talos:: source/region). Heuristic: "
        "'next launch' folds the Python glue before that launch into it. by_sync_source is "
        "the idle slice overlapped by host *Synchronize* calls (self_time of a sync ≠ its "
        "exposed idle). Each by_source/by_region row carries BOTH levers — exposed_idle_ms "
        "(starvation) and fusible_ms (bandwidth/compute reducible by fusion, runs at high "
        "util with no idle) — with a per-row 'lever' hint; the lever is chosen per hotspot, "
        "not globally, since a model mixes idle-bound and busy-bound regions.",
    }


# ─── device-level GPU utilisation (union across all streams) ────────


def build_gpu_util(
    parquetdir: str, start_ns: int, end_ns: int, n_buckets: int = 100
) -> dict:
    """Whole-device GPU utilisation: the union of busy time across ALL
    streams (kernel / memcpy / memset), bucketed like a stream's
    ``util.json``. Individual streams can each look moderate while the
    device is nearly saturated (their busy windows interleave) — or the
    reverse; this is the aggregate view the per-stream files can't show.
    Totals match ``exposed_idle.json``'s ``gpu`` block (same union)."""
    copy_kinds = _copy_kind_map(parquetdir)
    span = end_ns - start_ns
    bucket_ns = max(span // n_buckets, 1)

    events: list[tuple[int, int]] = []
    per_stream: dict = defaultdict(list)
    for s, e, stream, _kind, _cid in _gpu_events(
        parquetdir, copy_kinds, start_ns, end_ns
    ):
        es, ee = max(s, start_ns), min(e, end_ns)
        if ee > es:
            events.append((es, ee))
            per_stream[stream].append((es, ee))

    def _union(ivs):
        ivs.sort()
        merged: list[list[int]] = []
        for s, e in ivs:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    merged = _union(events)
    busy_ns = sum(e - s for s, e in merged)

    # per-bucket busy from the union intervals, clipped at bucket edges so
    # util ≤ 1.0 even when one interval spans several buckets
    bucket_busy = [0] * n_buckets
    for s, e in merged:
        i = int((s - start_ns) // bucket_ns)
        while s < e and i < n_buckets:
            b_end = start_ns + (i + 1) * bucket_ns
            seg = min(e, b_end) - s
            if seg > 0:
                bucket_busy[i] += seg
            s = min(e, b_end)
            i += 1

    util = {
        str(i): round(min(bucket_busy[i] / bucket_ns, 1.0), 4) for i in range(n_buckets)
    }
    vals = [util[str(i)] for i in range(n_buckets)]

    per_stream_mean = {
        str(k): round(sum(e - s for s, e in _union(v)) / span, 4)
        for k, v in sorted(per_stream.items(), key=lambda kv: str(kv[0]))
    }

    return {
        "window": {"start_ns": start_ns, "end_ns": end_ns, "duration_ms": _ms(span)},
        "n_buckets": n_buckets,
        "bucket_ns": bucket_ns,
        "gpu": {
            "busy_ms": _ms(busy_ns),
            "idle_ms": _ms(span - busy_ns),
            "util": round(busy_ns / span, 4) if span else 0.0,
        },
        "mean_util": round(sum(vals) / len(vals), 4) if vals else 0.0,
        "min_util": min(vals) if vals else 0.0,
        "max_util": max(vals) if vals else 0.0,
        "util": util,
        "per_stream_mean_util": per_stream_mean,
        "note": "device-level union of kernel/memcpy/memset across ALL streams; "
        "compare per_stream_mean_util against gpu.util to see how much "
        "of the device's busy time comes from stream overlap.",
    }
