# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-config "op call sequence" extractor from one or more nsys-rep traces.
#
# For every distinct cfg.label() NVTX range across the inputs, emits one CSV
# whose rows are ordered by op-invocation time and capture per-iteration
# averages of GPU time spent inside each (phase, parent stage chain, op)
# tuple on the **dynamicemb (dyn)** side.  TorchRec (trc) kernels are
# dropped entirely.
#
# Pipeline per kernel launch:
#
#   1. Reconstruct the full NVTX stack covering its start time, using a
#      sweep-line over CUPTI + NVTX events (proper push/pop nesting).
#   2. Classify each stack level by name:
#        cfg.label   (^T\d+_totalB...)        -> CSV file identity
#        dyn / trc                              -> filter (keep dyn, drop trc)
#        nsys_iter_N                            -> iter-id, used for averaging
#                                                  but NOT written to CSV
#        forward / backward                     -> 'phase' column
#        op:<name>                              -> 'op' column (innermost leaf)
#        anything else                          -> 'stage' chain (outer->inner,
#                                                  joined with '/')
#   3. Aggregate per (phase, parent_stages, op):
#        calls_per_iter  = avg #invocations per iter
#        avg_ms_per_iter = sum(ms) / iters_where_this_call_was_observed
#        total_ms        = full sum (sanity)
#        num_iters       = denominator used above
#        first_start     = earliest start time (for row ordering)
#   4. Multi-file: same cfg.label across inputs is MERGED (iter ids are
#      file-scoped so the average is over the union of observed iters).
#
# Kernels covered by the dyn NVTX scaffolding but not by any op:* range are
# labeled with the kernel name itself as the 'op' (instead of a placeholder),
# so each distinct kernel shows up as its own row and is individually
# identifiable in the chart / table.  The 'fallback_category' column still
# reports the closest KERNEL_NAME_PATTERNS bucket, useful for quickly
# spotting whether a missing op:* annotation would have been worth adding.
#
# Output filenames: opcalls__<sanitized-cfg-label>.csv, intentionally with a
# fixed prefix so downstream tools (e.g. nsys_sunburst.py) can recognize
# them.
#
# Usage:
#   python nsys_breakdown.py trace1.nsys-rep trace2.nsys-rep ...
#       -> opcalls__<label1>.csv, opcalls__<label2>.csv, ...
#
#   python nsys_breakdown.py trace.nsys-rep --out-dir /tmp/bd

import argparse
import bisect
import csv
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# KERNEL_NAME_PATTERNS pulled from the benchmark file via AST, used only as a
# fallback when an unwrapped kernel falls outside any op:* range.  We don't
# import the module directly because it pulls in torch/torchrec at import time.
def _load_kernel_name_patterns() -> Dict[str, List[str]]:
    import ast
    src_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "benchmark_batched_dynamicemb_tables.py",
    )
    tree = ast.parse(open(src_path).read())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "KERNEL_NAME_PATTERNS"
                for t in node.targets
            )
        ):
            return ast.literal_eval(node.value)
    return {}


KERNEL_NAME_PATTERNS: Dict[str, List[str]] = _load_kernel_name_patterns()


# ── nsys export ──────────────────────────────────────────────────────────────


def ensure_sqlite(nsys_rep: str) -> str:
    """Export .nsys-rep -> .sqlite if missing/stale; return sqlite path."""
    if not os.path.exists(nsys_rep):
        raise SystemExit(f"not found: {nsys_rep}")
    sqlite_path = re.sub(r"\.(nsys-rep|qdrep)$", ".sqlite", nsys_rep)
    if sqlite_path == nsys_rep:
        sqlite_path = nsys_rep + ".sqlite"
    if (
        os.path.exists(sqlite_path)
        and os.path.getmtime(sqlite_path) >= os.path.getmtime(nsys_rep)
    ):
        return sqlite_path
    print(f"[export] {nsys_rep} -> {sqlite_path}", file=sys.stderr)
    subprocess.run(
        ["nsys", "export", "--type=sqlite",
         f"--output={sqlite_path}", "--force-overwrite=true", nsys_rep],
        check=True,
    )
    return sqlite_path


# ── NVTX-name classification ────────────────────────────────────────────────


_CFG_RE = re.compile(r"^T\d+_totalB")
_ITER_RE = re.compile(r"^nsys_iter_(\d+)$")
_BACKEND_NAMES = {"dyn", "trc"}
_PHASE_NAMES = {"forward", "backward"}


def _classify_nvtx(name: str) -> Tuple[str, Optional[Any]]:
    """Return (kind, extra) where kind is one of:
       'cfg', 'backend', 'phase', 'iter', 'op', 'stage'.
    extra is the iter index (int) for 'iter', else None."""
    if not name:
        return "stage", None
    if _CFG_RE.match(name):
        return "cfg", None
    if name in _BACKEND_NAMES:
        return "backend", None
    if name in _PHASE_NAMES:
        return "phase", None
    m = _ITER_RE.match(name)
    if m:
        return "iter", int(m.group(1))
    if name.startswith("op:"):
        return "op", None
    return "stage", None


def _classify_kernel_fallback(name: str) -> str:
    """For kernels not covered by an op:* range, return a coarse bucket
    name based on KERNEL_NAME_PATTERNS substring matching."""
    lower = name.lower()
    for cat, patterns in KERNEL_NAME_PATTERNS.items():
        if any(p.lower() in lower for p in patterns):
            return cat
    return "(unwrapped)"


# ── load NVTX + kernels from sqlite ─────────────────────────────────────────


def _fetch_nvtx_and_kernels(sqlite_path: str):
    """Return (nvtx_events, kernels).
    nvtx_events: list of (start, end, name) -- NVTX push-pop ranges with
                  CPU-side timestamps.
    kernels:     list of (gpu_start, gpu_end, name, launch_cpu_start)
                  -- gpu_* are when the kernel actually ran (we use the
                  gap for 'duration_ns'); launch_cpu_start is when the
                  CPU called ``cudaLaunchKernel`` -- this is what we
                  match against NVTX, since both are CPU-side.

    Why use the CPU launch time for NVTX attribution rather than the
    kernel's GPU start: CUDA launches are asynchronous.  A Python
    sequence like
        nvtx_push("backward")
        out.backward(grad)
        nvtx_pop("backward")
    pops the NVTX range immediately after queueing the backward kernels;
    the GPU often hasn't started them yet, so their GPU-side timestamps
    fall AFTER the NVTX range closes -- attributing them to a parent
    range gives a misleading empty "backward" or worse, drops them
    entirely.  cudaLaunchKernel happens synchronously inside the NVTX
    bracket, so its CPU timestamp is the correct attribution anchor.

    Single global stack (no per-thread split): the benchmark uses one
    Python main thread for NVTX.  If a future workload uses worker
    threads, the sweep-line would need per-globalTid stacks (NVTX_EVENTS
    has globalTid, RUNTIME does too -- join through correlationId).
    """
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT n.start, n.end, COALESCE(s.value, n.text)
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.end IS NOT NULL AND n.end > n.start
          AND COALESCE(s.value, n.text) IS NOT NULL
        """
    )
    nvtx = cur.fetchall()
    cur.execute(
        """
        SELECT k.start, k.end, s.value, r.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
            ON r.correlationId = k.correlationId
        JOIN StringIds s ON s.id = k.shortName
        """
    )
    kernels = cur.fetchall()
    conn.close()
    return nvtx, kernels


# ── sweep-line: attribute each kernel to its NVTX stack ──────────────────────


def _attribute_kernels(nvtx, kernels):
    """For each kernel, compute its enclosing NVTX stack (outer->inner) by
    sweep-line over NVTX open/close + kernel-launch events.

    Each kernel is keyed in the sweep by its **CPU launch timestamp**
    (cudaLaunchKernel start), not its GPU execution start, so that async
    kernels queued inside a short NVTX range get attributed to that
    range even when they execute after the range closes (see
    _fetch_nvtx_and_kernels docstring for the rationale).

    Yields ``(gpu_start, gpu_end, k_name, [stack_of_nvtx_names])``
    -- the gpu_* are still used by callers for duration accounting.
    """
    OPEN, KERNEL, CLOSE = 0, 1, 2
    events = []
    for i, (s, e, n) in enumerate(nvtx):
        events.append((s, OPEN, i, n))
        events.append((e, CLOSE, i, n))
    for j, (gpu_s, gpu_e, name, launch_cpu) in enumerate(kernels):
        events.append((launch_cpu, KERNEL, j, (gpu_s, gpu_e, name)))
    # Sort key: (timestamp, kind_order) -- OPEN(0) < KERNEL(1) < CLOSE(2)
    # so a kernel launched exactly when an NVTX opens (or just before a
    # close) gets the right enclosing context.
    events.sort(key=lambda x: (x[0], x[1]))

    stack: List[Tuple[int, str]] = []      # (open_idx, name)

    for t, kind, idx, payload in events:
        if kind == OPEN:
            stack.append((idx, payload))
        elif kind == CLOSE:
            # pop matching idx (usually top; loop handles rare out-of-order)
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == idx:
                    stack.pop(i)
                    break
        else:  # KERNEL
            stack_names = [name for (_, name) in stack]
            yield payload[0], payload[1], payload[2], stack_names


# ── one event = one classified kernel record ────────────────────────────────


def _parse_record(k_start: int, k_end: int, k_name: str, stack: List[str]):
    """Return a dict describing the kernel's classified NVTX context, or
    None if the kernel should be skipped (in trc, or outside dyn iter)."""
    cfg_label = None
    in_dyn = False
    in_trc = False
    phase = None
    iter_id = None
    stages: List[str] = []
    op_leaf: Optional[str] = None

    for name in stack:
        kind, extra = _classify_nvtx(name)
        if kind == "cfg":
            cfg_label = name
        elif kind == "backend":
            if name == "dyn":
                in_dyn = True
            elif name == "trc":
                in_trc = True
        elif kind == "phase":
            phase = name
        elif kind == "iter":
            iter_id = extra
        elif kind == "op":
            op_leaf = name[3:]  # strip "op:" prefix
        elif kind == "stage":
            stages.append(name)

    # Filter rules.
    if in_trc:
        return None                # trc side, skip entirely
    if not in_dyn:
        return None                # outside dyn (setup / between iters)
    if cfg_label is None or phase is None or iter_id is None:
        return None                # missing context, can't average per iter

    # When no op:* NVTX wraps this launch, use the kernel name itself as
    # the op label.  Each distinct kernel then becomes its own row,
    # individually identifiable in the CSV / chart / table, rather than
    # being aggregated under a single "(unwrapped)" bucket.
    op_name = op_leaf if op_leaf is not None else k_name

    return {
        "cfg_label": cfg_label,
        "phase": phase,
        "iter_id": iter_id,
        "parent_stages": stages,   # outer -> inner
        "op": op_name,
        "is_wrapped": op_leaf is not None,
        "kernel_name": k_name,
        "fallback_cat": _classify_kernel_fallback(k_name)
                          if op_leaf is None else None,
        "duration_ns": k_end - k_start,
        "start": k_start,
    }


# ── aggregation across files ────────────────────────────────────────────────


def _accumulate(records, file_idx: int, acc):
    """Merge per-iter records into the global per-cfg accumulator.
    iter_id is keyed by (file_idx, raw_iter_id) so the same numeric iter
    across multiple input files counts as separate observations -- exactly
    what we want when averaging over the union.
    """
    for r in records:
        if r is None:
            continue
        cfg = r["cfg_label"]
        key = (r["phase"], tuple(r["parent_stages"]), r["op"])
        entry = acc[cfg].setdefault(key, {
            "calls": defaultdict(int),       # iter -> #invocations
            "ms":    defaultdict(float),     # iter -> total ms
            "first_start": None,             # for row ordering
            "kernel_names": set(),
            "fallback_cats": set(),
            "is_wrapped": r["is_wrapped"],
        })
        scoped_iter = (file_idx, r["iter_id"])
        entry["calls"][scoped_iter] += 1
        entry["ms"][scoped_iter] += r["duration_ns"] / 1e6
        if not r["is_wrapped"]:
            entry["kernel_names"].add(r["kernel_name"])
            if r["fallback_cat"]:
                entry["fallback_cats"].add(r["fallback_cat"])
        # Track earliest start across ALL iters/files so the row order is
        # stable even when iter 0 happens to be absent from one trace.
        if entry["first_start"] is None or r["start"] < entry["first_start"]:
            entry["first_start"] = r["start"]


def _finalize(acc) -> Dict[str, List[Dict[str, Any]]]:
    """Convert accumulator to per-cfg row lists, sorted by op call order."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cfg, by_key in acc.items():
        rows = []
        for (phase, parents, op), e in by_key.items():
            iters_observed = set(e["ms"].keys())
            n_iters = len(iters_observed)
            if n_iters == 0:
                continue
            total_ms = sum(e["ms"].values())
            total_calls = sum(e["calls"].values())
            rows.append({
                "phase": phase,
                "parent_stages": "/".join(parents),
                "op": op,
                "calls_per_iter": total_calls / n_iters,
                "avg_ms_per_iter": total_ms / n_iters,
                "total_ms": total_ms,
                "num_iters": n_iters,
                "kernel_names": ", ".join(sorted(e["kernel_names"]))
                                  if e["kernel_names"] else "",
                "fallback_category": ", ".join(sorted(e["fallback_cats"]))
                                     if e["fallback_cats"] else "",
                "_sort": e["first_start"],
            })
        rows.sort(key=lambda r: r["_sort"])
        for r in rows:
            r.pop("_sort", None)
        out[cfg] = rows
    return out


# ── csv output ──────────────────────────────────────────────────────────────


_CSV_COLS = [
    "phase",
    "parent_stages",
    "op",
    "calls_per_iter",
    "avg_ms_per_iter",
    "total_ms",
    "num_iters",
    "kernel_names",
    "fallback_category",
]


def _sanitize_filename(label: str) -> str:
    # cfg.label format includes '=' and '.' which are filesystem-safe on
    # Linux/macOS, but we still scrub anything outside [\w.=-] just in case.
    return re.sub(r"[^A-Za-z0-9_.=-]", "_", label)


def _write_csv(cfg_label: str, rows: List[Dict[str, Any]], out_dir: str) -> str:
    path = os.path.join(out_dir, f"opcalls__{_sanitize_filename(cfg_label)}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "phase": r["phase"],
                "parent_stages": r["parent_stages"],
                "op": r["op"],
                "calls_per_iter": f"{r['calls_per_iter']:.3f}",
                "avg_ms_per_iter": f"{r['avg_ms_per_iter']:.6f}",
                "total_ms": f"{r['total_ms']:.4f}",
                "num_iters": r["num_iters"],
                "kernel_names": r["kernel_names"],
                "fallback_category": r["fallback_category"],
            })
    return path


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-config op-call breakdown CSVs from one or more "
                    "nsys-rep traces.  TorchRec (trc) is dropped.",
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="one or more .nsys-rep (or .qdrep) files",
    )
    parser.add_argument(
        "--out-dir",
        help="output directory (default: directory of the first input)",
    )
    args = parser.parse_args()

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(args.inputs[0]))
    os.makedirs(out_dir, exist_ok=True)

    acc: Dict[str, Dict[Tuple, Dict[str, Any]]] = defaultdict(dict)

    for file_idx, rep in enumerate(args.inputs):
        sqlite = ensure_sqlite(rep)
        nvtx, kernels = _fetch_nvtx_and_kernels(sqlite)
        print(f"[load] {rep}: {len(nvtx)} nvtx + {len(kernels)} kernels",
              file=sys.stderr)
        records = (
            _parse_record(s, e, n, stack)
            for s, e, n, stack in _attribute_kernels(nvtx, kernels)
        )
        _accumulate(records, file_idx, acc)

    finalized = _finalize(acc)

    if not finalized:
        raise SystemExit(
            "no dyn-side op events found; either traces are trc-only, "
            "lack the cfg.label() / nsys_iter NVTX scaffolding, or "
            "predate the op:* instrumentation."
        )

    print(f"\n{'cfg.label':<60} {'rows':>5} {'iters':>6}")
    print("-" * 80)
    for cfg, rows in finalized.items():
        max_iters = max((r["num_iters"] for r in rows), default=0)
        path = _write_csv(cfg, rows, out_dir)
        print(f"{cfg[:60]:<60} {len(rows):>5} {max_iters:>6}  -> {path}")


if __name__ == "__main__":
    main()
