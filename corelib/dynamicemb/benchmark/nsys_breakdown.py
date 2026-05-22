# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-config x backend x phase x kernel-category timing breakdown from an
# nsys-rep file.  Reads the SQLite export, attributes each GPU kernel to
# its enclosing NVTX context (config label / dyn vs trc / forward vs
# backward), buckets kernel names via KERNEL_NAME_PATTERNS, and emits a
# CSV summary plus a stacked bar PNG.
#
# Usage:
#   python nsys_breakdown.py trace_all.nsys-rep
#       -> trace_all.breakdown.csv
#       -> trace_all.breakdown.png
#
#   python nsys_breakdown.py trace_all.nsys-rep \
#       --out-csv /tmp/bd.csv --out-png /tmp/bd.png \
#       --filter-config "cfr=0.8"          # only configs whose label matches

import argparse
import bisect
import csv
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Single source of truth: pull KERNEL_NAME_PATTERNS out of the benchmark
# module via AST, NOT `import` -- the benchmark file pulls in pytest, torch,
# torchrec etc. at import time, which we don't want here.
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
    raise SystemExit(
        "KERNEL_NAME_PATTERNS not found in benchmark_batched_dynamicemb_tables.py"
    )


KERNEL_NAME_PATTERNS: Dict[str, List[str]] = _load_kernel_name_patterns()
CATEGORY_ORDER: List[str] = list(KERNEL_NAME_PATTERNS.keys()) + ["other"]


# ── nsys export ──────────────────────────────────────────────────────────────


def ensure_sqlite(nsys_rep: str) -> str:
    """Export the nsys-rep to .sqlite if missing or stale."""
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
        [
            "nsys",
            "export",
            "--type=sqlite",
            f"--output={sqlite_path}",
            "--force-overwrite=true",
            nsys_rep,
        ],
        check=True,
    )
    return sqlite_path


# ── interval lookup ──────────────────────────────────────────────────────────


class IntervalIndex:
    """Bisect-based 'point in interval' lookup for disjoint sorted ranges.

    NVTX ranges at the same nesting level (e.g., all "forward" ranges across
    iters) don't overlap, so a single bisect over the sorted start times
    plus an end-time check identifies the enclosing range in O(log N).
    """

    def __init__(self, intervals: List[Tuple[int, int, str]]):
        intervals = sorted(intervals, key=lambda x: x[0])
        self._starts = [s for s, _, _ in intervals]
        self._ends = [e for _, e, _ in intervals]
        self._names = [n for _, _, n in intervals]

    def at(self, t: int) -> Optional[str]:
        if not self._starts:
            return None
        idx = bisect.bisect_right(self._starts, t) - 1
        if idx < 0 or self._ends[idx] < t:
            return None
        return self._names[idx]


# ── kernel classification ────────────────────────────────────────────────────


def classify_kernel(name: str) -> str:
    """Demangled kernel name -> KERNEL_NAME_PATTERNS bucket key (or 'other')."""
    lower = name.lower()
    for cat, patterns in KERNEL_NAME_PATTERNS.items():
        if any(p.lower() in lower for p in patterns):
            return cat
    return "other"


# ── load + attribute ─────────────────────────────────────────────────────────


CFG_LABEL_RE = re.compile(r"^T\d+_totalB")
_BACKEND_NAMES = {"dyn", "trc"}
_PHASE_NAMES = {"forward", "backward"}
_OP_PREFIX = "op:"


def load_breakdown(sqlite_path: str) -> List[dict]:
    """Return one row per GPU kernel launch with its NVTX context.

    Each row: ``{kernel, duration_ns, config, backend, phase, op}``.  Any
    context level not surrounding the kernel is ``None``.  ``op`` is the
    innermost ``op:*`` range name (without the prefix) when one covers the
    kernel; it serves as the preferred category source.  Kernels outside
    any ``op:*`` range have ``op=None`` and fall back to kernel-name
    pattern matching downstream.
    """
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    # Some nsys schemas use eventType 75 = NvtxPushPopRange; safer to filter by
    # presence of end > start instead of relying on the exact event-type code,
    # which changes across nsys releases.
    cur.execute(
        """
        SELECT n.start, n.end, s.value
        FROM NVTX_EVENTS n
        JOIN StringIds s ON s.id = n.textId
        WHERE n.end IS NOT NULL AND n.end > n.start
        """
    )
    raw_ranges = cur.fetchall()

    cfg_ranges, backend_ranges, phase_ranges, op_ranges = [], [], [], []
    for s, e, name in raw_ranges:
        if CFG_LABEL_RE.match(name):
            cfg_ranges.append((s, e, name))
        elif name in _BACKEND_NAMES:
            backend_ranges.append((s, e, name))
        elif name in _PHASE_NAMES:
            phase_ranges.append((s, e, name))
        elif name.startswith(_OP_PREFIX):
            # store without the "op:" prefix so the category column reads cleanly
            op_ranges.append((s, e, name[len(_OP_PREFIX):]))

    cfg_idx = IntervalIndex(cfg_ranges)
    backend_idx = IntervalIndex(backend_ranges)
    phase_idx = IntervalIndex(phase_ranges)
    op_idx = IntervalIndex(op_ranges)

    cur.execute(
        """
        SELECT k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        """
    )
    rows = []
    for k_start, k_end, k_name in cur:
        rows.append(
            {
                "kernel": k_name,
                "duration_ns": k_end - k_start,
                "config": cfg_idx.at(k_start),
                "backend": backend_idx.at(k_start),
                "phase": phase_idx.at(k_start),
                "op": op_idx.at(k_start),
            }
        )
    conn.close()
    return rows


# ── aggregation ──────────────────────────────────────────────────────────────


def aggregate(rows: List[dict]) -> Dict[Tuple[str, str, str, str], float]:
    """Sum duration_ns per (config, backend, phase, category).

    Category source preference:
      1. ``r["op"]`` -- innermost ``op:*`` NVTX range covering the kernel.
         This is the authoritative source when the Python code is
         instrumented; it doesn't depend on kernel-name heuristics.
      2. ``classify_kernel(r["kernel"])`` -- fall back to the
         ``KERNEL_NAME_PATTERNS`` substring matcher for traces that
         pre-date the ``op:*`` instrumentation, or for kernels that
         happen to launch outside the wrapped Python paths.
    """
    agg: Dict[Tuple[str, str, str, str], float] = defaultdict(float)
    for r in rows:
        cfg = r["config"] or "(no-config)"
        back = r["backend"] or "(no-backend)"
        phase = r["phase"] or "(no-phase)"
        cat = r.get("op") or classify_kernel(r["kernel"])
        agg[(cfg, back, phase, cat)] += r["duration_ns"]
    return agg


def write_csv(agg: Dict[Tuple[str, str, str, str], float], path: str) -> None:
    """Long-form CSV: one row per (config, backend, phase, category)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "backend", "phase", "category", "time_ms"])
        for key, ns in sorted(agg.items()):
            cfg, back, phase, cat = key
            w.writerow([cfg, back, phase, cat, f"{ns / 1e6:.4f}"])
    print(f"[wrote] {path} ({len(agg)} rows)")


# ── plotting ─────────────────────────────────────────────────────────────────


def plot_breakdown(
    agg: Dict[Tuple[str, str, str, str], float],
    out_png: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Drop rows where any context level is unattributed -- those are setup
    # kernels not wrapped in the benchmark's NVTX scaffolding.
    keys = [
        k for k in agg
        if not any(v.startswith("(no-") for v in k[:3])
    ]
    if not keys:
        print("[plot] no NVTX-attributed kernels found, skipping PNG", file=sys.stderr)
        return

    configs = sorted({k[0] for k in keys})
    backends = ["dyn", "trc"]
    phases = ["forward", "backward"]

    # Category order: CATEGORY_ORDER first (stable colors across runs for the
    # original 8 buckets), then any newly-discovered op:* categories from the
    # trace appended alphabetically.  Ensures op-level attribution from the
    # NVTX path doesn't get lost just because a name isn't in KERNEL_NAME_PATTERNS.
    discovered = sorted({k[3] for k in keys})
    category_order = list(CATEGORY_ORDER) + [
        c for c in discovered if c not in CATEGORY_ORDER
    ]

    palette = plt.get_cmap("tab20").colors
    cat_color = {cat: palette[i % len(palette)] for i, cat in enumerate(category_order)}

    # Figure: rows = configs, cols = (dyn fwd, dyn bwd, trc fwd, trc bwd)
    n_rows = len(configs)
    n_cols = len(backends) * len(phases)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(max(8.0, 1.2 * n_cols), 2.5 * n_rows),
        squeeze=False,
    )

    col_labels = [f"{b}\n{p}" for b in backends for p in phases]
    x = np.arange(n_cols)
    width = 0.7

    for row_idx, cfg in enumerate(configs):
        ax = axes[row_idx, 0]
        # Build a (category, n_cols) matrix of ms.
        mat = np.zeros((len(category_order), n_cols))
        for ci, cat in enumerate(category_order):
            for col_idx, (b, p) in enumerate(
                (b, p) for b in backends for p in phases
            ):
                mat[ci, col_idx] = agg.get((cfg, b, p, cat), 0.0) / 1e6

        bottoms = np.zeros(n_cols)
        for ci, cat in enumerate(category_order):
            vals = mat[ci]
            if np.all(vals == 0):
                continue
            ax.bar(
                x, vals, width, bottom=bottoms,
                label=cat, color=cat_color[cat],
                edgecolor="white", linewidth=0.3,
            )
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_ylabel("ms")
        # Short title -- full label can be very long, keep first segments.
        ax.set_title(cfg, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        # Annotate column totals.
        totals = mat.sum(axis=0)
        for xi, t in zip(x, totals):
            if t > 0:
                ax.annotate(
                    f"{t:.1f}",
                    xy=(xi, t),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )

    # One figure-level legend (shared across all subplots, only categories
    # that actually appear).  Use the per-figure category_order so op:*
    # categories from the trace also show up in the legend.
    used_cats = {k[3] for k in keys}
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=cat_color[c], label=c)
        for c in category_order
        if c in used_cats
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(legend_handles), 8),
        fontsize=9,
        frameon=False,
    )
    fig.suptitle(
        "nsys kernel-category breakdown  (per config x backend x phase, ms)",
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[wrote] {out_png}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-(config x backend x phase x kernel-category) "
                    "GPU-time breakdown from an nsys-rep file.",
    )
    parser.add_argument("nsys_rep", help="path to *.nsys-rep (or *.qdrep)")
    parser.add_argument(
        "--out-csv",
        help="output CSV path (default: <nsys_rep>.breakdown.csv)",
    )
    parser.add_argument(
        "--out-png",
        help="output PNG path (default: <nsys_rep>.breakdown.png)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip PNG generation",
    )
    parser.add_argument(
        "--filter-config",
        help="only keep configs whose label contains this substring",
    )
    args = parser.parse_args()

    base = re.sub(r"\.(nsys-rep|qdrep)$", "", args.nsys_rep)
    out_csv = args.out_csv or f"{base}.breakdown.csv"
    out_png = args.out_png or f"{base}.breakdown.png"

    sqlite_path = ensure_sqlite(args.nsys_rep)
    rows = load_breakdown(sqlite_path)
    print(f"[load] {len(rows)} kernel launches from {sqlite_path}", file=sys.stderr)

    if args.filter_config:
        rows = [r for r in rows if r["config"] and args.filter_config in r["config"]]
        print(f"[filter] {len(rows)} kernels match config substr "
              f"{args.filter_config!r}", file=sys.stderr)

    agg = aggregate(rows)
    write_csv(agg, out_csv)

    # Print a short stdout summary too (top categories per config/backend/phase).
    print()
    print(f"{'config':<55} {'backend':<8} {'phase':<10} {'cat':<22} {'ms':>9}")
    print("-" * 110)
    sorted_keys = sorted(agg.keys(), key=lambda k: (k[0], k[1], k[2], -agg[k]))
    last_group = None
    for key in sorted_keys:
        cfg, back, phase, cat = key
        group = (cfg, back, phase)
        if group != last_group:
            if last_group is not None:
                print()
            last_group = group
        ms = agg[key] / 1e6
        if ms < 0.01:
            continue
        print(f"{cfg[:55]:<55} {back:<8} {phase:<10} {cat:<22} {ms:>9.3f}")

    if not args.no_plot:
        plot_breakdown(agg, out_png)


if __name__ == "__main__":
    main()
