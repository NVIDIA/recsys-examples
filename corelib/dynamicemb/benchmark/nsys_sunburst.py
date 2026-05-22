# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-ring nested-donut (sunburst-style) visualization of a breakdown
# CSV produced by nsys_breakdown.py.  Four rings, inside → outside:
#     config → backend → phase → kernel category
# Each wedge's angle is proportional to its GPU time (ms), so heavier
# items look bigger.  Insertion order is preserved (no value-based
# resorting) so the outermost ring stays in the order the CSV gave us.
#
# Implemented in matplotlib so it works on hosts without a Chrome
# install (plotly Sunburst's PNG backend needs Chromium / kaleido).
#
# Usage:
#   # render a real breakdown
#   python nsys_sunburst.py trace_all.breakdown.csv
#       -> trace_all.sunburst.png
#
#   # demo mode (fake data, no input file needed)
#   python nsys_sunburst.py --demo
#       -> sunburst_demo.png
#
#   # filter / custom output
#   python nsys_sunburst.py bd.csv --filter-config "cfr=0.8" --out /tmp/a.png

import argparse
import colorsys
import csv
import math
import os
import random
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


# ── tree ────────────────────────────────────────────────────────────────────


class Node:
    __slots__ = ("label", "children", "value")

    def __init__(self, label: str, value: float = 0.0):
        self.label = label
        self.children: List["Node"] = []
        self.value = value

    def total(self) -> float:
        return self.value + sum(c.total() for c in self.children)


def add_path(root: Node, path: List[str], value: float) -> None:
    """Walk `root` along `path`, creating Nodes as needed; add `value` at leaf."""
    cur = root
    for seg in path[:-1]:
        nxt = next((c for c in cur.children if c.label == seg), None)
        if nxt is None:
            nxt = Node(seg)
            cur.children.append(nxt)
        cur = nxt
    leaf_label = path[-1]
    leaf = next((c for c in cur.children if c.label == leaf_label), None)
    if leaf is None:
        leaf = Node(leaf_label)
        cur.children.append(leaf)
    leaf.value += value


# ── CSV loading ─────────────────────────────────────────────────────────────


def tree_from_csv(
    csv_path: str,
    filter_config: Optional[str] = None,
) -> Node:
    """Build a config → backend → phase → category tree from breakdown CSV."""
    root = Node("ALL")
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            cfg = r["config"]
            back = r["backend"]
            phase = r["phase"]
            cat = r["category"]
            # Skip rows that aren't fully attributed by NVTX (setup, etc.)
            if any(v.startswith("(no-") for v in (cfg, back, phase)):
                continue
            if filter_config and filter_config not in cfg:
                continue
            ms = float(r["time_ms"])
            if ms <= 0:
                continue
            add_path(root, [cfg, back, phase, cat], ms)
    return root


# ── demo data ───────────────────────────────────────────────────────────────


def tree_demo() -> Node:
    """Hand-tuned random tree shaped like a realistic dynamicemb run."""
    random.seed(7)

    KERNEL_SEQ = {
        ("dyn", "forward"): [
            "segmented_unique", "hash_find", "load_from_flat",
            "init_for_admitted", "gather_embedding", "hash_insert",
        ],
        ("dyn", "backward"): [
            "reduce_grads", "optimizer_update", "store_to_flat",
        ],
        ("trc", "forward"): [
            "bounds_check", "embedding_lookup", "pooling",
        ],
        ("trc", "backward"): [
            "reduce_grads", "optimizer_update",
        ],
    }
    BASE_MS = {
        "segmented_unique":  0.4, "hash_find": 3.0, "load_from_flat": 2.2,
        "init_for_admitted": 0.3, "gather_embedding": 11.0, "hash_insert": 3.2,
        "reduce_grads":      0.6, "optimizer_update": 0.3, "store_to_flat": 0.1,
        "bounds_check":      0.4, "embedding_lookup":  3.5, "pooling":      0.2,
    }
    CONFIGS  = ["cfr=0.8", "cfr=1.0"]
    BACKENDS = ["dyn", "trc"]
    PHASES   = ["forward", "backward"]

    root = Node("ALL")
    for cfg in CONFIGS:
        for back in BACKENDS:
            for phase in PHASES:
                for step, kn in enumerate(KERNEL_SEQ[(back, phase)]):
                    base = BASE_MS[kn]
                    jitter = random.uniform(0.75, 1.25)
                    shrink = 0.4 if (cfg == "cfr=1.0" and kn in {
                        "hash_insert", "load_from_flat",
                        "store_to_flat", "hash_find",
                    }) else 1.0
                    ms = round(base * jitter * shrink, 3)
                    add_path(root, [cfg, back, phase, f"{step:02d}_{kn}"], ms)
    return root


# ── coloring ────────────────────────────────────────────────────────────────


_RING_PALETTES = [
    ["#76b900", "#9dc3e6", "#e07a5f", "#81b29a"],            # ring 1: config
    ["#558b00", "#1f77b4", "#a04030", "#427a5f"],            # ring 2: backend
    ["#f4a261", "#2a9d8f", "#e9c46a", "#264653"],            # ring 3: phase
]


def color_for(level: int, sibling_idx: int, n_siblings: int) -> tuple:
    if level < len(_RING_PALETTES):
        pal = _RING_PALETTES[level]
        return pal[sibling_idx % len(pal)]
    # outermost ring -- rainbow by sibling so neighbors don't blend
    h = (sibling_idx / max(n_siblings, 1)) * 0.85
    return colorsys.hsv_to_rgb(h, 0.55, 0.92)


# ── rendering ───────────────────────────────────────────────────────────────


RINGS = [
    # (r_outer, r_inner)
    (0.40, 0.20),   # ring 1: config
    (0.55, 0.40),   # ring 2: backend
    (0.70, 0.55),   # ring 3: phase
    (1.00, 0.70),   # ring 4: kernel category (outermost)
]


def _draw_node(ax, node: Node, angle_start: float, angle_end: float, depth: int):
    """angle_start/end in degrees, 0 = 12 o'clock, clockwise."""
    if depth >= len(RINGS) or angle_end <= angle_start:
        return
    r_outer, r_inner = RINGS[depth]
    tot = node.total()
    if tot <= 0:
        return

    cursor = angle_start
    n_sibs = len(node.children)
    for idx, child in enumerate(node.children):
        frac = child.total() / tot
        extent = (angle_end - angle_start) * frac
        c_start, c_end = cursor, cursor + extent

        # matplotlib's Wedge uses 0=3 o'clock, CCW.  We carry "0=12 o'clock,
        # CW" angles and convert: theta_mpl = 90 - angle_cw.
        w = Wedge(
            (0, 0), r_outer, 90 - c_end, 90 - c_start,
            width=r_outer - r_inner,
            facecolor=color_for(depth, idx, n_sibs),
            edgecolor="white", linewidth=1.2,
        )
        ax.add_patch(w)

        if extent >= 3.5:
            mid_angle = (c_start + c_end) / 2
            r_label = (r_outer + r_inner) / 2
            rad = math.radians(mid_angle)
            x = r_label * math.sin(rad)
            y = r_label * math.cos(rad)
            rotation = -mid_angle
            if 90 < mid_angle < 270:
                rotation += 180          # flip bottom-half labels
            ax.text(
                x, y, child.label,
                ha="center", va="center",
                rotation=rotation, rotation_mode="anchor",
                fontsize=8.5 if depth < 3 else 7.0,
                color="white" if depth in (0, 1) else "black",
            )

        _draw_node(ax, child, c_start, c_end, depth + 1)
        cursor = c_end


def render(root: Node, out_png: str, title: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")
    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-1.30, 1.20)
    ax.axis("off")

    total = root.total()
    if total <= 0:
        raise SystemExit("no positive-time leaves to draw")

    _draw_node(ax, root, 0.0, 360.0, 0)

    ax.text(0, 0, f"total\n{total:.1f} ms",
            ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(
        0, -1.18,
        "rings (inside → out): config → backend → phase → kernel  "
        "(wedge angle ∝ ms)",
        ha="center", va="top", fontsize=10, color="#444",
    )
    fig.suptitle(
        title or "GPU time breakdown",
        y=0.96, fontsize=14, fontweight="bold",
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"saved {out_png}")


# ── cli ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-ring nested donut from an nsys_breakdown CSV.",
    )
    parser.add_argument(
        "csv", nargs="?",
        help="path to breakdown CSV (omit when --demo is used)",
    )
    parser.add_argument(
        "--out",
        help="output PNG path (default: <csv-base>.sunburst.png, or "
             "sunburst_demo.png in --demo mode)",
    )
    parser.add_argument(
        "--filter-config",
        help="keep only configs whose label contains this substring",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="ignore the csv arg and render a hand-tuned fake-data tree",
    )
    args = parser.parse_args()

    if args.demo:
        root = tree_demo()
        out_png = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sunburst_demo.png",
        )
        title = "GPU time breakdown — fake demo data"
    else:
        if not args.csv:
            parser.error("csv path is required (or pass --demo)")
        root = tree_from_csv(args.csv, filter_config=args.filter_config)
        base = os.path.splitext(args.csv)[0]
        out_png = args.out or f"{base}.sunburst.png"
        title_suffix = f" — {args.filter_config}" if args.filter_config else ""
        title = f"GPU time breakdown{title_suffix}"

    render(root, out_png, title=title)


if __name__ == "__main__":
    main()
