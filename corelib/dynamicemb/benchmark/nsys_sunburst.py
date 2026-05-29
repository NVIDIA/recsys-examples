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
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# ── tree ────────────────────────────────────────────────────────────────────


class Node:
    __slots__ = ("label", "children", "value", "code", "color")

    def __init__(self, label: str, value: float = 0.0):
        self.label = label
        self.children: List["Node"] = []
        self.value = value
        # Set later on op-leaves: short alphabetic code (A, B, ...) shown
        # in the chart in place of the full op name.  Non-leaves stay None.
        self.code: Optional[str] = None
        # Override the auto color_for() result.  Used to assign unique
        # pastel colors to every op leaf across the WHOLE chart (the
        # default sibling-index palette restarts per parent, which can
        # leave two leaves under different stages with identical color).
        self.color: Optional[tuple] = None

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


# Stage names we collapse into the 3-bucket ring 2.  Matched as substrings
# (in order) against each segment of parent_stages.
_STAGE_BUCKETS = [
    ("dynamicemb_prefetch", "prefetch"),
    ("DynamicEmbeddingFunction.forward", "forward"),
    ("DynamicEmbeddingFunction.backward", "backward"),
]


def _simplified_stage(stage_chain: List[str], phase: str) -> str:
    """Map the (deep) parent_stages chain to one of prefetch/forward/backward.

    Walks the chain outermost->innermost; first matching bucket wins.
    If nothing matches (e.g., kernels that fired outside the named stage
    NVTX -- happens for some autograd-driven backward launches), fall
    back to the phase name so the layout stays uniformly 3-ring.
    """
    for seg in stage_chain:
        for needle, label in _STAGE_BUCKETS:
            if needle in seg:
                return label
    return phase


def tree_from_csv(
    csv_path: str,
    filter_config: Optional[str] = None,
) -> Node:
    """Build a hierarchy tree from a breakdown CSV.

    Auto-detects the CSV flavor by column names:

    * **opcalls** (per-config, from updated nsys_breakdown.py):
      columns ``phase,parent_stages,op,...,avg_ms_per_iter,...``.  One
      CSV per config -- the tree spans
      ``phase → <parent_stages segments split by '/'> → op``.  Wedge
      value is ``avg_ms_per_iter``, so the chart represents a single
      iteration's GPU-time breakdown.

    * **legacy** (cross-config, from older breakdown.py):
      columns ``config,backend,phase,category,time_ms``.  Tree spans
      ``config → backend → phase → category``.  Wedge value is
      ``time_ms`` aggregated across iters.  ``filter_config`` only
      applies here.
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    is_opcalls = "op" in fieldnames and "parent_stages" in fieldnames
    is_legacy = "category" in fieldnames and "time_ms" in fieldnames

    if not (is_opcalls or is_legacy):
        raise SystemExit(
            f"unrecognized CSV schema for {csv_path}; expected either "
            f"opcalls (phase,parent_stages,op,avg_ms_per_iter,...) or "
            f"legacy (config,backend,phase,category,time_ms).  got: "
            f"{fieldnames}"
        )

    root = Node("ALL")
    if is_opcalls:
        for r in rows:
            phase = r["phase"]
            parents = r["parent_stages"]
            op = r["op"]
            ms = float(r["avg_ms_per_iter"])
            if ms <= 0:
                continue
            # Collapse the deep stage chain into one of three coarse
            # buckets at ring 2: prefetch / forward / backward.  Skip
            # everything else (sub-stages, cub::* library ranges, etc.)
            # so the chart stays a fixed 3 rings.  The prefix-strip
            # ('dynamicemb_' / 'DynamicEmbeddingFunction.') yields
            # short readable labels.
            stage_chain = parents.split("/") if parents else []
            stage2 = _simplified_stage(stage_chain, phase)
            path = [phase, stage2, op]
            add_path(root, path, ms)
    else:  # legacy
        for r in rows:
            cfg = r["config"]
            back = r["backend"]
            phase = r["phase"]
            cat = r["category"]
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
            "segmented_unique",
            "hash_find",
            "load_from_flat",
            "init_for_admitted",
            "gather_embedding",
            "hash_insert",
        ],
        ("dyn", "backward"): [
            "reduce_grads",
            "optimizer_update",
            "store_to_flat",
        ],
        ("trc", "forward"): [
            "bounds_check",
            "embedding_lookup",
            "pooling",
        ],
        ("trc", "backward"): [
            "reduce_grads",
            "optimizer_update",
        ],
    }
    BASE_MS = {
        "segmented_unique": 0.4,
        "hash_find": 3.0,
        "load_from_flat": 2.2,
        "init_for_admitted": 0.3,
        "gather_embedding": 11.0,
        "hash_insert": 3.2,
        "reduce_grads": 0.6,
        "optimizer_update": 0.3,
        "store_to_flat": 0.1,
        "bounds_check": 0.4,
        "embedding_lookup": 3.5,
        "pooling": 0.2,
    }
    CONFIGS = ["cfr=0.8", "cfr=1.0"]
    BACKENDS = ["dyn", "trc"]
    PHASES = ["forward", "backward"]

    root = Node("ALL")
    for cfg in CONFIGS:
        for back in BACKENDS:
            for phase in PHASES:
                for step, kn in enumerate(KERNEL_SEQ[(back, phase)]):
                    base = BASE_MS[kn]
                    jitter = random.uniform(0.75, 1.25)
                    shrink = (
                        0.4
                        if (
                            cfg == "cfr=1.0"
                            and kn
                            in {
                                "hash_insert",
                                "load_from_flat",
                                "store_to_flat",
                                "hash_find",
                            }
                        )
                        else 1.0
                    )
                    ms = round(base * jitter * shrink, 3)
                    add_path(root, [cfg, back, phase, f"{step:02d}_{kn}"], ms)
    return root


# ── coloring ────────────────────────────────────────────────────────────────


_RING_PALETTES = [
    # ring 1 (phase: forward / backward) -- two clear light hues.
    ["#aed581", "#90caf9", "#ffab91", "#ce93d8"],
    # ring 2 (stage bucket: prefetch / forward / backward) -- mid-light,
    # picked to contrast against the phase ring above (greens-blues
    # behind warmer pastels here).
    ["#ffcc80", "#80deea", "#f48fb1", "#bcaaa4"],
    # ring 3 (kept for legacy 4-ring callers; ops use the HSV rainbow below).
    ["#fff59d", "#80cbc4", "#e1bee7", "#ffab91"],
]


def color_for(level: int, sibling_idx: int, n_siblings: int) -> tuple:
    """Pastel-leaning palette.

    Inner rings draw from ``_RING_PALETTES`` for stable colors across
    runs.  Outer (leaf) rings use a HSV rainbow with moderate saturation
    so adjacent ops never blend yet the whole chart stays light enough
    that black text reads on every wedge.
    """
    if level < len(_RING_PALETTES):
        pal = _RING_PALETTES[level]
        return pal[sibling_idx % len(pal)]
    # Outermost: hue spread across [0, 0.9] for distinct neighbors.
    # Lower saturation + high value -> pastel, readable with black text.
    h = (sibling_idx / max(n_siblings, 1)) * 0.90
    return colorsys.hsv_to_rgb(h, 0.40, 0.96)


# ── rendering ───────────────────────────────────────────────────────────────


def _tree_depth(node: Node, d: int = 0) -> int:
    """Max distance from this node to a leaf (root.children = depth 1)."""
    if not node.children:
        return d
    return max(_tree_depth(c, d + 1) for c in node.children)


def _collect_leaves_with_path(
    node: Node, path: Optional[List[str]] = None
) -> List[Tuple[List[str], "Node"]]:
    """DFS in tree-insertion (= CSV call) order; return (path, leaf_node)
    list where path is [root_label, ..., parent_label] (not including leaf)."""
    if path is None:
        path = []
    if not node.children:
        return [(path, node)]
    out: List[Tuple[List[str], Node]] = []
    new_path = path + [node.label]
    for c in node.children:
        out.extend(_collect_leaves_with_path(c, new_path))
    return out


def _letter_code(idx: int) -> str:
    """Excel-style 0->A, 25->Z, 26->AA, 27->AB, ..."""
    if idx < 26:
        return chr(ord("A") + idx)
    return _letter_code(idx // 26 - 1) + chr(ord("A") + idx % 26)


def _build_rings(depth: int) -> List[Tuple[float, float]]:
    """Distribute ``depth`` concentric rings between r=0.20 and r=1.00.

    The innermost ring is narrower (0.20-0.40) to leave room for the
    center label; outer rings get equal slices of the remaining radius.
    Falls back to the original 4-ring layout when depth==4 to keep the
    legacy plot pixel-stable.
    """
    if depth <= 0:
        return []
    if depth == 4:
        return [(0.40, 0.20), (0.55, 0.40), (0.70, 0.55), (1.00, 0.70)]
    inner = 0.20
    span_outer = 1.00 - inner
    # First ring is half-width to keep center area, then equal width.
    first_w = min(0.20, span_outer / (depth + 1))
    remain = span_outer - first_w
    other_w = remain / max(depth - 1, 1)
    rings = []
    r = inner
    for i in range(depth):
        w = first_w if i == 0 else other_w
        rings.append((r + w, r))
        r += w
    return rings


def _draw_node(
    ax,
    node: Node,
    angle_start: float,
    angle_end: float,
    depth: int,
    rings: List[Tuple[float, float]],
    grand_total: float,
):
    """angle_start/end in degrees, 0 = 12 o'clock, clockwise.

    ``grand_total`` is root.total() -- carried through recursion so each
    wedge's percentage label is computed relative to the WHOLE iteration
    rather than just its immediate parent.  Same wedge angle either way;
    only the printed pct changes.
    """
    if depth >= len(rings) or angle_end <= angle_start:
        return
    r_outer, r_inner = rings[depth]
    tot = node.total()
    if tot <= 0:
        return

    cursor = angle_start
    n_sibs = len(node.children)
    n_rings = len(rings)
    for idx, child in enumerate(node.children):
        frac = child.total() / tot
        extent = (angle_end - angle_start) * frac
        c_start, c_end = cursor, cursor + extent

        # Leaves always draw at the OUTERMOST ring; non-leaves draw at
        # their natural call-stack depth.  When a path is shallower than
        # the global max depth, the rings BETWEEN the parent's depth and
        # the outer ring stay blank for that angle -- the white background
        # shows through, matching the requested "blank gets filled with
        # white" behavior.
        is_leaf = not child.children
        draw_depth = n_rings - 1 if is_leaf else depth
        r_outer, r_inner = rings[draw_depth]

        # matplotlib's Wedge uses 0=3 o'clock, CCW.  We carry "0=12 o'clock,
        # CW" angles and convert: theta_mpl = 90 - angle_cw.
        # Leaves get a globally-unique pre-computed color (set in render()
        # before this descent); other nodes fall back to the per-ring
        # palette indexed by sibling order.
        fc = (
            child.color
            if child.color is not None
            else color_for(draw_depth, idx, n_sibs)
        )
        w = Wedge(
            (0, 0),
            r_outer,
            90 - c_end,
            90 - c_start,
            width=r_outer - r_inner,
            facecolor=fc,
            edgecolor="white",
            linewidth=1.2,
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
                rotation += 180  # flip bottom-half labels
            # Percentage = this wedge's value / total iter time.  Reads
            # as "this slice is X% of the whole iteration", consistent
            # across levels (parent + child pcts sum to parent's pct).
            pct = (child.total() / grand_total * 100.0) if grand_total > 0 else 0.0
            # For op leaves use the alphabetic code so even small wedges
            # remain identifiable; the full op name lives in the legend
            # table next to the chart.  Stage / phase rings use the
            # human-readable label.
            display_name = child.code if (is_leaf and child.code) else child.label
            label_text = (
                f"{display_name}\n{pct:.1f}%" if extent >= 6.0 else display_name
            )
            ax.text(
                x,
                y,
                label_text,
                ha="center",
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=8.5 if draw_depth < 3 else 7.0,
                color="black",  # all rings are now light pastels
                linespacing=0.95,
            )

        if not is_leaf:
            _draw_node(ax, child, c_start, c_end, depth + 1, rings, grand_total)
        cursor = c_end


def render(
    root: Node,
    out_png: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    units: str = "ms",
) -> None:
    total = root.total()
    if total <= 0:
        raise SystemExit("no positive-time leaves to draw")

    # Assign A, B, C, ... codes to op leaves in call (= CSV insertion) order.
    # The codes are what the chart shows; the legend table next to the chart
    # maps codes back to full op names + share of iter.  We also give every
    # leaf a globally-unique pastel color (spread across the HSV hue circle
    # rather than restarting per parent) so no two ops in the chart share
    # the same color, even when they live in different stage buckets.
    leaves_in_order = _collect_leaves_with_path(root)
    # Golden-ratio sampling on the hue circle: each new index jumps ~222°
    # (= 1 - 1/φ) from the previous one, so adjacent leaves never sit on
    # similar hues even when the chart packs >20 ops.  Add a saturation/
    # value triplet rotation as a secondary distinguishing axis so even
    # wedges that happen to share a hue family (every ~3 steps if N is
    # divisible) stay visually separable.
    GOLDEN_STEP = 1.0 - 1.0 / ((1 + 5**0.5) / 2)  # ≈ 0.381966
    sat_levels = (0.32, 0.46, 0.58)
    val_levels = (0.97, 0.92, 0.99)
    for i, (_, leaf) in enumerate(leaves_in_order):
        leaf.code = _letter_code(i)
        h = (0.07 + i * GOLDEN_STEP) % 1.0
        s = sat_levels[i % len(sat_levels)]
        v = val_levels[i % len(val_levels)]
        leaf.color = colorsys.hsv_to_rgb(h, s, v)

    depth = _tree_depth(root)
    rings = _build_rings(depth)

    # Top/bottom layout: sunburst on top (gets most of the height so it
    # renders large enough for narrow wedges to be visible), legend table
    # below.  Height ratio shifts a bit when there are many leaves so the
    # table doesn't get crushed.
    n_leaves = len(leaves_in_order)
    table_share = max(0.18, min(0.30, 0.10 + 0.012 * n_leaves))
    chart_share = 1.0 - table_share
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[chart_share, table_share],
        hspace=0.05,
    )
    ax_chart = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # ── chart ─────────────────────────────────────────────────────────
    ax_chart.set_aspect("equal")
    ax_chart.set_xlim(-1.20, 1.20)
    ax_chart.set_ylim(-1.30, 1.20)
    ax_chart.axis("off")

    _draw_node(ax_chart, root, 0.0, 360.0, 0, rings, total)

    ax_chart.text(
        0,
        0,
        f"total\n{total:.2f} {units}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax_chart.text(
        0,
        -1.18,
        subtitle
        or "rings (inside → out): phase → "
        "{prefetch / forward / backward} → op  "
        "(wedge angle ∝ avg ms / iter, dyn side only; "
        "pct = share of full iter)",
        ha="center",
        va="top",
        fontsize=9,
        color="#444",
    )

    # ── legend table ──────────────────────────────────────────────────
    ax_table.axis("off")
    # Sort by time spent, descending (so heaviest ops are at the top).
    rows_sorted = sorted(leaves_in_order, key=lambda x: -x[1].total())
    table_rows = []
    for path, leaf in rows_sorted:
        code = leaf.code
        # path = [root_label, phase, stage_bucket, ...]
        stage = path[2] if len(path) >= 3 else (path[1] if len(path) >= 2 else "")
        op_name = leaf.label
        ms = leaf.total()
        pct = ms / total * 100.0
        table_rows.append([code, stage, op_name, f"{ms:.3f}", f"{pct:.2f}%"])

    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=["#", "stage", "op", f"{units}", "% iter"],
        loc="upper center",
        cellLoc="left",
        colWidths=[0.06, 0.16, 0.50, 0.13, 0.13],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9 if len(table_rows) <= 25 else 8)
    tbl.scale(1, 1.3)
    # Header bold + shaded; '#' column centered.
    for col_idx, cell in enumerate(tbl[0, c] for c in range(5)):
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#e8e8e8")
    for r in range(1, len(table_rows) + 1):
        tbl[r, 0].set_text_props(ha="center", weight="bold")
        tbl[r, 3].set_text_props(ha="right")
        tbl[r, 4].set_text_props(ha="right")

    fig.suptitle(
        title or "GPU time breakdown",
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"saved {out_png}")
    plt.close(fig)


# ── cli ─────────────────────────────────────────────────────────────────────


def _detect_format(csv_path: str) -> str:
    """Return 'opcalls' or 'legacy' for the CSV at csv_path."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames or []
    if "op" in fns and "parent_stages" in fns:
        return "opcalls"
    if "category" in fns and "time_ms" in fns:
        return "legacy"
    raise SystemExit(f"unrecognized CSV schema: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-ring nested donut from nsys_breakdown CSV(s).  "
        "Accepts the per-config 'opcalls__*.csv' (one PNG per "
        "input) or the legacy cross-config 'breakdown.csv' "
        "(single PNG).",
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help="one or more breakdown CSVs (omit when --demo is used)",
    )
    parser.add_argument(
        "--out",
        help="output PNG path -- only honored when exactly one input CSV "
        "is given (or in --demo).  With multiple inputs the path is "
        "derived per-CSV as <csv-base>.sunburst.png",
    )
    parser.add_argument(
        "--out-dir",
        help="when given, write per-input PNGs into this dir instead of "
        "next to the CSV",
    )
    parser.add_argument(
        "--filter-config",
        help="legacy CSVs only: keep configs whose label contains this substring",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="ignore csv args and render a hand-tuned fake-data tree",
    )
    args = parser.parse_args()

    if args.demo:
        root = tree_demo()
        out_png = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sunburst_demo.png",
        )
        render(
            root,
            out_png,
            title="GPU time breakdown — fake demo data",
            subtitle="rings (inside → out): config → backend → phase → kernel  "
            "(wedge angle ∝ ms)",
            units="ms",
        )
        return

    if not args.csv:
        parser.error("at least one csv path is required (or pass --demo)")

    # Per-CSV rendering.
    for csv_path in args.csv:
        fmt = _detect_format(csv_path)
        root = tree_from_csv(csv_path, filter_config=args.filter_config)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        if args.out and len(args.csv) == 1:
            out_png = args.out
        elif args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out_png = os.path.join(args.out_dir, f"{base}.sunburst.png")
        else:
            out_png = f"{os.path.splitext(csv_path)[0]}.sunburst.png"

        if fmt == "opcalls":
            # opcalls CSV has been pre-filtered to one cfg.label; derive a
            # short config label for the figure title.
            cfg_label = base.replace("opcalls__", "")
            title = f"GPU time breakdown per iteration\n{cfg_label}"
            subtitle = (
                "rings (inside → out): phase → "
                "{prefetch / forward / backward} → op  "
                "(wedge angle ∝ avg ms / iter, dyn side only; "
                "pct = share of full iter)"
            )
            units = "ms/iter"
        else:
            title_suffix = f" — {args.filter_config}" if args.filter_config else ""
            title = f"GPU time breakdown{title_suffix}"
            subtitle = (
                "rings (inside → out): config → backend → phase → "
                "kernel  (wedge angle ∝ ms)"
            )
            units = "ms"

        render(root, out_png, title=title, subtitle=subtitle, units=units)


if __name__ == "__main__":
    main()
