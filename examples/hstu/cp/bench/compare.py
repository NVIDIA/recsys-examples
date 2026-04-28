# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Phase-0 reference benchmark comparator (plan T0.2).

Reads two JSONs (baseline + candidate) emitted by `baseline.py` and prints
a per-shape delta table. Exits non-zero if any shape regresses by more
than the configured threshold.

Default thresholds are set by SPEC §3 / plan §Global rule 3 tier:
    - small shapes (median < 1ms):   +10% tolerance
    - larger shapes (median ≥ 1ms):  +5% tolerance

The asymmetric tier exists because Python-side dispatch / autograd-Function
overhead is a fixed cost that swamps very small kernels.

Run:
    python examples/hstu/cp/bench/baseline.py --output /tmp/cur.json
    python examples/hstu/cp/bench/compare.py tasks/bench_baseline.json /tmp/cur.json
    # exit 0 ⇒ no regression; exit 1 ⇒ regression beyond threshold; exit 2 ⇒ error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SMALL_THRESHOLD_MS = 1.0  # median < this → small-shape tier
SMALL_TIER_PCT = 10.0  # +10% tolerance for small shapes
LARGE_TIER_PCT = 5.0  # +5% tolerance otherwise


def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _compare(baseline: dict, candidate: dict) -> tuple[int, list[str], list[str]]:
    """Returns (exit_code, table_lines, summary_lines)."""
    base_shapes = {s["label"]: s for s in baseline["shapes"]}
    cand_shapes = {s["label"]: s for s in candidate["shapes"]}

    table: list[str] = []
    table.append(
        f"{'shape':<24} {'base_ms':>10} {'cand_ms':>10} {'Δ%':>8} {'tier':>5} {'verdict':>10}"
    )
    table.append("-" * 78)

    regressions: list[str] = []
    new_shapes: list[str] = [lbl for lbl in cand_shapes if lbl not in base_shapes]
    removed_shapes: list[str] = [lbl for lbl in base_shapes if lbl not in cand_shapes]

    for lbl in sorted(set(base_shapes) | set(cand_shapes)):
        base = base_shapes.get(lbl)
        cand = cand_shapes.get(lbl)
        if base is None:
            assert cand is not None  # `lbl` came from one of the two sets
            table.append(
                f"{lbl:<24} {'(new)':>10} {cand['median_ms']:>10.3f} {'':>8} {'':>5} {'NEW':>10}"
            )
            continue
        if cand is None:
            table.append(
                f"{lbl:<24} {base['median_ms']:>10.3f} {'(removed)':>10} {'':>8} {'':>5} {'REMOVED':>10}"
            )
            continue
        b_ms = base["median_ms"]
        c_ms = cand["median_ms"]
        delta_pct = ((c_ms - b_ms) / b_ms) * 100.0 if b_ms > 0 else 0.0
        tier = "small" if b_ms < SMALL_THRESHOLD_MS else "large"
        threshold = SMALL_TIER_PCT if tier == "small" else LARGE_TIER_PCT
        if delta_pct > threshold:
            verdict = "REGRESS"
            regressions.append(
                f"{lbl}: +{delta_pct:.1f}% (tier={tier} threshold +{threshold:.1f}%)"
            )
        elif delta_pct < -threshold:
            verdict = "FASTER"
        else:
            verdict = "ok"
        table.append(
            f"{lbl:<24} {b_ms:>10.3f} {c_ms:>10.3f} {delta_pct:>+7.1f}% {tier:>5} {verdict:>10}"
        )

    summary: list[str] = []
    if regressions:
        summary.append(
            f"FAIL — {len(regressions)} shape(s) regressed beyond threshold:"
        )
        summary.extend(f"  - {r}" for r in regressions)
    if removed_shapes:
        summary.append(
            f"WARN — shape(s) present in baseline but missing from candidate: {', '.join(removed_shapes)}"
        )
    if new_shapes:
        summary.append(f"INFO — new shape(s) in candidate: {', '.join(new_shapes)}")
    if not regressions and not removed_shapes:
        summary.append(
            f"OK — all shapes within threshold "
            f"(small {SMALL_TIER_PCT:.0f}% / large {LARGE_TIER_PCT:.0f}%)"
        )

    exit_code = 1 if regressions else 0
    return exit_code, table, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="HSTU baseline benchmark comparator")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--small-threshold-ms",
        type=float,
        default=SMALL_THRESHOLD_MS,
        help="Median below this is treated as 'small' tier",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"ERROR: baseline not found: {args.baseline}", file=sys.stderr)
        return 2
    if not args.candidate.exists():
        print(f"ERROR: candidate not found: {args.candidate}", file=sys.stderr)
        return 2

    base = _load(args.baseline)
    cand = _load(args.candidate)
    print(
        f"baseline: commit={base.get('commit', '?')}  device={base.get('device', '?')}"
    )
    print(
        f"candidate: commit={cand.get('commit', '?')}  device={cand.get('device', '?')}"
    )
    if base.get("device") != cand.get("device"):
        print(
            "WARN: device labels differ — comparing across hardware is meaningless",
            file=sys.stderr,
        )

    exit_code, table, summary = _compare(base, cand)
    print()
    for line in table:
        print(line)
    print()
    for line in summary:
        print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
