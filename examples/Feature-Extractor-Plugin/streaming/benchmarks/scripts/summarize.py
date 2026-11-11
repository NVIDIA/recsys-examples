#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize FepKuaiRandBench results: per case x tier x mode medians, and CPU/GPU result equality.

usage: summarize.py RESULT_ROOT [COPY_DEST]
COPY_DEST (optional) receives result.json / plan.txt copies (small files only, no final.tsv).
"""
import json
import os
import shutil
import statistics
import sys
from collections import defaultdict

root = sys.argv[1]
dest = sys.argv[2] if len(sys.argv) > 2 else None
TIER = {"r20000": "low 20k/s", "r100000": "mid 100k/s", "r0": "high (saturated)"}
rows = defaultdict(list)
shas: defaultdict = defaultdict(lambda: defaultdict(set))
for case in sorted(os.listdir(root)):
    for mode in sorted(os.listdir(os.path.join(root, case))):
        for run in sorted(os.listdir(os.path.join(root, case, mode))):
            d = os.path.join(root, case, mode, run)
            f = os.path.join(d, "result.json")
            if not os.path.exists(f):
                print(f"MISSING/FAILED {case} {mode} {run}", file=sys.stderr)
                continue
            r = json.load(open(f))
            rate_key = run.split("-")[0]
            n = run.split("-")[1]
            tier = TIER.get(rate_key, rate_key) + f" {n}"
            rows[(case, tier, mode)].append(r)
            shas[(case, tier)][r["final_sha256"]].add(mode)
            if dest:
                o = os.path.join(dest, case, mode, run)
                os.makedirs(o, exist_ok=True)
                for name in ("result.json", "plan.txt"):
                    if os.path.exists(os.path.join(d, name)):
                        shutil.copy(os.path.join(d, name), o)


def med(rs, k):
    return statistics.median(x[k] for x in rs)


print(
    "| case | tier | mode | rounds | throughput rec/s (median) | p50 ms | p95 ms | p99 ms | out changelog rows | final rows |"
)
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
for (case, tier, mode), rs in sorted(rows.items()):
    print(
        f"| {case} | {tier} | {mode} | {len(rs)} | {med(rs,'throughput_records_per_s'):,.0f} | "
        f"{med(rs,'latency_p50_ms'):.2f} | {med(rs,'latency_p95_ms'):.2f} | {med(rs,'latency_p99_ms'):.2f} | "
        f"{med(rs,'out_changelog_rows'):,.0f} | {rs[0]['final_rows']:,} |"
    )
print()
print("## result equality (final table SHA-256 per case x tier)")
for (case, tier), m in sorted(shas.items()):
    status = "MATCH" if len(m) == 1 else "MISMATCH"
    detail = "; ".join(f"{sha[:16]}: {sorted(modes)}" for sha, modes in m.items())
    print(f"- {case} {tier}: {status} ({detail})")
print()
print("## per-round throughput")
for (case, tier, mode), rs in sorted(rows.items()):
    print(
        f"- {case} {tier} {mode}: "
        + ", ".join(f"{x['throughput_records_per_s']:,.0f}" for x in rs)
        + " | p99 "
        + ", ".join(f"{x['latency_p99_ms']:.1f}" for x in rs)
    )
