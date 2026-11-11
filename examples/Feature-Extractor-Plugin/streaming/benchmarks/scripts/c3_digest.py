#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Digest of c3-study.sh output.

usage: c3_digest.py ROOT   (the ROOT given to c3-study.sh)
"""
import os
import sys

os.chdir(sys.argv[1])
import glob
import json
import statistics as st


def ld(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


print("A c0-pass ceiling (source+exchange+sink only)")
for m in ["cpu", "gpu-batch"]:
    v = [
        ld(f)["throughput_records_per_s"]
        for f in sorted(glob.glob(f"A/c0-pass/{m}/*/result.json"))
    ]
    print(m, [int(x) for x in v], "median", int(st.median(v)))
print("B variance c3 high, 8 rounds")
for m in ["cpu", "gpu-batch"]:
    rs = [ld(f) for f in sorted(glob.glob(f"B/c3-topn/{m}/*/result.json"))]
    v = [r["throughput_records_per_s"] for r in rs]
    print(
        m,
        [int(x) for x in v],
        "median",
        int(st.median(v)),
        "min",
        int(min(v)),
        "max",
        int(max(v)),
    )
    if m == "gpu-batch":
        for r in rs:
            t = r["metrics"]["TopNRuntimeMetrics"]
            print(
                "  batches",
                t["batches"],
                "maxBatchRows",
                t.get("maxBatchRows"),
                "comp ms",
                round(t["computeNanos"] / 1e6),
                "h2d",
                round(t["h2dNanos"] / 1e6),
                "d2h",
                round(t["d2hNanos"] / 1e6),
                "thr",
                int(r["throughput_records_per_s"]),
                "p99",
                round(r["latency_p99_ms"], 1),
            )
print("C sweep rows x max-wait: median thr of 2 rounds | batches | kernel ms | p99")
for rows in [1024, 4096, 16384, 65536]:
    line = []
    for w in [1, 5, 20]:
        rs = [
            ld(f)
            for f in sorted(
                glob.glob(
                    f"C/c3-topn/gpu-batch/r0-n3000000-rows{rows}-w{w}-*/result.json"
                )
            )
        ]
        rs = [r for r in rs if r]
        if not rs:
            line.append("NA")
            continue
        t = rs[0]["metrics"]["TopNRuntimeMetrics"]
        line.append(
            f"{int(st.median(r['throughput_records_per_s'] for r in rs)/1000)}k b={t['batches']} k={round(t['computeNanos']/1e6)}ms p99={st.median(r['latency_p99_ms'] for r in rs):.0f}"
        )
    print(rows, " | ".join(line))
