#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare nsys device time with the Mega-Flink phase timers, per run of nsys-study.sh.

usage: nsys_digest.py ROOT   (the ROOT given to nsys-study.sh)
"""
import csv
import glob
import io
import json
import os
import sys

root = sys.argv[1]


def sections(path):
    txt = open(path).read().split("\n\n")
    out = {}
    for blk in txt:
        lines = blk.strip().split("\n")
        if len(lines) < 2:
            continue
        lines = [l for l in lines if not l.startswith(("Processing", "Generating"))]
        if len(lines) < 2:
            continue
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
        if not rows:
            continue
        out[tuple(rows[0])] = rows[1:]
    return out


for d in sorted(glob.glob(root + "/*/gpu*/*-ns")):
    case, mode = os.path.relpath(d, root).split("/")[0:2]
    r = json.load(open(d + "/result.json"))
    secs = sections(d + "/nsys-stats.csv")
    kern = h2d = d2h = 0
    nk = 0
    api = {}
    for h, rows in secs.items():
        if "Name" in h and "Instances" in h:
            for x in rows:
                kern += int(x[1])
                nk += int(x[2])
        if "Operation" in h and "Count" in h:
            for x in rows:
                if "Host-to-Device" in x[-1]:
                    h2d += int(x[1])
                if "Device-to-Host" in x[-1]:
                    d2h += int(x[1])
        if "Num Calls" in h:
            for x in rows:
                api[x[-1]] = int(x[1])
    wall = r["measured_seconds"] * 1e9
    sync = api.get("cudaEventSynchronize", 0) + api.get("cudaStreamSynchronize", 0)
    m = r["metrics"]
    mf = {"h2d": 0, "comp": 0, "d2h": 0, "r2c": 0, "c2r": 0}
    for cls, v in m.items():
        if (
            isinstance(v, dict)
            and v.get("batches")
            or (isinstance(v, dict) and v.get("kernels"))
        ):
            mf["h2d"] += v.get("h2dNanos", 0)
            mf["comp"] += v.get("computeNanos", 0)
            mf["d2h"] += v.get("d2hNanos", 0)
            mf["r2c"] += v.get("rowToColumnNanos", 0)
            mf["c2r"] += v.get("columnToRowNanos", 0)
    print(
        f"{case} {mode}: wall={wall/1e6:.0f}ms thr={r['throughput_records_per_s']:,.0f} | nsys device: kernel={kern/1e6:.1f} h2d={h2d/1e6:.1f} d2h={d2h/1e6:.1f} ms (kernels={nk}) sync-api={sync/1e6:.0f}ms | MegaFlink timers: h2d={mf['h2d']/1e6:.0f} comp={mf['comp']/1e6:.0f} d2h={mf['d2h']/1e6:.0f} r2c={mf['r2c']/1e6:.0f} c2r={mf['c2r']/1e6:.0f}"
    )
