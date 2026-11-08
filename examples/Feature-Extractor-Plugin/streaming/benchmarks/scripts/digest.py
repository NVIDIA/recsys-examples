#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-run breakdown digest of breakdown.sh output: JFR thread-stack shares and Mega-Flink phase timers.

usage: digest.py ROOT   (the ROOT given to breakdown.sh; GPU metrics appear only for Mega-Flink runs)
"""
import glob
import json
import os
import sys
from collections import defaultdict

root = sys.argv[1]
for d in sorted(glob.glob(root + "/*/*/*-bd")):
    p = os.path.relpath(d, root).split("/")
    case, mode, tier = p[0], p[1], ("high" if p[2].startswith("r0-") else "mid")
    try:
        j = json.load(open(d + "/jfr-breakdown.json"))
    except Exception:
        print(case, mode, tier, "JFR ERR")
        continue
    agg: defaultdict = defaultdict(int)
    for th, st in j.items():
        if isinstance(st, dict):
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    agg[k] += v
    tot = sum(agg.values()) or 1
    jf = " ".join(
        f"{k.split(' ')[0][:10]}={100*v/tot:.0f}%"
        for k, v in sorted(agg.items(), key=lambda x: -x[1])[:6]
    )
    r = json.load(open(d + "/result.json"))
    m = r.get("metrics", {})
    g = []
    for name, mm in m.items():
        if isinstance(mm, dict) and mm.get("batches", 0):

            def ms(k):
                return mm.get(k, 0) / 1e6

            g.append(
                f"{name[:-14]}: batches={mm['batches']} r2c={ms('rowToColumnNanos'):.0f} h2d={ms('h2dNanos'):.0f} comp={ms('computeNanos'):.0f} d2h={ms('d2hNanos'):.0f} c2r={ms('columnToRowNanos'):.0f}ms"
            )
    print(
        f"{case} {mode} {tier} thr={r['throughput_records_per_s']:,.0f} gpu_in_plan={r['gpu_in_plan']}\n   JFR: {jf}\n   "
        + ("\n   ".join(g) if g else "no gpu metrics")
    )
