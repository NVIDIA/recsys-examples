#!/usr/bin/env python3
"""Summarise tests/out/<run>/{directed,fuzz,coverage} into a table; exit 1 when fusion != unfused GPU anywhere."""
import collections
import glob
import json
import os
import sys

run = sys.argv[1]
bad = 0
print(f"=== {run} ===")
p = os.path.join(run, "directed", "directed.json")
if os.path.exists(p):
    d = json.load(open(p))
    t = d["totals"]
    print(
        "directed (edge-value grid): ",
        {k: v for k, v in t.items()},
        " coverage:",
        d["coverage"],
    )
    for label, lst in d["results"].items():
        if label.startswith("FUSION") and lst:
            bad += len(lst)
        for r in lst[:50]:
            print(
                f"  [{label}] {r['group']:9s} {r['template']}  rows={r.get('n_mismatch_rows','')}"
            )
fz = os.path.join(run, "fuzz", "failures")
if os.path.isdir(fz):
    c: collections.Counter = collections.Counter()
    uniq = collections.defaultdict(list)
    for f in glob.glob(fz + "/*.json"):
        r = json.load(open(f))
        c[r["label"]] += 1
        uniq[r["label"]].append(r.get("minimised_sql") or str(r["sql"])[:150])
    print("fuzz unique failures per comparison:", dict(c))
    for label, l in uniq.items():
        if label.startswith("FUSION"):
            bad += len(l)
        for s in sorted(set(l))[:40]:
            print(f"  [{label}] {s[:180]}")
hp = os.path.join(run, "fuzz", "hangs.jsonl")
if os.path.exists(hp):
    print("HANGS:", open(hp).read()[:500])
sys.exit(1 if bad else 0)
