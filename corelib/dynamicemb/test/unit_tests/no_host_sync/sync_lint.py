#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static host-sync pattern lister for DynamicEmb (standard library only, py3.6 compatible).

usage:
  sync_lint.py <dynamicemb-root> [--json out.json] [--diff baseline.json] [--ignore-tests]

Lists every line of corelib/dynamicemb/{dynamicemb,src} that matches a host-sync pattern, with a coarse label.
With --diff it prints what disappeared and what is new relative to a baseline run, matching hits by
(file, normalized text) so that line shifts do not matter. lint_baseline.json, next to this file, is the
list of hits of the tree before the sync removal (253 of them); test_sync_free_lint.py compares the current
tree with it and wants every new hit on its allow-list.
A hit is a CANDIDATE: ".item()" on a CPU tensor is not a sync. The tool only guarantees that nothing
matching these patterns is added or left behind unnoticed.
"""
import json
import os
import re
import sys

PY_PATTERNS = [
    ("item", r"\.item\s*\("),
    ("cpu", r"\.cpu\s*\("),
    ("tolist", r"\.tolist\s*\("),
    ("numpy", r"\.numpy\s*\("),
    ("sync_call", r"\.synchronize\s*\("),
    ("cuda_sync", r"torch\.cuda\.synchronize"),
    ("blocking_copy", r"non_blocking\s*=\s*False"),
    ("nonzero", r"\bnonzero\b"),
    ("unique", r"torch\.unique|\.unique\s*\("),
    ("masked_select", r"masked_select"),
    ("repeat_interleave", r"repeat_interleave"),
    ("any_all_in_cond", r"\b(if|while|assert)\b.*\.(any|all)\s*\(\s*\)"),
    (
        "int_bool_of_tensor",
        r"\b(int|bool|float)\s*\(\s*[A-Za-z_][\w\.\[\]]*\.(sum|max|min|any|all)\s*\(",
    ),
    ("flagged_compact", r"\bflagged_compact\s*\("),
    ("pinned_copy_helper", r"_copy_cuda_tensor_to_pinned_cpu|_scalar_item|_bool_item"),
    ("empty_cache", r"empty_cache\s*\("),
    ("pin_memory", r"pin_memory\s*=\s*True|\.pin_memory\s*\("),
    ("tensor_to_device", r"torch\.(tensor|as_tensor)\s*\(.*device\s*="),
    ("equal_allclose", r"torch\.(equal|allclose)\s*\("),
]
CPP_PATTERNS = [
    ("cudaStreamSynchronize", r"cudaStreamSynchronize|stream_synchronize"),
    ("cudaDeviceSynchronize", r"cudaDeviceSynchronize|cuCtxSynchronize"),
    ("cudaEventSynchronize", r"cudaEventSynchronize"),
    ("cudaMemcpy_sync", r"\bcudaMemcpy\s*\("),
    ("cudaMemcpyAsync_D2H", r"cudaMemcpyDeviceToHost|cudaMemcpyDefault"),
    (
        "cpu_item",
        r"\.cpu\s*\(\s*\)|\.item\s*<|\.item\s*\(\s*\)|\.to\s*\(\s*(at::)?kCPU|\.to\(torch::kCPU",
    ),
    (
        "alloc_free",
        r"\bcudaMalloc\s*\(|\bcudaFree\s*\(|cudaHostAlloc|cudaMallocHost|cudaFreeHost|cudaHostRegister|cudaHostUnregister",
    ),
    ("nonzero_unique", r"at::nonzero|at::unique|masked_select|torch::nonzero"),
    ("device_counter", r"DeviceCounter|DeviceTimestamp"),
    ("returns_host_count", r"\bint64_t\s+h_\w+|\bh_count\b"),
]


def scan(root, ignore_tests):
    hits = []
    for base in ("dynamicemb", "src"):
        top = os.path.join(root, base)
        for dp, dn, fn in os.walk(top):
            dn[:] = [d for d in dn if d not in ("__pycache__", "build")]
            for f in fn:
                path = os.path.join(dp, f)
                rel = os.path.relpath(path, root)
                if f.endswith(".py"):
                    pats = PY_PATTERNS
                elif f.endswith((".cu", ".cuh", ".h", ".cpp")):
                    pats = CPP_PATTERNS
                else:
                    continue
                with open(path, "r", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        s = line.strip()
                        if not s or s.startswith(("#", "//", "*", "/*")):
                            continue
                        for label, rx in pats:
                            if re.search(rx, line):
                                hits.append(
                                    {
                                        "file": rel,
                                        "line": i,
                                        "label": label,
                                        "text": s[:200],
                                    }
                                )
    return hits


def norm(h):
    return (h["file"], h["label"], re.sub(r"\s+", " ", h["text"]))


def diff_hits(old_hits, new_hits):
    """(gone, new): the hits of old_hits that new_hits lacks and the hits it adds, as
    lists of ((file, label, text), count)."""
    oc, nc = {}, {}
    for h in old_hits:
        oc[norm(h)] = oc.get(norm(h), 0) + 1
    for h in new_hits:
        nc[norm(h)] = nc.get(norm(h), 0) + 1
    gone = [(k, oc[k] - nc.get(k, 0)) for k in oc if oc[k] > nc.get(k, 0)]
    new = [(k, nc[k] - oc.get(k, 0)) for k in nc if nc[k] > oc.get(k, 0)]
    return gone, new


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    root = args[0]
    out = None
    base = None
    if "--json" in args:
        out = args[args.index("--json") + 1]
    if "--diff" in args:
        base = args[args.index("--diff") + 1]
    hits = scan(root, "--ignore-tests" in args)
    if out:
        with open(out, "w") as fh:
            json.dump(hits, fh, indent=1)
    by_label = {}
    for h in hits:
        by_label[h["label"]] = by_label.get(h["label"], 0) + 1
    print("total candidate hits: %d" % len(hits))
    for k in sorted(by_label, key=lambda k: -by_label[k]):
        print("  %-24s %d" % (k, by_label[k]))
    if base:
        with open(base) as fh:
            old = json.load(fh)
        gone, new = diff_hits(old, hits)
        print("\nREMOVED since baseline: %d" % sum(n for _, n in gone))
        for (f, lab, t), n in sorted(gone):
            print("  - %s [%s] x%d: %s" % (f, lab, n, t[:140]))
        print("\nNEW since baseline: %d" % sum(n for _, n in new))
        for (f, lab, t), n in sorted(new):
            print("  + %s [%s] x%d: %s" % (f, lab, n, t[:140]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
