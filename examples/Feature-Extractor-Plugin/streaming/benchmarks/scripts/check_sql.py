#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the SQL a driver run executed is the text in cases/<case>.sql.

usage: check_sql.py RESULT_ROOT       (every RESULT_ROOT/<case>/<mode>/<run>/result.json is checked)
The driver embeds its SQL; result.json records what was run. Whitespace (also inside parentheses) and comments are ignored.
Exit status 1 if any run differs.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "..", "cases")


def norm(s):
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", s))  # layout inside parentheses


def load(case):
    lines = [
        l
        for l in open(os.path.join(CASES, case + ".sql"))
        if not l.lstrip().startswith("--")
    ]
    stmts = [norm(x) for x in "".join(lines).split(";") if x.strip()]
    return stmts[:-1], stmts[-1]


def main(root):
    bad = checked = 0
    for case in sorted(os.listdir(root)):
        if not os.path.exists(os.path.join(CASES, case + ".sql")):
            continue
        views, query = load(case)
        for mode in sorted(os.listdir(os.path.join(root, case))):
            for run in sorted(os.listdir(os.path.join(root, case, mode))):
                f = os.path.join(root, case, mode, run, "result.json")
                if not os.path.exists(f):
                    continue
                r = json.load(open(f))
                got_views = [norm(v) for v in r["views"].split(" ; ") if v.strip()]
                checked += 1
                if norm(r["sql"]) != query or got_views != views:
                    bad += 1
                    print(f"DIFFERENT {case}/{mode}/{run}")
    print(f"checked {checked} runs, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
