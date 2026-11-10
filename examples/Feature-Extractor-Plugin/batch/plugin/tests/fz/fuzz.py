#!/usr/bin/env python3
"""Differential fuzzer: random expression trees x deterministic edge-heavy data, compared bit-exactly across modes.

  python3 fuzz.py --data DIR --out OUTDIR --cases 200 --seed 1 --modes cpu,gpu[,fused] [--groups ...] [--style mix|sql|df]
Pairs compared:  (cpu,gpu) -> KNOWN-DIFF candidates ;  (gpu,fused) -> FUSION BUGS ;  (cpu,fused) -> informational.
Failures are auto-minimised (tree reduction on the failing rows) and saved to OUTDIR/failures/*.json + summary.json.
"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff as D
import exprgen as X
from common import *


class Runner:
    def __init__(self, spark, tmp, modes):
        self.sp, self.tmp, self.modes = spark, tmp, modes
        self.n = 0
        self.timeout = 90
        self.cur_case = -1
        self.outdir = os.path.dirname(tmp)
        from pyspark.sql import functions as F

        self.F = F

    def on_hang(self, mode, exprs, pred):
        rec = {
            "case": self.cur_case,
            "mode": mode,
            "sql": [X.sql(e) for e in exprs],
            "filter": X.sql(pred) if pred is not None else None,
        }
        with open(os.path.join(self.outdir, "hangs.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            "HANG detected, aborting process (restart resumes after this case):",
            json.dumps(rec)[:300],
            flush=True,
        )
        os._exit(75)

    def build(self, src, exprs, pred, style):
        F = self.F
        d = src
        if pred is not None:
            d = d.filter(F.expr(X.sql(pred)) if style == "sql" else X.df(pred, F))
        if style == "sql":
            return d.selectExpr(
                "id", *[f"{X.sql(e)} AS c{i}" for i, e in enumerate(exprs)]
            )
        return d.select("id", *[X.df(e, F).alias(f"c{i}") for i, e in enumerate(exprs)])

    def run(self, mode, src, exprs, pred, style):
        """-> ('ok', arrow_table, flags) | ('err', message)"""
        self.n += 1
        out = os.path.join(self.tmp, f"o{self.n}_{mode}")
        import threading

        grp = f"fz{self.n}"
        timer = threading.Timer(
            self.timeout, lambda: self.sp.sparkContext.cancelJobGroup(grp)
        )
        hard = threading.Timer(
            self.timeout + 40, self.on_hang, args=(mode, exprs, pred)
        )
        hard.start()
        try:
            set_mode(self.sp, mode)
            self.sp.sparkContext.setJobGroup(grp, grp, True)
            timer.start()
            df = self.build(src, exprs, pred, style)
            flags = plan_flags(df)

            def once():
                shutil.rmtree(out, ignore_errors=True)
                df.write.mode("overwrite").parquet(out)
                return D.load(out)

            t = run_with_async_warm(
                self.sp, mode, once, lambda a, b: D.is_clean(D.compare(a, b))
            )
            return ("ok", t, flags)
        except Exception as e:  # noqa
            return (
                "err",
                D.short_err(e) + (" [TIMEOUT]" if "cancel" in str(e).lower() else ""),
            )
        finally:
            timer.cancel()
            hard.cancel()
            shutil.rmtree(out, ignore_errors=True)

    @staticmethod
    def differs(ra, rb):
        if ra[0] == "err" and rb[0] == "err":
            return False
        if ra[0] != rb[0]:
            return True
        return not D.is_clean(D.compare(ra[1], rb[1]))


# ---------------------------------------------------------------- minimiser
def paths(n, p=()):
    yield p
    for i, a in enumerate(n.args):
        yield from paths(a, p + (i,))


def get(n, p):
    for i in p:
        n = n.args[i]
    return n


def put(n, p, new):
    if not p:
        return new
    c = n.copy()
    c.args[p[0]] = put(n.args[p[0]], p[1:], new)
    return c


def candidates(tree):
    """reduction candidates, most aggressive first"""
    out = []
    for p in sorted(paths(tree), key=len):
        node = get(tree, p)
        for a in node.args:
            if a.t == node.t:
                out.append((put(tree, p, a), f"hoist@{p}"))
        if node.op not in ("col", "lit"):
            out.append(
                (put(tree, p, X.N("lit", [], node.t, _zero(node.t))), f"zero@{p}")
            )
            for cn in X.COLS.get(node.t, [])[:1]:
                out.append((put(tree, p, X.N("col", [], node.t, cn)), f"col@{p}"))
        if node.op in ("case", "casenoelse") and len(node.args) > 3:
            for i in range(0, len(node.args) - (1 if node.op == "case" else 0), 2):
                na = [a for j, a in enumerate(node.args) if j not in (i, i + 1)]
                if len(na) >= (3 if node.op == "case" else 2):
                    nn = node.copy()
                    nn.args = na
                    out.append((put(tree, p, nn), f"dropwhen@{p}"))
        if node.op in ("coalesce", "greatest", "least") and len(node.args) > 2:
            for i in range(len(node.args)):
                nn = node.copy()
                nn.args = node.args[:i] + node.args[i + 1 :]
                out.append((put(tree, p, nn), f"dropargs@{p}"))
        if node.op == "in" and len(node.args) > 2:
            for i in range(1, len(node.args)):
                nn = node.copy()
                nn.args = node.args[:i] + node.args[i + 1 :]
                out.append((put(tree, p, nn), f"dropin@{p}"))
    out.sort(key=lambda c: c[0].size())
    return out


def _zero(t):
    if t == "boolean":
        return False
    if t in X.INTT:
        return 0
    if t in X.FP_TYPES:
        return 0.0
    return "1970-01-01" if t == "date" else "1970-01-01 00:00:00"


def minimise(runner, sub, tree, fails, budget=120, log=print):
    cur = tree
    evals = 0
    improved = True
    while improved and evals < budget:
        improved = False
        for cand, why in candidates(cur):
            if cand.size() >= cur.size():
                continue
            evals += 1
            try:
                ok = fails(cand)
            except Exception:
                ok = False
            if ok:
                cur = cand
                improved = True
                break
            if evals >= budget:
                break
    return cur, evals


# ---------------------------------------------------------------- main
def pairs_for(modes):
    P = []
    if "cpu" in modes and "gpu" in modes:
        P.append(("cpu", "gpu", "gpu-vs-cpu"))
    if "gpu" in modes and "fused" in modes:
        P.append(("gpu", "fused", "FUSION-vs-unfused"))
    if "cpu" in modes and "fused" in modes:
        P.append(("cpu", "fused", "fused-vs-cpu"))
    if "cpu" in modes and "cpuansi" in modes:
        P.append(("cpu", "cpuansi", "harness-selftest"))
    if "gpu" in modes and "ast" in modes:
        P.append(("gpu", "ast", "ast-vs-gpu"))
    return P


def row_context(spark, data, ids, cols, runs, ncols_expr, limit=4):
    """small dict of inputs + outputs for first mismatching ids"""
    from pyspark.sql import functions as F

    ids = ids[:limit]
    inp = (
        spark.read.parquet(data)
        .filter(F.col("id").isin(ids))
        .select("id", *cols)
        .collect()
    )
    ctx = []
    for r in inp:
        ent = {
            "id": r["id"],
            "inputs": {c: D.fmt_val(r[c]) for c in cols},
            "outputs": {},
        }
        for m, res in runs.items():
            if res[0] == "ok":
                t = res[1]
                import numpy as np

                i = int(np.searchsorted(t.column("id").to_numpy(), r["id"]))
                ent["outputs"][m] = (
                    {
                        n: D.fmt_val(t.column(n)[i].as_py())
                        for n in t.column_names
                        if n != "id"
                    }
                    if i < t.num_rows
                    else "row-missing"
                )
            else:
                ent["outputs"][m] = res[1]
        ctx.append(ent)
    return ctx


def cols_used(tree):
    s = set()

    def w(n):
        if n.op == "col":
            s.add(n.x)
        for a in n.args:
            w(a)

    w(tree)
    return sorted(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=6, help="expressions per query")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--modes", default="cpu,gpu")
    ap.add_argument("--groups", default=",".join(X.ALL_GROUPS))
    ap.add_argument("--style", default="mix", choices=["mix", "sql", "df"])
    ap.add_argument("--filter-prob", type=float, default=0.3)
    ap.add_argument("--no-minimize", action="store_true")
    ap.add_argument("--min-budget", type=int, default=100)
    ap.add_argument("--local-n", type=int, default=4)
    ap.add_argument("--tz", default="UTC")
    ap.add_argument("--time-limit", type=float, default=0, help="seconds, 0 = none")
    a = ap.parse_args()
    modes = a.modes.split(",")
    groups = a.groups.split(",")
    os.makedirs(os.path.join(a.out, "failures"), exist_ok=True)
    tmp = os.path.join(a.out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    spark = build_session(
        use_gpu=any(m not in ("cpu", "cpuansi") for m in modes),
        local_n=a.local_n,
        tz=a.tz,
    )
    data_df = spark.read.parquet(a.data)
    R = Runner(spark, tmp, modes)
    P = pairs_for(modes)
    stats = collections.Counter()
    fails = []
    seen = {}
    t0 = time.time()
    cov = collections.Counter()
    print(
        f"[{now()}] fuzz start: cases={a.cases} seed={a.seed} modes={modes} pairs={[p[2] for p in P]}",
        flush=True,
    )
    start = 0
    pf = os.path.join(a.out, "progress.json")
    if os.path.exists(pf):
        start = json.load(open(pf)).get("next_case", 0)
    for c in range(start, a.cases):
        json.dump({"next_case": c + 1}, open(pf, "w"))
        R.cur_case = c
        if a.time_limit and time.time() - t0 > a.time_limit:
            print("time limit reached")
            break
        seed = a.seed * 100003 + c
        g = X.Gen(seed, max_depth=a.depth, groups=groups)
        exprs = [g.top() for _ in range(a.k)]
        pred = g.boolean(a.depth) if g.rng.random() < a.filter_prob else None
        style = (
            a.style if a.style != "mix" else ("sql" if g.rng.random() < 0.5 else "df")
        )
        runs = {m: R.run(m, data_df, exprs, pred, style) for m in modes}
        for m, r in runs.items():
            if r[0] == "ok":
                f = r[2]
                cov[m + ".queries"] += 1
                if m != "cpu" and f["gpu_project"] or f.get("fused_project"):
                    cov[m + ".gpu_project_in_plan"] += 1
                if f["fused_project"]:
                    cov[m + ".fused_project_in_plan"] += 1
                if pred is not None and (f["gpu_filter"] or f["fused_filter"]):
                    cov[m + ".gpu_filter_in_plan"] += 1
                if f["fused_filter"]:
                    cov[m + ".fused_filter_in_plan"] += 1
            else:
                cov[m + ".errors"] += 1
        stats["cases"] += 1
        for ma, mb, label in P:
            ra, rb = runs[ma], runs[mb]
            if ra[0] == "err" and rb[0] == "err":
                stats[label + ".both_err"] += 1
                continue
            if ra[0] == "ok" and rb[0] == "ok":
                cmp_ = D.compare(ra[1], rb[1])
                if D.is_clean(cmp_):
                    stats[label + ".ok"] += 1
                    continue
            else:
                cmp_ = {"rowset": None, "cols": {}}
            # ---- a failure: find failing items
            stats[label + ".FAIL"] += 1
            items = []
            if ra[0] != rb[0]:
                # isolate which expression (or the filter) triggers the error difference
                for i, e in enumerate(exprs):
                    if R.differs(
                        R.run(ma, data_df, [e], None, "sql"),
                        R.run(mb, data_df, [e], None, "sql"),
                    ):
                        items.append(("col", i))
                if pred is not None and R.differs(
                    R.run(ma, data_df, [], pred, "sql"),
                    R.run(mb, data_df, [], pred, "sql"),
                ):
                    items.append(("filter", None))
                if not items:
                    items = [("query-error", None)]
            else:
                if cmp_["rowset"] is not None:
                    items.append(("filter", None))
                for name in cmp_["cols"]:
                    items.append(("col", int(name[1:])))
            for kind, idx in items:
                tree = (
                    pred
                    if kind == "filter"
                    else (exprs[idx] if kind == "col" else None)
                )
                rec = {
                    "case": c,
                    "seed": seed,
                    "label": label,
                    "modes": [ma, mb],
                    "style": style,
                    "kind": kind,
                    "sql": X.sql(tree)
                    if tree is not None
                    else [X.sql(e) for e in exprs],
                    "filter_sql": X.sql(pred) if pred is not None else None,
                    "mismatch": None,
                }
                if ra[0] == "err" or rb[0] == "err":
                    rec["error"] = {
                        ma: ra[1] if ra[0] == "err" else None,
                        mb: rb[1] if rb[0] == "err" else None,
                    }
                else:
                    if kind == "col":
                        rec["mismatch"] = cmp_["cols"].get(f"c{idx}")
                    elif kind == "filter":
                        rec["mismatch"] = cmp_["rowset"]
                # minimise on the failing rows
                if "TIMEOUT" in json.dumps(rec.get("error", "")):
                    rec["note"] = "timeout (not minimised)"
                elif tree is not None and not a.no_minimize:
                    ids = (
                        (
                            rec["mismatch"]["ids"]
                            if kind == "col"
                            else (rec["mismatch"][2] if rec["mismatch"] else [])
                        )
                        if rec["mismatch"]
                        else []
                    )
                    subp = os.path.join(tmp, f"sub{c}_{kind}{idx}")
                    try:
                        # failing rows + a random sample so that minimisation keeps some context
                        from pyspark.sql import functions as F

                        samp = data_df.filter(
                            F.col("id").isin(ids)
                            | (F.pmod(F.col("id"), F.lit(97)) == 0)
                        ).limit(4000)
                        samp.write.mode("overwrite").parquet(subp)
                        sub = spark.read.parquet(subp)

                        def fails_f(t, kind=kind):
                            if kind == "filter":
                                ra2, rb2 = R.run(ma, sub, [], t, "sql"), R.run(
                                    mb, sub, [], t, "sql"
                                )
                            else:
                                ra2, rb2 = R.run(
                                    ma, sub, [t], pred if False else None, "sql"
                                ), R.run(mb, sub, [t], None, "sql")
                            return R.differs(ra2, rb2)

                        if fails_f(tree):
                            small, ev = minimise(
                                R, sub, tree, fails_f, budget=a.min_budget
                            )
                            rec["minimised_sql"] = X.sql(small)
                            rec["minimised_tree"] = small.to_json()
                            rec["min_evals"] = ev
                            rec["cols_used"] = cols_used(small)
                            r2 = {
                                ma: R.run(
                                    ma,
                                    sub,
                                    [] if kind == "filter" else [small],
                                    small if kind == "filter" else None,
                                    "sql",
                                ),
                                mb: R.run(
                                    mb,
                                    sub,
                                    [] if kind == "filter" else [small],
                                    small if kind == "filter" else None,
                                    "sql",
                                ),
                            }
                            if kind == "col":
                                m2 = (
                                    D.compare(r2[ma][1], r2[mb][1])
                                    if r2[ma][0] == "ok" and r2[mb][0] == "ok"
                                    else None
                                )
                                ids2 = (
                                    m2["cols"]["c0"]["ids"]
                                    if m2 and "c0" in m2["cols"]
                                    else []
                                )
                                if ids2:
                                    rec["rows"] = row_context(
                                        spark, subp, ids2, rec["cols_used"], r2, 1
                                    )
                        else:
                            rec["minimised_sql"] = None
                            rec[
                                "note"
                            ] = "not reproducible on the sampled sub-table / alone (context dependent)"
                    except Exception as e:
                        rec["minimise_error"] = str(e)[:300]
                    shutil.rmtree(subp, ignore_errors=True)
                key = (
                    label,
                    rec.get("minimised_sql") or rec["sql"]
                    if isinstance(rec["sql"], str)
                    else str(rec["error"]),
                )
                if key in seen:
                    stats[label + ".dup"] += 1
                    continue
                seen[key] = 1
                fails.append(rec)
                with open(
                    os.path.join(
                        a.out,
                        "failures",
                        f"{label}_{c}_{kind}{idx if idx is not None else ''}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(rec, f, indent=1, default=str)
                print(
                    f"[{now()}] FAIL {label} case {c} {kind}{idx}: {(rec.get('minimised_sql') or rec['sql'])[:200]}",
                    flush=True,
                )
        if c % 10 == 9:
            print(f"[{now()}] {c+1}/{a.cases} {dict(stats)}", flush=True)
    summary = {
        "args": vars(a),
        "stats": dict(stats),
        "coverage": dict(cov),
        "n_failures_unique": len(fails),
        "elapsed_s": round(time.time() - t0, 1),
        "failures": [
            {
                k: f.get(k)
                for k in ("case", "label", "kind", "minimised_sql", "sql", "style")
            }
            for f in fails
        ],
    }
    json.dump(
        summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str
    )
    print(
        "SUMMARY",
        json.dumps(
            {"stats": dict(stats), "coverage": dict(cov), "unique_failures": len(fails)}
        ),
        flush=True,
    )
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
