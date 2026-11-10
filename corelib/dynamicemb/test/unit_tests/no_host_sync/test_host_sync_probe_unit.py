# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests of host_sync_probe's analysis logic (no torch, no GPU).

A synthetic chrome trace stands in for the profiler's output, shaped like the
traces of torch 2.13.0a0+nv26.07 (cuda_runtime / gpu_memcpy / cpu_op /
user_annotation events with ts, dur, tid and args.correlation).

Run with pytest, or as a script:  python3 test_host_sync_probe_unit.py
"""
import fnmatch
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host_sync_probe as hsp  # noqa: E402
import sync_budgets  # noqa: E402

PS_FILE = "test/test_batched_dynamic_embedding_tables_v2.py"


def X(cat, name, ts, dur, tid=1, **args):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "ts": float(ts),
        "dur": float(dur),
        "tid": tid,
        "pid": 1,
        "args": args,
    }


def rt(name, ts, corr, dur=2.0, tid=1):
    return X("cuda_runtime", name, ts, dur, tid, correlation=corr, cbid=0)


def gpu_copy(kind, corr, ts=0.0):
    return X(
        "gpu_memcpy",
        "Memcpy %s" % kind,
        ts,
        1.0,
        tid=7,
        correlation=corr,
        bytes=8,
        stream=7,
    )


def synthetic_events():
    ev = []
    # ---- iteration 0 window [1000, 2000], iteration 1 window [3000, 4000]
    ev.append(X("user_annotation", "PROBE_ITER|0", 1000, 1000))
    ev.append(X("user_annotation", "PROBE_ITER|1", 3000, 1000))
    # E1: ATen .cpu() inside a C++ extension call made from the PS stand-in file
    ev.append(
        X(
            "user_annotation",
            "PROBE_EXT|flagged_compact|%s:900:find" % PS_FILE,
            1100,
            60,
        )
    )
    ev.append(X("cpu_op", "aten::_to_copy", 1105, 50))
    ev.append(X("cpu_op", "aten::copy_", 1108, 45))
    ev.append(rt("cudaMemcpyAsync", 1110, 10))
    ev.append(gpu_copy("DtoH (Device -> Pageable)", 10))
    ev.append(rt("cudaStreamSynchronize", 1130, 11))
    # E2: direct Tensor.item() from python (iteration 0), delivered with a warning marker
    ev.append(X("cpu_op", "aten::item", 1300, 60))
    ev.append(X("cpu_op", "aten::_local_scalar_dense", 1305, 50))
    ev.append(rt("cudaMemcpyAsync", 1310, 20))
    ev.append(gpu_copy("DtoH (Device -> Pinned)", 20))
    ev.append(rt("cudaStreamSynchronize", 1330, 21))
    ev.append(X("user_annotation", "PROBE_WARN|0", 1360, 0.1))
    # E3: explicit torch.cuda.synchronize()
    ev.append(
        X(
            "user_annotation",
            "PROBE_PYSYNC|torch.cuda.synchronize|dynamicemb/extendable_tensor.py:159:extend",
            1500,
            20,
        )
    )
    ev.append(rt("cudaDeviceSynchronize", 1505, 30))
    # E4: cudaMalloc on the hot path, no owner
    ev.append(rt("cudaMalloc", 1600, 40, dur=100))
    # E5: D2H into pageable memory, non_blocking: no sync call after it in the window
    ev.append(rt("cudaStreamIsCapturing", 1698, 50, dur=0.3))
    ev.append(rt("cudaMemcpyAsync", 1700, 51))
    ev.append(gpu_copy("DtoH (Device -> Pageable)", 51))
    # late python warning for E1: delivered when an enclosing handler returns
    ev.append(X("user_annotation", "PROBE_WARN|1", 1900, 0.1))
    # the barrier after the window: a sync that must NOT pair with E5 and is not counted
    ev.append(rt("cudaDeviceSynchronize", 2100, 60))
    # stale-container check: a direct sync after the ext range ended (E2 above) is not "inside" it
    # ---- iteration 1: one more direct item()
    ev.append(X("cpu_op", "aten::item", 3300, 60))
    ev.append(X("cpu_op", "aten::_local_scalar_dense", 3305, 50))
    ev.append(rt("cudaMemcpyAsync", 3310, 70))
    ev.append(gpu_copy("DtoH (Device -> Pinned)", 70))
    ev.append(rt("cudaStreamSynchronize", 3330, 71))
    ev.append(X("user_annotation", "PROBE_WARN|2", 3360, 0.1))
    ev.append(rt("cudaDeviceSynchronize", 4100, 80))  # barrier after window 1
    # noise that is not a sync
    ev.append(rt("cudaLaunchKernel", 1200, 90, dur=5))
    ev.append({"ph": "s", "cat": "ac2g", "id": 1, "ts": 1.0, "tid": 1})
    return ev


def fake_probe():
    p = hsp.Probe(use_profiler=True)
    p.trace_events = synthetic_events()
    cap = p.cap
    cap.warns = [
        {
            "idx": 0,
            "iter": 0,
            "tid": 1,
            "chain": [],
            "site": ("/x/dynamicemb/key_value_table.py", 1768, "_append_evicted"),
        },
        {
            "idx": 1,
            "iter": 0,
            "tid": 1,
            "chain": [],
            "site": ("/x/dynamicemb/batched_dynamicemb_tables.py", 1420, "forward"),
        },
        {
            "idx": 2,
            "iter": 1,
            "tid": 1,
            "chain": [],
            "site": ("/x/dynamicemb/key_value_table.py", 1768, "_append_evicted"),
        },
    ]
    cap.explicit = [
        {
            "iter": 0,
            "tid": 1,
            "api": "torch.cuda.synchronize",
            "chain": [],
            "site": ("/x/dynamicemb/extendable_tensor.py", 159, "extend"),
        },
    ]
    cap.chains = {"%s:900:find" % PS_FILE: ("/x/" + PS_FILE, 900, [])}
    return p


def test_analyze_trace_kinds_and_owners():
    tr = hsp.analyze_trace(synthetic_events(), 2)
    assert tr["windows"] == 2
    it0 = sorted(tr["per_iter"][0], key=lambda e: e["ts"])
    kinds = [(e["kind"], e["owner_typ"], bool(e["aten"])) for e in it0]
    assert kinds == [
        ("ss", "ext", True),  # E1: sync inside the ext range, under aten::copy_
        (
            "ss",
            None,
            True,
        ),  # E2: direct, under aten::item -- NOT hidden under E1's ended range
        ("ds", "pysync", False),  # E3
        ("al", None, False),  # E4
        ("pd", None, False),  # E5: not paired with the barrier sync outside the window
    ], kinds
    assert [e["kind"] for e in tr["per_iter"][1]] == ["ss"]


def test_pageable_copy_followed_by_sync_counts_once():
    tr = hsp.analyze_trace(synthetic_events(), 2)
    kinds = [e["kind"] for e in tr["per_iter"][0]]
    assert (
        kinds.count("pd") == 1 and kinds.count("ss") == 2
    )  # E1's pageable copy is not counted as pd


def test_summarize_totals_origin_and_matching():
    r = hsp._summarize(fake_probe(), 2, [hsp.PS_STANDIN])
    assert r["per_iter_total"] == [5, 1]
    assert r["per_iter_kind"]["ss"] == [2, 1]
    assert r["per_iter_kind"]["ds"] == [1, 0]
    assert r["per_iter_kind"]["al"] == [1, 0]
    assert r["per_iter_kind"]["pd"] == [1, 0]
    assert r["origin"]["cpp_aten"] == 0.5 and r["origin"]["explicit"] == 0.5
    assert r["origin"]["direct_aten"] == 1.0 and r["origin"]["direct_raw"] == 1.0
    # warning 0 and 2 name a direct python-level sync; warning 1 is the late duplicate of E1
    assert r["python_visible"] == {"warn": 1.0, "explicit": 0.5, "dup_late_warn": 0.5}
    assert r["inherent_site_per_iter"] == [1, 0]  # E1 sits in the PS stand-in file
    assert r["notes"] == [], r["notes"]
    py = [s for s in r["sites"] if s["source"] == "py_warn"]
    assert len(py) == 1 and py[0]["file"] == "dynamicemb/key_value_table.py"
    assert py[0]["total"] == 2 and py[0]["kinds"] == {"ss": 2}
    assert py[0]["ops"] == {"aten::_local_scalar_dense": 2}
    cpp = [s for s in r["sites"] if s["source"] == "cpp"]
    assert (
        len(cpp) == 1
        and cpp[0]["api"] == "flagged_compact"
        and cpp[0]["origin"] == "cpp_aten"
    )


def test_summarize_flags_unmatched_warning():
    p = fake_probe()
    p.cap.warns.append(
        {
            "idx": 3,
            "iter": 1,
            "tid": 1,
            "chain": [],
            "site": ("/x/dynamicemb/foo.py", 5, "g"),
        }
    )
    p.trace_events.append(X("user_annotation", "PROBE_WARN|3", 3400, 0.1))
    r = hsp._summarize(p, 2, [])
    assert any("matched no profiler sync event" in n for n in r["notes"]), r["notes"]


def result_of(config, scenario, probe=None, globs=()):
    r = hsp._summarize(probe or fake_probe(), 2, list(globs))
    r.update(
        config=config,
        scenario=scenario,
        mode="plain",
        status="ok",
        layout="x",
        regime={"ok": True, "note": ""},
        inherent=hsp.inherent_reason(config, scenario),
        inherent_sites=None,
    )
    return r


def zeroed(r):
    """``r`` with no sync of any kind in either iteration."""
    r.update(
        per_iter_total=[0, 0],
        per_iter_blocking=[0, 0],
        per_iter_kind={k: [0, 0] for k in hsp.KINDS},
        inherent_site_per_iter=[0, 0],
        sites=[],
    )
    r["mean"] = {k: 0.0 for k in list(hsp.KINDS) + ["total", "blocking"]}
    return r


def report_of(*results):
    return {
        "meta": {
            "time": "t",
            "torch": "x",
            "cuda": "y",
            "device": "z",
            "dynamicemb": "d",
            "git": "g",
            "iters": 2,
            "warmup": 1,
            "batch": {},
        },
        "results": list(results),
    }


def test_check_zero_site_level_and_case_level_inherent():
    # a case that is excused at call-site level (the table has no such entry today, so one is added):
    # only the syncs in the matching file are excused, the other 4/1 syncs are violations
    key = ("synthetic_ps*", "miss", hsp.PS_STANDIN)
    hsp.INHERENT[key] = "a stand-in's own host code"
    try:
        ps = result_of("synthetic_ps", "miss", globs=[hsp.PS_STANDIN])
        bad = hsp.check_zero(report_of(ps))
    finally:
        del hsp.INHERENT[key]
    assert (
        len(bad) == 1
        and "outside inherent sites" in bad[0]
        and "synthetic_ps/miss/plain" in bad[0]
    ), bad
    assert "key_value_table.py" in bad[0] or "dynamicemb" in bad[0]
    # a case excused as a whole (no entry of the table is one today, so one is added)
    whole = ("synthetic_whole*", "miss")
    hsp.INHERENT[whole] = "a whole-case excuse"
    try:
        g = result_of("synthetic_whole", "miss")
        assert g["inherent"] and hsp.check_zero(report_of(g)) == []
    finally:
        del hsp.INHERENT[whole]
    # table growth is not excused: the same numbers are a violation
    grew = result_of("hbm_only_growth", "growth")
    assert not grew["inherent"] and len(hsp.check_zero(report_of(grew))) == 1
    # the same numbers in a non-inherent scenario are a violation
    n = result_of("hbm_only", "all_hit")
    assert len(hsp.check_zero(report_of(n))) == 1
    # allow_inherent=False reports everything
    assert len(hsp.check_zero(report_of(g), allow_inherent=False)) == 1
    # a clean case passes
    clean = zeroed(result_of("hbm_only", "all_hit"))
    assert hsp.check_zero(report_of(clean)) == []
    # errors are violations, skipped cases are not
    err = {
        "config": "c",
        "scenario": "s",
        "mode": "plain",
        "status": "error",
        "error": "boom",
    }
    skp = {"config": "c", "scenario": "s", "mode": "plain", "status": "skipped"}
    assert hsp.check_zero(report_of(err, skp)) == ["c/s/plain: ERROR boom"]


def test_inherent_matching():
    # table growth is not inherent: a growth is queued, ordered against the other streams
    # and reads nothing back, in every layout and in the scenarios that grow a table
    for config in (
        "hbm_only_growth",
        "cache_storage_growth",
        "host_only_growth",
        "hbm_only_noevict",
        "cache_storage_noevict",
        "hybrid_growth",
        "host_only_noevict",
        "hbm_only_growth_wide",
        "host_only_growth_wide",
        "hbm_only_noevict_wide",
    ):
        for scenario in hsp.SCENARIO_DOC:
            assert hsp.inherent_reason(config, scenario) is None, (config, scenario)
    assert hsp.inherent_reason("hbm_only", "miss") is None
    # I1, the external PS of the host-sized contract, in every scenario but by call site: the
    # reads of DynamicEmb (sync_free.read_now) and the stand-in's own code are excused, the rest
    # of the case is not (see test_the_old_contract_ps_is_excused_by_call_site_only)
    for config in ("cache_ps", "cache_ps_sum"):
        for scenario in ("all_hit", "miss", "miss_evict", "tier2_hit"):
            assert hsp.inherent_reason(config, scenario) is None, (config, scenario)
    # nothing else is excused, the determinism mode and the growth events included: they read nothing
    for config in (
        "cache_storage",
        "hbm_only",
        "hybrid",
        "host_only",
        "cache_storage_metrics",
        "hybrid_retain",
        "cache_storage_determ",
        "hbm_only_determ",
        "hybrid_determ",
        "cache_storage_admit_determ",
    ):
        for scenario in hsp.SCENARIO_DOC:
            assert hsp.inherent_reason(config, scenario) is None, (config, scenario)
    # a PS of the padded contract reads nothing, so none of its cases has an entry
    for config in ("cache_dps", "cache_dps_sum", "host_dps", "cache_dps_chunked"):
        for scenario in hsp.SCENARIO_DOC:
            assert hsp.inherent_site_globs(config, scenario) == [], (config, scenario)
            assert hsp.inherent_reason(config, scenario) is None, (config, scenario)


def _syncs_in(*files):
    """A python-only probe whose iteration 0 made one explicit sync in each of ``files``
    (iteration 1 none)."""
    p = hsp.Probe(use_profiler=False)
    p.cap.explicit = [
        {
            "iter": 0,
            "tid": 1,
            "api": "Event.synchronize",
            "chain": [],
            "site": (path, 100 + i, "f"),
        }
        for i, path in enumerate(files)
    ]
    return p


def test_the_old_contract_ps_is_excused_by_call_site_only():
    """DynamicEmb's reads for a PS of the host-sized contract wait in sync_free.read_now and the
    stand-in's host code is in its own file: those are excused. A sync anywhere else in the same
    case (a raw .item() added to a PS flow, say) is a violation, which a whole-case excuse hid.
    """
    for config in ("cache_ps", "cache_ps_sum"):
        for scenario in ("all_hit", "miss", "miss_evict", "tier2_hit"):
            sites = hsp.inherent_site_globs(config, scenario)
            assert {glob for glob, _ in sites} == {hsp.READ_NOW_SITE, hsp.PS_STANDIN}
            assert all(why for _, why in sites)
    globs = [g for g, _ in hsp.inherent_site_globs("cache_ps", "miss")]
    excused = (
        "/x/dynamicemb/sync_free.py",
        "/x/dynamicemb/sync_free.py",
        "/x/" + PS_FILE,
    )
    ps = result_of("cache_ps", "miss", probe=_syncs_in(*excused), globs=globs)
    assert ps["per_iter_total"] == [3, 0] and ps["inherent_site_per_iter"] == [3, 0]
    assert hsp.check_zero(report_of(ps)) == []
    # the same number of syncs with one of them in the module's own code is a violation
    other = excused[:2] + ("/x/dynamicemb/types.py",)
    ps = result_of("cache_ps", "miss", probe=_syncs_in(*other), globs=globs)
    bad = hsp.check_zero(report_of(ps))
    assert len(bad) == 1 and "cache_ps/miss/plain" in bad[0], bad
    assert "outside inherent sites" in bad[0]


# --------------------------------------------------------------------------
# the acceptance counts the blocking kinds only
# --------------------------------------------------------------------------
PS_SITE = PS_FILE + ":900:find"
OWN_SITE = "dynamicemb/types.py:50:f"
GROWTH_SITE = "dynamicemb/extendable_tensor.py:227:extend"


def trace_of(*iterations):
    """A chrome trace with one window per iteration. An iteration is a list of
    (runtime or driver call, call site of the extension range it sits in, or None)."""
    ev = []
    corr = 100
    for i, calls in enumerate(iterations):
        t0 = 1000.0 + 2000.0 * i
        ev.append(X("user_annotation", "PROBE_ITER|%d" % i, t0, 1000))
        for j, (api, site) in enumerate(calls):
            ts = t0 + 100 + 150 * j
            corr += 1
            if site is not None:
                ev.append(X("user_annotation", "PROBE_EXT|f|%s" % site, ts - 10, 100))
            driver = api.startswith("cu") and not api.startswith("cuda")
            cat = "cuda_driver" if driver else "cuda_runtime"
            ev.append(X(cat, api, ts, 2.0, 1, correlation=corr, cbid=0))
    return ev


def probe_of(*iterations):
    p = hsp.Probe(use_profiler=True)
    p.trace_events = trace_of(*iterations)
    return p


def test_an_iteration_that_only_allocates_is_not_a_violation():
    """The first census of the merged tree: a growth maps pages (cuMemCreate, cuMemMap,
    cuMemSetAccess), registers a host buffer and allocates a key map, by design, and the
    ratchet bounds those events apart. check_zero summed them with the syncs and failed
    nine configs (growth and PS) that read zero blocking syncs."""
    p = probe_of(
        [
            ("cuMemCreate", GROWTH_SITE),
            ("cuMemMap", GROWTH_SITE),
            ("cuMemSetAccess", GROWTH_SITE),
            ("cudaHostRegister", GROWTH_SITE),
        ],
        [("cudaMalloc", None)],
    )
    r = result_of("hbm_only_growth_wide", "growth", probe=p)
    assert r["per_iter_total"] == [4, 1] and r["per_iter_kind"]["al"] == [4, 1]
    assert r["per_iter_blocking"] == [0, 0] and r["inherent_site_per_iter"] == [0, 0]
    assert r["mean"]["al"] == 2.5 and r["mean"]["total"] == 2.5
    assert (
        r["mean"]["blocking"] == 0
        and r["mean"]["blocking"] == sum(r["per_iter_blocking"]) / 2.0
    )
    assert hsp.check_zero(report_of(r)) == []
    assert hsp.check_zero(report_of(r), allow_inherent=False) == []
    assert hsp.effective_blocking_per_iter(r) == [0, 0]


def test_every_blocking_kind_is_a_violation_and_none_of_them_is_masked_by_allocations():
    for api, kind in (
        ("cudaStreamSynchronize", "ss"),
        ("cudaDeviceSynchronize", "ds"),
        ("cudaEventSynchronize", "es"),
        ("cudaMemcpy", "ms"),
    ):
        p = probe_of(
            [("cudaMalloc", None), ("cuMemSetAccess", GROWTH_SITE), (api, OWN_SITE)],
            [("cudaMalloc", None)],
        )
        r = result_of("hbm_only_growth_wide", "growth", probe=p)
        assert r["per_iter_kind"][kind] == [1, 0], kind
        assert r["per_iter_total"] == [3, 1] and r["per_iter_blocking"] == [1, 0], kind
        bad = hsp.check_zero(report_of(r))
        assert len(bad) == 1 and "0.50 blocking syncs/iter (max 1)" in bad[0], bad
        assert OWN_SITE in bad[0], bad  # the site of the sync, not the allocating one
    # the pageable D2H copy that no sync follows (pd) is a blocking kind as well
    r = result_of(
        "hbm_only", "miss"
    )  # fake_probe: the synthetic trace of the first tests
    assert r["per_iter_kind"]["pd"] == [1, 0] and r["per_iter_kind"]["al"] == [1, 0]
    assert r["per_iter_total"] == [5, 1] and r["per_iter_blocking"] == [4, 1]
    assert r["mean"]["blocking"] == 2.5 and r["mean"]["total"] == 3.0


def test_the_top_site_of_a_violation_is_one_that_blocks():
    """The allocator site of a growth outranks the site of the sync (it has more events)
    and must not be named as the place that syncs."""
    p = probe_of(
        [
            ("cudaMalloc", GROWTH_SITE),
            ("cudaMalloc", GROWTH_SITE),
            ("cudaMalloc", GROWTH_SITE),
            ("cudaStreamSynchronize", OWN_SITE),
        ],
        [],
    )
    r = result_of("hbm_only_growth_wide", "growth", probe=p)
    assert r["sites"][0]["site"] == GROWTH_SITE  # ranked first, by its count
    bad = hsp.check_zero(report_of(r))
    assert len(bad) == 1 and OWN_SITE in bad[0] and GROWTH_SITE not in bad[0], bad


def test_an_excused_site_that_allocates_does_not_excuse_a_sync_elsewhere():
    """The excuse of a call site is for the blocking syncs made there. The stand-in of a
    PS also grows the allocator (a dict of device tensors): those events must not be
    taken off the syncs of the module's own code in the same iteration."""
    globs = [hsp.PS_STANDIN]
    # iteration 0: the stand-in allocates twice, DynamicEmb's own code syncs once;
    # iteration 1: the stand-in syncs once (excused)
    p = probe_of(
        [
            ("cudaMalloc", PS_SITE),
            ("cudaMalloc", PS_SITE),
            ("cudaStreamSynchronize", OWN_SITE),
        ],
        [("cudaStreamSynchronize", PS_SITE)],
    )
    r = result_of("cache_ps", "miss", probe=p, globs=globs)
    assert r["per_iter_total"] == [3, 1] and r["per_iter_blocking"] == [1, 1]
    # the allocations at the excused site are not excused syncs
    assert r["inherent_site_per_iter"] == [0, 1]
    assert hsp.effective_blocking_per_iter(r) == [1, 0]
    bad = hsp.check_zero(report_of(r))
    assert len(bad) == 1 and "0.50 blocking syncs/iter (max 1)" in bad[0], bad
    assert OWN_SITE in bad[0], bad
    # without the sync of the module's own code the same case is clean, and so is one in
    # which every sync is at the excused site whatever else allocates
    p = probe_of(
        [
            ("cudaStreamSynchronize", PS_SITE),
            ("cudaMalloc", None),
            ("cudaMalloc", PS_SITE),
        ],
        [("cudaEventSynchronize", PS_SITE)],
    )
    r = result_of("cache_ps", "miss", probe=p, globs=globs)
    assert r["per_iter_blocking"] == [1, 1] and r["inherent_site_per_iter"] == [1, 1]
    assert hsp.check_zero(report_of(r)) == []
    # no excuse at all (allow_inherent=False): the syncs of the stand-in count
    assert len(hsp.check_zero(report_of(r), allow_inherent=False)) == 1


def test_the_excuse_of_one_iteration_does_not_pay_for_a_sync_of_another():
    """check_zero subtracts per iteration and never goes below zero: an excuse that is
    larger than the syncs of its iteration (a hand-made or an older report) must not
    cancel a violation of the next one."""
    r = dict(
        zeroed(result_of("cache_ps", "miss")),
        per_iter_blocking=[1, 1],
        inherent_site_per_iter=[2, 0],
    )
    assert hsp.effective_blocking_per_iter(r) == [0, 1]
    bad = hsp.check_zero(report_of(r))
    assert len(bad) == 1 and "0.50 blocking syncs/iter (max 1)" in bad[0], bad


def test_blocking_syncs_of_a_report_from_before_the_series_existed():
    """A saved report (--diff, --render) without per_iter_blocking is read from its per-kind
    series, and one that has neither from its totals."""
    kinds = {k: [0, 0] for k in hsp.KINDS}
    kinds.update(ss=[2, 1], ds=[1, 0], pd=[1, 0], al=[1, 0])
    old = {"per_iter_total": [5, 1], "per_iter_kind": kinds}
    assert hsp.blocking_per_iter(old) == [4, 1]
    assert hsp.blocking_per_iter({"per_iter_total": [3, 0]}) == [3, 0]
    assert hsp.blocking_per_iter(dict(old, per_iter_blocking=[7, 7])) == [7, 7]


def test_the_comparison_counts_blocking_syncs_and_says_so():
    """'still sync outside INHERENT' was a count of cases with allocator events."""
    before = result_of(
        "hbm_only_growth_wide",
        "growth",
        probe=probe_of(
            [("cudaStreamSynchronize", OWN_SITE)] * 3,
            [("cudaStreamSynchronize", OWN_SITE)] * 3,
        ),
    )
    only_allocates = result_of(
        "hbm_only_growth_wide",
        "growth",
        probe=probe_of([("cuMemSetAccess", GROWTH_SITE)], [("cudaMalloc", None)]),
    )
    cmp = hsp.compare_reports(report_of(before), report_of(only_allocates))
    (row,) = cmp["rows"]
    assert row["before"] == 3 and row["after"] == 0 and row["after_effective"] == 0
    assert len(cmp["improved"]) == 1 and cmp["remaining"] == []
    text = hsp.format_compare(cmp)
    assert "0 still make a blocking sync outside INHERENT" in text, text
    # a case that still blocks (and allocates) is counted
    still = result_of(
        "hbm_only_growth_wide",
        "growth",
        probe=probe_of(
            [("cudaStreamSynchronize", OWN_SITE), ("cudaMalloc", None)],
            [("cudaMalloc", None)],
        ),
    )
    cmp = hsp.compare_reports(report_of(before), report_of(still))
    assert len(cmp["remaining"]) == 1 and cmp["remaining"][0]["after_effective"] == 0.5
    assert "1 still make a blocking sync outside INHERENT" in hsp.format_compare(cmp)
    # a report from before the blocking metric existed compares on its totals
    older = dict(before, mean={"total": 6.0})
    cmp = hsp.compare_reports(report_of(older), report_of(only_allocates))
    assert cmp["rows"][0]["before"] == 6.0


def test_the_text_report_separates_the_blocking_syncs_from_the_allocator_events():
    r = result_of(
        "hbm_only_growth_wide",
        "growth",
        probe=probe_of(
            [("cuMemCreate", GROWTH_SITE), ("cuMemMap", GROWTH_SITE)] * 2,
            [("cudaMalloc", None)],
        ),
    )
    text = hsp.format_report(report_of(r), detail="none")
    assert "blk" in text and "allocator events" in text
    # the table row: tot is every event (2.5 a step), blk the blocking ones (none)
    row = next(
        ln.split()
        for ln in text.splitlines()
        if ln.startswith("hbm_only_growth_wide ") and "growth" in ln and "|" in ln
    )
    assert row[3:5] == ["2.50", "0"], row
    # the ranking of call sites names the ones that block first; a site that only allocates
    # (it may have more events and be in more cases) comes after them, marked as such
    blocks = result_of(
        "hbm_only",
        "miss",
        probe=probe_of(
            [("cudaStreamSynchronize", OWN_SITE)], [("cudaStreamSynchronize", OWN_SITE)]
        ),
    )
    ranked = hsp.format_top_sites(report_of(r, r, blocks)).splitlines()
    sites = [ln for ln in ranked if "cpp " in ln]
    assert OWN_SITE in sites[0] and "(alloc)" not in sites[0], sites
    assert GROWTH_SITE in sites[1] and sites[1].endswith("(alloc)"), sites
    # the pivot reads the blocking syncs: the growth column is 0, not 2.5
    pivot = next(
        ln.split()
        for ln in text.splitlines()
        if ln.startswith("hbm_only_growth_wide ") and "|" not in ln
    )
    assert pivot == ["hbm_only_growth_wide"] + ["-"] * 6 + ["0", "-"], pivot


def test_the_allocations_of_a_stand_in_are_excused_as_they_were_with_the_whole_case():
    """The sync is excused by call site, an allocation has none: a case with a stand-in's
    code in it (a dict of device tensors that grows the allocator) is not bounded for
    cudaMalloc events; every other case is, a growth event included (it gets an allowance
    per growth, see test_sync_ratchet.py)."""
    for config, scenario in (
        ("cache_ps", "all_hit"),
        ("cache_ps_sum", "tier2_hit"),
    ):
        assert hsp.allocations_excused(config, scenario), (config, scenario)
    for config, scenario in (
        ("cache_storage", "miss"),
        ("cache_dps", "tier2_hit"),
        ("host_dps", "miss"),
        ("hbm_only_growth", "all_hit"),
        ("hbm_only_growth", "growth"),
        ("cache_storage_determ", "all_hit"),
    ):
        assert not hsp.allocations_excused(config, scenario), (config, scenario)


def test_the_wide_growth_configs_cross_the_vmm_granule_and_the_narrow_ones_do_not():
    """A device value buffer is mapped in granules of 2 MiB and a growth inside the granule
    already mapped never reaches the driver calls that could wait (cuMemMap, cuMemSetAccess):
    the rows of the default width stay under it in the growth window, the wide ones do not.
    """
    granule = 2 << 20
    window_rows = 8192  # the most rows a table holds in the growth window (see plan_of)
    for name, cfg in hsp.CONFIGS.items():
        if not cfg.growth:
            continue
        dim = cfg.dim or hsp.DIM
        crosses = window_rows * dim * 4 > granule
        assert crosses == name.endswith("_wide"), (name, dim)
    for name in (
        "hbm_only_growth_wide",
        "host_only_growth_wide",
        "hbm_only_noevict_wide",
    ):
        assert hsp.CONFIGS[name].growth and hsp.CONFIGS[name].dim > hsp.DIM


def test_the_growth_configs_cover_every_layout_that_grows():
    layouts = {c.layout for c in hsp.CONFIGS.values() if c.growth}
    assert layouts == {"cache", "hbm", "host", "hybrid"}
    noevict = {c.layout for c in hsp.CONFIGS.values() if c.growth and c.noevict}
    assert noevict == {"cache", "hbm", "host"}  # HybridStorage cannot hold NO_EVICTION


def test_the_hybrid_growth_scenario_overflows_the_hbm_tier_into_the_host_tier():
    """Only the host tier of a hybrid grows, and from what it holds: the keys the HBM tier
    spills (the first hardware run of the merged tree saw no growth with an HBM tier that took
    every key). The growth scenario gives the HBM tier one bucket, fills it with the warm-up
    and moves the keys of the window to the host tier -- enough of them to double a key map
    that the warm-up has grown to the most it can reach -- while the other scenarios keep the
    HBM tier roomy: a miss there is a miss into a tier that is not full."""
    cfg = hsp.CONFIGS["hybrid_growth"]
    iters = sync_budgets.CENSUS_ITERS
    grow = hsp.plan_of(cfg, "growth")
    assert grow["hbm_rows"] == 1024  # one bucket of the HBM tier (bucket_capacity 1024)
    warm_keys = 3 * int(hsp.growth_warmup_scale(cfg) * hsp.UNIQ[0])
    assert warm_keys > grow["hbm_rows"], "the HBM tier is full when the window starts"
    # The host key map starts at init_capacity (1,024 rows) and the batches in flight of the
    # warm-up can double it twice (the bound of a step is its key slots); a growth in the window
    # needs the keys spilled before its last step, and the bound of that step, to pass the load
    # factor of that capacity.
    start = 4 * grow["init_capacity"]
    spilled = warm_keys + (iters - 1) * hsp.UNIQ[0] - grow["hbm_rows"]
    assert spilled + hsp.ROW_N[0] > 0.5 * start, (spilled, start)
    # the other scenarios hold every key of the warm-up and the window in the HBM tier
    for scenario in ("all_hit", "miss"):
        roomy = hsp.plan_of(cfg, scenario)
        assert roomy["hbm_rows"] > (3 + iters) * hsp.UNIQ[0], scenario
    # a small warm-up is for the others: nothing has grown when their window starts
    for name, other in hsp.CONFIGS.items():
        if other.growth and other.layout != "hybrid":
            assert hsp.growth_warmup_scale(other) == 0.1, name


def test_every_inherent_entry_excuses_a_case_the_probe_measures():
    cases = sync_budgets.case_keys()
    for key in hsp.INHERENT:
        config_glob, scenario_glob = key[0], key[1]
        assert any(
            fnmatch.fnmatch(c, config_glob) and fnmatch.fnmatch(s, scenario_glob)
            for c, s, _ in cases
        ), "the INHERENT entry %r matches no probe case" % (key,)
    assert all(reason for reason in hsp.INHERENT.values())


def test_text_report_renders():
    ps = result_of("cache_ps", "miss", globs=[hsp.PS_STANDIN])
    err = {
        "config": "c",
        "scenario": "s",
        "mode": "plain",
        "status": "error",
        "error": "boom",
        "traceback": "tb",
    }
    text = hsp.format_report(report_of(ps, err), sites=5, detail="full")
    assert "PIVOT mode=plain" in text and "cache_ps" in text and "(-0.5)" in text
    assert "py_warn" in text and "flagged_compact" in text and "ERROR boom" in text
    hsp.format_report(report_of(ps), detail="none")


def test_short_and_frames():
    assert hsp._short("/a/b/site-packages/torch/x.py") == "torch/x.py"
    assert (
        hsp._short("/opt/tree/dynamicemb/dynamicemb/key_value_table.py")
        == "dynamicemb/key_value_table.py"
    )
    assert hsp._is_skip_frame(hsp._THIS_FILE)
    assert hsp._is_skip_frame("/usr/lib/python3/dist-packages/torch/_tensor.py")
    assert not hsp._is_skip_frame("/x/dynamicemb/key_value_table.py")
    site, chain = hsp._resolve(sys._getframe(0))
    assert site[2] == "test_short_and_frames" and site[0].endswith(
        "test_host_sync_probe_unit.py"
    )


def test_static_scan_finds_known_patterns(tmp_path=None):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "dynamicemb"))
    os.makedirs(os.path.join(d, "src"))
    open(os.path.join(d, "dynamicemb", "a.py"), "w").write(
        "x = t.item()\n# y = t.cpu()\nz = int(a.sum())\n"
    )
    open(os.path.join(d, "src", "b.cu"), "w").write(
        "cudaStreamSynchronize(s);\n// cudaDeviceSynchronize();\n"
    )
    rows = hsp.static_scan(d)
    assert [(f, i) for f, i, _ in rows] == [
        ("dynamicemb/a.py", 1),
        ("dynamicemb/a.py", 3),
        ("src/b.cu", 1),
    ], rows


def test_config_table_is_consistent():
    assert len(hsp.CONFIGS) == len(set(hsp.CONFIGS))
    for c in hsp.CONFIGS.values():
        assert set(hsp.scenarios_of(c)) <= set(hsp.SCENARIO_DOC)
        for sc in hsp.scenarios_of(c):
            plan = hsp.plan_of(c, sc)
            assert plan["max_capacity"] > 0
            if c.layout in ("cache", "cache_ps", "hybrid"):
                # cache / HBM tier smaller than the table, a power-of-two fraction of it, bucket aligned
                assert plan["hbm_rows"] < plan["max_capacity"]
                assert (
                    plan["max_capacity"] % plan["hbm_rows"] == 0
                    and plan["hbm_rows"] % 1024 == 0
                )
            if c.growth:
                assert plan["init_capacity"] < plan["max_capacity"]
        assert ("growth" in hsp.scenarios_of(c)) == c.growth


def test_config_fields_are_known_values_and_consistent():
    """The fields that the module builder switches on (optimizer, dtype, key type,
    pooling, admission, score) take values it knows, several dims need pooling, and
    queue_ahead -- which sleeps on the stream and issues whole steps -- exists for the
    per-key-score configs only and runs in one mode."""
    known = {
        "optimizer": {"sgd", "adam", "adagrad", "rowwise_adagrad", "ftrl"},
        "dtype": {"fp32", "fp16", "bf16"},
        "keys": {"int64", "uint64"},
        "pooling": {"none", "sum", "mean"},
        "admit_kind": {"frequency", "probabilistic"},
        "score": {"default", "step", "customized", "lrulfu", "precise"},
    }
    for c in hsp.CONFIGS.values():
        for field, values in known.items():
            assert getattr(c, field) in values, (c.name, field)
        if len(set(c.dims)) > 1:
            assert c.pooling != "none", c.name  # a sequence lookup needs one dim
        if c.admit_kind != "frequency" or c.admit_init:
            assert c.admit, c.name
        assert ("queue_ahead" in hsp.scenarios_of(c)) == (c.score == "precise"), c.name
        # a per-key score needs a layout that supports it
        assert c.score != "precise" or c.layout in ("cache", "hbm"), c.name
    assert set(hsp.PLAIN_ONLY_SCENARIOS) <= set(hsp.SCENARIO_DOC)
    assert set(hsp.EVAL_SCENARIOS) <= set(hsp.PLAIN_ONLY_SCENARIOS)
    assert hsp.QUEUE_AHEAD_STEPS > 4  # more than the slots of the score check's ring


def ring_state(pending, slots=4, uncovered=True):
    return {"pending": pending, "slots": slots, "uncovered": uncovered}


def test_queue_ahead_regime_is_a_ring_that_was_full_after_every_iteration():
    """The zero of queue_ahead means something only if the host really ran a whole ring of
    checks ahead of the device: judged from the ring after the issue loop, not assumed.
    """
    cfg = hsp.CONFIGS["hbm_only_precise"]

    def regime(ring):
        return hsp.regime_check(cfg, "queue_ahead", {}, {}, 8, ring)

    ok, note = regime([ring_state(4)] * 8)
    assert ok, note
    # nothing recorded: not the regime
    assert not regime(None)[0] and not regime([])[0]
    # one iteration in which the ring was not full (a host slower than the sleep)
    assert not regime([ring_state(4)] * 7 + [ring_state(2)])[0]
    # full, but the last check found a free slot: the checks were not ahead of the device
    assert not regime([ring_state(4, uncovered=False)] * 8)[0]
    # a tree that has no such flag (before the late check never waited) is judged on the ring
    assert regime([ring_state(4, uncovered=None)] * 8)[0]
    # the note says how many iterations had it
    assert "2 of 3" in regime([ring_state(4), ring_state(1), ring_state(4)])[1]


def test_a_queue_ahead_case_whose_regime_did_not_happen_is_not_a_zero():
    """check_zero reports it though no sync was seen: the host was too slow, so no ring
    was ahead of the device and nothing was measured."""
    clean = zeroed(result_of("hbm_only_precise", "queue_ahead"))
    clean.update(inherent_site_per_iter=None)
    assert hsp.check_zero(report_of(clean)) == []
    vacuous = dict(
        clean, regime={"ok": False, "note": "ring full in 0 of 2 iterations"}
    )
    bad = hsp.check_zero(report_of(vacuous))
    assert len(bad) == 1 and "regime did not happen" in bad[0], bad
    assert "hbm_only_precise/queue_ahead/plain" in bad[0]
    assert "QUEUE_AHEAD_CYCLES" in bad[0]


def test_a_case_whose_scenario_did_not_happen_is_not_a_zero_in_any_scenario():
    """The first census of the merged tree had a growth case in which no table grew (the
    host tier of a hybrid) and a miss case whose tier was full: both read zero, and the
    acceptance said nothing, because only queue_ahead was held to its regime. A zero is a
    statement about a scenario; if it did not happen there is no statement."""
    clean = zeroed(result_of("hybrid_growth", "growth"))
    assert hsp.check_zero(report_of(clean)) == []
    for scenario in hsp.SCENARIO_DOC:
        case = dict(
            clean, scenario=scenario, regime={"ok": False, "note": "nothing grew"}
        )
        bad = hsp.check_zero(report_of(case))
        assert len(bad) == 1 and "regime did not happen" in bad[0], (scenario, bad)
        assert (
            "hybrid_growth/%s/plain" % scenario in bad[0] and "nothing grew" in bad[0]
        )
    # a case that is whole-case INHERENT is held to it as well (it is excused from the
    # syncs, not from being what its name says)
    whole = ("synthetic_whole*", "growth")
    hsp.INHERENT[whole] = "a whole-case excuse"
    try:
        case = dict(
            clean,
            config="synthetic_whole",
            inherent="a whole-case excuse",
            regime={"ok": False, "note": "nothing grew"},
        )
        bad = hsp.check_zero(report_of(case))
    finally:
        del hsp.INHERENT[whole]
    assert len(bad) == 1 and "regime did not happen" in bad[0], bad
    # the regime failing does not hide a sync of the same case: both are reported
    syncing = dict(
        result_of("hybrid_growth", "growth"),
        regime={"ok": False, "note": "nothing grew"},
    )
    bad = hsp.check_zero(report_of(syncing))
    assert len(bad) == 2 and "regime did not happen" in bad[0], bad
    assert "outside inherent sites" in bad[1], bad


def _tiers(**sizes):
    """tier_sizes of a module: name -> {"size": [t0, t1], "cap": [t0, t1]}."""
    return {name: {"size": list(v), "cap": [4096, 4096]} for name, v in sizes.items()}


def test_tier2_hit_is_held_to_where_the_keys_are():
    """The size of the backing tiers not changing says that no key was added to them, not that
    the keys of the window miss the top tier: a hybrid with LFU eviction held a fifth to three
    quarters of them in its HBM tier (tier_residency, the first hardware run of the series)
    and the case read zero as a tier-2 hit. The layouts with a PS had no check at all.
    """
    hybrid = hsp.CONFIGS["hybrid_lfu"]
    sizes = _tiers(hbm=(2048, 2048), host=(8992, 3456))
    ok, note = hsp.regime_check(hybrid, "tier2_hit", sizes, sizes, 8)
    assert ok, note  # the sizes alone are as they were
    leaky = {"hbm": [0.22, 0.74], "host": [0.78, 0.26]}
    ok, note = hsp.regime_check(hybrid, "tier2_hit", sizes, sizes, 8, residency=leaky)
    assert not ok and "holds [0.22, 0.74]" in note, note
    held = {"hbm": [0.0, 0.0], "host": [1.0, 1.0]}
    ok, note = hsp.regime_check(hybrid, "tier2_hit", sizes, sizes, 8, residency=held)
    assert ok, note
    # the size check still fails the case when a key was added below
    grown = _tiers(hbm=(2048, 2048), host=(9100, 3456))
    ok, _ = hsp.regime_check(hybrid, "tier2_hit", sizes, grown, 8, residency=held)
    assert not ok
    # a PS: no size check, but the keys come from the lower tier that has them -- the PS for
    # one table, the storage for the other (the storage holds what its capacity lets it)
    ps = hsp.CONFIGS["cache_ps"]
    tiers = _tiers(cache=(1024, 1024), storage=(4096, 3766), ps=(4590, 0))
    served = {"cache": [0.0, 0.0], "storage": [0.0, 0.987], "ps": [1.0, 0.013]}
    ok, note = hsp.regime_check(ps, "tier2_hit", tiers, tiers, 8, residency=served)
    assert ok, note
    nowhere = {"cache": [0.0, 0.0], "storage": [0.0, 0.0], "ps": [0.0, 0.0]}
    ok, note = hsp.regime_check(ps, "tier2_hit", tiers, tiers, 8, residency=nowhere)
    assert not ok and "a lower tier holds [0.0, 0.0]" in note, note
    # nothing to ask (an older tree): the check says so and passes on the sizes
    ok, note = hsp.regime_check(ps, "tier2_hit", tiers, tiers, 8, residency={})
    assert ok and "no backing-tier occupancy check" in note, note


def test_all_hit_and_eval_are_held_to_the_keys_being_in_the_top_tier():
    cfg = hsp.CONFIGS["hbm_only"]
    sizes = _tiers(storage=(345, 172))
    for scenario in ("all_hit", "eval"):
        ok, note = hsp.regime_check(
            cfg, scenario, sizes, sizes, 8, residency={"storage": [1.0, 0.99]}
        )
        assert ok, note
        # nothing is added, and nothing is held: a batch of keys the table refused
        ok, note = hsp.regime_check(
            cfg, scenario, sizes, sizes, 8, residency={"storage": [0.5, 1.0]}
        )
        assert not ok and "holds [0.5, 1.0]" in note, note
        # no residency (an older tree): the size check as before
        ok, note = hsp.regime_check(cfg, scenario, sizes, sizes, 8)
        assert ok, note


def test_miss_evict_is_held_for_both_tables():
    """Only the first table was checked, so a table whose tier never filled passed."""
    cfg = hsp.CONFIGS["hbm_only"]
    k = 8
    new = [k * hsp.UNIQ[0], k * hsp.UNIQ[1]]
    full = {"storage": {"size": [4090, 4090], "cap": [4096, 4096]}}
    before = {"storage": {"size": [4090, 4090], "cap": [4096, 4096]}}
    ok, note = hsp.regime_check(cfg, "miss_evict", before, full, k)
    assert ok, note
    # the second table is at a quarter, growing by what it is given: nothing is evicted there
    idle = {"storage": {"size": [4090, 1024 + new[1]], "cap": [4096, 4096]}}
    before = {"storage": {"size": [4090, 1024], "cap": [4096, 4096]}}
    ok, note = hsp.regime_check(cfg, "miss_evict", before, idle, k)
    assert not ok, note


def test_all_hit_scenarios_warm_up_long_enough_for_the_admitter():
    """A coin toss admits a key at each sight with probability p, so a batch that is to be
    all hit needs many steps; frequency admission needs its threshold, others a few."""
    plain = hsp.CONFIGS["cache_storage"]
    frequency = hsp.CONFIGS["cache_storage_admit"]
    probabilistic = hsp.CONFIGS["cache_storage_prob_admit"]
    assert not plain.admit and frequency.admit_kind == "frequency"
    assert hsp.admit_warmup_steps(plain) == 3
    assert hsp.admit_warmup_steps(frequency) >= hsp.ADMIT_THRESHOLD
    # the p of the probabilistic config is 0.5: a key is refused after n steps with 0.5**n
    assert 0.5 ** hsp.admit_warmup_steps(probabilistic) < 1e-6


if __name__ == "__main__":
    fns = [
        (n, f)
        for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    failed = 0
    for n, f in fns:
        try:
            f()
            print("PASS", n)
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback

            print("FAIL", n, repr(e))
            traceback.print_exc()
    print("%d/%d passed" % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
