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
    clean = result_of("hbm_only", "all_hit")
    clean.update(per_iter_total=[0, 0], inherent_site_per_iter=[0, 0], sites=[])
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
    clean = result_of("hbm_only_precise", "queue_ahead")
    clean.update(per_iter_total=[0, 0], inherent_site_per_iter=None)
    assert hsp.check_zero(report_of(clean)) == []
    vacuous = dict(
        clean, regime={"ok": False, "note": "ring full in 0 of 2 iterations"}
    )
    bad = hsp.check_zero(report_of(vacuous))
    assert len(bad) == 1 and "regime did not happen" in bad[0], bad
    assert "hbm_only_precise/queue_ahead/plain" in bad[0]
    # other scenarios keep their regime as a flag in the report, not a failure
    other = dict(clean, scenario="all_hit", regime={"ok": False, "note": "x"})
    assert hsp.check_zero(report_of(other)) == []


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
