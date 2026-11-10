# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The sync ratchet: a probe case may make no more host syncs than budgets.json lets it.

budgets.json starts as the census of the tree before the sync removal (the cache
layout reads 4 / 8 / 10 times per step on an all-hit / miss / miss+evict batch). Every
commit that removes syncs lowers the numbers it earned in the same diff, so a reviewer
sees the claim and nothing can quietly give it back; see sync_budgets.py for the file
and the command that lowers it. The ratchet also bounds the allocator events
(cudaMalloc/cudaFree, pages mapped or registered) of a measured window -- steady state
allocates nothing, a growth a few events -- and the transient memory a sync-free hit step
adds (B8).

Needs a GPU and the CUDA profiler for the census; the checks of the file itself run
anywhere. The probe's instruments are validated first (the ``instruments_ok``
fixture), so a zero is never an artefact of a dead profiler.

    pytest -q test_sync_ratchet.py                      # every config: about 3 minutes
    HSP_CONFIGS=cache_storage,hbm_only pytest -q ...    # some configs
    HSP_SCENARIOS=all_hit,miss HSP_MODES=plain ...      # some cases
    HSP_RATCHET_STRICT=1 pytest -q ...                  # also fail a case BELOW its budget:
                                                        # the commit gate (lower budgets.json)
    HSP_RATCHET_SLACK=0.5 pytest -q ...                 # tolerate this many extra syncs per
                                                        # iteration (a different GPU or torch)

The budgets are means over the iterations recorded in budgets.json ("_meta"), and the
census runs with those unless HSP_ITERS / HSP_WARMUP say otherwise.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import host_sync_probe as hsp  # noqa: E402
import sync_budgets as sb  # noqa: E402


# --------------------------------------------------------------------------
# budgets.json itself (CPU only)
# --------------------------------------------------------------------------
def test_budgets_cover_exactly_the_probe_cases():
    meta, budgets = sb.load_budgets()
    problems = sb.validate(meta, budgets)
    assert not problems, "\n".join(problems[:30])


def test_budgets_file_survives_the_regeneration_tool():
    """format_budgets writes what load_budgets reads back (so the tool's output is a
    reviewable diff of numbers only)."""
    meta, budgets = sb.load_budgets()
    text = sb.format_budgets(meta, budgets)
    data = json.loads(text)
    assert data.pop("_meta") == meta
    assert data == budgets


def _report(*cases):
    """A probe report with (config, scenario, mode, mean blocking syncs)."""
    results = []
    for config, scenario, mode, value in cases:
        results.append(
            {
                "config": config,
                "scenario": scenario,
                "mode": mode,
                "status": "ok",
                "mean": {sb.METRIC: value},
            }
        )
    return {"meta": {"label": "synthetic"}, "results": results}


def test_budgets_are_lowered_by_a_report_and_never_raised():
    previous = {"cache_storage": {"all_hit": {"plain": 4, "prefetch": 4}}}
    update = sb.budgets_from_report(
        _report(
            ("cache_storage", "all_hit", "plain", 2.0),
            ("cache_storage", "all_hit", "prefetch", 5.0),
            ("hbm_only", "all_hit", "plain", 0.125),
        ),
        previous,
    )
    assert update.budgets["cache_storage"]["all_hit"] == {"plain": 2, "prefetch": 4}
    assert update.budgets["hbm_only"]["all_hit"] == {"plain": 0.125}
    assert update.changes == [
        (("cache_storage", "all_hit", "plain"), 4, 2),
        (("hbm_only", "all_hit", "plain"), None, 0.125),
    ]
    assert len(update.refused) == 1 and "above its budget 4" in update.refused[0]
    assert previous == {
        "cache_storage": {"all_hit": {"plain": 4, "prefetch": 4}}
    }  # untouched

    raised = sb.budgets_from_report(
        _report(("cache_storage", "all_hit", "prefetch", 5.0)),
        previous,
        allow_raise=True,
    )
    assert (
        raised.budgets["cache_storage"]["all_hit"]["prefetch"] == 5
        and not raised.refused
    )


def test_a_census_clears_the_claims_it_has_measured():
    """budgets.json says which cells are claims, not measurements (_meta["unmeasured"]);
    --write must not leave that note behind for a config a census has measured."""
    meta = {
        "iters": 8,
        "unmeasured": {
            "what": "claims",
            "lowered_to_zero": ["a", "b"],
            "new_configs": ["c"],
        },
    }
    partial = sb.clear_measured_claims(
        meta, _report(("a", "s", "plain", 0.0), ("c", "s", "plain", 0.0))
    )
    assert partial["unmeasured"] == {
        "what": "claims",
        "lowered_to_zero": ["b"],
        "new_configs": [],
    }
    assert partial["iters"] == 8
    # a config the census skipped is still a claim
    skipped = _report(("a", "s", "plain", 0.0))
    skipped["results"].append(
        {"config": "b", "scenario": "s", "mode": "plain", "status": "skipped"}
    )
    assert sb.clear_measured_claims(meta, skipped)["unmeasured"]["lowered_to_zero"] == [
        "b"
    ]
    # everything measured: the note goes, with nothing else
    everything = _report(
        ("a", "s", "plain", 0.0), ("b", "s", "plain", 0.0), ("c", "s", "plain", 0.0)
    )
    assert sb.clear_measured_claims(meta, everything) == {"iters": 8}
    # the argument is not changed, and a file without the note is as it was
    assert meta["unmeasured"]["lowered_to_zero"] == ["a", "b"]
    assert sb.clear_measured_claims({"iters": 8}, everything) == {"iters": 8}


def test_a_partial_census_keeps_the_claims_of_the_cases_it_did_not_run():
    """A report made with ``--scenarios all_hit --modes plain`` measures some cases of a
    config and leaves the budgets of the others as they were, claims included: the note that
    they are claims has to stay until every case of the config has been measured."""
    meta = {"iters": 8, "unmeasured": {"what": "claims", "lowered_to_zero": ["a", "b"]}}
    expected = [("a", "s", "plain"), ("a", "s", "prefetch"), ("b", "s", "plain")]
    partial = _report(("a", "s", "plain", 0.0), ("b", "s", "plain", 0.0))
    kept = sb.clear_measured_claims(meta, partial, expected=expected)
    assert kept["unmeasured"]["lowered_to_zero"] == ["a"], "a lacks its prefetch case"
    full = _report(
        ("a", "s", "plain", 0.0), ("a", "s", "prefetch", 0.0), ("b", "s", "plain", 0.0)
    )
    assert sb.clear_measured_claims(meta, full, expected=expected) == {"iters": 8}
    # a case that is expected and was skipped is not a measurement either
    skipped = _report(("a", "s", "plain", 0.0), ("b", "s", "plain", 0.0))
    skipped["results"].append(
        {"config": "a", "scenario": "s", "mode": "prefetch", "status": "skipped"}
    )
    assert sb.clear_measured_claims(meta, skipped, expected=expected)["unmeasured"][
        "lowered_to_zero"
    ] == ["a"]
    # by default the expected cases are the ones the probe measures
    config, scenario, mode = sb.case_keys()[0]
    assert len([k for k in sb.case_keys() if k[0] == config]) > 1
    real = {"unmeasured": {"what": "claims", "lowered_to_zero": [config]}}
    one = _report((config, scenario, mode, 0.0))
    assert sb.clear_measured_claims(real, one) == real
    everything = _report(*[(c, s, m, 0.0) for (c, s, m) in sb.case_keys()])
    assert sb.clear_measured_claims(real, everything) == {}


def test_a_report_with_errors_or_skips_is_handled():
    report = _report(("cache_storage", "all_hit", "plain", 1.0))
    report["results"].append(
        {
            "config": "c",
            "scenario": "s",
            "mode": "plain",
            "status": "error",
            "error": "boom",
        }
    )
    report["results"].append(
        {"config": "c", "scenario": "t", "mode": "plain", "status": "skipped"}
    )
    update = sb.budgets_from_report(report, {})
    assert update.errors == ["c/s/plain: boom"]
    assert list(update.budgets) == ["cache_storage"]


def test_validate_flags_a_missing_unknown_and_malformed_budget():
    meta, budgets = sb.load_budgets()
    broken = json.loads(json.dumps(budgets))
    del broken["cache_storage"]["all_hit"]["plain"]
    broken["cache_storage"]["all_hit"]["bogus_mode"] = 1
    broken["hbm_only"]["miss"]["plain"] = -1
    broken["hbm_only"]["miss"]["prefetch"] = "0"
    problems = "\n".join(sb.validate(dict(meta, iters=meta["iters"] + 1), broken))
    assert "cache_storage/all_hit/plain: no budget" in problems
    assert (
        "cache_storage/all_hit/bogus_mode: budget for a case the probe does not measure"
        in problems
    )
    assert "hbm_only/miss/plain: budget -1 is not a number" in problems
    assert "hbm_only/miss/prefetch: budget '0' is not a number" in problems
    assert "_meta['iters']" in problems


def test_growth_events_counts_the_doublings_of_the_window():
    def tiers(*caps):
        return {"storage": {"size": [0] * len(caps), "cap": list(caps)}}

    grew = {"tiers_before": tiers(1024, 1024), "tiers_after": tiers(8192, 2048)}
    assert sb.growth_events(grew) == 3 + 1
    # a capacity that did not change, a missing tier and a missing record count for nothing
    assert sb.growth_events({"tiers_before": tiers(64), "tiers_after": tiers(64)}) == 0
    assert sb.growth_events({"tiers_before": {}, "tiers_after": tiers(2048, 2048)}) == 0
    assert sb.growth_events({"status": "error"}) == 0
    # not a power of two: it still grew, so at least once
    assert (
        sb.growth_events({"tiers_before": tiers(1000), "tiers_after": tiers(1001)}) == 1
    )
    # every tier counts, a hybrid's host tier as well
    both = {
        "tiers_before": {"hbm": {"cap": [4096]}, "host": {"cap": [1024]}},
        "tiers_after": {"hbm": {"cap": [4096]}, "host": {"cap": [2048]}},
    }
    assert sb.growth_events(both) == 1


# --------------------------------------------------------------------------
# the census (GPU)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("config", sb.census_configs())
def test_sync_ratchet(config, census, budgets):
    _, table = budgets
    slack = float(os.environ.get("HSP_RATCHET_SLACK", "0"))
    strict = os.environ.get("HSP_RATCHET_STRICT") == "1"
    report = census.get(config)
    problems = []
    over = []
    for r in report["results"]:
        tag = "%s/%s/%s" % (r["config"], r["scenario"], r["mode"])
        if r["status"] == "skipped":
            continue
        if r["status"] != "ok":
            problems.append("%s: ERROR %s" % (tag, r.get("error")))
            continue
        budget = table.get(r["config"], {}).get(r["scenario"], {}).get(r["mode"])
        if budget is None:
            problems.append("%s: no budget in budgets.json (see sync_budgets.py)" % tag)
            continue
        measured = r["mean"][sb.METRIC]
        if measured > budget + slack + sb.EPS:
            over.append(r)
            problems.append(
                "%s: %.4g host syncs per iteration, above its budget %.4g"
                % (tag, measured, budget)
            )
        elif strict and measured < budget - sb.EPS:
            problems.append(
                "%s: %.4g host syncs per iteration, below its budget %.4g: lower it in "
                "budgets.json (sync_budgets.py --from-report)" % (tag, measured, budget)
            )
        # a step that has reached steady state allocates nothing; the allocator may
        # still grow once, and a growth event allocates by design (a new key map, new
        # pages mapped or registered): a few events per growth, not one per step
        events = r["mean"]["al"] * r["iters"]
        allowed = sb.ALLOC_EVENTS_MAX + sb.ALLOC_EVENTS_PER_GROWTH * sb.growth_events(r)
        if events > allowed + sb.EPS and not hsp.allocations_excused(
            r["config"], r["scenario"]
        ):
            problems.append(
                "%s: %.3g allocator events (cudaMalloc/cudaFree, mapped or registered pages) "
                "in the measured window, at most %d allowed: %d, and %d for each of the %d "
                "growths of the window (a per-step allocation of a new size?)"
                % (
                    tag,
                    events,
                    allowed,
                    sb.ALLOC_EVENTS_MAX,
                    sb.ALLOC_EVENTS_PER_GROWTH,
                    sb.growth_events(r),
                )
            )
    detail = ""
    if over:
        detail = "\n\n" + hsp.format_report(
            {"meta": report["meta"], "results": over}, sites=8, detail="plain"
        )
    assert not problems, "\n".join(problems) + detail


# --------------------------------------------------------------------------
# transient memory of a hit step (GPU), B8
# --------------------------------------------------------------------------
# What an all-hit step may add to the memory it started with, in units of one
# [N, value_dim] buffer (N keys of the batch, duplicates included).
# ABSOLUTE: the whole step. The miss path runs on inert rows (the host does not know that
# nothing missed) and holds two such buffers (the loaded values and the evicted ones);
# the forward adds its output and the unique embeddings, the backward its gradients: at most
# about five of them are live at once, so this leaves room for the index arrays and a
# little more, and catches an extra full-size buffer or worse. A first estimate, to be
# tightened with a measurement: the test prints a MEMORY line for each config (pytest -s), and
# about 1.25 times the largest of them is the bound.
ABSOLUTE_BOUND = 8.0
# CHUNKED: what the miss path of the whole batch adds over a miss path that runs in 64-row
# chunks (a chunk's worth of buffers instead of the batch's): the two buffers and a half.
CHUNKED_BOUND = 2.5


@pytest.mark.parametrize("config", ["cache_storage", "hbm_only", "host_only", "hybrid"])
def test_hit_step_memory_growth_is_bounded(config, gpu, sf, monkeypatch):
    """The padded lookup paths run on arrays of the batch size and hold a [N, value_dim]
    values buffer and one of evicted values (B8). An all-hit step peaks under ABSOLUTE_BOUND
    buffers over its start, and under CHUNKED_BOUND over the same step with the miss path in
    64-row chunks (``sync_free.CHUNK_ROWS``, which makes the chunk buffers negligible).
    """
    torch = gpu
    # wide rows, as in a real model: the batch-sized index arrays are then small
    # against the [N, value_dim] buffers the bound is about
    monkeypatch.setattr(hsp, "DIM", 128)
    cfg = hsp.CONFIGS[config]
    device = torch.device("cuda", torch.cuda.current_device())
    mod = hsp.build_module(cfg, hsp.plan_of(cfg, "all_hit"), device.index)
    batch = hsp.BatchMaker(cfg, device, sb.CENSUS_SEED).fresh_batch()
    step = hsp._make_step(mod, cfg, "plain", True)
    mod.train()

    def peak_growth(chunk_rows):
        monkeypatch.setattr(sf, "CHUNK_ROWS", chunk_rows)
        for _ in range(4):  # the first steps insert the keys and warm the allocator
            step(batch)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(batch)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    chunked = peak_growth(64)
    whole = peak_growth(None)
    top_tier = mod._cache if mod._cache is not None else mod._storage
    itemsize = torch.empty(0, dtype=top_tier.embedding_dtype()).element_size()
    unit = batch.indices.numel() * top_tier.max_value_dim() * itemsize
    print(
        "MEMORY %s: an all-hit step peaks at %.2f buffers of N x value_dim over its start, "
        "%.2f with the miss path in 64-row chunks (one buffer is %d bytes)"
        % (config, whole / unit, chunked / unit, unit)
    )
    assert whole <= ABSOLUTE_BOUND * unit, (
        "%s: an all-hit step peaks at %d bytes over its start, %.2f buffers of N x value_dim "
        "(%d bytes); the bound is %.1f"
        % (config, whole, whole / unit, unit, ABSOLUTE_BOUND)
    )
    assert whole - chunked <= CHUNKED_BOUND * unit, (
        "%s: an all-hit step peaks at %d bytes over its start in one chunk and %d in 64-row "
        "chunks; the difference is %.2f buffers of N x value_dim (%d bytes), the bound is %.1f"
        % (config, whole, chunked, (whole - chunked) / unit, unit, CHUNKED_BOUND)
    )
