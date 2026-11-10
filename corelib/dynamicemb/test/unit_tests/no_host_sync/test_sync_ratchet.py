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
adds over the legacy path (B8).

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
        if events > allowed + sb.EPS and not hsp.inherent_reason(
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
@pytest.mark.parametrize("config", ["cache_storage", "hbm_only", "host_only", "hybrid"])
def test_hit_step_memory_growth_is_bounded(config, gpu, bdf, monkeypatch):
    """The sync-free lookup paths run on arrays of the batch size and hold a
    [N, value_dim] values buffer and one of evicted values (B8). Their peak memory over
    the legacy path's, in an all-hit step, stays under 2.5 x N x value_dim x itemsize.
    The legacy path is the kill switch ``bdf._SYNC_FREE = False``; a tree without it has
    nothing to compare with."""
    torch = gpu
    if not hasattr(bdf, "_SYNC_FREE"):
        pytest.skip(
            "this tree has no kill switch (bdf._SYNC_FREE), hence no legacy path to compare with"
        )
    # wide rows, as in a real model: the batch-sized index arrays are then small
    # against the [N, value_dim] buffers the bound is about
    monkeypatch.setattr(hsp, "DIM", 128)
    cfg = hsp.CONFIGS[config]
    device = torch.device("cuda", torch.cuda.current_device())
    mod = hsp.build_module(cfg, hsp.plan_of(cfg, "all_hit"), device.index)
    batch = hsp.BatchMaker(cfg, device, sb.CENSUS_SEED).fresh_batch()
    step = hsp._make_step(mod, cfg, "plain", True)
    mod.train()

    def peak_growth(sync_free):
        monkeypatch.setattr(bdf, "_SYNC_FREE", sync_free)
        for _ in range(4):  # the first steps insert the keys and warm the allocator
            step(batch)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(batch)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    legacy = peak_growth(False)
    masked = peak_growth(True)
    top_tier = mod._cache if mod._cache is not None else mod._storage
    itemsize = torch.empty(0, dtype=top_tier.embedding_dtype()).element_size()
    bound = 2.5 * batch.indices.numel() * top_tier.max_value_dim() * itemsize
    assert masked - legacy <= bound, (
        "%s: an all-hit step peaks at %d bytes over its start on the sync-free path and %d on "
        "the legacy path; the bound on the difference is %d (2.5 x N x value_dim x itemsize)"
        % (config, masked, legacy, bound)
    )
