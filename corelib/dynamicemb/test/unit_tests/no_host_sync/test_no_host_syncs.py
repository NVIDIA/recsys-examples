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

"""Acceptance test of the sync removal: DynamicEmb's train and eval step reads zero.

host_sync_probe builds BatchedDynamicEmbeddingTablesV2 modules in every storage layout,
drives them through scenarios (all hit, miss, miss and evict, eval, growth) and counts the
host syncs of each measured step; a case listed in host_sync_probe.INHERENT (an external
PS, a table growth event) is excused. Everything else must read zero, the configs with
DEMB_DETERMINISM_MODE (``*_determ``) included.

The removal lands in steps, so the test applies to a config once budgets.json (see
test_sync_ratchet.py) has brought every case of it that is not inherent to zero. Until
then it skips, naming the budget that is left; the ratchet is what holds the line in the
meantime. HSP_ASSERT_ZERO=1 applies the test to every config now: its failure message
ranks the call sites that still sync.

    pytest -q test_no_host_syncs.py
    HSP_ASSERT_ZERO=1 HSP_CONFIGS=cache_storage,hbm_only pytest -q test_no_host_syncs.py

test_probe_instruments validates the probe first (known syncs of every kind are seen,
the negative controls read zero), so a zero is never an artefact of a dead profiler.
Needs a GPU and the built extension; sync_budgets.py lists the knobs of the census
(HSP_CONFIGS, HSP_SCENARIOS, HSP_MODES, HSP_ITERS, HSP_WARMUP).
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import host_sync_probe as hsp  # noqa: E402
import sync_budgets as sb  # noqa: E402

GROWTH_CONFIGS = [name for name in sb.census_configs() if hsp.CONFIGS[name].growth]
VMM_GRANULE = 2 << 20
HOLD_SECONDS = 0.3


def test_probe_instruments(instruments_ok):
    """Known syncs are seen, negative controls read zero (see host_sync_probe.selftest)."""
    assert instruments_ok


def largest_open_budget(config, table):
    """The largest budget of a case of `config` that is not inherent, or None when the
    config has no such case."""
    open_budgets = [
        value
        for scenario, modes in table.get(config, {}).items()
        if not hsp.inherent_reason(config, scenario)
        for value in modes.values()
    ]
    return max(open_budgets) if open_budgets else None


@pytest.mark.parametrize("config", sb.census_configs())
def test_no_host_syncs(config, budgets, request):
    _, table = budgets
    if os.environ.get("HSP_ASSERT_ZERO") != "1":
        left = largest_open_budget(config, table)
        if left is None:
            pytest.skip(
                "every case of %s is inherent (see host_sync_probe.INHERENT)" % config
            )
        if left > sb.EPS:
            pytest.skip(
                "budgets.json still lets %s sync up to %.4g times per iteration outside its "
                "inherent cases; the acceptance test applies at zero (HSP_ASSERT_ZERO=1 now)"
                % (config, left)
            )
    report = request.getfixturevalue("census").get(config)
    violations = hsp.check_zero(report)
    assert not violations, (
        "\n".join(violations)
        + "\n\n"
        + hsp.format_report(report, sites=8, detail="plain")
    )


def _value_buffer_bytes(module) -> int:
    """The largest value buffer behind a module, in mapped bytes."""
    storage = module._storage
    states = [storage._state] if hasattr(storage, "_state") else list(storage.tables)
    return max(table.allocated_bytes() for state in states for table in state.tables)


@pytest.mark.parametrize("mode", hsp.MODES)
@pytest.mark.parametrize("config", GROWTH_CONFIGS)
def test_a_growing_step_returns_while_a_running_kernel_has_not_finished(
    config, mode, gpu
):
    """A step that grows a table never waits for the device, whatever the profiler calls it.

    The census above classifies what the profiler reports; a driver call that waits for the
    kernels in flight (``cuMemSetAccess``, the registration of host memory, an allocation) is
    not something a name can tell. This looks at the time the host comes back instead: a long
    sleep is queued on the stream the step runs on, a growing step is issued behind it, and
    when the host returns the sleep has not finished. If any call of the step had waited for
    the device, it would have. The tables of every growth config grow in the window
    (a hybrid's host tier, a cache's storage, host and device value buffers, narrow and wide
    rows: the wide ones map new pages, the narrow ones stay in the first granule).
    """
    torch = gpu
    cfg = hsp.CONFIGS[config]
    device = torch.device("cuda", torch.cuda.current_device())
    plan = hsp.plan_of(cfg, "growth")
    hold = int(
        HOLD_SECONDS * torch.cuda.get_device_properties(device).clock_rate * 1000
    )

    def make_step(module):
        if mode == "plain":
            stream = torch.cuda.current_stream(device)

            def step(b):
                out = module(b.indices, b.offsets)
                out.backward(torch.ones_like(out))

            return step, stream
        prefetch, forward = torch.cuda.Stream(device), torch.cuda.Stream(device)

        def step(b):
            prefetch.wait_stream(forward)
            with torch.cuda.stream(prefetch):
                module.prefetch(b.indices, b.offsets, forward)
            forward.wait_stream(prefetch)
            with torch.cuda.stream(forward):
                out = module(b.indices, b.offsets)
                out.backward(torch.ones_like(out))

        return step, forward

    def build():
        torch.manual_seed(sb.CENSUS_SEED)
        module = hsp.build_module(cfg, plan, device.index)
        module.train()
        maker = hsp.BatchMaker(cfg, device, sb.CENSUS_SEED)
        # all the batches before anything is queued: a copy from pageable memory waits for
        # the stream it is issued on
        warm = [maker.fresh_batch(scale=0.1) for _ in range(3)]
        measured = [maker.fresh_batch() for _ in range(sb.CENSUS_ITERS)]
        return module, warm, measured

    # the same steps once, unobserved: every kernel is loaded (with lazy loading the first
    # launch of a kernel can wait for the kernels already running) and the allocator is warm
    module, warm, measured = build()
    step, _ = make_step(module)
    for b in warm + measured:
        step(b)
    torch.cuda.synchronize()
    del module, step

    module, warm, measured = build()
    step, stream = make_step(module)
    for b in warm:
        step(b)
    torch.cuda.synchronize()
    before = hsp.tier_sizes(module)
    returned_early = []
    for b in measured:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(hold)
            held = torch.cuda.Event()
            held.record(stream)
        step(b)
        returned_early.append(not held.query())
        torch.cuda.synchronize()
    after = hsp.tier_sizes(module)

    grew, note = hsp.regime_check(cfg, "growth", before, after, len(measured))
    assert grew, "no table grew in the window: " + note
    if cfg.dim > hsp.DIM:
        assert _value_buffer_bytes(module) > VMM_GRANULE, (
            "the value buffers stayed inside the first 2 MiB granule: the wide rows do not "
            "reach the calls that map new pages"
        )
    assert all(returned_early), (
        "the host waited for the device in step(s) %s of %d: the step did not return before "
        "a sleep of %.1f s queued ahead of it had finished"
        % (
            [i for i, ok in enumerate(returned_early) if not ok],
            len(measured),
            HOLD_SECONDS,
        )
    )
