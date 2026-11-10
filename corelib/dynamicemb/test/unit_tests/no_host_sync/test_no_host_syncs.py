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
