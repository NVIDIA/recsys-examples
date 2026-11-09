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

"""budgets.json: how many host syncs per iteration each probe case may make.

test_sync_ratchet.py fails a case that syncs more than its budget. A commit that
removes syncs lowers the budgets it earned in the same diff, so a reviewer sees the
claim and no later commit can quietly give it back. The file starts as the census of
the tree before the sync removal.

    {"_meta": {...},
     "<config>": {"<scenario>": {"<mode>": <number>, ...}, ...}, ...}

The number is the mean, over the measured iterations of a case, of the host syncs
the probe counts (host_sync_probe.KIND_DOC: cudaStreamSynchronize, cudaDeviceSynchronize,
cudaEventSynchronize, synchronous cudaMemcpy and a D2H copy into pageable memory that no
stream sync follows), on every thread. cudaMalloc/cudaFree are not part of it; the
ratchet bounds them separately (ALLOC_EVENTS_MAX per window). The census runs with the
iterations, warm-up and seed recorded in "_meta": the means of the cases that grow a
table depend on them.

To lower the budgets after a change, on a GPU:

    python host_sync_probe.py --json census.json
    python sync_budgets.py --from-report census.json            # what would change
    python sync_budgets.py --from-report census.json --write    # lower budgets.json

A budget is never raised unless --allow-raise says so, and a report with errors
writes nothing. Needs only the standard library.
"""
import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import host_sync_probe as hsp  # noqa: E402

BUDGETS_PATH = os.path.join(HERE, "budgets.json")

# What a budget bounds: the mean of host_sync_probe's "blocking" total per iteration.
METRIC = "blocking"
# The census the budgets were taken with (the probe's own defaults).
CENSUS_ITERS = 8
CENSUS_WARMUP = 3
CENSUS_SEED = 1234
# cudaMalloc / cudaFree / cudaHostAlloc events allowed in one measured window of a
# case: the allocator may still grow once after the warm-up; a step that allocates
# every time shows up as at least `iters` of them.
ALLOC_EVENTS_MAX = 1
# Budgets are numbers written with at most 6 decimals; this is the comparison slack.
EPS = 1e-6

Case = Tuple[str, str, str]  # (config, scenario, mode)


# --------------------------------------------------------------------------
# environment knobs of the census tests (every one is optional)
#   HSP_CONFIGS=a,b  HSP_SCENARIOS=x,y  HSP_MODES=plain  HSP_ITERS=8  HSP_WARMUP=3
# --------------------------------------------------------------------------
def csv_env(name: str) -> Optional[List[str]]:
    """The comma-separated list in $name, or None when it is unset or empty."""
    return [x for x in os.environ.get(name, "").split(",") if x] or None


def census_configs() -> List[str]:
    """The probe configs the census tests run: all of them, or $HSP_CONFIGS."""
    return csv_env("HSP_CONFIGS") or list(hsp.CONFIGS)


def census_scenarios() -> Optional[List[str]]:
    return csv_env("HSP_SCENARIOS")


def census_modes() -> Tuple[str, ...]:
    return tuple(csv_env("HSP_MODES") or hsp.MODES)


def census_iters() -> int:
    return int(os.environ.get("HSP_ITERS", CENSUS_ITERS))


def census_warmup() -> int:
    return int(os.environ.get("HSP_WARMUP", CENSUS_WARMUP))


def case_keys() -> List[Case]:
    """Every (config, scenario, mode) the probe measures: budgets.json covers
    exactly these."""
    out: List[Case] = []
    for cfg in hsp.CONFIGS.values():
        for scenario in hsp.scenarios_of(cfg):
            for mode in hsp.MODES:
                if scenario in hsp.PLAIN_ONLY_SCENARIOS and mode != "plain":
                    continue  # prefetch is a no-op in eval; queue_ahead is plain only
                out.append((cfg.name, scenario, mode))
    return out


def load_budgets(path: str = BUDGETS_PATH) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(meta, budgets): budgets[config][scenario][mode] is the budget of a case."""
    with open(path) as f:
        data = json.load(f)
    meta = data.pop("_meta", {})
    return meta, data


def default_meta() -> Dict[str, Any]:
    return {
        "what": (
            "mean host syncs per iteration each probe case may make; "
            "see sync_budgets.py"
        ),
        "metric": METRIC,
        "iters": CENSUS_ITERS,
        "warmup": CENSUS_WARMUP,
        "seed": CENSUS_SEED,
        "alloc_events_max": ALLOC_EVENTS_MAX,
    }


def _number(x: float) -> Any:
    """A budget as written to the file: an int when it is one, else 6 decimals."""
    r = round(float(x), 6)
    return int(r) if abs(r - round(r)) < 1e-9 else r


def validate(meta: Dict[str, Any], budgets: Dict[str, Any]) -> List[str]:
    """Problems of a budgets file: wrong census settings, missing or unknown
    cases, numbers that are not numbers."""
    problems: List[str] = []
    for key, want in (
        ("metric", METRIC),
        ("iters", CENSUS_ITERS),
        ("warmup", CENSUS_WARMUP),
        ("seed", CENSUS_SEED),
        ("alloc_events_max", ALLOC_EVENTS_MAX),
    ):
        if meta.get(key) != want:
            problems.append(
                "_meta[%r] is %r, the census code uses %r" % (key, meta.get(key), want)
            )
    have: Dict[Case, Any] = {}
    for config, scenarios in budgets.items():
        if not isinstance(scenarios, dict):
            problems.append("%s: not a mapping of scenarios" % config)
            continue
        for scenario, modes in scenarios.items():
            if not isinstance(modes, dict):
                problems.append("%s/%s: not a mapping of modes" % (config, scenario))
                continue
            for mode, value in modes.items():
                have[(config, scenario, mode)] = value
    want_cases = set(case_keys())
    for case in sorted(want_cases - set(have)):
        problems.append("%s: no budget (the probe measures this case)" % "/".join(case))
    for case in sorted(set(have) - want_cases):
        problems.append(
            "%s: budget for a case the probe does not measure" % "/".join(case)
        )
    for case, value in sorted(have.items()):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not value >= 0
        ):
            problems.append(
                "%s: budget %r is not a number >= 0" % ("/".join(case), value)
            )
    return problems


class Update(NamedTuple):
    budgets: Dict[str, Any]
    changes: List[Tuple[Case, Optional[float], float]]  # (case, old, new)
    refused: List[str]  # raises that --allow-raise did not permit
    errors: List[str]  # cases the report could not measure


def budgets_from_report(
    report: Dict[str, Any],
    previous: Optional[Dict[str, Any]] = None,
    allow_raise: bool = False,
) -> Update:
    """The budgets after taking the measured means of a probe report
    (host_sync_probe --json) into ``previous``: a case's budget becomes what was
    measured, but is not raised unless allow_raise. Cases the report does not cover
    keep their budget."""
    budgets = copy.deepcopy(previous or {})
    changes: List[Tuple[Case, Optional[float], float]] = []
    refused: List[str] = []
    errors: List[str] = []
    for r in report["results"]:
        case = (r["config"], r["scenario"], r["mode"])
        tag = "/".join(case)
        if r["status"] == "skipped":
            continue
        if r["status"] != "ok":
            errors.append("%s: %s" % (tag, r.get("error")))
            continue
        value = _number(r["mean"][METRIC])
        old = budgets.get(case[0], {}).get(case[1], {}).get(case[2])
        if old is not None and value > old + EPS and not allow_raise:
            refused.append("%s: measured %s, above its budget %s" % (tag, value, old))
            continue
        if old is None or abs(value - old) > EPS:
            changes.append((case, old, value))
        budgets.setdefault(case[0], {}).setdefault(case[1], {})[case[2]] = value
    return Update(budgets, changes, refused, errors)


def format_budgets(meta: Dict[str, Any], budgets: Dict[str, Any]) -> str:
    """The text of budgets.json: one line per scenario, configs and scenarios in the
    probe's order, so that a diff shows exactly the numbers that changed."""
    config_order = [c for c in hsp.CONFIGS if c in budgets] + sorted(
        c for c in budgets if c not in hsp.CONFIGS
    )
    lines = ["{", ' "_meta": %s,' % json.dumps(meta, indent=1).replace("\n", "\n ")]
    for ci, config in enumerate(config_order):
        scenarios = budgets[config]
        scenario_order = [s for s in hsp.SCENARIO_DOC if s in scenarios] + sorted(
            s for s in scenarios if s not in hsp.SCENARIO_DOC
        )
        lines.append(' "%s": {' % config)
        for si, scenario in enumerate(scenario_order):
            modes = scenarios[scenario]
            mode_order = [m for m in hsp.MODES if m in modes] + sorted(
                m for m in modes if m not in hsp.MODES
            )
            cells = ", ".join(
                '"%s": %s' % (m, json.dumps(modes[m])) for m in mode_order
            )
            lines.append(
                '  "%s": {%s}%s'
                % (scenario, cells, "," if si + 1 < len(scenario_order) else "")
            )
        lines.append(" }%s" % ("," if ci + 1 < len(config_order) else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Lower budgets.json to what a probe report measured.",
    )
    ap.add_argument(
        "--from-report",
        required=True,
        metavar="CENSUS.json",
        help="host_sync_probe --json output",
    )
    ap.add_argument(
        "--budgets",
        default=BUDGETS_PATH,
        help="the budgets file (default: the one next to this script)",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="write the file (default: only print what would change)",
    )
    ap.add_argument(
        "--allow-raise",
        action="store_true",
        help="also take measurements above the current budget",
    )
    args = ap.parse_args(argv)

    with open(args.from_report) as f:
        report = json.load(f)
    if os.path.exists(args.budgets):
        meta, previous = load_budgets(args.budgets)
    else:
        meta, previous = {}, {}
    update = budgets_from_report(report, previous, args.allow_raise)

    for case, old, new in update.changes:
        print(
            "%-46s %8s -> %s" % ("/".join(case), "(new)" if old is None else old, new)
        )
    print("%d budget(s) changed" % len(update.changes))
    for line in update.refused:
        print("NOT RAISED  " + line)
    for line in update.errors:
        print("NOT MEASURED  " + line)
    if update.refused or update.errors:
        print(
            "nothing written: fix the regression (or pass --allow-raise) and measure again"
        )
        return 1
    if args.write:
        new_meta = dict(
            default_meta(), **{k: v for k, v in meta.items() if k not in default_meta()}
        )
        new_meta["measured"] = {
            k: report.get("meta", {}).get(k)
            for k in ("label", "time", "device", "torch", "cuda", "python")
            if report.get("meta", {}).get(k)
        }
        problems = validate(new_meta, update.budgets)
        if problems:
            print(
                "the result would not be a valid budgets file (is the report partial?):"
            )
            for p in problems[:20]:
                print("  " + p)
            return 1
        with open(args.budgets, "w") as f:
            f.write(format_budgets(new_meta, update.budgets))
        print("wrote %s" % args.budgets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
