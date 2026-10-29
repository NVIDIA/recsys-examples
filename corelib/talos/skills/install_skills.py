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

"""Install the skills shipped in this package into the current project.

    python -m talos.install_skills <agent> [<agent> ...] [--skip-observer-build]

Symlinks every populated skill into `./.<agent>/skills/` in the current
working directory. Run from the root of the project you want the skills in.

This is a module entry point rather than a console script on purpose: it
works straight after `pip install` regardless of whether the pip scripts
directory happens to be on PATH.
"""

from __future__ import annotations

import argparse
import sys

from talos import AGENTS, _build_observer, _install_skills

_ENV_HELP = """\
Environment variables:
  TORCH_PERF_ANALYSIS_ENABLE      Master switch for the profiler hooks in a
                                  training script (enable/start/stop/step).
                                  Unset (default) = every call is a no-op.
Ablation switches for the C++ observer (TALOS_CPP_*) are documented in
torch_perf_analysis/python/profiler.py.
"""


def _agent_token(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in parts if p not in AGENTS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"invalid agent(s): {', '.join(bad)} — choose from {', '.join(AGENTS)}"
        )
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m talos.install_skills",
        description="Symlink every ready skill into ./.<agent>/skills/ in the current directory.",
        epilog=_ENV_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-observer-build",
        action="store_true",
        help="don't pre-compile the C++ profiling observer after linking",
    )
    parser.add_argument(
        "agents",
        nargs="+",
        type=_agent_token,
        metavar="{" + ",".join(AGENTS) + "}",
        help=(
            f"target agent(s) — one or more of {', '.join(AGENTS)}. "
            f"Pass as separate args ({' '.join(AGENTS)}) "
            f"or comma-separated ({','.join(AGENTS)})."
        ),
    )
    args = parser.parse_args(argv)

    agents = [a for tok in args.agents for a in tok]
    rc = _install_skills(agents)
    if rc == 0 and not args.skip_observer_build:
        rc = _build_observer()
    return rc


if __name__ == "__main__":
    sys.exit(main())
