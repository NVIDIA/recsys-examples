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

"""Workspace management for the torch-optimize skill.

    python3 -m talos.torch_optimize.python.workspace init [--root DIR]

Creates the run-local ``.torch_optimize/`` workspace that the Main Loop
Agent, the Optimizer, and the Judge all read and write during a run; its
layout is described in ``workspace_layout.md``. Safe to re-run: if a
``.torch_optimize/`` already exists it is renamed to a timestamped backup
first — a fresh run never inherits or silently overwrites a prior one's
state, and nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE_DIRNAME = ".torch_optimize"

# `init` creates the workspace root only. Everything inside it
# (loop_<n>/, round_<k>/, ...) is created by the agents as they go; the
# layout is documented in workspace_layout.md.


def _backup_path(root: Path) -> Path:
    """A ``.torch_optimize_backup_<timestamp>`` path under root, not yet
    taken (suffixed with a counter in the unlikely event of a collision)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{WORKSPACE_DIRNAME}_backup_{ts}"
    n = 1
    while candidate.exists():
        n += 1
        candidate = root / f"{WORKSPACE_DIRNAME}_backup_{ts}_{n}"
    return candidate


def init_workspace(root: Path) -> Path:
    """Create a fresh ``<root>/.torch_optimize/`` (the root only; the
    agents create everything inside it). If one already exists, it is
    renamed to a timestamped backup first — never deleted, never merged
    into."""
    ws = root / WORKSPACE_DIRNAME
    if ws.exists():
        backup = _backup_path(root)
        ws.rename(backup)
        sys.stderr.write(f"[torch-optimize] backed up existing {ws} -> {backup}\n")

    ws.mkdir(parents=True)
    sys.stderr.write(f"[torch-optimize] created {ws}\n")
    return ws


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="talos.torch_optimize.python.workspace",
        description="Manage the .torch_optimize/ runtime workspace.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser(
        "init",
        help="create a fresh .torch_optimize/ (backs up any existing one first)",
    )
    p_init.add_argument(
        "--root",
        default=".",
        help="directory to create .torch_optimize/ under (default: current directory)",
    )

    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"[torch-optimize] --root not a directory: {root}")
    print(init_workspace(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
