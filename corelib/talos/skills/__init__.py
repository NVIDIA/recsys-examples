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

"""talos: agent skills for PyTorch performance work.

Talos' own entry points are modules, so nothing depends on the pip scripts
dir being on PATH:

    python -m talos.install_skills <agent> [<agent> ...]
    python -m talos.compile_observer

`install_skills` symlinks every populated skill into the current project's
`.<agent>/skills/` directory. Supported agents are listed in `AGENTS` below
— currently `claude` and `codex`.

The one exception is `veloq`: it is a native binary bundled in the wheel and
cannot be run with `python -m`, so it stays a console script (see
`_veloq_main` below and [project.scripts] in pyproject.toml).

Adding a new agent = add its slug to `AGENTS`. The agent decides where to
look for skills; we just drop them under `.<agent>/skills/` in the project.

Adding a new skill = add one entry to `SKILLS` below (and one subpackage).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

__version__ = "0.1.0"

_PKG = __name__

AGENTS = ("claude", "codex")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_PREFIX = f"[{_PKG}]"


def log(msg: str) -> None:
    print(f"{_LOG_PREFIX} {msg}", file=sys.stderr)


def log_warn(msg: str) -> None:
    print(f"{_LOG_PREFIX} WARN {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Skill registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skill:
    slug: str  # directory name under .claude/skills/<slug>
    subpackage: str  # name of the subpackage under talos (e.g. "torch_optimize")

    def root(self) -> Path:
        # Subpackage dir on disk. Resolve from this file's path rather
        # than importing the subpackage — `import torch_optimize` pulls
        # in torch, and the CLI is supposed to be stdlib-only.
        return Path(__file__).resolve().parent / self.subpackage

    def is_populated(self) -> bool:
        return (self.root() / "SKILL.md").is_file()


SKILLS: list[Skill] = [
    Skill(slug="torch-optimize", subpackage="torch_optimize"),
    Skill(slug="torch-perf-analysis", subpackage="torch_perf_analysis"),
    # Bundled with veloq — SKILL.md + references/ are downloaded by setup.py
    # at wheel-build time. The slug uses dashes so the on-disk Claude skill
    # directory matches veloq's own convention; the subpackage uses
    # underscores so it's importable as a Python package.
    Skill(slug="nsys-profile-analysis", subpackage="nsys_profile_analysis"),
    Skill(slug="ncu-profile-analysis", subpackage="ncu_profile_analysis"),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _veloq_main() -> None:
    """Console-script entry: exec into the bundled veloq binary.

    Registered as `veloq` in pyproject.toml's [project.scripts]. After
    `pip install talos`, the wheel's bin/veloq sits next to this
    file; we replace the current Python process with it via os.execv so
    signals and exit codes flow through transparently.
    """
    binary = Path(__file__).resolve().parent / "bin" / "veloq"
    if not binary.is_file() and sys.platform == "win32":
        binary = binary.with_suffix(".exe")
    if not binary.is_file():
        sys.stderr.write(
            f"{_LOG_PREFIX} ERROR bundled veloq binary not found at {binary}\n"
            f"{_LOG_PREFIX} reinstall with: pip install --force-reinstall talos\n"
        )
        sys.exit(1)
    os.execv(str(binary), [str(binary), *sys.argv[1:]])


def _build_observer() -> int:
    """Pre-compile the C++ profiling observer in this environment.

    Runs as the tail of `python -m talos.install_skills` so a broken
    toolchain surfaces at
    install time, not at training time (the build is cached; training-time
    loads then hit the cache). No torch in this environment is fine — the
    build is skipped with a warning and first training use JIT-compiles
    instead. A failed build with torch present is an error (non-zero)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        log_warn(
            "torch not importable here; skipping the observer pre-build "
            "(it will JIT-compile on first profiled run)."
        )
        return 0
    log("pre-compiling the C++ profiling observer (cached after first build) ...")
    from talos.torch_perf_analysis.python.profiler import compile_observer_main

    return compile_observer_main()


def _install_skills(agents: list[str]) -> int:
    conflicts = 0
    for agent in agents:
        base = Path.cwd() / f".{agent}" / "skills"
        for skill in SKILLS:
            if not skill.is_populated():
                print(
                    f"skip {skill.slug}: SKILL.md not found at {skill.root() / 'SKILL.md'}",
                    file=sys.stderr,
                )
                continue
            target = base / skill.slug
            if _link(skill.root(), target):
                print(f"  [{agent}] {skill.slug} -> {target}")
            else:
                conflicts += 1
    _print_install_hint()
    if conflicts:
        print(
            f"\n{conflicts} skill(s) were not linked because a file or "
            f"directory of the same name already exists.",
            file=sys.stderr,
        )
    return 1 if conflicts else 0


def _print_install_hint() -> None:
    """Show where the package lives and, if the bundled `veloq` binary is not
    on PATH, print a ready-to-paste `export`.

    The skills invoke `veloq` bare (hundreds of call sites), and it is a
    native binary, so unlike this package's own entry points it cannot fall
    back to `python -m`. It is the only thing that still needs PATH."""
    pkg_path = Path(__file__).resolve().parent
    print()
    print(f"package installed at: {pkg_path}")
    print("run it with:          python -m talos.install_skills <agent>")

    found = shutil.which("veloq")
    if found:
        print(f"veloq available at:   {found}")
        return

    import sysconfig

    candidates = []
    for scheme in ("posix_user", "posix_prefix"):
        try:
            candidates.append(Path(sysconfig.get_paths(scheme)["scripts"]))
        except KeyError:
            pass
    scripts_dir = next(
        (p for p in candidates if (p / "veloq").exists()),
        candidates[0] if candidates else None,
    )
    if scripts_dir is None:
        return
    print(f"veloq script dir:     {scripts_dir}  (not on PATH)")
    print("the skills call `veloq` bare; to enable it, run:")
    print(f'  export PATH="{scripts_dir}:$PATH"')


def _link(src: Path, target: Path) -> bool:
    """Point ``target`` at ``src``. Returns False and leaves the path alone if
    something that is not one of our symlinks is already sitting there — a
    project's own skill of the same name is the user's, not ours to delete."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if Path(os.readlink(target)) == src:
            return True
        target.unlink()
    elif target.exists():
        kind = "directory" if target.is_dir() else "file"
        print(f"  refusing to replace existing {kind}: {target}", file=sys.stderr)
        print("  (move or delete it yourself, then re-run)", file=sys.stderr)
        return False
    os.symlink(src, target)
    return True
