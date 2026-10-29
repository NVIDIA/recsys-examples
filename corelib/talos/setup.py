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

"""Wheel-build hook: download the bundled veloq binary + skill tarball.

Project metadata still lives in pyproject.toml. This file only adds a
custom `build_py` (and `bdist_wheel`) that, before setuptools sweeps
package contents into the wheel, fetches:

  1. veloq native binary  → skills/bin/veloq
  2. veloq-skills.tar.gz  → skills/nsys_profile_analysis/{SKILL.md, references/}
                              and skills/ncu_profile_analysis/{SKILL.md, references/}

Both come from veloq's public GitHub Releases (github.com/lucifer1004/veloq)
— same source veloq's own install.sh uses, so the binary and the skills are
guaranteed to match.

Configuration (from pyproject.toml's [tool.talos]):
  veloq-version    GitHub release tag to pin (required).
  veloq-base-url   Optional release-download base URL override.

Target-platform selection:
  VELOQ_PLATFORM env var picks one of {x86_64-linux, aarch64-linux,
  x86_64-macos, aarch64-macos}. Default = current build host.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:
    import tomli as tomllib  # build-only dep, declared in [build-system].requires

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

HERE = Path(__file__).resolve().parent
SKILLS_ROOT = HERE / "skills"
BIN_DIR = SKILLS_ROOT / "bin"
BIN_DEST = BIN_DIR / "veloq"

DEFAULT_BASE_URL = "https://github.com/lucifer1004/veloq/releases/download"

# (target key → veloq asset filename, wheel platform tag)
PLATFORMS = {
    "x86_64-linux": ("veloq-x86_64-linux", "manylinux2014_x86_64"),
    "aarch64-linux": ("veloq-aarch64-linux", "manylinux2014_aarch64"),
    "x86_64-macos": ("veloq-x86_64-macos", "macosx_11_0_x86_64"),
    "aarch64-macos": ("veloq-aarch64-macos", "macosx_11_0_arm64"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Config + platform resolution
# ─────────────────────────────────────────────────────────────────────────────


def _read_config() -> dict:
    with open(HERE / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    section = cfg.get("tool", {}).get("talos", {})
    if "veloq-version" not in section:
        raise SystemExit(
            "pyproject.toml is missing [tool.talos].veloq-version — "
            'pin a release tag (e.g. "v0.2.1") so the build can fetch a '
            "matching binary and skill tarball."
        )
    return section


def _detect_host_platform() -> str:
    sysname = sys.platform
    mach = platform.machine().lower()
    if sysname.startswith("linux"):
        if mach == "x86_64":
            return "x86_64-linux"
        if mach in ("aarch64", "arm64"):
            return "aarch64-linux"
    elif sysname == "darwin":
        if mach == "x86_64":
            return "x86_64-macos"
        if mach in ("aarch64", "arm64"):
            return "aarch64-macos"
    raise SystemExit(
        f"unsupported build host {sysname}/{mach}; set VELOQ_PLATFORM explicitly "
        f"to one of {sorted(PLATFORMS)}"
    )


def _resolve_platform() -> str:
    env = os.environ.get("VELOQ_PLATFORM")
    if env:
        if env not in PLATFORMS:
            raise SystemExit(
                f"VELOQ_PLATFORM={env!r} unrecognized; choose from {sorted(PLATFORMS)}"
            )
        return env
    return _detect_host_platform()


# ─────────────────────────────────────────────────────────────────────────────
# Fetch helpers
# ─────────────────────────────────────────────────────────────────────────────

# A single, ordinary path component: no "..", no "/", no absolute paths, no
# empty names. Archive members are untrusted input.
_SAFE_COMPONENT = re.compile(r"(?!\.\.?$)[^/\\]+")


def _download(url: str, dest: Path, asset: str, digests: dict) -> None:
    """Fetch ``url`` to ``dest`` and refuse to keep it unless it matches the
    sha256 pinned for ``asset`` in pyproject.toml's [tool.talos.sha256]."""
    expected = digests.get(asset)
    if not expected:
        raise SystemExit(
            f"[veloq-build] no sha256 pinned for '{asset}'. Add it under "
            f"[tool.talos.sha256] in pyproject.toml before building."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"[veloq-build] GET {url}\n")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            digest.update(chunk)
            out.write(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"[veloq-build] checksum mismatch for {asset}\n"
            f"  expected {expected}\n  actual   {actual}\n"
            f"Refusing to package it. Either the release was re-cut (update the "
            f"pin) or the download was tampered with."
        )
    sys.stderr.write(f"[veloq-build] sha256 OK {asset}\n")


def _fetch_binary(base_url: str, version: str, asset: str, digests: dict) -> None:
    url = f"{base_url}/{version}/{asset}"
    _download(url, BIN_DEST, asset, digests)
    BIN_DEST.chmod(0o755)


def _fetch_skills(base_url: str, version: str, digests: dict) -> None:
    """Pull veloq-skills.tar.gz and explode it under skills/<pkg_name>/.

    The tarball entries look like:
        .claude/skills/<skill-slug>/SKILL.md
        .claude/skills/<skill-slug>/references/*.md

    We strip the leading `.claude/skills/` prefix and remap the skill slug
    to its Python-package form (`-` → `_`) so it slots into the talos
    namespace as a real subpackage. SKILL.md / references/ get written
    *inside* the existing `__init__.py`-bearing source-tree directory.
    """
    url = f"{base_url}/{version}/veloq-skills.tar.gz"
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _download(url, tmp_path, "veloq-skills.tar.gz", digests)
        skills_root = SKILLS_ROOT.resolve()
        with tarfile.open(tmp_path, "r:gz") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 3 or parts[0] != ".claude" or parts[1] != "skills":
                    continue
                slug = parts[2]  # e.g. "nsys-profile-analysis"
                if not _SAFE_COMPONENT.fullmatch(slug):
                    sys.stderr.write(f"[veloq-build] skip unsafe slug: {member.name}\n")
                    continue
                pkg = slug.replace("-", "_")  # e.g. "nsys_profile_analysis"
                rest = parts[3:]
                if not rest or not all(_SAFE_COMPONENT.fullmatch(c) for c in rest):
                    if rest:
                        sys.stderr.write(
                            f"[veloq-build] skip unsafe path: {member.name}\n"
                        )
                    continue
                target = SKILLS_ROOT / pkg / Path(*rest)
                # Belt and braces: the component check above already rejects
                # "..", but confirm the resolved destination stays inside the
                # package tree before anything is created or opened.
                try:
                    resolved = target.resolve()
                    resolved.relative_to(skills_root / pkg)
                except (ValueError, OSError):
                    sys.stderr.write(
                        f"[veloq-build] skip escaping member: {member.name}\n"
                    )
                    continue
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    continue
                with open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    finally:
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom commands
# ─────────────────────────────────────────────────────────────────────────────


class BuildWithVeloq(build_py):
    """Fetch veloq artifacts before the standard build_py pass."""

    def run(self):
        cfg = _read_config()
        version = cfg["veloq-version"]
        base_url = cfg.get("veloq-base-url", DEFAULT_BASE_URL)
        plat_key = _resolve_platform()
        asset = PLATFORMS[plat_key][0]

        sys.stderr.write(
            f"[veloq-build] platform={plat_key}  version={version}  asset={asset}\n"
        )
        digests = cfg.get("sha256", {})
        _fetch_binary(base_url, version, asset, digests)
        _fetch_skills(base_url, version, digests)
        super().run()


class BdistWheelTagged(bdist_wheel):
    """Mark the wheel as platform-specific and pin the platform tag."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False  # not a pure-Python wheel

    def get_tag(self):
        wheel_tag = PLATFORMS[_resolve_platform()][1]
        return "py3", "none", wheel_tag


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

setup(cmdclass={"build_py": BuildWithVeloq, "bdist_wheel": BdistWheelTagged})
