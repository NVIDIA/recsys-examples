"""Runs the script-style CPU tests ported from the original code base (tests/scripts/*.py), each in a fresh
interpreter with PYTHONPATH=<repo>/src, and requires exit code 0.  These scripts install monkey patches and fake
framework modules at module level, so they are not imported into the pytest process."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
SCRIPTS = os.path.join(HERE, "scripts")
NAMES = sorted(f for f in os.listdir(SCRIPTS) if f.endswith(".py"))


@pytest.mark.parametrize("name", NAMES)
def test_script(name, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASRT_")}
    env["PYTHONPATH"] = os.pathsep.join([SRC, SCRIPTS])
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, name)], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=900)
    if r.returncode == 77:   # the script needs an optional external source (e.g. slime's fully_async_rollout.py)
        pytest.skip(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "skipped by the script")
    assert r.returncode == 0, f"{name} exit {r.returncode}\n--- stdout\n{r.stdout[-3000:]}\n--- stderr\n{r.stderr[-3000:]}"
