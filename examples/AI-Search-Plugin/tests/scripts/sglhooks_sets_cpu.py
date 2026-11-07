"""which SGLang engine hooks each asrt verl mode turns on when ASRT_SGL_HOOKS is unset
(= the user dropped 'sglhooks'): only what the mode needs, never the implicit
nosync/fastload defaults.  With ASRT_SGL_HOOKS preset (sglhooks installed first) the mode adds to it.
Old code: kvfresh -> fastload,kvfresh,nosync ; keepkv -> fastload,kvepoch,nosync ; inplace -> + nosync."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
CODE = """
import os
from ai_search_plugin.adapters import verl_rollout as VR, verl_engine as VE
{call}
print("HOOKS=" + os.environ.get("ASRT_SGL_HOOKS", ""))
"""
CASES = {
    "kvfresh": ("VR.install_kvfresh()", {"kvfresh"}),
    "keepkv": ("VR.install_keepkv()", {"kvepoch"}),
    "inplace": ("VE.install_inplace()", {"kvepoch", "inplace", "fastload"}),
}


def _hooks(call, preset=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASRT_")}
    env["PYTHONPATH"] = PY
    if preset is not None:
        env["ASRT_SGL_HOOKS"] = preset
    r = subprocess.run([sys.executable, "-c", CODE.format(call=call)], env=env, capture_output=True, text=True,
                       timeout=120)
    line = [l for l in r.stdout.splitlines() if l.startswith("HOOKS=")]
    assert line, (r.returncode, r.stdout[-500:], r.stderr[-1500:])
    return {h for h in line[-1][6:].split(",") if h}


def test_sets():
    for name, (call, want) in CASES.items():
        got = _hooks(call)
        assert got == want, (name, got, want)
        got = _hooks(call, preset="nosync,fastload")
        assert got == want | {"nosync", "fastload"}, (name, got)
        print(f"{name}: unset -> {sorted(want)}; preset nosync,fastload -> {sorted(got)}")


if __name__ == "__main__":
    test_sets()
    print("ok")
