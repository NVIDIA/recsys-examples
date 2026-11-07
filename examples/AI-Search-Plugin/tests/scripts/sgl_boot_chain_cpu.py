"""ai_search_plugin/adapters/sgl_boot/sitecustomize.py next to other sitecustomize files, in fresh interpreters:
  (1) a shim that chains to "the first other sitecustomize on sys.path" and a third and a fourth file that do the
      same / nothing: no RecursionError, each file runs exactly once, engine hooks registered once
      (an earlier version hit RecursionError + half-installed hooks);
  (2) a shim that does not chain back: same result;
  (3) hook registration failing: a helper interpreter (never imports sglang) logs ERROR and runs on; a process that
      imports sglang.srt.managers.* (= SGLang engine) exits 78 there; ASRT_SGL_HOOKS_STRICT=0 runs on.
Two copies of the file on one PYTHONPATH (e.g. a site copy + the packaged one) are covered by test_both_copies."""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
_BOOT = os.path.join(PY, "ai_search_plugin", "adapters", "sgl_boot", "sitecustomize.py")
BOOTS = {"verl": _BOOT, "slime": _BOOT}   # one packaged file; both framework paths use it

CHAINER = '''import os, sys
print("RAN %s", flush=True)
_HERE = os.path.dirname(os.path.abspath(__file__))
for d in sys.path:                       # = a typical site shim that chains to the next sitecustomize
    if not d or os.path.abspath(d) == _HERE:
        continue
    f = os.path.join(d, "sitecustomize.py")
    if os.path.isfile(f):
        exec(compile(open(f).read(), f, "exec"), {"__name__": "sitecustomize_chained", "__file__": f})
        break
'''
PLAIN = 'print("RAN %s", flush=True)\n'
# registry-aware chainer that marks the NEXT file in the shared registry BEFORE running it (an earlier version's case: it
# sgl_boot then read "my path is in the registry" as re-entry and silently skipped its asrt hooks)
MARKER = '''import os, sys
print("RAN %s", flush=True)
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAN = sys.__dict__.setdefault("_asrt_sitecustomize_ran", set())
_RAN.add(os.path.abspath(__file__))
for d in sys.path:
    f = os.path.abspath(os.path.join(d or ".", "sitecustomize.py"))
    if not d or os.path.dirname(f) == _HERE or f in _RAN or not os.path.isfile(f):
        continue
    _RAN.add(f)
    exec(compile(open(f).read(), f, "exec"), {"__name__": "sitecustomize_chained", "__file__": f})
    break
'''
PROBE = "import sys; print('MAIN', flush=True); print('ASRT_HOOKS', sys.__dict__.get('_fake_asrt_hooks', 0), flush=True)"


def _tree(boot, files):
    root = tempfile.mkdtemp(prefix="sglboot-")
    dirs = []
    for name, body in [("boot", None)] + files:
        os.makedirs(os.path.join(root, name))
        dst = os.path.join(root, name, "sitecustomize.py")
        if body is None:
            shutil.copy(boot, dst)
        else:
            open(dst, "w").write(body % name)
        dirs.append(os.path.join(root, name))
    # fake sglang.srt.managers package for the engine-process check
    pkg = os.path.join(root, "fakepkgs", "sglang", "srt", "managers")
    os.makedirs(pkg)
    for d in (os.path.join(root, "fakepkgs", "sglang"), os.path.join(root, "fakepkgs", "sglang", "srt"), pkg):
        open(os.path.join(d, "__init__.py"), "w").write("")
    open(os.path.join(pkg, "scheduler.py"), "w").write("X = 1\n")
    # fake asrt package: register_import_hooks counts its calls
    fa = os.path.join(root, "fakeasrt", "ai_search_plugin", "adapters")
    os.makedirs(fa)
    open(os.path.join(root, "fakeasrt", "ai_search_plugin", "__init__.py"), "w").write("")
    open(os.path.join(fa, "__init__.py"), "w").write("")
    open(os.path.join(fa, "slime_sglang_hooks.py"), "w").write(
        "import sys\ndef register_import_hooks():\n    sys._fake_asrt_hooks = sys.__dict__.get('_fake_asrt_hooks', 0) + 1\n")
    return root, dirs


def _run(dirs, env_extra, tail, code="print('MAIN', flush=True)"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASRT_")}
    env.update(env_extra)
    env["PYTHONPATH"] = ":".join(dirs + tail)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)


def _order(kind, names):
    """names: order on PYTHONPATH, 'boot' = the sgl_boot copy under test."""
    bodies = {"chainer": CHAINER, "marker": MARKER, "third": CHAINER, "fourth": PLAIN, "plain": PLAIN}
    root = tempfile.mkdtemp(prefix="sglboot-o-")
    dirs = []
    for n in names:
        d = os.path.join(root, n)
        os.makedirs(d)
        if n == "boot":
            shutil.copy(BOOTS[kind], os.path.join(d, "sitecustomize.py"))
        else:
            open(os.path.join(d, "sitecustomize.py"), "w").write(bodies[n] % n)
        dirs.append(d)
    _, tmp = _tree(BOOTS[kind], [])
    fake_asrt = os.path.join(os.path.dirname(tmp[0]), "fakeasrt")
    r = _run(dirs, {"ASRT_SGL_HOOKS": "kvfresh"}, [fake_asrt], code=PROBE)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "RecursionError" not in out, (kind, names, r.returncode, out[-1500:])
    for n in names:
        if n != "boot":
            assert r.stdout.count("RAN " + n) == 1, (kind, names, n, r.stdout)
    assert "ASRT_HOOKS 1" in r.stdout, (kind, names, "asrt hooks must register exactly once", r.stdout, r.stderr[-800:])


def test_orders(kind):
    # one naive chainer (= a typical site shim) or one registry-marking chainer next to sgl_boot, in both orders, plus
    # plain files after (two naive chainers can re-run each other directly, which no sgl_boot logic can prevent)
    # (a plain sitecustomize ends the chain - normal Python behaviour, as without any chaining)
    for names in (["boot", "chainer", "plain"], ["chainer", "boot", "plain"],
                  ["marker", "boot", "plain"], ["boot", "marker", "plain"], ["boot", "plain"]):
        _order(kind, names)
    print(f"[{kind}] 5 chain orders (naive chainer, registry-marking chainer, plain; both orders): each file once, "
          f"asrt hooks registered exactly once")


def test_both_copies():
    """verl copy and slime copy on the same PYTHONPATH (+ a chaining shim between and a plain file after): no
    recursion, every file once, and asrt hooks registered exactly once per copy that really runs ('twice in
    total' must be the registry's doing only if both copies run; here the second copy is reached through the chain)."""
    root = tempfile.mkdtemp(prefix="sglboot-both-")
    dirs = []
    for n, src in (("bootverl", BOOTS["verl"]), ("shim", None), ("bootslime", BOOTS["slime"]), ("plain", None)):
        d = os.path.join(root, n)
        os.makedirs(d)
        if src:
            shutil.copy(src, os.path.join(d, "sitecustomize.py"))
        else:
            open(os.path.join(d, "sitecustomize.py"), "w").write((CHAINER if n == "shim" else PLAIN) % n)
        dirs.append(d)
    _, tmp = _tree(BOOTS["verl"], [])
    fake_asrt = os.path.join(os.path.dirname(tmp[0]), "fakeasrt")
    r = _run(dirs, {"ASRT_SGL_HOOKS": "kvfresh"}, [fake_asrt], code=PROBE)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "RecursionError" not in out, (r.returncode, out[-1500:])
    assert r.stdout.count("RAN shim") == 1 and r.stdout.count("RAN plain") == 1, r.stdout
    n_hooks = int(r.stdout.split("ASRT_HOOKS")[1].split()[0])
    assert n_hooks in (1, 2), r.stdout
    print(f"[both copies] verl copy -> shim -> slime copy -> plain: each file once, no recursion; "
          f"register_import_hooks called {n_hooks}x (the second call must be harmless: finder is added once)")


def test_chain(kind):
    files = [("shim", CHAINER), ("third", CHAINER), ("fourth", PLAIN)]
    root, dirs = _tree(BOOTS[kind], files)
    r = _run(dirs, {"ASRT_SGL_HOOKS": "kvfresh"}, [PY])
    out = r.stdout + r.stderr
    assert "RecursionError" not in out and r.returncode == 0, (kind, r.returncode, out[-2000:])
    for name in ("shim", "third", "fourth"):
        assert r.stdout.count("RAN " + name) == 1, (kind, name, r.stdout)
    assert "not installed" not in out and "MAIN" in r.stdout, (kind, out[-2000:])
    # shim that does not chain back
    root, dirs = _tree(BOOTS[kind], [("shim", PLAIN), ("third", PLAIN)])
    r = _run(dirs, {"ASRT_SGL_HOOKS": "kvfresh"}, [PY])
    assert r.returncode == 0 and r.stdout.count("RAN shim") == 1, (kind, r.returncode, r.stdout, r.stderr[-800:])
    print(f"[{kind}] boot -> chaining shim -> chaining third -> fourth: each once, no recursion; non-chaining shim ok")


def test_strict(kind):
    root, dirs = _tree(BOOTS[kind], [("shim", CHAINER)])
    fake = os.path.join(root, "fakepkgs")
    env = {"ASRT_SGL_HOOKS": "kvfresh"}           # asrt NOT on the path -> registration fails
    r = _run(dirs, env, [fake])                   # helper interpreter: logs, runs on
    assert r.returncode == 0 and "ERROR sgl hooks not installed" in r.stderr and "MAIN" in r.stdout, \
        (kind, r.returncode, r.stderr[-800:])
    eng = "import sglang.srt.managers.scheduler; print('MAIN', flush=True)"
    helper = "import sglang.srt.managers.io_struct; print('MAIN', flush=True)"   # non-engine verl process
    open(os.path.join(root, "fakepkgs", "sglang", "srt", "managers", "io_struct.py"), "w").write("Y = 1\n")
    r = _run(dirs, env, [fake], code=helper)      # imports a sglang.srt.managers module that is NOT an engine: runs on
    assert r.returncode == 0 and "MAIN" in r.stdout, (kind, r.returncode, r.stderr[-800:])
    r = _run(dirs, env, [fake], code=eng)         # SGLang engine process: stopped
    assert r.returncode == 78 and "without its asrt hooks" in r.stderr and "MAIN" not in r.stdout, \
        (kind, r.returncode, r.stderr[-800:])
    r = _run(dirs, dict(env, ASRT_SGL_HOOKS_STRICT="0"), [fake], code=eng)
    assert r.returncode == 0 and "MAIN" in r.stdout, (kind, r.returncode, r.stderr[-800:])
    print(f"[{kind}] registration failure: helper interpreter and non-engine sglang importer log and run on; SGLang engine exits 78; STRICT=0 runs on")


if __name__ == "__main__":
    test_both_copies()
    for k in ("verl", "slime"):
        test_chain(k)
        test_orders(k)
        test_strict(k)
    print("ok")
