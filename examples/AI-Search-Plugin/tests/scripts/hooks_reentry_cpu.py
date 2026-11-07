"""asrt _hooks finder next to another wrapping finder that also delegates to every other
meta-path finder (like a user's logging shim): importing the module both hook must not recurse, and both
post-import hooks must fire once.  Old _hooks (no re-entrancy guard): RecursionError."""
import importlib
import importlib.abc
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from ai_search_plugin.adapters import _hooks  # noqa: E402

d = tempfile.mkdtemp(prefix="hooks-reentry-")
with open(os.path.join(d, "asrt_reentry_target.py"), "w") as f:
    f.write("X = 1\n")
sys.path.insert(0, d)
fired = []


class _OtherLoader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec) if hasattr(self.inner, "create_module") else None

    def exec_module(self, module):
        self.inner.exec_module(module)
        fired.append("other")


class OtherFinder(importlib.abc.MetaPathFinder):   # = such a shim's finder
    TARGET = "asrt_reentry_target"

    def find_spec(self, name, path, target=None):
        if name != self.TARGET:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            s = f.find_spec(name, path, target)
            if s is not None:
                if s.loader is not None and hasattr(s.loader, "exec_module"):
                    s.loader = _OtherLoader(s.loader)
                return s
        return None


def test_no_recursion_both_hooks_fire():
    _hooks.when_imported("asrt_reentry_target", lambda m: fired.append("asrt"))
    sys.meta_path.insert(0, OtherFinder())       # in front of asrt's finder, like the real launch order
    try:
        import asrt_reentry_target  # noqa: F401
    except RecursionError:
        raise AssertionError("RecursionError: finders recurse into each other")
    assert sorted(fired) == ["asrt", "other"], fired
    print("two delegating finders: no recursion, both post-import hooks fired once")


def test_asrt_first_then_other():
    """reverse order (asrt's finder in front): still no recursion, each hook once."""
    with open(os.path.join(d, "asrt_reentry_target2.py"), "w") as f:
        f.write("Y = 2\n")
    fired.clear()
    OtherFinder.TARGET = "asrt_reentry_target2"
    _hooks.when_imported("asrt_reentry_target2", lambda m: fired.append("asrt"))
    i = next(i for i, f in enumerate(sys.meta_path) if isinstance(f, _hooks._Finder))
    other = next(f for f in sys.meta_path if isinstance(f, OtherFinder))
    sys.meta_path.remove(other)
    sys.meta_path.insert(i + 1, other)          # asrt first, other right after it
    importlib.invalidate_caches()               # the file was written after the directory listing was cached
    import asrt_reentry_target2  # noqa: F401
    assert sorted(fired) == ["asrt", "other"], fired
    print("asrt finder first: no recursion, both hooks once")


if __name__ == "__main__":
    test_no_recursion_both_hooks_fire()
    test_asrt_first_then_other()
    print("ok")
