"""Tiny post-import hook utility shared by the verl / AReaL / ROLL adapters.

`when_imported("pkg.mod", fn)` runs `fn(module)` right after `pkg.mod` is
imported (or immediately if it already is). Lets an adapter be activated
early (e.g. from `VERL_USE_EXTERNAL_MODULES`, a sitecustomize, or a worker
init hook) and still patch framework modules that are imported later, in
every process, without touching framework source.
"""
import importlib.abc
import importlib.util
import sys
import threading
from typing import Callable, Dict, List

_pending: Dict[str, List[Callable]] = {}
_lock = threading.RLock()
_finder = None


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec) if hasattr(self.inner, "create_module") else None

    def exec_module(self, module):
        self.inner.exec_module(module)
        _fire(module.__name__, module)

    def __getattr__(self, name):  # e.g. get_source / is_package used by tooling
        return getattr(self.inner, name)


_tls = threading.local()


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        with _lock:
            if name not in _pending:
                return None
        # re-entrancy guard: another wrapping finder that also delegates to "every other finder
        # in sys.meta_path" (e.g. a user's logging shim) calls back into us for the same module -> infinite
        # recursion (RecursionError importing verl.trainer.ppo.v1.trainer_base).  While we resolve
        # `name` in this thread, a nested lookup of the same name is not ours: return None so the other finder goes on
        # to the real path finder; both wrappers still wrap the loader on the way back.
        active = getattr(_tls, "active", None)
        if active is None:
            active = _tls.active = set()
        if name in active:
            return None
        active.add(name)
        # only the finders AFTER us: the import system already asked the ones before us (they declined, or one of them
        # is the wrapping finder that is delegating to us right now - asking it again would wrap the loader twice)
        mp = list(sys.meta_path)
        later = mp[mp.index(self) + 1:] if self in mp else mp
        try:
            for f in later:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                        spec.loader = _Loader(spec.loader)
                    return spec
            return None
        finally:
            active.discard(name)


def _fire(name, module):
    with _lock:
        fns = _pending.pop(name, [])
    for fn in fns:
        try:
            fn(module)
        except Exception as e:  # never break the framework import
            print(f"[asrt] post-import hook for {name} failed: {e!r}", flush=True)


def when_imported(name: str, fn: Callable) -> None:
    global _finder
    mod = sys.modules.get(name)
    if mod is not None and getattr(getattr(mod, "__spec__", None), "_initializing", False) is False:
        fn(mod)
        return
    with _lock:
        _pending.setdefault(name, []).append(fn)
        if _finder is None:
            _finder = _Finder()
            sys.meta_path.insert(0, _finder)


_KEEP_SKIP = ("__wrapped__", "__asrt_patched__")


def _fn(obj):
    """The plain function behind a staticmethod / classmethod / bound method, else obj."""
    return getattr(obj, "__func__", obj)


def keep_attrs(new, orig):
    """Make `new` (a replacement for `orig`) carry `orig`'s identity and function attributes, so frameworks that
    read metadata off the method still see it - e.g. verl's @register puts the dispatch/execute mode under
    `attrs_3141562937` (MAGIC_ATTR) in the function's __dict__, and RayWorkerGroup binds only methods that have it
    (a wrapper without it -> "'RayWorkerGroup' object has no attribute 'train_mini_batch'"). Attributes `new`
    already has win (its own markers, its __wrapped__ chain); staticmethod/classmethod/bound methods are unwrapped
    on both sides. Returns `new`."""
    dst, src = _fn(new), _fn(orig)
    for a in ("__module__", "__name__", "__qualname__", "__doc__"):
        try:
            if hasattr(src, a):
                setattr(dst, a, getattr(src, a))
        except (AttributeError, TypeError):
            pass
    own = getattr(dst, "__dict__", None)
    if own is None:
        return new
    for k, v in list(getattr(src, "__dict__", {}).items()):
        if k in _KEEP_SKIP or k in own:
            continue
        try:
            setattr(dst, k, v)
        except (AttributeError, TypeError):
            pass
    return new


_wrap_keep_attrs = keep_attrs


def set_kept(cls, name: str, new):
    """setattr(cls, name, new) with keep_attrs against the method it replaces (own or inherited)."""
    old = cls.__dict__.get(name) if name in cls.__dict__ else getattr(cls, name, None)
    if old is not None:
        keep_attrs(new, old)
    setattr(cls, name, new)
    return new


def patch_method(cls, name: str, make: Callable) -> int:
    """Replace `cls.name` by `make(orig)` on cls and on every live class with the
    same __name__ that carries its own copy of the method (Ray actor wrapper
    subclasses copy methods). Idempotent per class. Returns #classes patched."""
    import gc
    targets = [cls]
    for o in gc.get_objects():
        try:
            if type(o) is type and o is not cls and type.__getattribute__(o, "__name__") == cls.__name__ \
                    and name in type.__getattribute__(o, "__dict__"):
                targets.append(o)
        except Exception:
            continue
    n = 0
    for C in targets:
        orig = C.__dict__.get(name) or getattr(C, name)
        if getattr(orig, "__asrt_patched__", False):
            continue
        new = keep_attrs(make(orig), orig)
        _fn(new).__asrt_patched__ = True
        _fn(new).__wrapped__ = orig
        setattr(C, name, new)
        n += 1
    return n
