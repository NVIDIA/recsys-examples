"""CPU test: asrt's monkeypatches of verl worker methods keep the function attributes verl's @register
stores on the method (MAGIC_ATTR = dispatch / execute mode / blocking), so RayWorkerGroup still binds them.

Bug (verl separate_async PPO): verl_placement wrapped TrainingWorker.train_mini_batch without
copying the attribute -> the critic worker group had no train_mini_batch
("'RayWorkerGroup' object has no attribute 'train_mini_batch'").

(a) always: a stand-in for verl's @register; patch_method / set_kept / keep_attrs keep the attribute, the
    wrapper's own markers and __wrapped__ chain, async-ness, on the class, an inheriting subclass, a
    multiple-inheritance subclass and a Ray-style copy subclass (same __name__, own copy of the method).
(b) when verl imports (VERL_SRC or an installed verl): install the asrt verl plugin (ASRT_VERL=full), then run
    verl's own binders - WorkerGroup._bind_worker_method (what RayWorkerGroup does) and
    _bind_workers_method_to_parent (colocated WorkerDict) - over TrainingWorker (the critic / new-engine actor)
    and ActorRolloutRefWorker: every registered method is still bound, with the same dispatch attributes.
Run: python3 tests/scripts/verl_keep_attrs_cpu.py
"""
import asyncio
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ai_search_plugin.adapters._hooks import keep_attrs, patch_method, set_kept  # noqa: E402

MAGIC = "attrs_3141562937"


def register(dispatch_mode="ONE_TO_ALL", execute_mode="ALL", blocking=True):
    """Shape of verl.single_controller.base.decorator.register: a functools.wraps wrapper + MAGIC_ATTR dict."""
    import functools

    def deco(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*a, **k):
                return await func(*a, **k)
        else:
            @functools.wraps(func)
            def wrapper(*a, **k):
                return func(*a, **k)
        setattr(wrapper, MAGIC, {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking})
        return wrapper
    return deco


def part_a():
    class Base:
        @register(dispatch_mode={"dispatch_fn": 1, "collect_fn": 2}, blocking=False)
        def train_mini_batch(self, data):
            return ("tmb", data)

        @register()
        async def update_weights(self, step=None):
            return ("uw", step)

        @register()
        def save_checkpoint(self, path):
            return ("save", path)

    class Mixin:
        pass

    class Critic(Mixin, Base):      # multiple inheritance, inherits the methods
        pass

    attrs_tmb = getattr(Base.train_mini_batch, MAGIC)

    def make_tmb(orig):
        def train_mini_batch(self, data):
            return orig(self, data)
        train_mini_batch._asrt_mark = 1
        return train_mini_batch

    def make_uw(orig):
        async def update_weights(self, step=None):
            return await orig(self, step)
        return update_weights

    assert patch_method(Base, "train_mini_batch", make_tmb) >= 1
    patch_method(Base, "update_weights", make_uw)
    for C in (Base, Critic):
        m = C.train_mini_batch
        assert getattr(m, MAGIC) is attrs_tmb, C
        assert m._asrt_mark == 1 and m.__name__ == "train_mini_batch" and m.__wrapped__ is not None
        assert C().train_mini_batch(3) == ("tmb", 3)
        assert hasattr(C.update_weights, MAGIC) and inspect.iscoroutinefunction(C.update_weights)
        assert asyncio.run(C().update_weights(5)) == ("uw", 5)

    # Ray-style copy: a subclass with the same __name__ that carries its own copy of the method
    def copy_of(base):
        class Base(base):            # noqa: F811  (same __name__ on purpose)
            pass
        Base.save_checkpoint = base.__dict__["save_checkpoint"]
        return Base
    RayCopy = copy_of(Base)

    def make_save(orig):
        def save_checkpoint(self, path):
            return orig(self, path)
        return save_checkpoint
    patch_method(Base, "save_checkpoint", make_save)
    for C in (Base, RayCopy, Critic):
        assert hasattr(C.save_checkpoint, MAGIC) and getattr(C.save_checkpoint, "__asrt_patched__", False), C

    # set_kept: a plain assignment-style patch (ceiling / offpol), also when the method is inherited
    def save_checkpoint(self, path):
        return ("mine", path)
    save_checkpoint._own = True
    set_kept(Critic, "save_checkpoint", save_checkpoint)
    assert hasattr(Critic.save_checkpoint, MAGIC) and Critic.save_checkpoint._own

    # keep_attrs: the wrapper's own attributes / __wrapped__ win over the original's
    def inner():
        pass

    def orig():
        pass
    orig.__wrapped__ = "orig-inner"
    setattr(orig, MAGIC, {"dispatch_mode": "x"})
    orig.flag = "orig"

    def new():
        pass
    new.__wrapped__ = inner
    new.flag = "new"
    keep_attrs(new, orig)
    assert new.__wrapped__ is inner and new.flag == "new" and getattr(new, MAGIC) == {"dispatch_mode": "x"}
    # staticmethod / classmethod on both sides
    sm_orig = staticmethod(orig)
    sm_new = staticmethod(lambda: None)
    keep_attrs(sm_new, sm_orig)
    assert hasattr(sm_new.__func__, MAGIC)
    print("(a) keep_attrs / patch_method / set_kept keep @register metadata: OK")


def part_b():
    src = os.environ.get("VERL_SRC")
    if src:
        sys.path.insert(0, src)
    try:
        import verl.workers.engine_workers as W
        from verl.single_controller.base.decorator import MAGIC_ATTR
        from verl.single_controller.base.worker_group import WorkerGroup
        from verl.single_controller.ray import base as RB
    except Exception as e:
        print(f"(b) skipped: verl not importable ({e!r})")
        return False
    classes = [W.TrainingWorker, W.ActorRolloutRefWorker]

    def registered(C):
        return {n: getattr(getattr(C, n), MAGIC_ATTR) for n in dir(C)
                if callable(getattr(C, n, None)) and hasattr(getattr(C, n), MAGIC_ATTR)}
    before = {C: registered(C) for C in classes}
    os.environ.setdefault("ASRT_VERL", "full")
    from ai_search_plugin.adapters import verl as AV
    feats = AV.install("full")
    import importlib
    for mod in ("verl.workers.engine.megatron.transformer_impl", "verl.checkpoint_engine.base"):
        try:   # fire the post-import hooks that patch these modules (place / inplace / profile)
            importlib.import_module(mod)
        except Exception:
            pass
    patched = []
    for C in classes:
        after = registered(C)
        assert set(after) == set(before[C]), (C.__name__, set(before[C]) ^ set(after))
        for n, a in after.items():
            assert a == before[C][n], (C.__name__, n)
            if getattr(getattr(C, n), "__asrt_patched__", False):
                patched.append(f"{C.__name__}.{n}")

    # verl's own binders over the critic class (separate worker group) and the colocated WorkerDict
    class FakeWG(WorkerGroup):
        def __init__(self):
            pass

        def execute_all(self, *a, **k):
            pass
        execute_rank_zero = execute_all
    for C in classes:
        wg = FakeWG()
        names = wg._bind_worker_method(C, lambda self, name, **k: (lambda *a, **kk: name))
        assert set(names) == set(before[C]), (C.__name__, set(before[C]) ^ set(names))
        assert hasattr(wg, "train_mini_batch") or C is not W.TrainingWorker

    class Dict:
        pass
    RB._bind_workers_method_to_parent(Dict, "critic", W.TrainingWorker)
    assert hasattr(Dict, "critic_train_mini_batch")
    assert getattr(Dict.critic_train_mini_batch, MAGIC_ATTR) == before[W.TrainingWorker]["train_mini_batch"]
    print(f"(b) verl {W.__file__.split('/verl/workers')[0]}: features {sorted(feats)}; patched registered methods "
          f"{patched}; all {sum(len(v) for v in before.values())} registered methods bound with the same "
          "dispatch attributes (worker group + colocated WorkerDict): OK")
    return True


if __name__ == "__main__":
    part_a()
    part_b()
    print("verl keep-attrs cpu test OK")
