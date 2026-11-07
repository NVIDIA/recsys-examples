"""CPU test of the cuDNN FROST part of ai_search_plugin.adapters.jitcache: complete entries go level 1 -> shared store
(publish) and shared store -> a fresh level 1 (seed); an incomplete entry (kernel.o without its entry.json commit
marker) is never copied; nothing already present is rewritten.
    python3 asrt/tests/test_jitcache_cudnn_cpu.py"""
import os
import sys
import tempfile


def mk(root, env, ent, complete=True):
    d = os.path.join(root, "v2", env, ent)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(root, "v2", env, "manifest.json"), "w").write('{"device_name": "GPU-A"}')
    open(os.path.join(d, "kernel.o"), "wb").write(os.urandom(1000))
    if complete:
        open(os.path.join(d, "entry.json"), "w").write('{"key": "%s"}' % ent)


def main():
    t = tempfile.mkdtemp(prefix="jc-cudnn-")
    store = os.path.join(t, "store")
    os.environ["ASRT_JITCACHE_DIR"] = store
    os.environ["ASRT_JITCACHE"] = "cudnn"
    os.environ.pop("CUDNN_FRONTEND_COMPILED_CACHE", None)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    import ai_search_plugin.adapters.jitcache as J  # auto-installs (cudnn item only)
    # run 1 (cold): level 1 = XDG of run 1
    os.environ["XDG_CACHE_HOME"] = os.path.join(t, "run1")
    l1 = J.cudnn_l1()
    mk(l1, "envA", "e1")
    mk(l1, "envA", "e2")
    mk(l1, "envA", "e3", complete=False)   # being written
    assert J.publish() == 2
    l2 = J.cudnn_l2()
    assert sorted(os.listdir(os.path.join(l2, "v2", "envA"))) == ["e1", "e2", "manifest.json"]
    assert J.publish() == 0                # nothing new
    # run 2 (warm): fresh level 1
    os.environ["XDG_CACHE_HOME"] = os.path.join(t, "run2")
    assert J.cudnn_seed() == 2
    l1b = J.cudnn_l1()
    for e in ("e1", "e2"):
        for f in ("kernel.o", "entry.json"):
            assert open(os.path.join(l1b, "v2", "envA", e, f), "rb").read() == open(os.path.join(l1, "v2", "envA", e, f), "rb").read()
    assert os.path.isfile(os.path.join(l1b, "v2", "envA", "manifest.json"))
    assert not os.path.exists(os.path.join(l1b, "v2", "envA", "e3"))
    leftovers = [f for dp, _, fs in os.walk(t) for f in fs if ".tmp." in f]
    assert not leftovers, leftovers
    print("OK", J.stats())


if __name__ == "__main__":
    main()
