""" Persistent JIT cache: Triton binaries + Triton autotune choices (+ optionally Inductor) kept across runs.

Why: step 0 of every run spent 290-340 s in training; the JIT probe (Y-ours-f1s-pin, ASRT_JIT_PROBE=1) shows ~284 s of
it is Triton compile + autotune (FLA GDN kernels, FLA conv1d/l2norm, TE MoE permute/unpermute/sort_chunks: 19 autotunes,
~200 compiles of ~1.3 s each), on every rank, on every run - because each run gets a private, empty TRITON_CACHE_DIR
(shared dirs were dropped after a run died reading a half-written shared triton file).  All those autotune keys are
shape-independent once the FLA NB key is pinned (ASRT_FLA_NB_PIN), so one persistent store covers every later run.

What: a two-level Triton cache manager (installed in-process via triton.knobs, NOT via env, so child processes that
cannot import asrt are unaffected):
  * level 1 = the framework's own TRITON_CACHE_DIR (unchanged; private per run);
  * level 2 = a shared persistent store  <ASRT_JITCACHE_DIR>/<gpu>-triton<ver>-torch<ver>/triton/<key>/
    - miss in level 1, hit in level 2 -> the files are copied into level 1 (never executed from the shared dir);
    - a group (kernel) is only taken from level 2 when every child file is there (children are published before the
      group file, each by write-to-temp + os.replace, so a visible group is complete); else it is recompiled;
    - every new level-1 file is also published to level 2 (temp + os.replace; failures only counted).
  * Triton autotune results are made persistent for every Autotuner (cache_results=True, the TRITON_CACHE_AUTOTUNING
    path of triton>=3.3; FLA already asks for it, TE's permute kernels do not), so a warm run skips the benchmark and
    compiles only the chosen config (a level-2 hit too).
  * optional `inductor`: TORCHINDUCTOR_CACHE_DIR -> <store>/inductor (FX graph + autotune caches; Inductor writes
    atomically).  Only has an effect where torch.compile runs (slime full plan uses ASRT_JITFIX=eager).

 `cudnn`: cuDNN frontend FROST compiled kernels (gdn_cudnn's cuDNN GDN, ~43 s of step 0) under
<store root>/cudnn_frontend/compiled_plans (the frontend's own env_hash keys versions + GPU), same copy-in / publish.

Switch (one env var): ASRT_JITCACHE_DIR=<dir> turns it on (slime: set it in the job environment; verl: the
`jitcache` feature of ASRT_VERL).  ASRT_JITCACHE=off|<items> with items
triton,autotune,inductor (default triton,autotune,inductor).  One-line use anywhere: `import ai_search_plugin.adapters.jitcache`
(installs when ASRT_JITCACHE_DIR is set).  Counters: jitcache-<pid>.json in ASRT_LOG_DIR at exit and via stats().
Numerics: a cache hit loads the very binary/config a previous run built; the autotune choice is the recorded one
(a cold run picks per rank by timing, i.e. already non-deterministic), kernels are the same code.
"""
import atexit
import json
import os
import re
import threading
import time
import uuid

_STATE = {"installed": False, "items": [], "root": None, "tag": None}
_C = {"l2_file_hit": 0, "l2_group_hit": 0, "l2_group_partial": 0, "l2_miss": 0, "l2_put": 0, "l2_put_skip": 0,
      "l2_put_err": 0, "l2_read_err": 0, "bytes_in": 0, "bytes_out": 0, "copy_s": 0.0, "autotune_forced": 0,
      "cudnn_seeded": 0, "cudnn_published": 0, "cudnn_err": 0}
_LOCK = threading.Lock()
DEFAULT_ITEMS = ("triton", "autotune", "inductor", "cudnn")


def _log(msg):
    r = os.environ.get("RANK", "0")
    if r in ("0", "") or os.environ.get("ASRT_JITCACHE_VERBOSE") == "1":
        print(f"[asrt] jitcache {msg}", flush=True)


def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_")[:80]


def _tag():
    """<gpu>-triton<ver>-torch<ver>, computed lazily (first compile: CUDA is up by then)."""
    if _STATE["tag"]:
        return _STATE["tag"]
    gpu = os.environ.get("ASRT_JITCACHE_GPU", "")
    if not gpu:
        try:
            import torch
            gpu = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "nogpu"
        except Exception:
            gpu = "unknown"
    try:
        import triton
        tv = triton.__version__
    except Exception:
        tv = "x"
    try:
        import torch
        pv = torch.__version__
    except Exception:
        pv = "x"
    _STATE["tag"] = _sanitize(f"{gpu.replace('NVIDIA ', '')}-triton{tv}-torch{pv}")
    return _STATE["tag"]


def store_dir(sub=""):
    root = _STATE["root"] or os.environ.get("ASRT_JITCACHE_DIR", "")
    d = os.path.join(root, _tag(), sub) if sub else os.path.join(root, _tag())
    return d


def _atomic_write(path, data):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".tmp.{os.getpid()}.{uuid.uuid4().hex}.{os.path.basename(path)[:60]}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _make_manager():
    from triton.runtime.cache import FileCacheManager

    class TwoLevelCacheManager(FileCacheManager):
        """Level 1: knobs.cache.dir (framework's dir); level 2: shared persistent store (copy-in, publish-out)."""

        def __init__(self, key, override=False, dump=False):
            super().__init__(key, override=override, dump=dump)
            self._l2 = None if (override or dump) else os.path.join(store_dir("triton"), key)

        # ---- reads
        def get_file(self, filename):
            p = super().get_file(filename)
            if p or not self._l2:
                return p
            sp = os.path.join(self._l2, filename)
            try:
                if os.path.exists(sp):
                    t = time.perf_counter()
                    data = _read(sp)
                    out = FileCacheManager.put(self, data, filename)
                    with _LOCK:
                        _C["l2_file_hit"] += 1
                        _C["bytes_in"] += len(data)
                        _C["copy_s"] += time.perf_counter() - t
                    return out
            except Exception:
                with _LOCK:
                    _C["l2_read_err"] += 1
            return None

        def get_group(self, filename):
            g = super().get_group(filename)
            if g or not self._l2:
                return g
            sg = os.path.join(self._l2, f"__grp__{filename}")
            try:
                if not os.path.exists(sg):
                    with _LOCK:
                        _C["l2_miss"] += 1
                    return None
                t = time.perf_counter()
                child = json.loads(_read(sg)).get("child_paths") or {}
                blobs = {}
                for c, p in child.items():
                    sp = os.path.join(self._l2, os.path.basename(p))
                    if not os.path.exists(sp):
                        with _LOCK:
                            _C["l2_group_partial"] += 1
                        return None
                    blobs[c] = (os.path.basename(p), _read(sp))
                local = {c: FileCacheManager.put(self, data, fname) for c, (fname, data) in blobs.items()}
                FileCacheManager.put(self, json.dumps({"child_paths": local}), f"__grp__{filename}", binary=False)  # level 1 only
                with _LOCK:
                    _C["l2_group_hit"] += 1
                    _C["bytes_in"] += sum(len(d) for _, d in blobs.values())
                    _C["copy_s"] += time.perf_counter() - t
                return local
            except Exception:
                with _LOCK:
                    _C["l2_read_err"] += 1
                return None

        # ---- writes (level 1 as usual, then publish to level 2)
        def put(self, data, filename, binary=True):
            out = super().put(data, filename, binary)
            if self._l2:
                self._publish(data, filename)
            return out

        def put_group(self, filename, group):
            # FileCacheManager.put_group -> self.put("__grp__...") -> published last (after every child): a group
            # visible in level 2 is complete.  Its child paths are level-1 paths; readers map them by basename.
            return super().put_group(filename, group)

        def _publish(self, data, filename):
            sp = os.path.join(self._l2, filename)
            try:
                if os.path.exists(sp) and not filename.startswith("__grp__"):
                    with _LOCK:
                        _C["l2_put_skip"] += 1
                    return
                b = data if isinstance(data, bytes) else str(data).encode()
                _atomic_write(sp, b)
                with _LOCK:
                    _C["l2_put"] += 1
                    _C["bytes_out"] += len(b)
            except Exception:
                with _LOCK:
                    _C["l2_put_err"] += 1

    return TwoLevelCacheManager


def _install_triton():
    from triton import knobs
    cur = knobs.cache.manager_class
    if cur is not None and getattr(cur, "__name__", "") == "TwoLevelCacheManager":
        return "already"
    if cur is not None:
        return f"SKIP: a custom TRITON_CACHE_MANAGER is already set ({cur!r})"
    knobs.cache.manager_class = _make_manager()  # python-level knob: no env propagation to children
    return "on"


def _install_autotune():
    from triton.runtime import autotuner as ta
    if getattr(ta.Autotuner.run, "_asrt_jitcache", False):
        return "already"
    orig = ta.Autotuner.run

    def run(self, *a, **k):
        if not getattr(self, "cache_results", True):
            self.cache_results = True
            with _LOCK:
                _C["autotune_forced"] += 1
        return orig(self, *a, **k)
    run._asrt_jitcache = True
    ta.Autotuner.run = run
    return "on"


def _install_inductor():
    d = store_dir("inductor") if os.environ.get("ASRT_JITCACHE_GPU") else None
    if d is None:  # tag needs the GPU name; resolve without touching CUDA: nvidia-smi-free fallback = env/torch lazily
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                d = store_dir("inductor")
        except Exception:
            pass
    if d is None:
        d = os.path.join(_STATE["root"], "inductor-" + _sanitize(_torch_triton()))
    os.makedirs(d, exist_ok=True)
    prev = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = d
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    return f"on ({d}; was {prev})"


# ----------------------------------------------------------------------------- cuDNN frontend FROST kernels
# cudnn-frontend (>=1.32) keeps every compiled FROST kernel (the cuDNN native GDN of gdn_cudnn: ~43 s of step 0) in
# <root>/v2/<env_hash>/<entry_hash>/{kernel.o, entry.json} + <root>/v2/<env_hash>/manifest.json, root =
# $CUDNN_FRONTEND_COMPILED_CACHE or $XDG_CACHE_HOME/cudnn_frontend/compiled_plans (per run here).  env_hash already
# keys frontend/cutlass-dsl/tvm-ffi versions, CUDA driver and GPU (name, cc, SMs, L2); entry.json is its commit marker
# (written after kernel.o).  Same two-level scheme as Triton: level 1 stays the framework's dir; complete entries of
# the shared store are copied in before the first compile, new complete entries are published after training calls.
CUDNN_SUB = "cudnn_frontend"


def cudnn_l1():
    base = os.environ.get("CUDNN_FRONTEND_COMPILED_CACHE")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "cudnn_frontend", "compiled_plans")
    return base


def cudnn_l2():
    return os.path.join(_STATE["root"] or os.environ.get("ASRT_JITCACHE_DIR", ""), CUDNN_SUB, "compiled_plans")


def _sync_cudnn(src, dst, counter):
    """Copy every COMPLETE entry (entry.json present) of src that dst lacks: kernel.o first, entry.json last, each by
    temp file + os.replace; manifest.json per environment.  Returns the number of entries copied."""
    n = 0
    if not src or not dst or not os.path.isdir(src):
        return 0
    for schema in sorted(os.listdir(src)):
        sd = os.path.join(src, schema)
        if schema.startswith(".") or not os.path.isdir(sd):
            continue
        for env in sorted(os.listdir(sd)):
            ed = os.path.join(sd, env)
            man = os.path.join(ed, "manifest.json")
            if env.startswith(".") or not os.path.isdir(ed) or not os.path.isfile(man):
                continue
            for ent in sorted(os.listdir(ed)):
                e = os.path.join(ed, ent)
                rec, obj = os.path.join(e, "entry.json"), os.path.join(e, "kernel.o")
                if not os.path.isdir(e) or not os.path.isfile(rec) or not os.path.isfile(obj):
                    continue
                out = os.path.join(dst, schema, env, ent)
                if os.path.isfile(os.path.join(out, "entry.json")):
                    continue
                try:
                    if not os.path.isfile(os.path.join(dst, schema, env, "manifest.json")):
                        _atomic_write(os.path.join(dst, schema, env, "manifest.json"), _read(man))
                    data = _read(obj)
                    r = _read(rec)
                    _atomic_write(os.path.join(out, "kernel.o"), data)
                    _atomic_write(os.path.join(out, "entry.json"), r)
                    n += 1
                    with _LOCK:
                        _C["bytes_in" if counter == "cudnn_seeded" else "bytes_out"] += len(data) + len(r)
                except Exception:
                    with _LOCK:
                        _C["cudnn_err"] += 1
    with _LOCK:
        _C[counter] += n
    return n


def cudnn_seed():
    t = time.perf_counter()
    n = _sync_cudnn(cudnn_l2(), cudnn_l1(), "cudnn_seeded")
    with _LOCK:
        _C["copy_s"] += time.perf_counter() - t
    return n


def publish():
    """Publish new level-1 cuDNN FROST entries to the shared store (Triton publishes itself on every put).
    Called after training calls (when a training hook calls it) and at exit; cheap when nothing is new."""
    if "cudnn" not in _STATE["items"]:
        return 0
    return _sync_cudnn(cudnn_l1(), cudnn_l2(), "cudnn_published")


def _install_cudnn():
    n = cudnn_seed()
    atexit.register(publish)
    return f"on (seeded {n} entries {cudnn_l2()} -> {cudnn_l1()})"


def _torch_triton():
    try:
        import torch
        import triton
        return f"triton{triton.__version__}-torch{torch.__version__}"
    except Exception:
        return "x"


def stats():
    with _LOCK:
        out = dict(_C)
    out["copy_s"] = round(out["copy_s"], 3)
    out.update(items=_STATE["items"], store=os.path.join(_STATE["root"] or "", _STATE["tag"] or "?"))
    return out


def _dump():
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "jitcache-%d.json" % os.getpid()), "w") as f:
            json.dump(stats(), f)
    except Exception:
        pass


def install(items=None):
    """Idempotent; returns {item: status}.  Off unless ASRT_JITCACHE_DIR is set (or root passed via env)."""
    if _STATE["installed"]:
        return {"jitcache": "already"}
    root = os.environ.get("ASRT_JITCACHE_DIR", "")
    spec = items if items is not None else os.environ.get("ASRT_JITCACHE", "")
    if not root or spec in ("0", "off"):
        return {"jitcache": "off"}
    want = [x.strip() for x in (spec or ",".join(DEFAULT_ITEMS)).split(",") if x.strip() and x.strip() not in ("1", "on")]
    want = want or list(DEFAULT_ITEMS)
    _STATE.update(installed=True, root=root, items=want)
    rep = {}
    for it, fn in (("triton", _install_triton), ("autotune", _install_autotune), ("inductor", _install_inductor),
                   ("cudnn", _install_cudnn)):
        if it not in want:
            continue
        try:
            rep[it] = fn()
        except Exception as e:
            rep[it] = f"FAILED: {e!r}"[:200]
    atexit.register(_dump)
    _log(f"root {root} items {want} -> {json.dumps(rep)}")
    return rep


def megatron_init(args=None, *rest, **kwargs):
    """slime `--custom-megatron-init-path ai_search_plugin.adapters.jitcache.megatron_init` (or chained through
    ASRT_CHAIN_MEGATRON_INIT)."""
    return install()


if os.environ.get("ASRT_JITCACHE_DIR") and os.environ.get("ASRT_JITCACHE", "") not in ("0", "off"):
    install()
