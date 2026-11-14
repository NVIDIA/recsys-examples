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

"""Global-hook NVTX attribution for an nsys-profiled PyTorch workload.

You do **not** wrap a specific model. Call ``enable()`` once at program
start, bracket the capture window with ``start()`` / ``stop()``, and mark
each step with ``step()``::

    from talos.torch_perf_analysis import talos_profiler

    talos_profiler.enable()    # ONCE, at program start

    for step, batch in enumerate(loader):
        if step == 50: talos_profiler.start()   # open capture window
        if step == 60: talos_profiler.stop()    # close it
        talos_profiler.step()                    # mark a step boundary
        loss = model(batch); loss.backward(); opt.step(); opt.zero_grad()

Emission is done by a **C++ RecordFunction observer** (``talos_observer.cpp``),
registered at the ATen dispatcher via ``at::addGlobalCallback``. Per-op work
never enters the Python interpreter, and a global callback fires on *every*
thread — framework worker threads and the C++ autograd threads included — so
no threading patch is involved.

What it produces in the trace (all NVTX, default domain):

* ``ta:step_<N>``                  one per ``step()`` call
* ``talos::<func>#<path>@<Class>!!<file>:<line>``
                                   one per outermost dispatched op, tagged
                                   with the op, the enclosing module's
                                   path+class, and the user call site
                                   (cwd-relative path, ``/`` shown as ``.``)
* ``talos_seq:<op>`` marks               nested autograd-recorded ops; the seq
                                   travels in the 64-bit NVTX payload
* ``XxxBackward0`` / ``autograd::engine::evaluate_function`` ranges on the
  autograd threads, seq in the payload — ``stats.py`` joins backward to the
  forward label by seq and stamps the attribution over each
  ``evaluate_function`` range.

Message strings are NVTX *registered strings* (one registration per unique
label, seq in the payload), so trace string interning does not scale with
event count.

**Compilation.** The observer is JIT-compiled in the target environment on
first use (``torch.utils.cpp_extension.load``; ~1 min, cached under
``~/.cache/torch_extensions``). To surface build problems at install time
instead of at training time, run this once after ``pip install``::

    python -m talos.compile_observer   # exits non-zero if the build fails

If the extension cannot be built at runtime, ``enable()``/``start()`` warn
and disable profiling — the training run itself never breaks.

**Module hooks are context only — they emit no NVTX.** A global
``nn.Module`` forward pre/post hook pair keeps a per-thread stack of the
module ``(path, class)`` and mirrors the top into the observer's C++
thread-local, so each op label carries its enclosing module. Naming is
cached on the module instance on first sight (``_ta_info``).

**Kill switch.** Everything is gated by ``TORCH_PERF_ANALYSIS_ENABLE``.
Unset (default) ⇒ ``enable`` / ``start`` / ``stop`` / ``step`` /
:func:`nvtx_range` are all no-ops, so the calls can live in the training
script permanently at zero cost.

Ablation switches (default on): ``TALOS_CPP_SITELINE=0`` drops file:line,
``TALOS_CPP_SEQ_MARKS=0`` drops the seq marks, ``TALOS_CPP_SKIPLIST=0`` labels
no-kernel ops too. ``TALOS_CPP_OWN_DOMAIN=1`` moves emission to a separate
"talos" NVTX domain, which detaches it from the workload's own nvtx-path
tree — attribution in memcpy/sync artifacts is lost, so keep it off unless
you are isolating our ranges on purpose.
"""

from __future__ import annotations

import os
import os.path as _osp
import sys
import threading
from contextlib import contextmanager

import torch
import torch.nn as nn

# ─── Env gating ─────────────────────────────────────────────────────

_ENABLE_ENV = "TORCH_PERF_ANALYSIS_ENABLE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "").lower() in _TRUTHY


# ─── C++ observer backend ────────────────────────────────────────────

_CPP_SITELINE_ENV = "TALOS_CPP_SITELINE"  # =0 → skip file:line (ablation)
_CPP_SEQ_MARKS_ENV = "TALOS_CPP_SEQ_MARKS"  # =0 → skip talos_seq: marks (ablation)
_CPP_SKIPLIST_ENV = "TALOS_CPP_SKIPLIST"  # =0 → label no-kernel ops too
_CPP_OWN_DOMAIN_ENV = "TALOS_CPP_OWN_DOMAIN"  # =1 → separate "talos" NVTX domain


_cpp_ext = None  # loaded extension module
_cpp_active = False  # capture window open
_cpp_failed = False  # compile/load failed → profiling disabled (warned)


def _load_cpp_ext():
    """Compile/load the C++ observer (cached by torch's extension builder)."""
    global _cpp_ext
    if _cpp_ext is not None:
        return _cpp_ext
    from torch.utils import cpp_extension as _ce

    src = _osp.join(_osp.dirname(_osp.abspath(__file__)), "talos_observer.cpp")
    cuda_inc = _osp.join(_ce.CUDA_HOME or "/usr/local/cuda", "include")
    _cpp_ext = _ce.load(
        name="talos_observer",
        sources=[src],
        extra_cflags=["-O3"],
        extra_include_paths=[cuda_inc],
        verbose=False,
    )
    prefixes = [p for p in (_PKG_DIR, _STDLIB_DIR, _TORCH_DIR) if p]
    _cpp_ext.set_internal_prefixes(prefixes)
    _cpp_ext.set_cwd(_CWD_PREFIX)
    return _cpp_ext


# ─── Label vocabulary ───────────────────────────────────────────────

# The observer's ``talos::`` label layout is
# ``talos::<func>#<path>@<Class>!!<file>:<line>`` (or ``talos::<func>!!<file>:<line>``
# when no module is on the stack); it is built in talos_observer.cpp. Only the
# step marker is emitted from Python:
_STEP_PREFIX = "ta:step_"


# DDP / FSDP / DataParallel expose the real model at ``.module``; we
# unwrap so paths read ``<RootClass>.x`` rather than
# ``<RootClass>.module.x``. Match by class name so a user submodule
# literally named ``module`` isn't mistaken for a wrapper.
_WRAPPER_CLASSES = frozenset(
    {"DistributedDataParallel", "FullyShardedDataParallel", "DataParallel"}
)


# ─── Frame-filter prefixes for the C++ observer's file:line resolver ─

_sep = _osp.sep
_PKG_DIR = _osp.dirname(_osp.abspath(__file__)) + _sep
_STDLIB_DIR = _osp.dirname(os.__file__) + _sep
try:
    _TORCH_DIR = _osp.dirname(_osp.abspath(torch.__file__)) + _sep
except Exception:  # pragma: no cover
    _TORCH_DIR = None

# Call sites under the working directory are shown with the cwd-relative
# path, '/' replaced by '.' (veloq joins nvtx_path segments with '/', so a
# slash inside a label would corrupt path parsing); files outside the cwd
# keep the bare basename. The path rewrite itself happens in the C++
# observer (resolve_site); Python only supplies the cwd prefix.
_CWD_PREFIX = os.getcwd() + _sep


# ─── NVTX ───────────────────────────────────────────────────────────


def _push(label: str) -> None:
    torch.cuda.nvtx.range_push(label)


def _pop() -> None:
    torch.cuda.nvtx.range_pop()


# ─── Thread-local context stack ─────────────────────────────────────
#
# A stack of ``(path, class)`` for the modules / nvtx_range regions
# currently on the call path. Maintained by the module hooks and
# nvtx_range; the top of the stack is mirrored into the C++ observer's
# thread-local (``set_ctx``) so op labels carry their enclosing module.

_tls = threading.local()


def _stack() -> list:
    s = getattr(_tls, "stack", None)
    if s is None:
        s = _tls.stack = []
    return s


# ─── Naming cache (built on the instance, on first sight) ───────────

_index_lock = threading.Lock()


def _unwrap(m: nn.Module) -> nn.Module:
    while type(m).__name__ in _WRAPPER_CLASSES and isinstance(
        getattr(m, "module", None), nn.Module
    ):
        m = m.module
    return m


def _index_subtree(top: nn.Module) -> None:
    """Stamp ``top`` and all descendants with ``_ta_info = (path, class)``.
    The path is rooted at the (unwrapped) root's *class name*, e.g.
    ``Model.encoder.0.linear``. The wrapper (DDP) shares the root info."""
    real = _unwrap(top)
    root_prefix = type(real).__name__
    root_info = (root_prefix, root_prefix)
    top._ta_info = root_info
    for name, m in real.named_modules():
        if name == "":
            m._ta_info = root_info
        else:
            m._ta_info = (f"{root_prefix}.{name}", type(m).__name__)


def _ensure_indexed(module: nn.Module) -> tuple:
    """Return ``module._ta_info``, indexing its subtree once if needed.
    Locked + double-checked so concurrent first-sight from worker threads
    can't observe a half-stamped tree."""
    with _index_lock:
        info = module.__dict__.get("_ta_info")
        if info is None:
            _index_subtree(module)
            info = module.__dict__["_ta_info"]
        return info


# ─── Global module hooks (context only — no NVTX) ───────────────────


def _module_pre(module, args):
    info = module.__dict__.get("_ta_info") or _ensure_indexed(module)
    _stack().append(info)
    if _cpp_active:  # mirror top-of-stack into the C++ thread-local
        _cpp_ext.set_ctx(f"{info[0]}@{info[1]}")


def _module_post(module, args, output):
    st = _stack()
    if st:
        st.pop()
    if _cpp_active:
        _cpp_ext.set_ctx(f"{st[-1][0]}@{st[-1][1]}" if st else "")


# ─── Manual region annotation (context only) ────────────────────────


@contextmanager
def nvtx_range(name: str, cls: str = "Region"):
    """Give a non-Module code region its own ``(name, class)`` context.

    Like a module hook, this only pushes context — it emits no NVTX range
    itself. Any op dispatched inside it is then labeled
    ``talos::<func>#<name>@<cls>`` (instead of inheriting the enclosing
    module), so you can attribute a free function — an optimizer step, a
    mask filter — to a name + class of your choosing. ``stats.py``'s seq
    join then synthesizes a matching backward attribution. No-op unless
    ``TORCH_PERF_ANALYSIS_ENABLE``.
    """
    if not _enabled():
        yield
        return
    _stack().append((name, cls))
    if _cpp_active:
        _cpp_ext.set_ctx(f"{name}@{cls}")
    try:
        yield
    finally:
        st = _stack()
        st.pop()
        if _cpp_active:
            _cpp_ext.set_ctx(f"{st[-1][0]}@{st[-1][1]}" if st else "")


# ─── Profiler ───────────────────────────────────────────────────────


class TalosProfiler:
    """Process-global profiler. ``enable()`` (once, up front) extends
    coverage to worker threads; ``start()`` / ``stop()`` bracket the nsys
    capture window; ``step()`` marks one step.

    Reused across steps; a single instance is exported as the module
    singleton :data:`talos_profiler`."""

    def __init__(self):
        self._installed = False
        self._active = False
        self._handles: list = []
        self._step = 0
        self._step_pushed = False

    # ── worker-thread coverage (call once, up front) ──

    def enable(self) -> None:
        """Load (JIT-compiling if needed) the C++ observer extension.

        Call **once at program start**. The observer is a process-global
        dispatcher callback, so every thread — framework workers and the
        C++ autograd threads — is covered with no threading patch.
        Compiling here (instead of inside :meth:`start`, mid-training)
        keeps the capture window clean. Idempotent; no-op unless
        ``TORCH_PERF_ANALYSIS_ENABLE``. If the extension cannot be built
        (no compiler, missing headers), a warning is printed and profiling
        is disabled — the training run itself is never broken. To surface
        build failures at install time instead, run
        ``python -m talos.compile_observer`` once after ``pip install``."""
        global _cpp_failed
        if not _enabled():
            return
        if _cpp_failed:
            return
        try:
            _load_cpp_ext()
            sys.stderr.write(
                "[torch_perf_analysis] observer extension loaded; all "
                "threads covered.\n"
            )
        except Exception as e:  # no compiler / missing headers / ABI issue
            _cpp_failed = True
            sys.stderr.write(
                f"[torch_perf_analysis] WARN: observer extension unavailable "
                f"({type(e).__name__}: {e}); profiling DISABLED for this "
                f"run.\n"
            )

    # ── capture window ──

    def start(self) -> None:
        """Install the global module hooks (context), register the C++
        observer callback, open its gate, and ``cudaProfilerStart``.
        No-op unless enabled / already active / extension unavailable."""
        global _cpp_active, _cpp_failed
        if not _enabled():
            sys.stderr.write(
                f"[torch_perf_analysis] disabled; set {_ENABLE_ENV}=1 to enable.\n"
            )
            return
        if self._active or _cpp_failed:
            return

        # Fresh stack on the driver thread (defensive against a prior
        # unbalanced run).
        _tls.stack = []
        self._step = 0
        self._step_pushed = False

        self._install()
        try:
            ext = _load_cpp_ext()
            ext.start(
                os.environ.get(_CPP_SITELINE_ENV, "1") != "0",
                os.environ.get(_CPP_SEQ_MARKS_ENV, "1") != "0",
                os.environ.get(_CPP_SKIPLIST_ENV, "1") != "0",
                os.environ.get(_CPP_OWN_DOMAIN_ENV, "") == "1",
            )
        except Exception as e:
            _cpp_failed = True
            self._remove()
            sys.stderr.write(
                f"[torch_perf_analysis] WARN: observer extension unavailable "
                f"({type(e).__name__}: {e}); profiling DISABLED for this "
                f"run.\n"
            )
            return
        _cpp_active = True
        torch.cuda.profiler.start()
        self._active = True
        sys.stderr.write("[torch_perf_analysis] capture started.\n")

    def stop(self) -> None:
        """Reverse of :meth:`start`. No-op unless active."""
        global _cpp_active
        if not self._active:
            return
        torch.cuda.profiler.stop()
        if self._step_pushed:  # a step left open by an exception path
            _pop()
            self._step_pushed = False
        if _cpp_active:
            _cpp_active = False
            _cpp_ext.stop()
            sys.stderr.write(
                f"[torch_perf_analysis] observer stats: {dict(_cpp_ext.stats())}\n"
            )
        self._remove()
        self._active = False
        sys.stderr.write("[torch_perf_analysis] capture stopped.\n")

    # ── per-step ──

    def step(self) -> None:
        """Mark a step boundary: close the open ``ta:step_N`` range (if any)
        and open the next. Call once per training iteration. No-op outside
        the capture window."""
        if not self._active:
            return
        if self._step_pushed:
            _pop()  # close the previous step
        _push(f"{_STEP_PREFIX}{self._step}")  # special format, no caller suffix
        self._step += 1
        self._step_pushed = True

    # ── hook (de)registration ──

    def _install(self) -> None:
        if self._installed:
            return
        M = nn.modules.module
        try:
            post = M.register_module_forward_hook(_module_post, always_call=True)
        except TypeError:  # older torch without always_call
            post = M.register_module_forward_hook(_module_post)
        self._handles = [
            M.register_module_forward_pre_hook(_module_pre),
            post,
        ]
        self._installed = True

    def _remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._installed = False


# Process-global singleton — import this directly.
talos_profiler = TalosProfiler()


def compile_observer_main() -> int:
    """Entry point for ``python -m talos.compile_observer``: build the C++
    observer NOW, in this environment, and exit non-zero on failure.

    Run once right after ``pip install talos`` (e.g. as a Dockerfile
    step) so a broken toolchain / header mismatch surfaces at install time
    instead of silently disabling profiling at training time. The build is
    cached (torch extensions cache), so training-time loads are instant."""
    try:
        _load_cpp_ext()
    except Exception as e:
        sys.stderr.write(
            f"[torch_perf_analysis] observer build FAILED: "
            f"{type(e).__name__}: {e}\n"
        )
        return 1
    sys.stderr.write("[torch_perf_analysis] observer built and cached.\n")
    return 0
