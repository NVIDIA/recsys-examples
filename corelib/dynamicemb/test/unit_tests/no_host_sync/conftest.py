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

"""Shared helpers and fixtures of the no-host-sync tests.

Nothing at module level needs torch, dynamicemb or a GPU: they are imported inside
functions and fixtures, so the CPU-only tests of this directory (the AST lint, the
checks of budgets.json, the probe's trace analysis) collect and run anywhere, and the
GPU tests skip with a reason.

What is here:

* ``no_sync()``: a context manager that turns a host sync inside it into an error.
  PyTorch's sync-debug mode raises for ``.item()``, ``.cpu()``, ``nonzero``, a
  pageable D2H copy and the like, but not for ``torch.cuda.synchronize()`` or
  ``Event.synchronize()``; those, and ``Tensor.item/tolist/__bool__/cpu`` on a CUDA
  tensor, are patched here to raise ``AssertionError("host sync outside
  sf.read_now")`` unless they are issued by ``sf.read_now``, ``sf.pull_to_host`` or
  ``DeferredRead.wait`` (the audited reads).  It cannot see a sync raised inside a
  C++ extension call: that is what the probe's census is for.  Build the inputs
  before the ``with`` and look at the outputs after it, since ``repr`` of a CUDA
  tensor reads it.
* ``count_syncs(fn)``: the probe's census of one callable.
* fixtures: ``no_sync``, ``count_syncs`` (the two above), ``gpu``, ``sf`` and ``bdf``
  (torch, dynamicemb.sync_free and the lookup-function module, or a skip), ``instruments_ok`` (the probe's
  self test, so a zero means something), ``census`` (one probe report per config,
  shared by the ratchet and the acceptance test), ``budgets`` and ``read_now_calls``
  (how many times ``sf.read_now`` ran).

Tests in other directories can ``sys.path.insert`` this directory and
``from conftest import no_sync``.
"""
import contextlib
import functools
import importlib
import os
import sys
import threading
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import host_sync_probe as hsp  # noqa: E402
import sync_budgets as sb  # noqa: E402


# --------------------------------------------------------------------------
# optional imports
# --------------------------------------------------------------------------
def optional_import(name: str) -> Any:
    """``importlib.import_module(name)``, or skip the running test (fixture time)."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


def _need_cuda() -> Any:
    torch = optional_import("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch


# --------------------------------------------------------------------------
# no_sync(): host syncs inside the block are errors
# --------------------------------------------------------------------------
_ALLOW = threading.local()  # depth of audited reads (sf.read_now, ...) on this thread
_PATCH_LOCK = threading.Lock()
_PATCH_DEPTH = 0
_UNDO: List[Tuple[Any, str, Any, bool]] = []
_ALLOWED_CODES: set = set()

# What may block, by name, in dynamicemb.sync_free (the audited reads).
_SF_FUNCTIONS = ("read_now", "pull_to_host")
_SF_METHODS = (("DeferredRead", "wait"),)


def _called_from_audited_read() -> bool:
    """True when an audited read of sync_free is on the call stack. This catches a
    sync issued by a function the caller imported by name (so that patching the module
    attribute did not wrap it)."""
    frame = sys._getframe(2)
    for _ in range(16):
        if frame is None:
            return False
        if frame.f_code in _ALLOWED_CODES:
            return True
        frame = frame.f_back
    return False


def _sync_allowed() -> bool:
    return getattr(_ALLOW, "depth", 0) > 0 or _called_from_audited_read()


def _forbid(label: str, orig: Callable, only_cuda_tensors: bool) -> Callable:
    @functools.wraps(orig)
    def guard(*args, **kwargs):
        # a Tensor method called on a CPU tensor is not a sync
        may_sync = not only_cuda_tensors or args[0].is_cuda
        if may_sync and not _sync_allowed():
            raise AssertionError("host sync outside sf.read_now: %s" % label)
        return orig(*args, **kwargs)

    return guard


def _allow(orig: Callable) -> Callable:
    """Mark `orig` (an audited read) as the one place a sync may happen, and keep
    PyTorch's sync-debug mode quiet inside it."""

    @functools.wraps(orig)
    def audited(*args, **kwargs):
        import torch

        _ALLOW.depth = getattr(_ALLOW, "depth", 0) + 1
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("default")
        try:
            return orig(*args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(mode)
            _ALLOW.depth -= 1

    return audited


def _patch(owner: Any, name: str, wrap: Callable[[Callable], Callable]) -> None:
    had_own = name in vars(owner)
    orig = getattr(owner, name)
    setattr(owner, name, wrap(orig))
    _UNDO.append((owner, name, orig, had_own))


def _audit(owner: Any, name: str) -> None:
    """Let the audited read ``owner.name`` of sync_free sync."""
    fn = getattr(owner, name, None)
    if fn is None:
        return
    code = getattr(getattr(fn, "__wrapped__", fn), "__code__", None)
    if code is not None:
        _ALLOWED_CODES.add(code)
    _patch(owner, name, _allow)


def _install_guards() -> None:
    import torch

    try:
        sf = importlib.import_module("dynamicemb.sync_free")
    except ImportError:  # a tree without the sync-free helpers: every sync is forbidden
        sf = None
    if sf is not None:
        for name in _SF_FUNCTIONS:
            _audit(sf, name)
        for cls_name, name in _SF_METHODS:
            if hasattr(sf, cls_name):
                _audit(getattr(sf, cls_name), name)
    for owner, name, label in (
        (torch.cuda.Event, "synchronize", "Event.synchronize"),
        (torch.cuda.Stream, "synchronize", "Stream.synchronize"),
        (torch.cuda, "synchronize", "torch.cuda.synchronize"),
    ):
        _patch(owner, name, lambda f, lbl=label: _forbid(lbl, f, False))
    for name in ("item", "tolist", "__bool__", "cpu"):
        _patch(
            torch.Tensor,
            name,
            lambda f, lbl="Tensor.%s on a CUDA tensor" % name: _forbid(lbl, f, True),
        )


def _remove_guards() -> None:
    while _UNDO:
        owner, name, orig, had_own = _UNDO.pop()
        if had_own:
            setattr(owner, name, orig)
        else:
            delattr(owner, name)
    _ALLOWED_CODES.clear()


@contextlib.contextmanager
def no_sync(debug_mode: str = "error"):
    """Run the block with host syncs turned into errors (see the module docstring)."""
    global _PATCH_DEPTH
    import torch

    previous = torch.cuda.get_sync_debug_mode()
    with _PATCH_LOCK:
        if _PATCH_DEPTH == 0:
            _install_guards()
        _PATCH_DEPTH += 1
    torch.cuda.set_sync_debug_mode(debug_mode)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)
        with _PATCH_LOCK:
            _PATCH_DEPTH -= 1
            if _PATCH_DEPTH == 0:
                _remove_guards()


# --------------------------------------------------------------------------
# count_syncs(): the probe's census of one callable
# --------------------------------------------------------------------------
class SyncCount(NamedTuple):
    total: float  # mean syncs per call, every kind (allocations included)
    blocking: float  # mean per call of ss + ds + es + ms + pd (what the ratchet bounds)
    by_kind: Dict[str, float]  # ss ds es ms pd al, see host_sync_probe.KIND_DOC
    per_iter: List[int]
    sites: List[Dict[str, Any]]  # ranked call sites, see host_sync_probe._summarize

    def describe(self, top: int = 6) -> str:
        """A compact text for assertion messages."""
        rows = [
            "%s syncs/call at %s" % (("%.3g" % s["per_iter"]), s["site"])
            for s in self.sites[:top]
        ]
        return "total=%.3g blocking=%.3g kinds=%s\n  %s" % (
            self.total,
            self.blocking,
            {k: v for k, v in self.by_kind.items() if v},
            "\n  ".join(rows) or "(no call site)",
        )


def count_syncs(fn: Callable[[int], Any], iters: int = 4, warmup: int = 2) -> SyncCount:
    """Mean host syncs per call of ``fn(i)``: ``warmup`` unmeasured calls (i = -1),
    then ``iters`` measured ones, by the probe's instruments (CUDA profiler, sync-debug
    warnings, interposed synchronize calls)."""
    r = hsp.measure_callable(fn, iters=iters, warmup=warmup)
    mean = r["mean"]
    return SyncCount(
        total=mean["total"],
        blocking=mean["blocking"],
        by_kind={k: mean[k] for k in hsp.KINDS},
        per_iter=list(r["per_iter_total"]),
        sites=r["sites"],
    )


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def gpu():
    """torch, once a CUDA device is known to be there (the test skips otherwise)."""
    return _need_cuda()


@pytest.fixture
def sf():
    """dynamicemb.sync_free (the test skips on a tree that does not have it yet)."""
    _need_cuda()
    return optional_import("dynamicemb.sync_free")


@pytest.fixture
def bdf():
    """dynamicemb.batched_dynamicemb_function (the test skips without a GPU or the
    built extension)."""
    _need_cuda()
    return optional_import("dynamicemb.batched_dynamicemb_function")


@pytest.fixture(name="no_sync")
def _no_sync_fixture():
    _need_cuda()
    return no_sync


@pytest.fixture(name="count_syncs")
def _count_syncs_fixture():
    _need_cuda()
    return count_syncs


@pytest.fixture(scope="session")
def instruments_ok():
    """The probe's self test: known syncs are seen and the negative controls read
    zero. Without it a zero census proves nothing."""
    _need_cuda()
    optional_import("dynamicemb")
    ok, rows = hsp.selftest(iters=4, verbose=False)
    if ok:
        return True
    bad = [r for r in rows if not r.get("ok", True)]
    # A dead profiler reads zero for the negative controls and fails every
    # positive one with a note that no iteration window was seen (or raises).
    dead = [
        r
        for r in bad
        if any(("raised" in n) or ("saw 0/" in n) for n in r.get("notes", []))
    ]
    detail = "\n".join(
        "  %s: expected %s got %s %s"
        % (r["case"], r["expected"], r["got"], r.get("notes", ""))
        for r in bad
    )
    if len(dead) == len(bad) and os.environ.get("HSP_REQUIRE_PROFILER") != "1":
        pytest.skip(
            "the CUDA profiler does not run here, so no census can be taken "
            "(HSP_REQUIRE_PROFILER=1 makes this an error):\n" + detail
        )
    pytest.fail(
        "host_sync_probe's self test failed, so a zero would mean nothing:\n" + detail
    )


class Census:
    """One probe report per config, kept for the session: the ratchet and the
    acceptance test read the same measurement."""

    def __init__(self) -> None:
        self._reports: Dict[Tuple, Dict[str, Any]] = {}

    def get(self, config: str) -> Dict[str, Any]:
        scenarios, modes = sb.census_scenarios(), sb.census_modes()
        iters, warmup = sb.census_iters(), sb.census_warmup()
        key = (config, tuple(scenarios or ()), modes, iters, warmup)
        if key not in self._reports:
            self._reports[key] = hsp.measure(
                [config],
                scenarios=scenarios,
                modes=modes,
                iters=iters,
                warmup=warmup,
                seed=sb.CENSUS_SEED,
            )
        return self._reports[key]


@pytest.fixture(scope="session")
def census(instruments_ok):
    return Census()


@pytest.fixture(scope="session")
def budgets():
    """(meta, budgets) of budgets.json."""
    return sb.load_budgets()


@pytest.fixture
def read_now_calls():
    """How many times ``sf.read_now`` ran since the fixture was made."""
    sf = optional_import("dynamicemb.sync_free")

    class Delta:
        start = sf.READ_NOW_CALLS

        @property
        def delta(self) -> int:
            return sf.READ_NOW_CALLS - self.start

    return Delta()
