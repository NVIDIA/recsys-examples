# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""What an erase that runs beside training is made of.

Three pieces, none of which knows what is being erased:

* ``TableAccessGate`` -- who may touch the tables right now, on the host (a lock)
  and on the device (a CUDA event handed from one holder to the next).
* ``AsyncEraser`` -- one background thread that runs the erase, so that the call
  that asks for it returns at once.
* ``EraseHandle`` -- what the caller keeps: wait for it, or make a stream wait.

The gate and the worker belong to one ``BatchedDynamicEmbeddingTablesV2`` -- the fused
tables of one storage, which is what an erase and a prefetch contend for. A model
(a TorchRec embedding collection, say) can hold several of them, each with its own.

``erase_by_score_async`` is the call a user makes on a model: it starts an erase on every
``BatchedDynamicEmbeddingTablesV2`` it finds that has a table to erase from, and returns
one handle for all of them. The method of the same name on
``BatchedDynamicEmbeddingTablesV2`` does it for that one.

An erase takes the gate once per chunk of its scan (see
``DynamicEmbStorage.erase_by_score``), a prefetch takes it around the part of it
that reads or changes the tables. Forward and backward do not take it: a key they
use was pinned by the prefetch, and an erase spares what is pinned.
"""

import collections
import contextlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, ContextManager, List, Optional, Sequence, Tuple

import torch
from dynamicemb.dynamicemb_config import DynamicEmbScoreStrategy
from dynamicemb.key_value_table import ERASE_CHUNK_SLOTS, ERASE_MAX_CHUNKS_IN_FLIGHT
from torch import nn


class FairLock:
    """A lock that goes to the threads that wait for it in the order they came.

    ``threading.Lock`` is free for an instant between one holder letting go and the
    woken waiter -- which the system still has to schedule, some microseconds -- taking
    it, and a thread that is running can take it in that instant. A thread that lets
    go and asks again at once (a worker launching chunk after chunk) therefore wins
    every time and a waiter is kept out until the loop ends. Here ``release`` hands the
    lock to the first waiter *itself*, and the lock is never free while someone waits:
    whoever asks next joins the queue behind. With two parties -- one training thread,
    one erase worker -- each waits for at most the section the other is in.

    Not reentrant: a thread that asks for a lock it holds waits for itself.
    """

    def __init__(self) -> None:
        self._state = threading.Lock()  # guards the two fields below, held for a moment
        self._held = False
        # One lock per waiter, taken by it, opened by the thread that hands the lock over.
        self._waiters: "collections.deque[threading.Lock]" = collections.deque()

    def acquire(self) -> None:
        with self._state:
            if not self._held and not self._waiters:
                self._held = True
                return
            baton = threading.Lock()
            baton.acquire()
            self._waiters.append(baton)
        try:
            self._wait_for_hand_over(baton)
        except BaseException:
            # Interrupted while waiting. Still queued: leave the queue. Already handed
            # the lock: it is ours and we are not going to use it, so pass it on.
            with self._state:
                try:
                    self._waiters.remove(baton)
                    handed_over = False
                except ValueError:
                    handed_over = True
            if handed_over:
                self.release()
            raise

    def _wait_for_hand_over(self, baton: "threading.Lock") -> None:
        baton.acquire()

    def release(self) -> None:
        with self._state:
            if not self._held:
                raise RuntimeError("release of a lock that is not held")
            if self._waiters:
                # Still held, now by the first waiter: nobody can slip in between.
                self._waiters.popleft().release()
            else:
                self._held = False

    def locked(self) -> bool:
        return self._held

    def __enter__(self) -> "FairLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class TableAccessGate:
    """Mutual exclusion on the tables of one ``BatchedDynamicEmbeddingTablesV2``, host side
    and device side.

    A lock orders the *launches*, not the kernels: two threads that launch on two
    streams in lock order may still run out of that order on the device. So the
    holder, on leaving, records an event on the stream it launched on, and the next
    holder's stream waits for it before anything it launches. The kernels of the
    holders are then one chain on the device, in the order of the lock, while the
    streams of forward and backward (which never enter) are not part of it.

    The lock is held while launching, not while waiting for the device, and it is fair
    (:class:`FairLock`): a thread that waits for it is let in after the section that
    holds it, not after however many the other thread can run in a row. What the
    other thread has already queued on the device is bounded by the erase itself (see
    ``ERASE_MAX_CHUNKS_IN_FLIGHT``).

    ``gate()`` returns the context manager of one section; a section runs on, and
    orders, the current stream of the thread that enters it.
    """

    def __init__(self) -> None:
        self._lock = FairLock()
        self._last_access: Optional[torch.cuda.Event] = None

    @contextlib.contextmanager
    def __call__(self):
        with self._lock:
            stream = torch.cuda.current_stream()
            if self._last_access is not None:
                stream.wait_event(self._last_access)
            try:
                yield
            finally:
                # Also after a failure: whatever was launched is still on the
                # stream, and the next holder has to come after it.
                event = torch.cuda.Event()
                event.record(stream)
                self._last_access = event

    def __reduce__(self):
        # A lock and an event belong to one process and one device; a copy of the
        # ``BatchedDynamicEmbeddingTablesV2`` (copy.deepcopy, pickle) starts with a gate of
        # its own.
        return (TableAccessGate, ())


class EraseHandle:
    """An erase that was asked for, to be waited for when its result is needed.

    The work is launched by a background thread, so there are two moments: the
    kernels are *launched* (the thread is done with them), then they *finish* on the
    device. ``wait_stream`` needs the first and orders a stream after the second;
    ``wait`` and ``result`` need both. Any of ``wait``, ``wait_stream``, ``done``
    and ``result`` raises what the worker raised.
    """

    def __init__(
        self,
        future: "Future[Optional[torch.Tensor]]",
        event: Optional[torch.cuda.Event],
    ) -> None:
        self._future = future
        self._event = event
        self._result: Optional[Tuple[int, int]] = None

    @classmethod
    def completed(cls) -> "EraseHandle":
        """A handle for work that had nothing to do: finished, and ``(0, 0)``."""
        future: "Future[Optional[torch.Tensor]]" = Future()
        future.set_result(None)
        handle = cls(future, None)
        handle._result = (0, 0)
        return handle

    def done(self) -> bool:
        """Whether the erase has finished on the device. Does not block."""
        if not self._future.done():
            return False
        self._future.result()  # raises what the worker raised
        return self._event is None or self._event.query()

    def wait(self) -> None:
        """Block the calling thread until the erase has finished on the device."""
        self._future.result()
        if self._event is not None:
            self._event.synchronize()

    def wait_stream(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Make ``stream`` (default: the current one) wait for the erase.

        Blocks the calling thread until the worker has launched every kernel -- an
        event that was not recorded yet orders nothing -- but not until they have
        run: that is left to the stream.
        """
        self._future.result()
        if self._event is not None:
            (stream or torch.cuda.current_stream()).wait_event(self._event)

    def result(self) -> Tuple[int, int]:
        """``(erased, spared)``: keys removed, and keys left because a batch held them.

        Blocks until the erase has finished.
        """
        if self._result is None:
            self.wait()
            counts = self._future.result()
            assert counts is not None
            erased, spared = counts.cpu().tolist()
            self._result = (int(erased), int(spared))
        return self._result


class EraseHandleGroup(EraseHandle):
    """The erases one call started, one on each ``BatchedDynamicEmbeddingTablesV2``, waited
    for as one.

    The same four questions as ``EraseHandle``, which it is a kind of, so that a caller
    holds one type whether it erased from one ``BatchedDynamicEmbeddingTablesV2`` or from a
    model; ``result`` adds the
    counts up. It has no future or event of its own, so it does not call the parent's
    ``__init__`` and answers every question through the handles it holds.
    """

    def __init__(self, handles: "list[EraseHandle]") -> None:  # noqa: super-init-not-called
        self._handles = list(handles)

    def done(self) -> bool:
        # Every one is asked, so that one that failed raises even if another is running.
        return all([handle.done() for handle in self._handles])

    def wait(self) -> None:
        for handle in self._handles:
            handle.wait()

    def wait_stream(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        for handle in self._handles:
            handle.wait_stream(stream)

    def result(self) -> Tuple[int, int]:
        results = [handle.result() for handle in self._handles]
        return (sum(erased for erased, _ in results), sum(spared for _, spared in results))


class AsyncEraser:
    """One background thread per ``BatchedDynamicEmbeddingTablesV2`` that runs its erases, one
    at a time, in order. (Erases of different ones run in parallel: they share nothing.)

    The thread is made when the first erase is asked for and ends with the process.
    It sets the device and the stream itself: both are per thread in CUDA and in
    PyTorch, so the caller's do not carry over.

    What the thread calls must not hold the GIL while it waits for the device (a
    binding that does would stop the training thread from launching anything), so
    it calls only bindings that release it and PyTorch operations.
    """

    def __init__(self) -> None:
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()
        self._streams: "dict[int, torch.cuda.Stream]" = {}

    def own_stream(self, device_id: int) -> torch.cuda.Stream:
        """The stream erases run on when the caller names none: one per device, made
        when first asked for and kept, so that it is no stream of the training."""
        with self._executor_lock:
            if device_id not in self._streams:
                self._streams[device_id] = torch.cuda.Stream(device=device_id)
            return self._streams[device_id]

    def submit(
        self,
        work: Callable[[torch.Tensor], None],
        stream: torch.cuda.Stream,
    ) -> EraseHandle:
        """Run ``work(counts)`` on the worker thread, with ``stream`` current.

        ``counts`` is two int64 on the stream's device, zero; ``work`` adds to them
        (erased, spared) and launches, it does not wait for the device. The handle
        comes back at once.
        """
        event = torch.cuda.Event()
        device = stream.device

        def run() -> torch.Tensor:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                counts = torch.zeros(2, dtype=torch.int64, device=device)
                work(counts)
                event.record(stream)
            return counts

        return EraseHandle(self._thread().submit(run), event)

    def _thread(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="dynamicemb-erase"
                )
            return self._executor

    def shutdown(self, wait: bool = True) -> None:
        """Let what was asked for finish, then end the thread. A later ``submit`` starts a new one."""
        with self._executor_lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait)

    def __reduce__(self):
        return (AsyncEraser, ())


def table_section(
    gate: Optional[Callable[[], ContextManager]],
) -> ContextManager:
    """The section of a gate, or nothing where there is no gate."""
    return contextlib.nullcontext() if gate is None else gate()


def check_table_names(table_names: Optional[Sequence[str]]) -> None:
    """Raise if ``table_names`` is not ``None`` or a non-empty sequence of names."""
    if isinstance(table_names, str):
        raise TypeError(
            f"table_names must be a sequence of table names, not a str: {table_names!r}."
        )
    if table_names is not None and len(table_names) == 0:
        raise ValueError(
            "table_names is empty: leave it out (None) to erase from every table "
            "that qualifies."
        )


def check_erase_arguments(
    score_strategy: DynamicEmbScoreStrategy,
    threshold: int,
    stream: Optional[torch.cuda.Stream],
    chunk_slots: int,
    max_in_flight: int,
    table_names: Optional[Sequence[str]] = None,
) -> None:
    """Raise if what an erase by score is asked for cannot mean anything, whatever the
    tables are. Both calls make it first, so a wrong argument is an error even where
    there is no table to erase from. (Whether the stream is on the tables' device, and
    whether ``table_names`` are tables that can be erased, is for each call to check, as
    it needs the tables.)"""
    check_table_names(table_names)
    if not isinstance(score_strategy, DynamicEmbScoreStrategy):
        raise TypeError(
            "score_strategy must be a DynamicEmbScoreStrategy, got "
            f"{type(score_strategy).__name__}."
        )
    if score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        raise ValueError(
            "NO_EVICTION keeps no score to erase by: name the strategy that "
            "scores the keys, e.g. CUSTOMIZED for (NO_EVICTION, CUSTOMIZED)."
        )
    if isinstance(threshold, bool) or not isinstance(threshold, int):
        raise TypeError(f"threshold must be an int, got {type(threshold).__name__}.")
    if not 0 <= threshold < 1 << 64:
        raise ValueError(f"threshold must be in [0, 2**64), got {threshold}.")
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError(
            f"stream must be a torch.cuda.Stream or None, got {type(stream).__name__}."
        )
    for name, value in (("chunk_slots", chunk_slots), ("max_in_flight", max_in_flight)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}.")


def erase_by_score_async(
    model: nn.Module,
    score_strategy: DynamicEmbScoreStrategy,
    threshold: int,
    stream: Optional[torch.cuda.Stream] = None,
    chunk_slots: int = ERASE_CHUNK_SLOTS,
    max_in_flight: int = ERASE_MAX_CHUNKS_IN_FLIGHT,
    table_names: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> EraseHandle:
    """Erase, in the background of training, the keys whose score is below ``threshold``
    from every dynamic embedding table of ``model`` that scores its keys by ``score_strategy``,
    or only from the tables named in ``table_names``.

    A table qualifies when its score strategy has ``score_strategy`` as a member, so
    ``CUSTOMIZED`` takes ``CUSTOMIZED`` and ``(NO_EVICTION, CUSTOMIZED)`` tables, and both
    ``TIMESTAMP`` and ``LFU`` take ``(TIMESTAMP, LFU)`` tables; a table with no score
    (``NO_EVICTION`` alone) never does. With no table that qualifies the call is not an
    error: the handle is done at once and its result is ``(0, 0)``.

    A ``(TIMESTAMP, LFU)`` table keeps two score words, its last-access timestamp and its
    frequency; ``score_strategy`` says which one is compared (either tuple order), the
    other is not looked at, and an erased key loses both.

    Parameters
    ----------
    model : nn.Module
        The model containing dynamic embedding tables.
    score_strategy : DynamicEmbScoreStrategy
        Which tables to erase from, and how their scores read: ``threshold`` is in the
        unit of that score. For ``TIMESTAMP`` take it from
        ``dynamicemb.device_timestamp()``; for ``(NO_EVICTION, CUSTOMIZED)`` and
        ``(NO_EVICTION, PRECISE_CUSTOMIZED)`` tables, the scores of live keys must be at
        least 1, because an erased slot is cleared to 0. For ``LFU`` the score is the raw
        access count, which never decays (a ``score_function`` only changes how evictions
        rank), and a key just admitted has a small one: an erase by ``LFU`` alone also
        removes the keys that came in recently.
    threshold : int
        A key is erased when its score is *strictly below* it. In ``[0, 2**64)``.
    stream : torch.cuda.Stream, optional
        The stream the erase launches on, on the tables' device. By default each
        ``BatchedDynamicEmbeddingTablesV2``
        uses a stream of its own, made on first use and kept, which lets the erase overlap
        the training. The training's stream, or the prefetch stream, also works, the erase
        then queueing among the training's kernels without overlap. Work the training has
        already queued on its stream is not waited for either way: the erase is best
        effort, and a prefetch that already ran is ahead of it.
    chunk_slots : int
        The most slots one chunk of the scan covers, in whole buckets (a number below a
        bucket is a bucket). The training may take the tables between two chunks, and a
        chunk's kernel is how long its own kernels can wait: a smaller chunk disturbs the
        training less and the scan takes longer. Default ``ERASE_CHUNK_SLOTS`` (2**22).
    max_in_flight : int
        At most this many chunks are launched and not yet finished on the device. What
        the training takes the tables for next waits for them: fewer disturb it less and
        slow the scan more. Default 2; at least 1.
    table_names : Sequence[str], optional
        Erase only from these tables, by name, wherever they are in ``model``; each must
        qualify for ``score_strategy``. ``None`` (the default) is every table that
        qualifies. A name that is in no module, or names a table that does not qualify,
        and an empty list, are errors. Tables that need different thresholds -- a TTL of a
        day on one and of 90 days on another -- take one call each.
    strict : bool
        With ``strict=False``, a name in ``table_names`` that no module of this
        rank's ``model`` holds is skipped instead of being an error, so one call
        works on every rank when tables are not on every rank (``TABLE_ROW_WISE``).
        A rank that holds none of the names erases nothing and its result is
        ``(0, 0)``. A name that is held but does not qualify is an error either
        way. Default ``True``.

    Returns
    -------
    EraseHandle
        ``done()``, ``wait()``, ``wait_stream(stream)`` and ``result()``, which is
        ``(erased, spared)`` summed over the ``BatchedDynamicEmbeddingTablesV2`` it started on. ``spared`` are keys left because a
        prefetched batch still held them. A failure in the background is raised from those
        calls.

    Notes
    -----
    Everything is checked before anything starts: an error leaves nothing erased.
    Supported are HBM-only storage and a cache over host storage (only the storage behind
    the cache is erased; the cache is not read or changed). ``HybridStorage``,
    host-only storage and external storage (PS) are refused when a table qualifies.

    The erase can overlap training (forward, backward, prefetch) and nothing else: wait
    for the handle before :func:`DynamicEmbDump`, :func:`DynamicEmbLoad`, an incremental
    dump, a replay or ``pop_*``. Best effort: it is not atomic with a training step.
    """
    # Not at the top of the file: dump_load imports the tables, and they import this file.
    from dynamicemb.dump_load import get_dynamic_emb_module

    check_erase_arguments(
        score_strategy, threshold, stream, chunk_slots, max_in_flight, table_names
    )
    modules: List[Tuple[nn.Module, Optional[List[str]]]] = []
    found = set()
    for module in get_dynamic_emb_module(model):
        names = None
        if table_names is not None:
            names = [name for name in table_names if name in module.table_names]
            if not names:
                continue
            found.update(names)
        if module.get_erase_table_ids(score_strategy, names):
            if stream is not None and stream.device != torch.device(
                "cuda", module.device_id
            ):
                raise ValueError(
                    f"stream is on {stream.device}, the tables are on cuda:{module.device_id}."
                )
            modules.append((module, names))
    if table_names is not None and strict:
        missing = [name for name in table_names if name not in found]
        if missing:
            raise ValueError(
                f"table_names {missing} are not tables of any dynamic embedding module "
                "of the model."
            )
    # Everything was accepted above, so each of these starts.
    return EraseHandleGroup(
        [
            module.erase_by_score_async(
                score_strategy,
                threshold,
                stream,
                chunk_slots,
                max_in_flight,
                table_names=names,
            )
            for module, names in modules
        ]
    )
