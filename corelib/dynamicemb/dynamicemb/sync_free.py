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

"""Helpers that let DynamicEmb's per-step code run without a host sync.

The step code keeps its arrays at the length ``N`` the host already knows (the
number of looked-up keys) instead of cutting them to the number of unique keys,
which only the device knows. The part past the real entries, the tail, is made
inert:

* a key that must not take part in a step is the reserved all-ones key
  ``EMPTY_KEY`` (``-1`` as int64, ``0xFF..FF`` as uint64): every table kernel
  answers "not found" / "illegal" for it without touching a table;
* a row or slot that must not be touched is exactly ``-1``;
* a table id that does not exist is ``-1``.

``canonical_unique``, ``inert_evicted``, ``inert_collected``, ``mask_keys``,
``initialize_masked`` and ``padded_optimizer_table_ids`` produce and honour that
convention. Each uses what the compiled extension does itself when it was built
with it (the module flags below) and an equivalent PyTorch fallback otherwise;
the results are the same. The flags are read at call time through this module,
so tests set them to run the fallbacks.

The miss path allocates ``[rows, value_dim]`` buffers. Where they would not fit
``BUFFER_BYTES`` (``DYNAMICEMB_MISS_PATH_BUFFER_BYTES``) the path runs in consecutive row
chunks (``row_chunks``) instead of taking another code path: the number of launches
grows with the number of chunks, the number of reads does not.

What the host genuinely has to know is read late: ``DeferredRead`` is a ring of
pinned landing slots with events and never blocks. A read that cannot wait goes
through one audited place, ``read_now`` (``DeferredRead.wait`` and
``pull_to_host`` for public APIs that return host data), so grepping for those
names lists every blocking read of the package. ``sync_allowed()`` is True while
the calling thread is inside one of them.
"""

import contextlib
import os
import threading
import warnings
from typing import Callable, List, Optional, Tuple, Union

import dynamicemb_extensions
import torch

# The reserved key every table kernel rejects (``Bucket::is_valid``).
EMPTY_KEY: int = -1


def _extension_flag(name: str) -> bool:
    # An extension built before the attribute existed lacks it; the fallbacks
    # below give the same results.
    return bool(getattr(dynamicemb_extensions, name, False))


# segmented_unique / expand_table_ids already leave EMPTY_KEY, frequency 0 and
# table id -1 on the tail.
TAIL_INERT: bool = _extension_flag("segmented_unique_tail_inert")
# table_insert_and_evict / table_insert_collect_evicted already fill their
# evicted arrays past the eviction count with EMPTY_KEY, index -1, table id -1.
EVICT_TAIL_INERT: bool = _extension_flag("evicted_tail_inert")
# The initializers skip a row whose index is negative.
INIT_SKIPS_NEGATIVE: bool = _extension_flag("initializer_skips_negative_index")
# The padded-buffer optimizers skip a row whose table id is negative.
PADDED_OPT_SKIPS_NEGATIVE: bool = _extension_flag(
    "padded_optimizer_skips_negative_table_id"
)

# How a step decides whether its miss path has to run. Neither mode waits for the GPU:
#   "always": launch it every step and read nothing;
#   "poll":   skip it only when a non-blocking copy already landed a zero.
# "always" is the default: a step whose keys all hit runs the miss path on inert rows,
# which changes nothing, and the CPU never waits for the GPU. DYNAMICEMB_MISS_PATH_GATE
# selects the other. Nothing else does: neither the storage, an external one included, nor
# the determinism mode. What a class that counts its own rows reads is its contract
# (types.OldContractStorage), not the gate's.
_MISS_GATE_MODES = ("always", "poll")
_DEFAULT_MISS_GATE = "always"


def _miss_gate_from_env() -> str:
    mode = (os.environ.get("DYNAMICEMB_MISS_PATH_GATE") or _DEFAULT_MISS_GATE).lower()
    if mode == "read":
        # A mode of the first masked paths, which blocked on the count of misses. A
        # script that still asks for it keeps running, without the wait.
        warnings.warn(
            "DYNAMICEMB_MISS_PATH_GATE=read was removed: no miss-gate mode waits for the "
            f"GPU any more ({', '.join(_MISS_GATE_MODES)}); using {_DEFAULT_MISS_GATE!r}.",
            stacklevel=2,
        )
        return _DEFAULT_MISS_GATE
    if mode not in _MISS_GATE_MODES:
        raise ValueError(
            "DYNAMICEMB_MISS_PATH_GATE must be one of "
            f"{', '.join(_MISS_GATE_MODES)}, got {mode!r}."
        )
    return mode


MISS_GATE: str = _miss_gate_from_env()

# The bytes of ``[rows, value_dim]`` buffers one chunk of the miss path may allocate
# (``row_chunks``): 4 GiB, which is what the memory guard of the first masked paths
# allowed for two buffers of a whole batch.
_DEFAULT_BUFFER_BYTES = 4 << 30


def _buffer_bytes_from_env() -> int:
    text = os.environ.get("DYNAMICEMB_MISS_PATH_BUFFER_BYTES")
    if not text:
        return _DEFAULT_BUFFER_BYTES
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(
            "DYNAMICEMB_MISS_PATH_BUFFER_BYTES must be a positive number of bytes, "
            f"got {text!r}."
        )
    return value


BUFFER_BYTES: int = _buffer_bytes_from_env()
# Rows per chunk whatever the budget says: a test hook (a module attribute, read at call
# time, never an environment variable). 0 and None leave the budget in charge.
CHUNK_ROWS: Optional[int] = None
_WARNED_SMALL_CHUNK = False
# Buffers that are not split into chunks and were named once as over the budget.
_WARNED_UNCHUNKED: set = set()

# Variables of the first masked paths that nothing reads any more. A script that still sets
# one gets a warning and nothing else: no variable selects a path that reads.
_RETIRED_VARIABLES = {
    "DYNAMICEMB_SYNC_FREE": "a kill switch that sent every lookup down the paths that read "
    "counts from the device is gone, no variable selects such a path",
    "DYNAMICEMB_SYNC_FREE_MAX_BYTES": "the budget is DYNAMICEMB_MISS_PATH_BUFFER_BYTES now, "
    "and a batch over it is split into chunks, not sent down another path",
}


def _warn_retired_variables() -> None:
    for name, what in _RETIRED_VARIABLES.items():
        if os.environ.get(name) is not None:
            warnings.warn(
                f"{name} is not read any more: {what}.", UserWarning, stacklevel=3
            )


_warn_retired_variables()

# Number of read_now calls so far; tests snapshot it to assert how many blocking
# reads a step costs.
READ_NOW_CALLS: int = 0
_READ_NOW_LOCK = threading.Lock()

# Per-thread state: ``allow_sync`` is True while the thread is inside an audited
# blocking read, ``scratch`` holds the pinned landing buffers of read_now.
_TLS = threading.local()


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def as_i64(t: torch.Tensor) -> torch.Tensor:
    """``t`` as int64 without a copy; uint64 keys are reinterpreted bit for bit.

    uint64 has no advanced indexing in PyTorch and only partial ``where``
    support, so key arrays go through this view before they are gathered or
    selected.
    """
    if t.dtype == torch.int64:
        return t
    if t.dtype == torch.uint64:
        return t.view(torch.int64)
    raise TypeError(f"keys are int64 or uint64, got {t.dtype}")


def mask_keys(keys: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """``keys`` where ``keep`` is True, ``EMPTY_KEY`` elsewhere; same dtype and length.

    One launch. The lanes that are masked out become inert for every table
    kernel, so the call needs no compaction and no count.
    """
    masked = torch.where(keep, as_i64(keys), EMPTY_KEY)
    return masked if keys.dtype == torch.int64 else masked.view(keys.dtype)


def valid_keys(keys: torch.Tensor) -> torch.Tensor:
    """bool: the key is not ``EMPTY_KEY``.

    For metrics and evicted arrays, whose inert entries carry ``EMPTY_KEY``. It is
    not the ``valid`` mask of the unique arrays: that one comes from the table ids
    (the keys of a raw tail are unwritten).
    """
    return as_i64(keys) != EMPTY_KEY


# ---------------------------------------------------------------------------
# Padded arrays
# ---------------------------------------------------------------------------


def canonical_unique(
    unique_keys: torch.Tensor, unique_table_ids: torch.Tensor, num_tables: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make the raw ``N``-long unique arrays obey the padded-array convention.

    The inputs are the outputs of ``segmented_unique`` and ``expand_table_ids``.
    Returns ``(keys, table_ids, valid)``: ``valid`` is True for the real entries
    and the tail has ``EMPTY_KEY`` and table id ``-1`` (``-1`` wraps into range in
    a gather, ``num_tables`` would not).
    """
    if TAIL_INERT:
        return unique_keys, unique_table_ids, unique_table_ids >= 0
    # Decide from the ids before rewriting them: the tail's id is ``num_tables``
    # (keys unwritten) from an extension that predates the tail fill and ``-1``
    # from one that has it, which a test can still send down this branch. Neither
    # is a table.
    valid = (unique_table_ids >= 0) & (unique_table_ids < num_tables)
    return (
        mask_keys(unique_keys, valid),
        torch.where(valid, unique_table_ids, -1),
        valid,
    )


def inert_evicted(
    num_evicted: torch.Tensor,
    keys: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    table_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The evicted arrays of ``insert_and_evict``, tails inert; the count stays unread.

    The arrays are ``N`` long and only the first ``num_evicted`` (int64[1] on the
    device) entries are real. Past them: key ``EMPTY_KEY``, index ``-1``, score
    ``0`` and table id ``-1``. An entry below the count is kept as it is, a
    refused insert (index ``-(i+1)``) included.
    """
    if EVICT_TAIL_INERT:
        return keys, indices, scores, table_ids
    positions = torch.arange(keys.numel(), dtype=torch.int64, device=keys.device)
    live = positions < num_evicted
    return (
        mask_keys(keys, live),
        torch.where(live, indices, -1),
        torch.where(live, scores, 0),
        torch.where(live, table_ids, -1),
    )


def inert_collected(
    num_evicted: torch.Tensor, keys: torch.Tensor, table_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``inert_evicted`` for ``table_insert_collect_evicted``'s keys and table ids."""
    if EVICT_TAIL_INERT:
        return keys, table_ids
    positions = torch.arange(keys.numel(), dtype=torch.int64, device=keys.device)
    live = positions < num_evicted
    return mask_keys(keys, live), torch.where(live, table_ids, -1)


def initialize_masked(
    initializer: Callable[..., None],
    out: torch.Tensor,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    dense: bool = False,
) -> None:
    """Run ``initializer`` on the rows of ``out`` where ``mask`` is True, in place.

    Every other row of ``out`` is left untouched. ``out`` is ``[N, dim]`` with
    ``stride(1) == 1`` (a column slice of the value block is fine); ``keys``,
    ``table_ids`` and ``mask`` are ``N`` long, one entry per row of ``out``. A
    masked-out row may carry ``EMPTY_KEY`` and table id ``-1``.

    ``dense`` is for an initializer written for index lists that name only the rows to
    write (a user's ``non_admitted_initializer`` that did not declare
    ``skips_negative_index``): the number of rows in ``mask`` is read once, by that
    initializer's contract, and it is called with exactly those indices (not at all when
    there are none).
    """
    n = out.size(0)
    if n == 0:
        return
    rows = torch.arange(n, dtype=torch.int64, device=out.device)
    if dense:
        count = int(read_now(mask.sum())[0])
        if count:
            (picked,) = compact_by_count(mask, count, rows)
            initializer(out, keys, table_ids, picked)
        return
    if INIT_SKIPS_NEGATIVE:
        # A row whose index is -1 is skipped before anything else is read.
        initializer(out, keys, table_ids, torch.where(mask, rows, -1))
        return
    # Initialize every row of a scratch block, then copy back the masked ones, in row
    # chunks so that the scratch stays within the budget. The clamp keeps a table id of
    # -1 inside the table-parameter tensor; such a row is discarded anyway.
    for start, stop in row_chunks(n, out.size(1) * out.element_size(), buffers=1):
        scratch = torch.empty(
            stop - start, out.size(1), dtype=out.dtype, device=out.device
        )
        initializer(
            scratch,
            keys[start:stop],
            table_ids[start:stop].clamp(min=0),
            rows[: stop - start],
        )
        out[start:stop].copy_(
            torch.where(mask[start:stop].unsqueeze(1), scratch, out[start:stop])
        )


def padded_optimizer_table_ids(table_ids: torch.Tensor) -> torch.Tensor:
    """The table ids to hand ``update_for_padded_buffer`` for an ``N``-long batch.

    With support in the extension the tail's ``-1`` is skipped there. Without it
    the ids are clamped so the tail reads table 0's width: those rows are never
    persisted (their keys are ``EMPTY_KEY``), so what the optimizer does to them
    does not matter.
    """
    return table_ids if PADDED_OPT_SKIPS_NEGATIVE else table_ids.clamp(min=0)


def _take_rows(a: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if a.dtype == torch.uint64:
        return torch.index_select(a.view(torch.int64), 0, order).view(torch.uint64)
    return torch.index_select(a, 0, order)


def compact_by_count(
    mask: torch.Tensor, count: int, *arrays: Optional[torch.Tensor]
) -> List[Optional[torch.Tensor]]:
    """The rows of ``arrays`` where ``mask`` is True, in order, as exact-length arrays.

    ``count`` is the number of True entries as a host integer, which the caller
    read (``read_now``) from the same mask. A stable sort brings the selected
    rows to the front, so there is no ``nonzero`` and no boolean-mask index. A
    ``None`` array stays ``None``.

    Not dead code: the adapters of a participant that is host-sized by its contract
    compact with it (``types._live_rows``, ``initialize_masked(dense=True)``, and through
    ``_live_rows`` ``OldContractStorage`` and ``embedding_admission.admit_padded``).
    """
    selected_first = torch.sort(mask.to(torch.int8), descending=True, stable=True)
    order = selected_first.indices[:count]
    return [None if a is None else _take_rows(a, order) for a in arrays]


def row_chunks(n: int, row_bytes: int, buffers: int = 2) -> List[Tuple[int, int]]:
    """Consecutive ``(start, stop)`` covering ``[0, n)``, each short enough that
    ``buffers`` buffers of ``row_bytes`` per row fit ``BUFFER_BYTES`` (at least one row).

    ``[(0, n)]`` when everything fits, which is every batch the 4 GiB memory guard let
    through (``2 * n * row_bytes <= BUFFER_BYTES``), and for ``n == 0``. Pure host
    arithmetic: the length of a chunk never depends on the data.
    """
    rows = max(1, int(CHUNK_ROWS or BUFFER_BYTES // max(1, buffers * row_bytes)))
    if n <= rows:
        return [(0, n)]
    global _WARNED_SMALL_CHUNK
    if not CHUNK_ROWS and rows < 16 and not _WARNED_SMALL_CHUNK:
        # A budget this small is a misconfiguration: say so once.
        _WARNED_SMALL_CHUNK = True
        warnings.warn(
            f"DYNAMICEMB_MISS_PATH_BUFFER_BYTES={BUFFER_BYTES} allows {rows} row(s) "
            f"per chunk for {n} keys: the miss path will launch about "
            f"{-(-n // rows)} chunks per step.",
            UserWarning,
        )
    return [(start, min(start + rows, n)) for start in range(0, n, rows)]


def warn_if_unchunked_over_budget(
    n: int, row_bytes: int, what: str, buffers: int = 2
) -> None:
    """Say, once per ``what`` and process, that a buffer that is not split into chunks is
    over the budget: ``n`` rows of ``row_bytes`` (``buffers`` of them, as ``row_chunks``
    counts).

    Evaluation allocates the unique rows of a pooled lookup, and a DEFAULT-mode forward
    its value block, with one row per key of the batch, duplicates included: they cannot
    run in chunks (the gather that follows would have to accumulate across them), so
    ``DYNAMICEMB_MISS_PATH_BUFFER_BYTES`` does not bound them, and a batch that fits no
    longer sends them down a path of their own. The caller says so before it allocates,
    so that an allocation that fails has its cause named. Pure host arithmetic: nothing
    is read, and nothing changes but the warning.
    """
    if buffers * n * row_bytes <= BUFFER_BYTES or what in _WARNED_UNCHUNKED:
        return
    _WARNED_UNCHUNKED.add(what)
    warnings.warn(
        f"{what} holds {n} rows of {row_bytes} bytes at once ({buffers} such buffers are "
        f"over DYNAMICEMB_MISS_PATH_BUFFER_BYTES={BUFFER_BYTES}). It has one row per key of "
        "the batch, duplicates included, and is not split into chunks, so that budget does "
        "not bound it: if the allocation fails, send fewer keys per step.",
        UserWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Audited blocking reads
# ---------------------------------------------------------------------------


def sync_allowed() -> bool:
    """Whether the calling thread is inside one of the audited blocking reads.

    Those are ``read_now``, ``DeferredRead.wait`` and ``pull_to_host``. The test
    fixture that forbids every other host sync lets a call through while this is
    True.
    """
    return getattr(_TLS, "allow_sync", False)


@contextlib.contextmanager
def _allow_sync():
    previous = sync_allowed()
    _TLS.allow_sync = True
    try:
        yield
    finally:
        _TLS.allow_sync = previous


# What ``DeferredRead.enqueue`` hands out: an int, or None when no slot was free.
_Ticket = Optional[int]


class _Slot:
    """One landing slot of a ``DeferredRead``."""

    __slots__ = ("buf", "event", "ticket")

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype) -> None:
        self.buf = torch.empty(shape, dtype=dtype, pin_memory=True)
        self.event = torch.cuda.Event(blocking=True)
        # The ticket of the enqueue that owns the slot, None while it is free.
        self.ticket: _Ticket = None


class DeferredRead:
    """A ring of pinned landing slots for device values the host reads late.

    Pinned memory and events are allocated here, never in a step. ``enqueue``
    starts an asynchronous copy of a device tensor into a free slot and records
    the slot's event on the current stream; the host polls ``ready`` and reads
    ``value`` once it landed. Nothing blocks except ``wait``, which is for public
    APIs that return host data.

    A slot is busy from ``enqueue`` until ``release``. A released slot is reused
    only after its copy has landed, so releasing without waiting (a reset) is
    safe. One ring belongs to one owner (a table state, a module). A copy of the
    owner (``copy.deepcopy``, pickle) gets a fresh ring that does not know the
    original's tickets (``ready``, ``value`` and ``wait`` raise ``ValueError``
    for them), so an owner that keeps tickets of its own must not copy them.
    """

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...]] = (1,),
        dtype: torch.dtype = torch.int64,
        slots: int = 4,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        if slots < 1:
            raise ValueError(f"a ring needs at least one slot, got {slots}")
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.slots = int(slots)
        self._ring = [_Slot(self.shape, dtype) for _ in range(self.slots)]
        self._serial = 0
        # Re-entrant: a finalizer or signal handler on the same thread that
        # releases a ticket must not deadlock against an enqueue it interrupted.
        self._lock = threading.RLock()

    def enqueue(self, dev: torch.Tensor) -> _Ticket:
        """Start copying ``dev`` into a free slot and return the slot's ticket.

        ``dev`` is a CUDA tensor of this ring's ``shape`` and ``dtype``, produced on
        the current stream (otherwise call ``dev.record_stream`` with it first).
        Returns None when every slot is in flight: the caller decides later. It
        never blocks and never allocates.
        """
        if not dev.is_cuda or tuple(dev.shape) != self.shape or dev.dtype != self.dtype:
            raise ValueError(
                f"this ring takes a CUDA tensor of shape {self.shape} and dtype "
                f"{self.dtype}, got {tuple(dev.shape)} {dev.dtype} on {dev.device}"
            )
        with self._lock:
            for slot in self._ring:
                # A released slot may still have its copy in flight.
                if slot.ticket is None and slot.event.query():
                    break
            else:
                return None
            slot.buf.copy_(dev.detach(), non_blocking=True)
            slot.event.record(torch.cuda.current_stream(dev.device))
            self._serial += 1
            slot.ticket = self._serial
            return slot.ticket

    def _slot_of(self, ticket: _Ticket) -> _Slot:
        for slot in self._ring:
            if slot.ticket is not None and slot.ticket == ticket:
                return slot
        raise ValueError(f"ticket {ticket!r} is not in flight in this ring")

    def ready(self, ticket: int) -> bool:
        """Whether the copy of ``ticket`` has landed."""
        return self._slot_of(ticket).event.query()

    def value(self, ticket: int) -> torch.Tensor:
        """The slot's pinned CPU tensor; its contents are defined only once ``ready``.

        It is the slot itself, not a copy: clone it before ``release`` if it is kept.
        """
        return self._slot_of(ticket).buf

    def release(self, ticket: _Ticket) -> None:
        """Give the slot of ``ticket`` back. An unknown or None ticket is ignored."""
        with self._lock:
            for slot in self._ring:
                if slot.ticket is not None and slot.ticket == ticket:
                    slot.ticket = None

    def free_slots(self) -> int:
        """Slots not owned by a ticket (host bookkeeping only, no CUDA call)."""
        return sum(1 for slot in self._ring if slot.ticket is None)

    def wait(self, ticket: int) -> torch.Tensor:
        """Block until ``ticket`` landed and return its pinned CPU tensor.

        Only for public APIs that return host data (pop, flush, a metrics getter).
        """
        slot = self._slot_of(ticket)
        with _allow_sync():
            slot.event.synchronize()
        return slot.buf

    def __reduce__(self):
        # Pinned memory and events belong to one process and one device: a copy of
        # the owner (copy.deepcopy, pickle) gets a ring of its own.
        return (DeferredRead, (self.shape, self.dtype, self.slots))


def _read_scratch(
    device: torch.device, numel: int
) -> Tuple[torch.Tensor, torch.cuda.Event]:
    """The calling thread's pinned buffer and event for ``device``, grown if needed."""
    cache = getattr(_TLS, "scratch", None)
    if cache is None:
        cache = _TLS.scratch = {}
    entry = cache.get(device)
    if entry is None or entry[0].numel() < numel:
        have = 0 if entry is None else entry[0].numel()
        buf = torch.empty(max(numel, 2 * have, 64), dtype=torch.int64, pin_memory=True)
        event = torch.cuda.Event(blocking=True) if entry is None else entry[1]
        entry = cache[device] = (buf, event)
    return entry


def read_now(*dev_tensors: torch.Tensor) -> List[torch.Tensor]:
    """THE blocking read: the host needs these values now.

    Only inherent syncs come here: a class written against exact-length tensors
    that wants host sizes (an external storage, a user admitter, counter or
    initializer that did not declare the padded contract: ``types._live_rows``,
    ``initialize_masked(dense=True)``), a public API that returns host data. The
    tensors are flattened, cast to int64 and concatenated on the current stream,
    copied to host in one pinned transfer, and waited for with one event. Returns
    CPU int64 tensors shaped like the inputs; a CPU input is returned as it is. The
    pinned buffer is reused (one per thread and device, grown to the largest
    request), so a steady state allocates nothing; the results do not alias it.

    ``READ_NOW_CALLS`` counts the calls.
    """
    global READ_NOW_CALLS
    with _READ_NOW_LOCK:
        READ_NOW_CALLS += 1
    results: List[torch.Tensor] = list(dev_tensors)
    on_device = [i for i, t in enumerate(dev_tensors) if t.is_cuda]
    if not on_device:
        return results
    device = dev_tensors[on_device[0]].device
    flats = []
    for i in on_device:
        if dev_tensors[i].device != device:
            raise ValueError("read_now reads the tensors of one device per call")
        flats.append(dev_tensors[i].detach().reshape(-1).to(torch.int64))
    total = sum(f.numel() for f in flats)
    if total == 0:
        for i in on_device:
            results[i] = torch.empty(tuple(dev_tensors[i].shape), dtype=torch.int64)
        return results
    packed = flats[0] if len(flats) == 1 else torch.cat(flats)
    buf, event = _read_scratch(device, total)
    landing = buf[:total]
    landing.copy_(packed, non_blocking=True)
    event.record(torch.cuda.current_stream(device))
    with _allow_sync():
        event.synchronize()
    host = landing.clone()
    start = 0
    for i in on_device:
        n = dev_tensors[i].numel()
        results[i] = host[start : start + n].reshape(tuple(dev_tensors[i].shape))
        start += n
    return results


def pull_to_host(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """Copy CUDA tensors to pinned CPU tensors of the same dtype and shape.

    For cold paths only (checkpoint, export, pop): the destinations are allocated
    per call, one asynchronous copy is issued for each tensor and a single event
    wait follows. A CPU input is returned as it is.
    """
    results: List[torch.Tensor] = []
    devices: List[torch.device] = []
    for t in tensors:
        if not t.is_cuda:
            results.append(t)
            continue
        host = torch.empty(tuple(t.shape), dtype=t.dtype, pin_memory=True)
        host.copy_(t.detach(), non_blocking=True)
        results.append(host)
        if t.device not in devices:
            devices.append(t.device)
    events = []
    for device in devices:
        event = torch.cuda.Event(blocking=True)
        event.record(torch.cuda.current_stream(device))
        events.append(event)
    with _allow_sync():
        for event in events:
            event.synchronize()
    return results


class _StageSlot:
    """One staging buffer of a ``StagedUpload``, allocated on first use."""

    __slots__ = ("buf", "event")

    def __init__(self) -> None:
        self.buf = None
        self.event = None


class StagedUpload:
    """Host-to-device uploads through a ring of pinned staging buffers.

    For cold paths only (load, replay), where the data to upload is pageable CPU
    memory. ``upload`` copies a CPU tensor into the next staging buffer, starts an
    asynchronous copy to the device and records an event, so the host moves on to
    the next batch while the transfer runs. It waits only when it comes back to a
    buffer whose transfer is still in flight. Columns that share one object take
    turns in the ring; give each column its own object, or a longer ``ring``, to
    keep every one of them double buffered.
    """

    def __init__(self, device: Union[torch.device, str, int], ring: int = 2) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"StagedUpload uploads to a CUDA device, got {device}")
        if ring < 2:
            raise ValueError(f"a staging ring needs at least two buffers, got {ring}")
        self.device = device
        self.ring = int(ring)
        self._slots = [_StageSlot() for _ in range(self.ring)]
        self._next = 0

    def upload(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """``cpu_tensor`` on the device. The result is valid on the current stream."""
        if cpu_tensor.is_cuda:
            return cpu_tensor.to(self.device, non_blocking=True)
        if cpu_tensor.numel() == 0:
            return torch.empty(
                tuple(cpu_tensor.shape), dtype=cpu_tensor.dtype, device=self.device
            )
        slot = self._slots[self._next]
        self._next = (self._next + 1) % self.ring
        # The buffer is rewritten only after its previous transfer completed.
        if slot.event is not None and not slot.event.query():
            with _allow_sync():
                slot.event.synchronize()
        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
        if slot.buf is None or slot.buf.numel() < nbytes:
            have = 0 if slot.buf is None else slot.buf.numel()
            slot.buf = torch.empty(
                max(nbytes, 2 * have), dtype=torch.uint8, pin_memory=True
            )
        staged = slot.buf[:nbytes].view(cpu_tensor.dtype).view(cpu_tensor.shape)
        staged.copy_(cpu_tensor)
        uploaded = staged.to(self.device, non_blocking=True)
        if slot.event is None:
            slot.event = torch.cuda.Event(blocking=True)
        slot.event.record(torch.cuda.current_stream(self.device))
        return uploaded

    def __reduce__(self):
        # Staging buffers and events are not copied; see DeferredRead.__reduce__.
        return (StagedUpload, (self.device, self.ring))


# ---------------------------------------------------------------------------
# Miss-path gate
# ---------------------------------------------------------------------------


def miss_gate_mode(rng: bool) -> str:
    """How this step decides whether to launch its miss path: always or poll.

    ``MISS_GATE``, except that "poll" becomes "always" when the miss path draws random
    numbers (``rng``): a seeded probabilistic admitter must not consume a
    timing-dependent number of draws. Nothing else decides it, not the storage and not
    the determinism mode.
    """
    if rng and MISS_GATE == "poll":
        return "always"
    return MISS_GATE


# What ``MissGate.begin`` returns: None, or ``(ticket, miss count on the device)``.
_Pending = Optional[Tuple[_Ticket, torch.Tensor]]


class MissGate:
    """Decides whether a step's miss path has to be launched, without waiting for the GPU.

    One per table state; it owns its ring. The decision never changes a result:
    a miss path that finds no miss is a no-op, so a skipped launch only saves work.
    """

    def __init__(self) -> None:
        self.ring = DeferredRead((1,), torch.int64, slots=4)

    def begin(self, is_miss: torch.Tensor, *, rng: bool = False) -> _Pending:
        """Start counting the misses of ``is_miss`` (bool[N]) for ``needed``.

        Returns what ``needed`` takes: None for "always" (no device work) or when
        the ring has no free slot (nothing to poll, so the miss path simply runs),
        otherwise ``(ticket, count)``. Call it as early as possible, right after the
        mask exists, so a polled count has time to land.
        """
        if miss_gate_mode(rng) == "always":
            return None
        count = is_miss.sum().reshape(1)
        ticket = self.ring.enqueue(count)
        return None if ticket is None else (ticket, count)

    def needed(self, pending: _Pending) -> bool:
        """True when the miss path has to be launched: unless a polled count that has
        already landed is zero. A late copy costs a launch, never a result."""
        if pending is None:
            return True
        ticket, _ = pending
        landed_zero = (
            self.ring.ready(ticket) and _first_int(self.ring.value(ticket)) == 0
        )
        self.ring.release(ticket)
        return not landed_zero


class _AlwaysGate:
    """The gate of a storage that has no table state (an external one in DEFAULT mode):
    it has no ring to poll, so the miss path always runs, which is what ``MISS_GATE``
    "always" does for everyone else. Same interface as :class:`MissGate`."""

    def begin(self, is_miss: torch.Tensor, *, rng: bool = False) -> _Pending:
        return None

    def needed(self, pending: _Pending) -> bool:
        return True


ALWAYS_GATE = _AlwaysGate()


def _first_int(host_tensor: torch.Tensor) -> int:
    return int(host_tensor.reshape(-1)[0])
