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

"""The helpers of dynamicemb/sync_free.py, the one home of everything the sync-free paths share.

Pure helpers (``as_i64``, ``mask_keys``, ``valid_keys``, ``compact_by_count``,
``canonical_unique``, ``miss_gate_mode``) are tested on CPU tensors as well as on the GPU. The
rest needs a CUDA device: the inert evicted arrays, ``initialize_masked``, the ring of pinned
reads (``DeferredRead``: slots, wrap, never blocking behind a long kernel, ``ready`` flipping,
a copied module getting a fresh ring), ``read_now`` (one blocking read, counted), ``pull_to_host``,
``StagedUpload`` and ``MissGate``.

Each fallback is exercised by setting the module flag the extension would set
(``TAIL_INERT``, ``EVICT_TAIL_INERT``, ``INIT_SKIPS_NEGATIVE``, ``PADDED_OPT_SKIPS_NEGATIVE``), so
both branches are tested on whatever extension is built. The flags are read at call time through
the module, which is what these tests rely on. The file skips on a tree without
dynamicemb/sync_free.py.
"""
import copy
import inspect
import pickle
import time

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)
try:
    from dynamicemb import sync_free as sf
except ImportError as e:
    pytest.skip(
        "dynamicemb.sync_free is not in this tree (yet): %s" % (e,),
        allow_module_level=True,
    )

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
DEVICES = ["cpu"] + (["cuda"] if HAS_CUDA else [])
KEY_DTYPES = [torch.int64, torch.uint64]
EMPTY = -1


def keys_of(values, dtype, device):
    """Key tensor of `dtype` holding the given int64 bit patterns."""
    return torch.tensor(values, dtype=torch.int64, device=device).view(dtype)


def as_i64_cpu(t):
    t = t.detach().cpu()
    return t.view(torch.int64) if t.dtype == torch.uint64 else t


# --------------------------------------------------------------------------
# keys and masks
# --------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_as_i64_is_a_view(device):
    plain = torch.tensor([5, -1, 7], dtype=torch.int64, device=device)
    out = sf.as_i64(plain)
    assert (
        out.dtype == torch.int64
        and out.data_ptr() == plain.data_ptr()
        and torch.equal(out, plain)
    )

    unsigned = keys_of([5, -1, -4, 1 << 62], torch.uint64, device)
    out = sf.as_i64(unsigned)
    assert out.dtype == torch.int64 and out.data_ptr() == unsigned.data_ptr(), "no copy"
    assert out.tolist() == [
        5,
        -1,
        -4,
        1 << 62,
    ], "the bit patterns, so 0xFF..FF reads -1"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", KEY_DTYPES)
def test_mask_keys_replaces_the_rest_by_the_reserved_key(device, dtype):
    values = [5, -1, 7, 9, 1 << 62, -(1 << 62), 0, 3]
    keys = keys_of(values, dtype, device)
    keep = torch.tensor(
        [True, True, False, True, False, True, False, True], device=device
    )
    out = sf.mask_keys(keys, keep)
    assert out.dtype == dtype and out.shape == keys.shape
    expected = [v if k else EMPTY for v, k in zip(values, keep.tolist())]
    assert (
        as_i64_cpu(out).tolist() == expected
    ), "EMPTY is -1 as int64 and 0xFF..FF as uint64"
    assert as_i64_cpu(keys).tolist() == values, "the input is not modified"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", KEY_DTYPES)
def test_valid_keys_is_everything_but_the_reserved_key(device, dtype):
    keys = keys_of([5, EMPTY, 7, -2, 0], dtype, device)
    out = sf.valid_keys(keys)
    assert out.dtype == torch.bool
    assert out.cpu().tolist() == [True, False, True, True, True]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pattern", ["mixed", "none", "all"])
def test_compact_by_count_keeps_the_rows_of_the_mask_in_order(device, pattern):
    n = 12
    g = torch.Generator().manual_seed(3)
    mask_cpu = {
        "mixed": torch.rand(n, generator=g) < 0.5,
        "none": torch.zeros(n, dtype=torch.bool),
        "all": torch.ones(n, dtype=torch.bool),
    }[pattern]
    mask = mask_cpu.to(device)
    count = int(mask_cpu.sum())
    positions = torch.arange(n, device=device)
    unsigned_keys = keys_of(list(range(100, 100 + n)), torch.uint64, device)
    values = torch.arange(n * 3, dtype=torch.float32, device=device).reshape(n, 3)
    out = sf.compact_by_count(mask, count, positions, unsigned_keys, values, None)
    assert len(out) == 4 and out[3] is None
    keep = mask_cpu
    assert out[0].cpu().tolist() == positions.cpu()[keep].tolist()
    assert as_i64_cpu(out[1]).tolist() == as_i64_cpu(unsigned_keys)[keep].tolist()
    assert out[1].dtype == torch.uint64
    assert out[2].shape == (count, 3) and torch.equal(out[2].cpu(), values.cpu()[keep])


# --------------------------------------------------------------------------
# the padded unique arrays
# --------------------------------------------------------------------------
def check_canonical(keys, ids, valid, u):
    """keys[i] == EMPTY <=> ids[i] == -1 <=> not valid[i] <=> i >= U."""
    tail = (torch.arange(keys.numel()) >= u).tolist()
    assert (as_i64_cpu(keys) == EMPTY).tolist() == tail
    assert (ids.cpu() == EMPTY).tolist() == tail
    assert (~valid.cpu()).tolist() == tail


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", KEY_DTYPES)
@pytest.mark.parametrize("u", [0, 3, 8])
def test_canonical_unique_rewrites_the_tail_the_unpatched_kernels_leave(
    monkeypatch, device, dtype, u
):
    """TAIL_INERT False: the tail of the raw output holds live-looking keys and `num_tables` as
    table id; the fallback turns it inert."""
    monkeypatch.setattr(sf, "TAIL_INERT", False)
    num_tables, n = 3, 8
    head_keys = [11, 12, 13, 14, 15, 16, 17, 18][:u]
    head_ids = [0, 0, 1, 1, 1, 2, 2, 2][:u]
    keys = keys_of(head_keys + [99, 98, 97, 96, 95, 94, 93, 92][: n - u], dtype, device)
    ids = torch.tensor(
        head_ids + [num_tables] * (n - u), dtype=torch.int64, device=device
    )
    out_keys, out_ids, valid = sf.canonical_unique(keys, ids, num_tables)
    assert (
        out_keys.dtype == dtype
        and out_keys.shape == keys.shape
        and valid.dtype == torch.bool
    )
    assert as_i64_cpu(out_keys).tolist()[:u] == head_keys
    assert out_ids.cpu().tolist()[:u] == head_ids
    check_canonical(out_keys, out_ids, valid, u)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", KEY_DTYPES)
@pytest.mark.parametrize("u", [0, 3, 8])
def test_canonical_unique_fallback_accepts_the_tail_of_a_patched_kernel(
    monkeypatch, device, dtype, u
):
    """TAIL_INERT False on an extension that did make the tail inert (EMPTY keys and the id -1,
    the capability turned off on a rebuilt extension to run the fallbacks). `ids < num_tables` is
    true of -1, so the fallback has to class a negative id as not a table too; the result is the
    one of the TAIL_INERT path."""
    num_tables, n = 3, 8
    head_keys = [11, 12, 13, 14, 15, 16, 17, 18][:u]
    head_ids = [0, 0, 1, 1, 1, 2, 2, 2][:u]
    keys = keys_of(head_keys + [EMPTY] * (n - u), dtype, device)
    ids = torch.tensor(head_ids + [-1] * (n - u), dtype=torch.int64, device=device)
    monkeypatch.setattr(sf, "TAIL_INERT", False)
    out_keys, out_ids, valid = sf.canonical_unique(keys, ids, num_tables)
    assert as_i64_cpu(out_keys).tolist() == as_i64_cpu(keys).tolist()
    assert out_ids.cpu().tolist() == ids.cpu().tolist()
    check_canonical(out_keys, out_ids, valid, u)
    monkeypatch.setattr(sf, "TAIL_INERT", True)
    keys_t, ids_t, valid_t = sf.canonical_unique(keys, ids, num_tables)
    assert as_i64_cpu(keys_t).tolist() == as_i64_cpu(out_keys).tolist()
    assert ids_t.cpu().tolist() == out_ids.cpu().tolist()
    assert valid_t.cpu().tolist() == valid.cpu().tolist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", KEY_DTYPES)
@pytest.mark.parametrize("u", [0, 3, 8])
def test_canonical_unique_trusts_an_inert_tail(monkeypatch, device, dtype, u):
    """TAIL_INERT True: the kernels did it; `valid` is the table id being a table."""
    monkeypatch.setattr(sf, "TAIL_INERT", True)
    n = 8
    head_keys = [11, 12, 13, 14, 15, 16, 17, 18][:u]
    keys = keys_of(head_keys + [EMPTY] * (n - u), dtype, device)
    ids = torch.tensor(
        [0, 0, 1, 1, 1, 2, 2, 2][:u] + [-1] * (n - u), dtype=torch.int64, device=device
    )
    out_keys, out_ids, valid = sf.canonical_unique(keys, ids, 3)
    assert as_i64_cpu(out_keys).tolist() == as_i64_cpu(keys).tolist()
    assert out_ids.cpu().tolist() == ids.cpu().tolist()
    check_canonical(out_keys, out_ids, valid, u)


@cuda_only
@pytest.mark.parametrize("dtype", KEY_DTYPES)
@pytest.mark.parametrize("count", [0, 3, 8])
@pytest.mark.parametrize("evict_tail_inert", [False, True])
def test_inert_evicted(monkeypatch, dtype, count, evict_tail_inert):
    monkeypatch.setattr(sf, "EVICT_TAIL_INERT", evict_tail_inert)
    n = 8
    keys = keys_of(list(range(100, 100 + n)), dtype, "cuda")
    indices = torch.arange(50, 50 + n, dtype=torch.int64, device="cuda")
    scores = torch.arange(7, 7 + n, dtype=torch.int64, device="cuda")
    ids = torch.arange(1, 1 + n, dtype=torch.int64, device="cuda") % 3
    num_evicted = torch.tensor([count], dtype=torch.int64, device="cuda")
    out = sf.inert_evicted(num_evicted, keys, indices, scores, ids)
    assert len(out) == 4
    o_keys, o_indices, o_scores, o_ids = out
    assert o_keys.dtype == dtype
    if (
        evict_tail_inert
    ):  # the kernel filled the tail: the arrays are handed on as they are
        assert as_i64_cpu(o_keys).tolist() == as_i64_cpu(keys).tolist()
        assert (
            o_indices.tolist() == indices.tolist()
            and o_scores.tolist() == scores.tolist()
        )
        assert o_ids.tolist() == ids.tolist()
        return
    assert as_i64_cpu(o_keys).tolist() == list(range(100, 100 + count)) + [EMPTY] * (
        n - count
    )
    assert o_indices.tolist() == list(range(50, 50 + count)) + [-1] * (n - count)
    assert o_scores.tolist() == list(range(7, 7 + count)) + [0] * (n - count)
    assert o_ids.tolist() == ids.tolist()[:count] + [-1] * (n - count)


@cuda_only
@pytest.mark.parametrize("dtype", KEY_DTYPES)
@pytest.mark.parametrize("count", [0, 5, 8])
@pytest.mark.parametrize("evict_tail_inert", [False, True])
def test_inert_collected(monkeypatch, dtype, count, evict_tail_inert):
    monkeypatch.setattr(sf, "EVICT_TAIL_INERT", evict_tail_inert)
    n = 8
    keys = keys_of(list(range(100, 100 + n)), dtype, "cuda")
    ids = torch.arange(1, 1 + n, dtype=torch.int64, device="cuda") % 3
    num_evicted = torch.tensor([count], dtype=torch.int64, device="cuda")
    o_keys, o_ids = sf.inert_collected(num_evicted, keys, ids)
    assert o_keys.dtype == dtype
    if evict_tail_inert:
        assert (
            as_i64_cpu(o_keys).tolist() == as_i64_cpu(keys).tolist()
            and o_ids.tolist() == ids.tolist()
        )
        return
    assert as_i64_cpu(o_keys).tolist() == list(range(100, 100 + count)) + [EMPTY] * (
        n - count
    )
    assert o_ids.tolist() == ids.tolist()[:count] + [-1] * (n - count)


@cuda_only
@pytest.mark.parametrize("skips_negative", [False, True])
def test_padded_optimizer_table_ids(monkeypatch, skips_negative):
    monkeypatch.setattr(sf, "PADDED_OPT_SKIPS_NEGATIVE", skips_negative)
    ids = torch.tensor([0, 2, 1, -1, -1], dtype=torch.int64, device="cuda")
    out = sf.padded_optimizer_table_ids(ids)
    assert out.tolist() == (ids.tolist() if skips_negative else [0, 2, 1, 0, 0])
    assert ids.tolist() == [0, 2, 1, -1, -1], "the input is not modified"


class RecordingInitializer:
    """Stands in for a MultiTableInitializer: row r of the buffer that `indices` names becomes
    keys[r] % 1000 + 0.5 * table_ids[r] + column. A row whose index is negative is skipped, as
    the C++ initializer does when the extension says initializer_skips_negative_index; without
    that capability a negative index or table id is an error, as in the unpatched extension.
    """

    def __init__(self, skips_negative: bool):
        self.skips_negative = skips_negative
        self.calls = 0

    def __call__(self, buffer, keys, table_ids, indices):
        self.calls += 1
        assert (
            buffer.dim() == 2 and buffer.stride(1) == 1
        ), "the initializer needs unit column stride"
        assert keys.size(0) == table_ids.size(0) == buffer.size(0)
        if not self.skips_negative:
            assert bool(
                (indices >= 0).all()
            ), "an unpatched initializer would write at a negative index"
            assert bool(
                (table_ids >= 0).all()
            ), "an unpatched initializer reads table_params[table_id]"
        rows = indices[indices >= 0]
        assert bool(
            (table_ids[rows] >= 0).all()
        ), "a row that is initialized has a table"
        keys_i64 = keys.view(torch.int64) if keys.dtype == torch.uint64 else keys
        base = (keys_i64[rows] % 1000).to(torch.float32) + 0.5 * table_ids[rows].to(
            torch.float32
        )
        columns = torch.arange(
            buffer.size(1), dtype=torch.float32, device=buffer.device
        )
        buffer[rows] = base.unsqueeze(1) + columns


@cuda_only
@pytest.mark.parametrize("dtype", KEY_DTYPES)
@pytest.mark.parametrize("skips_negative", [False, True])
@pytest.mark.parametrize("pattern", ["mixed", "none", "all"])
def test_initialize_masked_touches_only_the_rows_of_the_mask(
    monkeypatch, dtype, skips_negative, pattern
):
    monkeypatch.setattr(sf, "INIT_SKIPS_NEGATIVE", skips_negative)
    n, emb, extra = 16, 8, 4
    g = torch.Generator().manual_seed(5)
    mask_cpu = {
        "mixed": torch.rand(n, generator=g) < 0.5,
        "none": torch.zeros(n, dtype=torch.bool),
        "all": torch.ones(n, dtype=torch.bool),
    }[pattern]
    mask = mask_cpu.cuda()
    # the tail (here: the rows that are not in the mask) has the reserved key and a table id of -1
    keys_list = [EMPTY if not m else 100 + i for i, m in enumerate(mask_cpu.tolist())]
    ids_list = [-1 if not m else i % 3 for i, m in enumerate(mask_cpu.tolist())]
    keys = keys_of(keys_list, dtype, "cuda")
    ids = torch.tensor(ids_list, dtype=torch.int64, device="cuda")
    block = torch.full(
        (n, emb + extra), 555.0, device="cuda"
    )  # embedding columns and optimizer state
    block[:, :emb] = -777.0
    out = block[
        :, :emb
    ]  # a column slice of the value block: stride(1) == 1, rows not contiguous
    initializer = RecordingInitializer(skips_negative)
    sf.initialize_masked(initializer, out, keys, ids, mask)
    assert initializer.calls >= 1
    got = block.cpu()
    for r in range(n):
        if mask_cpu[r]:
            expected = (
                (keys_list[r] % 1000)
                + 0.5 * ids_list[r]
                + torch.arange(emb, dtype=torch.float32)
            )
            assert torch.equal(got[r, :emb], expected), "row %d is initialized" % r
        else:
            assert (got[r, :emb] == -777.0).all(), (
                "row %d is not in the mask: untouched" % r
            )
    assert (got[:, emb:] == 555.0).all(), "the columns outside the slice are untouched"


@cuda_only
@pytest.mark.parametrize("dtype", KEY_DTYPES)
def test_initialize_masked_fallback_equals_the_capability_path(monkeypatch, dtype):
    """With and without initializer_skips_negative_index the result is bit-identical."""
    n, emb = 32, 8
    g = torch.Generator().manual_seed(9)
    mask = (torch.rand(n, generator=g) < 0.4).cuda()
    keys = keys_of(list(range(1000, 1000 + n)), dtype, "cuda")
    ids = (torch.arange(n, device="cuda") % 4).to(torch.int64)
    results = []
    for skips_negative in (False, True):
        monkeypatch.setattr(sf, "INIT_SKIPS_NEGATIVE", skips_negative)
        out = torch.full((n, emb), -1.0, device="cuda")
        sf.initialize_masked(RecordingInitializer(skips_negative), out, keys, ids, mask)
        results.append(out.cpu())
    assert torch.equal(results[0], results[1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", KEY_DTYPES)
@pytest.mark.parametrize("chunk", [1, 5, 16, 100])
def test_initialize_masked_fallback_in_row_chunks_equals_one_block(
    monkeypatch, device, dtype, chunk
):
    """The scratch of the fallback is bounded by the miss-path budget: initialized and copied
    back in row chunks it gives the result of one block (an initializer sees only its own
    chunk's keys, table ids and row numbers)."""
    monkeypatch.setattr(sf, "INIT_SKIPS_NEGATIVE", False)
    n, emb = 32, 8
    g = torch.Generator().manual_seed(9)
    mask = (torch.rand(n, generator=g) < 0.4).to(device)
    keys = keys_of(list(range(1000, 1000 + n)), dtype, device)
    ids = (torch.arange(n, device=device) % 4).to(torch.int64)
    results = []
    for rows in (None, chunk):
        monkeypatch.setattr(sf, "CHUNK_ROWS", rows)
        out = torch.full((n, emb), -1.0, device=device)
        sf.initialize_masked(RecordingInitializer(False), out, keys, ids, mask)
        results.append(out.cpu())
    assert torch.equal(results[0], results[1])


# --------------------------------------------------------------------------
# the miss gate's mode
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gate", ["always", "poll"])
@pytest.mark.parametrize("rng", [False, True])
def test_miss_gate_mode(monkeypatch, gate, rng):
    monkeypatch.setattr(sf, "MISS_GATE", gate)
    assert sf.miss_gate_mode(rng) == ("always" if (rng and gate == "poll") else gate)


@pytest.mark.parametrize("gate", ["always", "poll"])
@pytest.mark.parametrize("rng", [False, True])
def test_the_determinism_mode_does_not_change_the_gate(monkeypatch, gate, rng):
    monkeypatch.setattr(sf, "MISS_GATE", gate)
    monkeypatch.delenv("DEMB_DETERMINISM_MODE", raising=False)
    without = sf.miss_gate_mode(rng)
    monkeypatch.setenv("DEMB_DETERMINISM_MODE", "ON")
    assert sf.miss_gate_mode(rng) == without


def test_no_storage_can_choose_the_gate_and_the_gate_has_no_read_in_it():
    """What kind of storage is below a step does not decide how the step waits (it used to
    be a parameter, ``ps``, that made the gate read), and nothing in the gate is a read.
    """
    assert list(inspect.signature(sf.miss_gate_mode).parameters) == ["rng"]
    assert list(inspect.signature(sf.MissGate.begin).parameters) == [
        "self",
        "is_miss",
        "rng",
    ]
    for obj in (sf.miss_gate_mode, sf.MissGate, sf._AlwaysGate):
        assert "read_now" not in inspect.getsource(obj), obj


# --------------------------------------------------------------------------
# the ring of pinned reads
# --------------------------------------------------------------------------
def spin_cycles_for(seconds: float) -> int:
    """The argument of torch.cuda._sleep that keeps the stream busy for about `seconds`."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    probe = 10_000_000
    torch.cuda._sleep(probe)  # warm up the launch
    start.record()
    torch.cuda._sleep(probe)
    end.record()
    end.synchronize()
    return int(probe * seconds * 1000.0 / max(start.elapsed_time(end), 1e-3))


@cuda_only
def test_the_ring_hands_out_its_slots_and_wraps():
    ring = sf.DeferredRead((2,), torch.int64, slots=3)
    assert ring.free_slots() == 3
    values = [torch.tensor([i, 10 * i], device="cuda") for i in range(5)]
    tickets = [ring.enqueue(values[i]) for i in range(3)]
    assert all(t is not None for t in tickets) and len(set(tickets)) == 3
    assert ring.free_slots() == 0
    assert (
        ring.enqueue(values[3]) is None
    ), "every slot is in flight: no ticket, no wait, no allocation"
    torch.cuda.synchronize()
    for i, t in enumerate(tickets):
        assert ring.ready(t) and ring.value(t).tolist() == [i, 10 * i]
    ring.release(tickets[1])
    assert ring.free_slots() == 1
    t3 = ring.enqueue(values[3])
    assert t3 is not None and ring.free_slots() == 0
    torch.cuda.synchronize()
    assert ring.value(t3).tolist() == [3, 30]
    assert ring.value(tickets[0]).tolist() == [0, 0] and ring.value(
        tickets[2]
    ).tolist() == [2, 20], "kept"
    for t in (tickets[0], tickets[2], t3):
        ring.release(t)
    assert ring.free_slots() == 3

    for i in range(20):  # many laps
        t = ring.enqueue(torch.tensor([i, -i], device="cuda"))
        assert t is not None
        torch.cuda.synchronize()
        assert ring.value(t).tolist() == [i, -i]
        ring.release(t)
    assert ring.free_slots() == 3


@cuda_only
def test_the_value_has_the_ring_dtype_and_is_pinned():
    ring = sf.DeferredRead((4,), torch.int32, slots=2)
    t = ring.enqueue(torch.arange(4, dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    v = ring.value(t)
    assert v.dtype == torch.int32 and v.is_pinned() and v.tolist() == [0, 1, 2, 3]
    ring.release(t)


@cuda_only
def test_enqueue_does_not_wait_for_a_kernel_in_flight():
    ring = sf.DeferredRead((1,), torch.int64, slots=2)
    seconds = 0.5
    cycles = spin_cycles_for(seconds)
    dev = torch.tensor([21], device="cuda")
    # the first launch of a kernel waits for the device (CUDA lazy module loading): make the
    # multiplication that the observed call makes once before the stream is busy
    _ = dev * 2
    torch.cuda.synchronize()
    torch.cuda._sleep(cycles)  # the stream is busy for about half a second
    t0 = time.perf_counter()
    ticket = ring.enqueue(dev * 2)  # the copy is queued behind the kernel
    host_seconds = time.perf_counter() - t0
    assert ticket is not None
    assert host_seconds < 0.25 * seconds, "enqueue waited for the stream"
    assert not ring.ready(ticket), "the copy cannot have landed while the kernel runs"
    torch.cuda.synchronize()
    assert ring.ready(ticket), "ready flips once the stream got past the copy"
    assert ring.value(ticket).tolist() == [42]
    ring.release(ticket)


@cuda_only
def test_the_event_is_recorded_on_the_stream_that_issued_the_copy():
    """enqueue records its event on the CURRENT stream, after the copy. With another stream
    current (the prefetch pipeline runs under `torch.cuda.stream(prefetch_stream)`) and busy,
    the ticket is not ready until that stream is, and the value is the one produced there. An
    event on any other stream (the default one, the stream the ring was built under) reads ready
    at once, before the copy has run, and the slot still holds what was there before."""
    ring = sf.DeferredRead((1,), torch.int64, slots=2)
    # every slot holds a landed 0 first: the stale value that a too early ready() would expose
    zeros = [
        ring.enqueue(torch.zeros(1, dtype=torch.int64, device="cuda")) for _ in range(2)
    ]
    torch.cuda.synchronize()
    for t in zeros:
        assert ring.ready(t) and ring.value(t).tolist() == [0]
        ring.release(t)
    source = torch.tensor([21], device="cuda")
    # the first launch of a kernel waits for the device (lazy module loading): make the
    # multiplication that the observed call makes once before `side` is busy
    _ = source * 2
    side = torch.cuda.Stream()
    cycles = spin_cycles_for(0.4)
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(cycles)  # `side` is busy for about 0.4 s
        ticket = ring.enqueue(source * 2)
    assert ticket is not None
    assert not ring.ready(
        ticket
    ), "the copy is queued behind the kernel on `side`: an event on another stream is ready"
    side.synchronize()
    assert ring.ready(ticket), "ready flips once `side` got past the copy"
    assert ring.value(ticket).tolist() == [42]
    ring.release(ticket)


@cuda_only
def test_wait_blocks_until_the_copy_landed():
    ring = sf.DeferredRead((1,), torch.int64, slots=2)
    torch.cuda._sleep(spin_cycles_for(0.1))
    ticket = ring.enqueue(torch.tensor([7], device="cuda"))
    assert ring.wait(ticket).tolist() == [7]
    assert ring.ready(ticket)
    ring.release(ticket)


@cuda_only
def test_a_copied_module_gets_a_fresh_ring():
    ring = sf.DeferredRead((3,), torch.int32, slots=5)
    ticket = ring.enqueue(torch.arange(3, dtype=torch.int32, device="cuda"))
    assert ring.free_slots() == 4
    for clone in (
        copy.deepcopy(ring),
        copy.copy(ring),
        pickle.loads(pickle.dumps(ring)),
    ):
        assert isinstance(clone, sf.DeferredRead) and clone is not ring
        assert (tuple(clone.shape), clone.dtype, clone.slots) == (
            tuple(ring.shape),
            ring.dtype,
            ring.slots,
        )
        assert (
            clone.free_slots() == 5
        ), "a fresh ring: nothing in flight, nothing shared"
    torch.cuda.synchronize()
    assert ring.value(ticket).tolist() == [0, 1, 2]
    ring.release(ticket)


@cuda_only
def test_the_ring_never_blocks_or_allocates_in_steady_state(
    count_syncs, instruments_ok
):
    ring = sf.DeferredRead((1,), torch.int64, slots=2)
    dev = torch.tensor([5], device="cuda")

    def cycle(i):
        ticket = ring.enqueue(dev + i)
        ready = ring.ready(ticket)
        assert ready in (True, False)
        ring.release(ticket)

    result = count_syncs(cycle, iters=6, warmup=3)
    assert result.blocking == 0 and result.by_kind["al"] == 0, result.describe()


# --------------------------------------------------------------------------
# the audited reads
# --------------------------------------------------------------------------
@cuda_only
def test_read_now_is_one_counted_blocking_read(read_now_calls):
    a = torch.arange(6, dtype=torch.int64, device="cuda").reshape(3, 2)
    b = torch.tensor([True, False, True, True, False], device="cuda")
    c = torch.tensor([9, 8, 7, 6], dtype=torch.int32, device="cuda")
    out = sf.read_now(a, b, c)
    assert read_now_calls.delta == 1, "one call, however many tensors"
    assert len(out) == 3
    for got, want in zip(out, (a, b, c)):
        assert not got.is_cuda and got.shape == want.shape
        assert torch.equal(got.to(torch.int64), want.cpu().to(torch.int64))


@cuda_only
def test_read_now_costs_one_sync(count_syncs, instruments_ok):
    x = torch.arange(4, device="cuda")
    y = torch.ones(3, dtype=torch.int64, device="cuda")
    result = count_syncs(lambda i: sf.read_now(x + i, y), iters=4, warmup=3)
    assert result.blocking == 1.0, result.describe()


@cuda_only
def test_read_now_passes_cpu_tensors_through():
    host = torch.tensor([1, 2, 3])
    (out,) = sf.read_now(host)
    assert torch.equal(out, host)


@cuda_only
def test_pull_to_host_is_one_wait_into_pinned_memory(count_syncs, instruments_ok):
    a = torch.arange(10, dtype=torch.int64, device="cuda")
    b = torch.tensor([[1.5, 2.5], [3.5, 4.5]], device="cuda")
    out = sf.pull_to_host(a, b)
    assert all(t.is_pinned() and not t.is_cuda for t in out)
    assert out[0].dtype == torch.int64 and out[0].tolist() == list(range(10))
    assert out[1].dtype == torch.float32 and out[1].tolist() == [[1.5, 2.5], [3.5, 4.5]]
    result = count_syncs(lambda i: sf.pull_to_host(a + i, b), iters=3, warmup=2)
    assert result.blocking == 1.0, result.describe()


@cuda_only
def test_the_audited_reads_are_the_only_ones_no_sync_lets_through(no_sync):
    x = torch.tensor([3, 4], device="cuda")
    ring = sf.DeferredRead((2,), torch.int64, slots=2)
    ticket = ring.enqueue(x)
    host = torch.tensor([1])  # one element: .item() of a longer tensor is an error
    with no_sync():
        (read,) = sf.read_now(x)
        waited = ring.wait(ticket)
        assert host.item() == 1 and host.cpu() is host, "a CPU tensor is not a sync"
        for forbidden in (
            lambda: x.item(),
            lambda: x.cpu(),
            lambda: x.tolist(),
            lambda: torch.cuda.synchronize(),
            lambda: torch.cuda.current_stream().synchronize(),
        ):
            with pytest.raises(AssertionError, match="host sync outside sf.read_now"):
                forbidden()
    assert read.tolist() == [3, 4] and waited.tolist() == [3, 4]
    assert x[0].item() == 3, "the patches are gone after the block"
    ring.release(ticket)


@cuda_only
@pytest.mark.parametrize("sizes", [(8,), (3, 100, 17, 4096, 1, 250)])
def test_staged_upload_keeps_every_upload_intact(sizes):
    """Uploads issued back to back, all synchronized afterwards: no staging buffer is reused
    before its copy has landed."""
    upload = sf.StagedUpload(torch.device("cuda"), ring=2)
    sources = [
        torch.arange(n, dtype=torch.int64) * (i + 2) for i, n in enumerate(sizes * 3)
    ]
    uploaded = [upload.upload(s) for s in sources]
    torch.cuda.synchronize()
    for src, dev in zip(sources, uploaded):
        assert dev.is_cuda and torch.equal(dev.cpu(), src)


# --------------------------------------------------------------------------
# the miss gate
# --------------------------------------------------------------------------
def masks(n=8, miss_at=3):
    none = torch.zeros(n, dtype=torch.bool, device="cuda")
    some = none.clone()
    some[miss_at] = True
    return none, some


@cuda_only
def test_the_always_gate_does_no_device_work(monkeypatch, read_now_calls):
    monkeypatch.setattr(sf, "MISS_GATE", "always")
    gate = sf.MissGate()
    none, some = masks()
    for is_miss in (none, some):
        pending = gate.begin(is_miss)
        assert pending is None and gate.needed(pending) is True
    assert read_now_calls.delta == 0


@cuda_only
@pytest.mark.parametrize("gate", ["always", "poll"])
def test_the_gate_reads_nothing_in_the_determinism_mode(
    monkeypatch, read_now_calls, gate
):
    monkeypatch.setenv("DEMB_DETERMINISM_MODE", "ON")
    monkeypatch.setattr(sf, "MISS_GATE", gate)
    miss_gate = sf.MissGate()
    none, some = masks()
    pending = miss_gate.begin(none)
    torch.cuda.synchronize()
    assert miss_gate.needed(pending) is (gate == "always")
    assert miss_gate.needed(miss_gate.begin(some)) is True
    assert read_now_calls.delta == 0


@cuda_only
def test_the_poll_gate_skips_on_a_landed_zero_only(monkeypatch, read_now_calls):
    monkeypatch.setattr(sf, "MISS_GATE", "poll")
    gate = sf.MissGate()
    none, some = masks()
    pending = gate.begin(none)
    torch.cuda.synchronize()
    assert gate.needed(pending) is False, "a landed zero proves there is no miss"
    pending = gate.begin(some)
    torch.cuda.synchronize()
    assert gate.needed(pending) is True, "a landed non-zero"
    assert gate.ring.free_slots() == gate.ring.slots, "needed gives its ticket back"
    assert read_now_calls.delta == 0, "polling never blocks"


@cuda_only
def test_the_poll_gate_runs_the_path_when_the_count_has_not_landed(
    monkeypatch, read_now_calls
):
    monkeypatch.setattr(sf, "MISS_GATE", "poll")
    gate = sf.MissGate()
    none, _ = masks()
    # the first launch of a kernel waits for the device (lazy module loading): begin() once
    # before the stream is busy
    warm = gate.begin(none)
    torch.cuda.synchronize()
    gate.needed(warm)
    torch.cuda._sleep(spin_cycles_for(0.3))
    pending = gate.begin(none)
    assert (
        gate.needed(pending) is True
    ), "not landed yet: nothing is proven, the path runs"
    torch.cuda.synchronize()
    assert gate.ring.free_slots() == gate.ring.slots, "the slot is free again"
    assert read_now_calls.delta == 0


@cuda_only
def test_the_poll_gate_follows_the_stream_the_count_was_produced_on(
    monkeypatch, read_now_calls
):
    """The miss count of a step is produced on the stream the step runs on. While that stream is
    busy the count has not landed, and a slot that still holds an older 0 must not read as "no
    miss": needed() is True then, and True once the count landed, because it is not zero. A skip
    on a stale 0 would leave the keys of a step uninserted and uninitialized."""
    monkeypatch.setattr(sf, "MISS_GATE", "poll")
    gate = sf.MissGate()
    none, some = masks()
    slots = gate.ring.slots
    pendings = [gate.begin(none) for _ in range(slots)]  # every slot gets a landed 0
    torch.cuda.synchronize()
    assert [gate.needed(p) for p in pendings] == [False] * slots
    assert gate.ring.free_slots() == slots
    side = torch.cuda.Stream()
    cycles = spin_cycles_for(0.4)
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(cycles)
        pending = gate.begin(some)
    assert (
        gate.needed(pending) is True
    ), "the count is queued behind the kernel on `side`: nothing is proven"
    side.synchronize()
    with torch.cuda.stream(side):
        pending = gate.begin(some)
    side.synchronize()
    assert gate.needed(pending) is True, "a landed non-zero count"
    assert read_now_calls.delta == 0, "polling never blocks"


@cuda_only
def test_the_poll_gate_cannot_skip_without_a_ticket(monkeypatch):
    """More steps in flight than the ring has slots: those that got no ticket run the path."""
    monkeypatch.setattr(sf, "MISS_GATE", "poll")
    gate = sf.MissGate()
    none, _ = masks()
    slots = gate.ring.slots
    pendings = [gate.begin(none) for _ in range(slots + 2)]
    torch.cuda.synchronize()
    answers = [gate.needed(p) for p in pendings]
    assert answers == [False] * slots + [True, True]


@cuda_only
def test_a_probabilistic_admitter_is_never_skipped_by_the_poll_gate(monkeypatch):
    monkeypatch.setattr(sf, "MISS_GATE", "poll")
    gate = sf.MissGate()
    none, _ = masks()
    pending = gate.begin(none, rng=True)
    torch.cuda.synchronize()
    assert gate.needed(pending) is True
