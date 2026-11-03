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

"""The pieces an erase beside training is made of: the gate, the worker, the handle;
and then all of them on a real module, with a prefetch on one side and a scan on the
other.

The first part needs no tables. The second submits the scan to the module's worker
directly, in small chunks, so that the two sides meet over and over.
"""

import contextlib
import copy
import pickle
import threading
import time

import pytest
import torch
from dynamicemb.async_erase import (
    AsyncEraser,
    EraseHandle,
    FairLock,
    TableAccessGate,
    table_section,
)
from helpers import BUCKET, build_de_table, found, train_step

# Long enough for a launch to be over long before the kernel is: about 30 ms.
SLEEP_CYCLES = 50_000_000
WAIT = 10  # seconds a test waits for something that should happen at once


def wait_until(condition):
    deadline = time.monotonic() + WAIT
    while not condition():
        assert time.monotonic() < deadline, "did not happen"
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_kernels_of_a_section_run_after_those_of_the_section_before(device):
    """Two sections on two streams, launched in order: the second sees what the
    first wrote, although the first one's kernel is still sleeping when the second
    launches. (Without the gate it does not: the control.)"""
    gate = TableAccessGate()
    first, second = torch.cuda.Stream(), torch.cuda.Stream()

    def run(section):
        x = torch.zeros(1 << 20, device=device)
        seen = torch.empty_like(x)
        torch.cuda.synchronize()
        with torch.cuda.stream(first), section():
            torch.cuda._sleep(SLEEP_CYCLES)
            x.fill_(1)
        with torch.cuda.stream(second), section():
            seen.copy_(x)
        torch.cuda.synchronize()
        return bool((seen == 1).all())

    assert not run(contextlib.nullcontext), "the test would not notice a missing gate"
    assert run(gate)


def test_the_lock_is_held_for_the_whole_section():
    gate = TableAccessGate()
    inside, release, order = threading.Event(), threading.Event(), []

    def holder():
        with gate():
            inside.set()
            release.wait(WAIT)
            order.append("holder")

    def other():
        with gate():
            order.append("other")

    first = threading.Thread(target=holder)
    first.start()
    assert inside.wait(WAIT)
    second = threading.Thread(target=other)
    second.start()
    time.sleep(0.2)
    assert order == [], "entered while another section was open"
    release.set()
    first.join(WAIT)
    second.join(WAIT)
    assert order == ["holder", "other"]

    # What prefetch enters: the gate's section, or nothing where there is no gate.
    with table_section(gate):
        assert gate._lock.locked()
    assert not gate._lock.locked()
    with table_section(None):
        pass


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------


def test_a_fair_lock_goes_to_the_waiters_in_order_and_is_never_free_while_one_waits():
    """The waiters get it in the order they came. And the lock is handed over, not freed:
    a thread that lets go and asks again at once -- which a free lock would give to it --
    comes after all of them."""
    lock = FairLock()
    lock.acquire()
    order, threads = [], []
    for i in range(5):

        def waiter(i=i):
            with lock:
                order.append(i)

        thread = threading.Thread(target=waiter)
        thread.start()
        threads.append(thread)
        wait_until(lambda: len(lock._waiters) == i + 1)
    lock.release()
    assert lock.locked(), "free between the hand-over and the first waiter running"
    lock.acquire()
    order.append("asked again")
    lock.release()
    for thread in threads:
        thread.join(WAIT)
    assert order == [0, 1, 2, 3, 4, "asked again"]
    assert not lock.locked()


def test_a_fair_lock_that_is_not_held_cannot_be_released():
    with pytest.raises(RuntimeError, match="not held"):
        FairLock().release()


class Interrupting(FairLock):
    """A lock whose waiters are interrupted as the thread's name says: ``"before"``
    raises while still queued, ``"after"`` once the lock was handed over. (By name, not by
    order: which of two threads that queued first calls this first is not settled.)"""

    def __init__(self, modes):
        super().__init__()
        self.modes = modes

    def _wait_for_hand_over(self, baton):
        mode = self.modes.get(threading.current_thread().name)
        if mode == "before":
            raise KeyboardInterrupt
        baton.acquire()
        if mode == "after":
            raise KeyboardInterrupt


def test_a_waiter_interrupted_before_it_was_handed_the_lock_leaves_the_queue():
    lock = Interrupting({"MainThread": "before"})
    lock.acquire()
    with pytest.raises(KeyboardInterrupt):
        lock.acquire()  # would wait for itself; interrupted instead
    assert len(lock._waiters) == 0
    lock.release()
    assert not lock.locked()


def test_a_waiter_interrupted_after_it_was_handed_the_lock_passes_it_on():
    lock = Interrupting({"first": "after"})
    lock.acquire()
    got, errors = [], []

    def first():
        try:
            lock.acquire()
        except KeyboardInterrupt as error:
            errors.append(error)

    def second():
        with lock:
            got.append("second")

    one = threading.Thread(target=first, name="first")
    one.start()
    wait_until(lambda: len(lock._waiters) == 1)
    two = threading.Thread(target=second, name="second")
    two.start()
    wait_until(lambda: len(lock._waiters) == 2)
    lock.release()  # to the first, which is interrupted and passes it to the second
    one.join(WAIT)
    two.join(WAIT)
    assert len(errors) == 1 and got == ["second"]
    assert not lock.locked()


def test_a_thread_in_a_hot_loop_does_not_keep_another_out_of_the_gate(device):
    """The worker takes the gate for a chunk, lets go and asks for the next at once; the
    training thread, asking meanwhile, is let in after the section that holds it, not
    after the loop. The loop takes about a second; the training thread waits, each time,
    far less than that. (It is timed, not counted in laps: the thread that was let in
    needs the GIL to go on, and the loop may take a few milliseconds of laps meanwhile.)
    """
    gate = TableAccessGate()
    laps = [0]

    def hot():
        for _ in range(40_000):
            with gate():
                laps[0] += 1

    thread = threading.Thread(target=hot)
    thread.start()
    wait_until(lambda: laps[0] > 0)
    waits = []
    for _ in range(20):
        asked = time.perf_counter()
        with gate():
            waits.append(time.perf_counter() - asked)
    still_running = thread.is_alive()
    thread.join(WAIT * 3)
    assert laps[0] == 40_000
    assert still_running, "the loop was over before the test got through"
    assert max(waits) < 0.1, waits


def test_a_failed_section_lets_go_of_the_gate():
    gate = TableAccessGate()
    with pytest.raises(RuntimeError, match="boom"):
        with gate():
            raise RuntimeError("boom")
    assert gate._last_access is not None, "what was launched is still on the stream"
    with gate():  # would hang if the lock were still held
        pass


@pytest.mark.parametrize(
    "copy_it", [copy.copy, copy.deepcopy, lambda g: pickle.loads(pickle.dumps(g))]
)
def test_a_copy_of_a_gate_or_a_worker_is_a_new_one(copy_it):
    """A module is copied (deepcopy, pickle) and a lock cannot be."""
    gate = TableAccessGate()
    with gate():
        pass
    new_gate = copy_it(gate)
    assert isinstance(new_gate, TableAccessGate) and new_gate is not gate
    assert new_gate._last_access is None
    eraser = AsyncEraser()
    eraser.own_stream(torch.cuda.current_device())
    new_eraser = copy_it(eraser)
    assert isinstance(new_eraser, AsyncEraser) and new_eraser._streams == {}


# ---------------------------------------------------------------------------
# The worker and the handle
# ---------------------------------------------------------------------------


@pytest.fixture
def eraser():
    """A worker that is shut down after the test, whether it passed or not."""
    eraser = AsyncEraser()
    yield eraser
    eraser.shutdown()


def test_the_work_runs_on_the_worker_with_the_given_stream_and_zeroed_counts(
    device, eraser
):
    stream = torch.cuda.Stream()
    seen = {}

    def work(counts):
        seen["thread"] = threading.current_thread()
        seen["stream"] = torch.cuda.current_stream()
        seen["device"] = torch.cuda.current_device()
        seen["counts"] = (counts.dtype, counts.shape, counts.device, counts.tolist())

    eraser.submit(work, stream).wait()
    assert seen["thread"] is not threading.main_thread()
    assert seen["stream"] == stream
    assert seen["device"] == device.index
    assert seen["counts"] == (torch.int64, (2,), device, [0, 0])


def test_erases_run_one_at_a_time_in_the_order_they_were_asked_for(device, eraser):
    stream = torch.cuda.Stream()
    log, active, most = [], [0], [0]

    def make(n):
        def work(counts):
            active[0] += 1
            most[0] = max(most[0], active[0])
            time.sleep(0.05)
            log.append(n)
            active[0] -= 1

        return work

    handles = [eraser.submit(make(n), stream) for n in range(4)]
    for handle in handles:
        handle.wait()
    assert log == [0, 1, 2, 3]
    assert most[0] == 1


def test_result_is_the_counts_the_work_made(device, eraser):
    handle = eraser.submit(
        lambda counts: counts.add_(torch.tensor([3, 2], device=device)),
        torch.cuda.Stream(),
    )
    assert handle.result() == (3, 2)
    assert handle.result() == (3, 2)  # again: from what it kept


def test_a_handle_with_nothing_to_wait_for():
    handle = EraseHandle.completed()
    assert handle.done()
    handle.wait()
    handle.wait_stream()
    assert handle.result() == (0, 0)


def test_the_three_moments_of_an_erase(device, eraser):
    """The call returns while the worker is still launching; then the worker is done
    and the kernel is still running; then the kernel is done. ``done`` is about the
    last, ``wait_stream`` about the second, ``wait`` the last."""
    release = threading.Event()

    def work(counts):
        release.wait(WAIT)
        torch.cuda._sleep(SLEEP_CYCLES * 4)

    handle = eraser.submit(work, torch.cuda.Stream())
    assert not handle.done(), "the call waited for the work"
    release.set()
    handle.wait_stream()  # launched
    assert not handle.done(), "the kernel is still running"
    handle.wait()
    assert handle.done()


def wait_by_polling_done(handle):
    while not handle.done():
        time.sleep(0.01)


def fail(counts):
    raise ValueError("no such table")


@pytest.mark.parametrize(
    "ask",
    [
        lambda handle: handle.wait(),
        lambda handle: handle.wait_stream(),
        lambda handle: handle.result(),
        wait_by_polling_done,
    ],
    ids=["wait", "wait_stream", "result", "done"],
)
def test_what_the_worker_raised_is_raised_by_every_way_of_asking(device, eraser, ask):
    handle = eraser.submit(fail, torch.cuda.Stream())
    with pytest.raises(ValueError, match="no such table"):
        ask(handle)


def test_wait_stream_waits_for_the_launch_and_orders_the_stream_after_the_kernels(
    device, eraser
):
    """It blocks until the worker has launched everything -- an event not recorded
    yet orders nothing -- and the stream then runs after the kernels, which were
    still sleeping when the call returned."""
    release, box = threading.Event(), []

    def work(counts):
        release.wait(WAIT)
        torch.cuda._sleep(SLEEP_CYCLES)
        counts[0] = 7
        box.append(counts)

    handle = eraser.submit(work, torch.cuda.Stream())
    other = torch.cuda.Stream()
    returned = threading.Event()

    def ask():
        handle.wait_stream(other)
        returned.set()

    asker = threading.Thread(target=ask)
    asker.start()
    time.sleep(0.2)
    assert not returned.is_set(), "returned before the work was launched"
    release.set()
    assert returned.wait(WAIT)
    with torch.cuda.stream(other):
        seen = box[0][0].clone()
    torch.cuda.synchronize()
    assert int(seen) == 7
    asker.join(WAIT)


# ---------------------------------------------------------------------------
# On a module: a prefetch on one side, a scan on the other
# ---------------------------------------------------------------------------

OLD_SCORE, NEW_SCORE, THRESHOLD = 10, 200, 100
OLD = list(range(1, 20_001))
NEW_KEY_STEPS = 100


def new_de_table(device):
    """One table that scores keys as the caller says, with room to grow."""
    return build_de_table(device.index, init_capacity=1 << 15)


def scan(module, chunk_guard, chunk_slots=BUCKET):
    """Ask the module's worker to erase what is below the threshold, in small chunks."""
    return module._eraser.submit(
        lambda counts: module._storage.erase_by_score(
            THRESHOLD, [0], counts, chunk_slots=chunk_slots, chunk_guard=chunk_guard
        ),
        torch.cuda.Stream(),
    )


class WatchedGate:
    """The module's gate, counting the sections it is entered for -- one per chunk of
    a scan -- and, if told to, stopping the worker inside the ``at``-th until released.
    """

    def __init__(self, gate, at=None):
        self.gate, self.at, self.entered = gate, at, 0
        self.inside, self.release = threading.Event(), threading.Event()

    @contextlib.contextmanager
    def __call__(self):
        with self.gate():
            self.entered += 1
            if self.entered == self.at:
                self.inside.set()
                self.release.wait(WAIT)
            yield


@pytest.fixture
def module_with_old_keys(device):
    module = new_de_table(device)
    for start in range(0, len(OLD), 2000):
        train_step(module, OLD[start : start + 2000], device, score=OLD_SCORE)
    assert int(module._storage.key_index_map.size(0)) == len(OLD)
    return module


def test_a_prefetch_waits_for_the_chunk_the_scan_is_in_not_for_the_scan(
    device, module_with_old_keys
):
    """The worker stops inside a chunk, holding the gate. A training step -- whose
    prefetch needs the gate -- cannot get past it; when the worker goes on the step
    completes, between two chunks of the same scan, with new keys that the table
    has to grow for. The scan, on its restart, still erases every old key once
    and none of the new ones."""
    module = module_with_old_keys
    state = module._storage._state
    new_keys = list(range(100_001, 120_001))
    parked = WatchedGate(module._table_gate, at=3)
    generation_before = state.key_map_generation
    handle = scan(module, parked)
    assert parked.inside.wait(WAIT)

    errors = []

    def training():
        try:
            train_step(module, new_keys, device, score=NEW_SCORE)
        except BaseException as error:  # reported by the test thread
            errors.append(error)

    step = threading.Thread(target=training)
    step.start()
    time.sleep(0.3)
    assert step.is_alive(), "the prefetch went through while a chunk held the gate"
    parked.release.set()
    step.join(WAIT * 3)
    assert not step.is_alive() and not errors, errors

    assert handle.result() == (len(OLD), 0)
    assert (
        state.key_map_generation > generation_before
    ), "no expansion: no restart tried"
    assert not bool(found(module, OLD, device).any())
    assert bool(found(module, new_keys, device).all())
    assert int(module._storage.key_index_map.size(0)) == len(new_keys)


def test_training_goes_on_while_the_scan_runs_and_nothing_it_uses_is_lost(
    device, module_with_old_keys
):
    """No coordination at all: erases of the old keys one after the other, while the
    training thread works through new keys -- each batch grows the table a little more
    and restarts the scan -- and re-reads some old ones at the new score, which saves
    them. Every output is the right one, and in the end the table holds what training
    touched and nothing below the threshold.

    The scans are not asked to finish within some number of steps: how many steps they
    take is how fast the worker is against the training, which is not what this is
    about. Training goes on until they are done, or a minute has passed. New keys come
    for the first steps only, the same ones again after, so that the table never fills
    and nothing is evicted.
    """
    module = module_with_old_keys
    watched = WatchedGate(module._table_gate)
    handles = [scan(module, watched, chunk_slots=8 * BUCKET) for _ in range(3)]
    kept = OLD[:500]
    new_keys, steps = [], 0
    deadline = time.monotonic() + 60
    while steps < 20 or not all(h.done() for h in handles):
        if steps < NEW_KEY_STEPS:
            batch = list(range(200_001 + steps * 2000, 200_001 + (steps + 1) * 2000))
            new_keys += batch
        else:
            batch = new_keys[-2000:]
        train_step(module, kept + batch, device, score=NEW_SCORE)
        steps += 1
        assert (
            time.monotonic() < deadline
        ), f"the erases never finished: {watched.entered} chunks in {steps} steps"

    print(f"stress: {steps} steps, {watched.entered} chunks")
    erased = sum(h.result()[0] for h in handles)
    # A kept key was either re-read before the scan reached it (saved) or erased
    # and put back by the read: either way it is there; the rest of the old keys
    # are erased exactly once, by whichever scan got to them first.
    assert bool(found(module, kept + new_keys, device).all())
    assert not bool(found(module, OLD[500:], device).any())
    assert erased >= len(OLD) - len(kept)
    assert erased <= len(OLD)
