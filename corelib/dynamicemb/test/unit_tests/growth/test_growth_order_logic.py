# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The logic of ``GrowthOrder`` with stand-in streams: who waits for whom, and when.

CPU only, no torch: the module is loaded by its path and given streams that log the
calls made on them. What this cannot show is that the ordering holds on a GPU (that is
``test_growth_order.py``); it shows that the right waits are issued, once, and none that
could deadlock.
"""

import ast
import copy
import importlib.util
import os
import pickle
import sys
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # corelib/dynamicemb
PATH = os.path.join(ROOT, "dynamicemb", "growth_order.py")


def load_module():
    spec = importlib.util.spec_from_file_location("growth_order_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # a class of it has to be picklable
    spec.loader.exec_module(module)
    return module


growth_order = load_module()


class Log(list):
    """Every call made on a stand-in, in order: (who, what, argument)."""


class FakeEvent:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def record(self, stream):
        self.log.append((stream.name, "record", self.name))


class FakeStream:
    def __init__(self, log, name, handle, gone=False):
        self.log, self.name, self.cuda_stream, self.gone = log, name, handle, gone

    def wait_event(self, event):
        self._check()
        self.log.append((self.name, "wait_event", event.name))

    def wait_stream(self, other):
        # What torch does: an event recorded on `other`, which this stream waits for.
        other._check()
        self.log.append((self.name, "wait_stream", other.name))

    def _check(self):
        if self.gone:
            raise RuntimeError("CUDA error: invalid resource handle")


class FakeCuda:
    """What GrowthOrder takes from torch: the current stream, new events, the capture flag."""

    def __init__(self):
        self.log = Log()
        self.streams = {}
        self.current = None
        self.is_capturing = False
        self.events = 0

    def stream(self, name, handle=None, gone=False):
        stream = FakeStream(self.log, name, handle or len(self.streams) + 1, gone)
        self.streams[name] = stream
        return stream

    def current_stream(self, device):
        return self.current

    def new_event(self):
        self.events += 1
        return FakeEvent(self.log, "growth%d" % self.events)

    def capturing(self):
        return self.is_capturing

    def on(self, stream):
        self.current = stream
        return stream


@pytest.fixture
def cuda():
    return FakeCuda()


@pytest.fixture
def order(cuda):
    return growth_order.GrowthOrder("cuda:0", cuda=cuda)


def use(order, cuda, stream):
    cuda.on(stream)
    order.wait_for_growth()


def test_a_table_nobody_grew_orders_nothing(order, cuda):
    f, p = cuda.stream("F"), cuda.stream("P")
    use(order, cuda, f)
    use(order, cuda, p)
    assert cuda.log == []
    assert order.serial == 0
    assert set(order.users()) == {f, p}


def test_the_growth_stream_waits_for_every_other_user_before_it_starts(order, cuda):
    f, g, p = cuda.stream("F"), cuda.stream("G"), cuda.stream("P")
    for stream in (f, g, p):
        use(order, cuda, stream)
    with order.section(p):
        # Inside the section the growth stream has already been told to wait for the
        # work F and G have queued (H1), and not for itself.
        assert cuda.log == [("P", "wait_stream", "F"), ("P", "wait_stream", "G")]
        cuda.log.append(("P", "launch", "rehash"))
    # The end of the growth is recorded on P after its last launch, and the others
    # are ordered after it at once (H2).
    assert cuda.log[2:] == [
        ("P", "launch", "rehash"),
        ("P", "record", "growth1"),
        ("F", "wait_event", "growth1"),
        ("G", "wait_event", "growth1"),
    ]
    assert order.serial == 1


def test_a_stream_is_ordered_after_each_growth_once(order, cuda):
    f, p = cuda.stream("F"), cuda.stream("P")
    use(order, cuda, f)
    with order.section(p):
        pass
    cuda.log.clear()
    # F was ordered when the growth ended: using the table again waits for nothing.
    use(order, cuda, f)
    use(order, cuda, f)
    assert cuda.log == []
    # A stream that came after the growth has not seen it: it waits, once.
    late = cuda.stream("L")
    use(order, cuda, late)
    use(order, cuda, late)
    assert cuda.log == [("L", "wait_event", "growth1")]
    # The next growth is a new event for everybody.
    cuda.log.clear()
    with order.section(p):
        pass
    assert ("F", "wait_event", "growth2") in cuda.log
    assert ("L", "wait_event", "growth2") in cuda.log


def test_a_growth_stream_orders_itself_after_a_growth_that_ran_on_another_stream(
    order, cuda
):
    p, q = cuda.stream("P"), cuda.stream("Q")
    with order.section(p):
        pass
    cuda.log.clear()
    with order.section(q):
        # Q is not one of the streams that was ordered at the end of the first growth
        # (it was not registered), so the second growth reads what the first built only
        # if Q waits for it before it starts.
        assert cuda.log == [("Q", "wait_event", "growth1")]


def test_the_growth_stream_is_not_asked_to_wait_for_its_own_event(order, cuda):
    p = cuda.stream("P")
    use(order, cuda, p)
    with order.section(p):
        pass
    assert cuda.log == [("P", "record", "growth1")]
    cuda.log.clear()
    use(order, cuda, p)
    assert cuda.log == []


def test_nested_sections_are_one_growth(order, cuda):
    f, p = cuda.stream("F"), cuda.stream("P")
    use(order, cuda, f)
    with order.section(p):
        with order.section(p):
            cuda.log.append(("P", "launch", "values"))
        cuda.log.append(("P", "launch", "keys"))
    assert cuda.log == [
        ("P", "wait_stream", "F"),
        ("P", "launch", "values"),
        ("P", "launch", "keys"),
        ("P", "record", "growth1"),
        ("F", "wait_event", "growth1"),
    ]
    assert order.serial == 1


def test_two_growths_in_a_row_are_two_events(order, cuda):
    p = cuda.stream("P")
    for _ in range(2):
        with order.section(p):
            pass
    assert order.serial == 2
    assert [e for e in cuda.log if e[1] == "record"] == [
        ("P", "record", "growth1"),
        ("P", "record", "growth2"),
    ]


def test_a_growth_that_fails_still_orders_the_others_after_what_it_launched(
    order, cuda
):
    f, p = cuda.stream("F"), cuda.stream("P")
    use(order, cuda, f)
    with pytest.raises(ValueError, match="out of memory"):
        with order.section(p):
            cuda.log.append(("P", "launch", "half"))
            raise ValueError("out of memory")
    assert cuda.log[-2:] == [("P", "record", "growth1"), ("F", "wait_event", "growth1")]
    # ... and the next growth starts from a clean depth: it is a growth, not a nested one
    cuda.log.clear()
    with order.section(p):
        pass
    assert ("P", "wait_stream", "F") in cuda.log


def test_a_stream_that_was_destroyed_is_dropped_not_fatal(order, cuda):
    f, p = cuda.stream("F"), cuda.stream("P")
    ext = cuda.stream("EXT")
    for stream in (f, ext, p):
        use(order, cuda, stream)
    ext.gone = True  # an external stream the owner destroyed
    with order.section(p):
        pass  # the wait for EXT raised RuntimeError at the start and at the end
    assert ext not in order.users()
    assert set(order.users()) == {f, p}
    # F, the live one, is ordered as usual
    assert ("F", "wait_event", "growth1") in cuda.log
    assert not any(entry[0] == "EXT" for entry in cuda.log)


def test_a_stream_under_capture_is_not_asked_to_wait(order, cuda):
    f, p = cuda.stream("F"), cuda.stream("P")
    with order.section(p):
        pass
    cuda.log.clear()
    cuda.is_capturing = True
    use(order, cuda, f)
    assert cuda.log == []
    # not marked as ordered: the first call outside the capture waits
    cuda.is_capturing = False
    use(order, cuda, f)
    assert cuda.log == [("F", "wait_event", "growth1")]


def test_the_depth_is_per_thread(order, cuda):
    """A growth that another thread starts while this thread is inside one is a growth
    of its own, not a nested one: it waits for the streams."""
    f, p, q = cuda.stream("F"), cuda.stream("P"), cuda.stream("Q")
    use(order, cuda, f)
    seen = []

    def other_thread():
        with order.section(q):
            seen.append(list(cuda.log))

    with order.section(p):
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
    assert ("Q", "wait_stream", "F") in seen[0]


def test_a_copy_of_the_order_starts_fresh(order, cuda):
    p = cuda.stream("P")
    with order.section(p):
        pass
    for clone in (copy.deepcopy(order), pickle.loads(pickle.dumps(order))):
        assert clone.serial == 0 and clone.users() == ()
        assert clone.device == "cuda:0"


def test_the_module_needs_no_torch_to_be_imported():
    with open(PATH) as f:
        tree = ast.parse(f.read())
    top_level = [
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert "torch" not in top_level


def test_it_names_no_blocking_call():
    """The order is made of queued waits and recorded events. A call that waits for the
    device would defeat it."""
    with open(PATH) as f:
        source = f.read()
    for blocking in (
        ".synchronize(",
        ".item(",
        ".cpu(",
        ".query(",
        "torch.cuda.synchronize",
    ):
        assert blocking not in source, blocking
