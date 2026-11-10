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

"""Orders the growth of a table against the streams that use the table.

A table that grows replaces its key map (a rehash) and may extend its value buffers. The
host calls that do it only *queue* the work on the stream that decided to grow, the
growth stream, and return. Everything else that touches the table runs on other streams
(the forward and the backward of the training, an evaluation, a dump), so four things have
to be ordered on the device, and none of them may make the host wait:

* **H1, the growth comes after the work queued before it.** The rehash reads the key map
  (keys, scores, rows, reference counters) that a backward on another stream may still be
  writing. At the start of a growth the growth stream waits, on the device, for what every
  stream that uses the table has queued so far.
* **H2, later work comes after the growth.** A stream that uses the table after the growth
  must not read a key map that is still being built. A stream waits for the end of the last
  growth the first time it uses the table after it (``wait_for_growth``), and every stream
  that has used the table is ordered after it when the growth ends.
* **H3, the replaced key map is not reused early.** The caching allocator takes a freed
  block back as soon as the stream it was allocated on has passed the free, which says
  nothing about the growth stream; the table the growth replaces is therefore marked as
  used by the growth stream (``LinearBucketTable.record_stream``).
* **Pointers do not change.** A value buffer keeps its address when it grows, so no kernel
  that was queued earlier holds a stale one.

One rule that is not about growth sits beside them, found by the gate that drives growths
under the order of the prefetch pipeline (``growth/gates``, E3): what a prefetch hands to the
forward stream is not reused early. The tensors a prefetch makes on its stream are read by the
forward and the backward on another, and the host lets go of them once the backward has been
queued, so the allocator would give them to the next prefetch while a late backward still
reads them (an illegal memory access, with or without a growth). A prefetch records them as
used by the forward stream (``batched_dynamicemb_function.record_prefetch_state_on``).

``wait_for_growth`` is the one call a user of the table makes: the forward, the backward and
the evaluation of the module, and every public method of the storages that reads the
table. It registers the current stream (for H1 of a later growth) and orders it after the
last growth (H2), at the cost of a lock and a dict lookup; it never waits for the device.
The growth itself runs inside ``section``.

Nothing here needs torch until it is called, so that the logic can be tested with stand-in
streams.
"""

import collections
import contextlib
import threading
from typing import Any, Dict, Iterator, Tuple


class _TorchCuda:
    """The CUDA facts the order needs, taken from torch when asked for."""

    @staticmethod
    def current_stream(device: Any) -> Any:
        import torch

        return torch.cuda.current_stream(device)

    @staticmethod
    def new_event() -> Any:
        import torch

        return torch.cuda.Event()

    @staticmethod
    def capturing() -> bool:
        import torch

        return torch.cuda.is_current_stream_capturing()


class GrowthOrder:
    """The streams that use one table state, and the end of its last growth.

    One per table state that can grow (a state that cannot has none and pays nothing).
    ``device`` is the state's device. A copy of the state (``copy.deepcopy``, pickle) gets
    a fresh order: streams and events belong to one process.
    """

    def __init__(self, device: Any, cuda: Any = None) -> None:
        self.device = device
        self._cuda = _TorchCuda if cuda is None else cuda
        self._lock = threading.Lock()
        # Stream handle -> the stream object, for every stream that used the table. The
        # streams of PyTorch come from a bounded pool, so this stays small; an external
        # stream that was destroyed is dropped when the first wait for it fails.
        self._users: "collections.OrderedDict[int, Any]" = collections.OrderedDict()
        # Stream handle -> serial of the last growth the stream was ordered after.
        self._seen: Dict[int, int] = {}
        self._serial = 0
        # Recorded on the growth stream after the last kernel of the last growth.
        self._done: Any = None
        # The depth of the sections the calling thread is in: a growth decision that
        # grows the key map and the value buffers makes two calls, which are one growth.
        self._local = threading.local()

    def __reduce__(self) -> Tuple[Any, ...]:
        return (GrowthOrder, (self.device,))

    @property
    def serial(self) -> int:
        """How many growths have ended."""
        return self._serial

    def users(self) -> Tuple[Any, ...]:
        """The streams that used the table so far (a snapshot)."""
        with self._lock:
            return tuple(self._users.values())

    def wait_for_growth(self) -> None:
        """Register the current stream as a user of the table and order it after the last
        growth (H2): all work submitted to the stream from now on comes after the
        growth's kernels. Cheap, and the host never waits."""
        stream = self._cuda.current_stream(self.device)
        handle = stream.cuda_stream
        with self._lock:
            self._users.setdefault(handle, stream)
            self._order_after_growth(stream, handle)

    def _order_after_growth(self, stream: Any, handle: int) -> None:
        # self._lock is held.
        done = self._done
        if done is None or self._seen.get(handle) == self._serial:
            return
        # sync-ok: a Python bool (the capture state of the stream), not a tensor
        if self._cuda.capturing():
            # A stream under capture cannot wait for an event of the past. Whoever
            # captures a step of a table that grows warms it up (and grows it) first.
            return
        stream.wait_event(done)
        self._seen[handle] = self._serial

    @contextlib.contextmanager
    def section(self, stream: Any) -> Iterator[None]:
        """One growth, whose kernels the caller launches on ``stream``.

        On entry ``stream`` waits for the work every other user stream has queued so far
        (H1). On exit the end of the growth is recorded on ``stream`` and every user stream
        is ordered after it (H2). Nested sections of one thread are one growth: only the
        outermost does anything. The exit also runs when the growth failed half way:
        whatever was launched is on ``stream``, and the others must come after it.
        """
        depth = getattr(self._local, "depth", 0)
        self._local.depth = depth + 1
        try:
            if depth == 0:
                self._begin(stream)
            yield
        finally:
            self._local.depth = depth
            if depth == 0:
                self._end(stream)

    def _begin(self, stream: Any) -> None:
        handle = stream.cuda_stream
        with self._lock:
            # The growth stream also comes after the last growth, if that ran elsewhere.
            self._order_after_growth(stream, handle)
            others = [(h, s) for h, s in self._users.items() if h != handle]
            for h, other in others:
                try:
                    stream.wait_stream(other)
                except RuntimeError:
                    self._forget(h)

    def _end(self, stream: Any) -> None:
        event = self._cuda.new_event()
        event.record(stream)
        handle = stream.cuda_stream
        with self._lock:
            self._done = event
            self._serial += 1
            self._seen[handle] = self._serial
            for h, other in list(self._users.items()):
                if h == handle:
                    continue
                try:
                    other.wait_event(event)
                except RuntimeError:
                    self._forget(h)
                else:
                    self._seen[h] = self._serial

    def _forget(self, handle: int) -> None:
        # self._lock is held: the stream is gone, nothing is queued on it.
        self._users.pop(handle, None)
        self._seen.pop(handle, None)
