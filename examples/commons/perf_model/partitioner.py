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
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import heapq
import os
from typing import Any, List, Tuple, Union

import numpy as np

try:
    from torch import Tensor
    from torch.cuda import nvtx
except ImportError:
    Tensor = None
    nvtx = None

# Optional C++ accelerator. Same output as the Python implementation but
# releases the GIL for the entire compute, so the main thread can keep
# submitting CUDA kernels while KK runs in a background ThreadPoolExecutor.
# Set ``KK_FORCE_PYTHON=1`` to bypass the C++ path (useful for parity tests).
#
# Resolution order:
#   1. Honour ``KK_FORCE_PYTHON=1`` → no native module.
#   2. Top-level import — the location used by ``python setup.py install``
#      inside the container (``/usr/local/lib/.../dist-packages``).
#   3. Sibling .so next to the ``perf_model`` package — the location used by
#      ``python setup.py build_ext --inplace`` during dev iteration.
_FORCE_PYTHON = os.environ.get("KK_FORCE_PYTHON", "0") == "1"
_kk_cpu_ops = None
if not _FORCE_PYTHON:
    try:
        import kk_cpu_ops as _kk_cpu_ops  # type: ignore[import-not-found,no-redef]
    except ImportError:
        import glob as _glob
        import importlib.util as _importlib_util
        import sys as _sys

        _so_glob = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "kk_cpu_ops*.so",
        )
        _matches = sorted(_glob.glob(_so_glob))
        if _matches:
            _spec = _importlib_util.spec_from_file_location("kk_cpu_ops", _matches[0])
            if _spec is not None and _spec.loader is not None:
                # exec_module can raise on a stale / ABI-incompatible .so
                # (e.g. left over from a different Python or torch ABI).
                # Catch broadly so the partitioner module still imports and
                # falls back to the pure-Python implementation in that case.
                try:
                    _kk_cpu_ops = _importlib_util.module_from_spec(_spec)
                    _spec.loader.exec_module(_kk_cpu_ops)
                    # Register so ``import kk_cpu_ops`` elsewhere reuses the
                    # same module object instead of failing or re-loading.
                    _sys.modules.setdefault("kk_cpu_ops", _kk_cpu_ops)
                except Exception:
                    _kk_cpu_ops = None


def karmarkar_karp(
    workloads: Union[np.ndarray, List[int], Tensor], k_partitions: int, equal_size: bool
):
    """K-way load-balanced partitioning via Karmarkar-Karp.

    Returns ``k_partitions`` lists of original indices.  When the C++ accelerator
    ``kk_cpu_ops`` is importable, the heavy heap traversal runs without the
    GIL; otherwise the pure-Python fallback below is used.  Output is
    bit-identical between the two paths (same tie-breaking).
    """
    if nvtx is not None:
        nvtx.range_push("karmarkar_karp")
    try:
        # Normalize to a plain Python list of ints.  Tensors / ndarrays both
        # have ``.tolist()``; built-in lists do not, so a hasattr check picks
        # the right branch.
        if hasattr(workloads, "tolist"):
            workloads = workloads.tolist()

        if _kk_cpu_ops is not None:
            partitions = _kk_cpu_ops.karmarkar_karp(workloads, k_partitions, equal_size)
        else:
            partitions = _karmarkar_karp_python(workloads, k_partitions, equal_size)

        if equal_size:
            for partition in partitions:
                assert len(partition) * k_partitions == len(
                    workloads
                ), f"{len(partition)} * {k_partitions} != {len(workloads)}"
        return partitions
    finally:
        if nvtx is not None:
            nvtx.range_pop()  # karmarkar_karp


def _karmarkar_karp_python(workloads: List[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items: List[Tuple[int, int]] = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, workload) in enumerate(items):
                self.sets[i].add(idx=idx, val=workload)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, workload) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(workload)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_workloads = sorted([(workload, i) for i, workload in enumerate(workloads)])
    states_pq: List[Any] = []
    if equal_size:
        assert (
            len(workloads) % k_partitions == 0
        ), f"{len(workloads)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_workloads), k_partitions):
            items = []
            for i in range(k_partitions):
                workload, idx = sorted_workloads[offset + i]
                items.append((idx, workload))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for workload, idx in sorted_workloads:
            heapq.heappush(states_pq, State(items=[(idx, workload)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    return states_pq[0].get_partitions()


if __name__ == "__main__":
    workloads = [100, 200, 300, 400, 500]
    k_partitions = 2
    equal_size = False
    partitions = karmarkar_karp(
        workloads=workloads, k_partitions=k_partitions, equal_size=equal_size
    )
