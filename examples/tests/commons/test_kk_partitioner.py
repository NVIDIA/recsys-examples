"""Parity + sanity tests for the C++/Python Karmarkar-Karp partitioner.

Necessity: if the C++ port has a tie-breaking divergence from Python, every
production rank that imports ``kk_cpu_ops`` would silently produce a
different partition than the Python reference — leading to inconsistent
load-balanced batches across ranks during training.  These tests fail
loudly on any such divergence.

Sufficiency: the parity test exercises both ``equal_size=True`` (production
path) and ``equal_size=False`` modes, sweeps multiple ``k_partitions`` and
input sizes, and includes adversarial inputs (all-equal workloads, ties on
sum, ties on size) that exercise every tie-breaking branch of
``Set.__lt__`` and ``State.__lt__``.  Without the C++ extension built, the
parity tests are skipped (so CI on a fresh checkout still passes).
"""

from __future__ import annotations

import random
from typing import List

import pytest

# Reference: the pure-Python helper inside partitioner.py — always available,
# never goes through the C++ path even if kk_cpu_ops is importable.
from commons.perf_model import partitioner as _partitioner  # noqa: E402

_python_kk = _partitioner._karmarkar_karp_python


def _cpp_kk():
    """Return the C++ ``karmarkar_karp`` callable, or skip the test if the
    extension is not built.

    We piggyback on partitioner.py's resolution logic so we exercise the same
    install/inplace lookup the production code uses.
    """
    if _partitioner._kk_cpu_ops is None:
        pytest.skip("kk_cpu_ops C++ extension not built (or KK_FORCE_PYTHON=1)")
    return _partitioner._kk_cpu_ops.karmarkar_karp


def _canonicalize(partitions: List[List[int]]) -> List[List[int]]:
    """KK output ordering is meaningful (set 0 is the heaviest); we compare
    raw orderings to catch tie-breaking divergence, but for the
    "covers the same indices" sanity check we sort within each partition."""
    return [sorted(p) for p in partitions]


# ---------------------------------------------------------------------------
# Parity: C++ output must exactly equal Python output (same tie-breaks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 4, 8, 16])
@pytest.mark.parametrize("multiple", [1, 4, 32, 128])
def test_parity_equal_size_random(k: int, multiple: int) -> None:
    kk_cpp = _cpp_kk()
    n = k * multiple
    rng = random.Random(0xC0FFEE + k * 1000 + multiple)
    workloads = [rng.randint(1, 10_000) for _ in range(n)]

    py_out = _python_kk(workloads, k, equal_size=True)
    cpp_out = kk_cpp(workloads, k, True)

    assert cpp_out == py_out, (
        f"C++ and Python KK diverged (k={k}, n={n}). "
        f"py={py_out[:2]}... cpp={cpp_out[:2]}..."
    )


@pytest.mark.parametrize("k", [2, 4, 8])
def test_parity_unequal_size(k: int) -> None:
    kk_cpp = _cpp_kk()
    rng = random.Random(0xBEEF + k)
    # Non-multiple of k on purpose.
    n = k * 7 + 3
    workloads = [rng.randint(1, 1_000) for _ in range(n)]

    py_out = _python_kk(workloads, k, equal_size=False)
    cpp_out = kk_cpp(workloads, k, False)
    assert cpp_out == py_out


def test_parity_all_equal_workloads() -> None:
    """Adversarial: every Set has identical ``sum`` at every merge step, so
    the tie-break falls through to ``len(items)`` and then to lex compare on
    items.  Any difference in how C++ vs Python lex-compares ``(idx, val)``
    pairs would surface here."""
    kk_cpp = _cpp_kk()
    k = 4
    workloads = [42] * (k * 8)
    py_out = _python_kk(workloads, k, equal_size=True)
    cpp_out = kk_cpp(workloads, k, True)
    assert cpp_out == py_out


def test_parity_size_tie_branch() -> None:
    """Adversarial: distinct sums, then deliberate ties on ``len(items)``
    after the first merge to exercise the size tie-break."""
    kk_cpp = _cpp_kk()
    # Crafted so several Sets land on the same sum but different item counts.
    workloads = [1, 1, 2, 2, 3, 3, 4, 4] * 2  # 16 items, k=4
    py_out = _python_kk(workloads, 4, equal_size=True)
    cpp_out = kk_cpp(workloads, 4, True)
    assert cpp_out == py_out


# ---------------------------------------------------------------------------
# Sufficiency: invariants on the output regardless of which path produced it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 8])
def test_partitions_cover_all_indices_exactly_once(k: int) -> None:
    kk_cpp = _cpp_kk()
    n = k * 16
    rng = random.Random(1234)
    workloads = [rng.randint(1, 100) for _ in range(n)]
    out = kk_cpp(workloads, k, True)
    flat = sorted(i for part in out for i in part)
    assert flat == list(range(n)), "partitions must cover [0, n) exactly once"


def test_equal_size_partition_balance_is_better_than_naive() -> None:
    """The whole point of KK is that the max-min spread should be smaller than
    the naive 'first n/k samples per rank' chunking on skewed input."""
    kk_cpp = _cpp_kk()
    k = 8
    n = k * 32
    rng = random.Random(0xDADA)
    # Skewed: a few heavy samples + lots of light ones.
    workloads = [rng.randint(1, 5) for _ in range(n)]
    for _ in range(n // 10):
        workloads[rng.randrange(n)] = rng.randint(1000, 5000)

    naive_loads = [sum(workloads[r * (n // k) : (r + 1) * (n // k)]) for r in range(k)]
    naive_spread = max(naive_loads) - min(naive_loads)

    partitions = kk_cpp(workloads, k, True)
    kk_loads = [sum(workloads[i] for i in p) for p in partitions]
    kk_spread = max(kk_loads) - min(kk_loads)

    assert (
        kk_spread < naive_spread
    ), f"KK should reduce spread vs naive; got kk={kk_spread} naive={naive_spread}"


def test_invalid_inputs_raise() -> None:
    kk_cpp = _cpp_kk()
    with pytest.raises(Exception):  # noqa: B017 -- pybind11 error type may vary
        kk_cpp([1, 2, 3, 4, 5], 4, True)  # n=5 not divisible by k=4
    with pytest.raises(Exception):  # noqa: B017
        kk_cpp([1, 2, 3, 4], 0, True)  # k=0


def test_gil_released_during_compute() -> None:
    """If the C++ path does not release the GIL, this test will deadlock or
    massively under-run the expected concurrency.  We run a batch of KKs in a
    background thread and a tight Python compute loop in the main thread —
    if GIL is released, both make wall-clock progress in parallel.

    We measure 'main thread iterations completed while KK runs' and compare
    to a baseline where the main thread runs alone for the same wall-clock
    duration.  >50% throughput proves the GIL is released.

    To stay robust under CI load:
      * the worker runs N=8 KKs back-to-back so per-KK thread-create /
        scheduling overhead is amortised below 2%.
      * the worker signals a ``threading.Event`` after acquiring the GIL
        and right before its first compute, and the main thread anchors
        its overlap deadline to *that* moment instead of ``thread.start()``
        — eliminating the scheduling-jitter window.
    """
    kk_cpp = _cpp_kk()
    import threading
    import time

    # Large input so the C++ KK takes long enough to overlap (~ms-scale).
    k = 8
    n = k * 4096
    rng = random.Random(7)
    workloads = [rng.randint(1, 10_000) for _ in range(n)]

    n_reps = 8

    # Warm up
    kk_cpp(workloads, k, True)

    # Time baseline: main-thread tight loop alone for N KK durations
    t0 = time.perf_counter()
    for _ in range(n_reps):
        kk_cpp(workloads, k, True)
    kk_wall = time.perf_counter() - t0

    def tight_python_work(deadline: float) -> int:
        # Simple Python work that needs the GIL constantly.
        n_iters = 0
        x = 0
        while time.perf_counter() < deadline:
            for _ in range(1_000):
                x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            n_iters += 1
        return n_iters

    baseline_iters = tight_python_work(time.perf_counter() + kk_wall)

    # Now run KK in background, measure main-thread iters concurrently.
    # The worker fires ``started`` after thread scheduling completes so the
    # main thread can anchor its overlap window to the actual compute start,
    # not to ``thread.start()``.
    result_holder: list = []
    started = threading.Event()

    def run_kk() -> None:
        started.set()
        for _ in range(n_reps):
            result_holder.append(kk_cpp(workloads, k, True))

    th = threading.Thread(target=run_kk)
    th.start()
    started.wait()
    t_start = time.perf_counter()
    overlapped_iters = tight_python_work(t_start + kk_wall)
    th.join()

    assert len(result_holder) == n_reps, "KK thread did not produce all results"
    # If GIL is released, main thread should retain at least ~50% throughput
    # of the GIL-free baseline.  With the Python KK fallback, this number
    # would typically be <5% due to GIL contention.
    ratio = overlapped_iters / max(baseline_iters, 1)
    assert ratio > 0.5, (
        f"main-thread throughput dropped to {ratio:.1%} while KK ran — "
        f"C++ KK does not appear to release the GIL "
        f"(baseline={baseline_iters} overlap={overlapped_iters})"
    )
