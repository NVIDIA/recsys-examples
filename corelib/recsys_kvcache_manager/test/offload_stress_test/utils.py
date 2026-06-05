from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

# Hook names for legacy single-offload profiler paths (not profiler_offload.py burst).
FLEXKV_CSV_METRICS: Tuple[str, ...] = (
    "flexkv.offload_kvcache_launch",
    "flexkv.offload_kvcache_wait",
    "flexkv.finish_task",
    "flexkv.cancel_task",
    "flexkv.client.put_async",
    "flexkv.client.try_wait",
    "flexkv.client.wait",
)

# profiler_offload.py burst benchmark (see profiler_offload.LAYER_METRICS).
PROFILER_OFFLOAD_LAYER_METRICS: Tuple[str, ...] = (
    "kvcache_manager.offload_try_wait",
    "host_kvstorage_manager.offload_kvcache_wait",
    "host_kvstorage_manager.finish_task",
    "flexkv_client.try_wait",
    "flexkv_client.wait",
)

# (exclusive_key, plot_label) — stack bottom → top
EXCLUSIVE_STACK: Tuple[Tuple[str, str], ...] = (
    ("put_async", "put_async"),
    ("launch_shell", "launch"),
    ("try_wait", "try_wait"),
    ("client_wait", "wait"),
)

EXCLUSIVE_SUM_KEYS: Tuple[str, ...] = tuple(key for key, _ in EXCLUSIVE_STACK)

_LAUNCH_KEYS = (
    "flexkv.offload_kvcache_launch",
    "offload_kvcache_launch_ms",
    "sum_offload_kvcache_launch_ms",
)
_WAIT_KEYS = (
    "flexkv.offload_kvcache_wait",
    "offload_kvcache_wait_ms",
    "sum_offload_kvcache_wait_ms",
)
_PUT_KEYS = (
    "flexkv.client.put_async",
    "client_put_async_ms",
    "sum_client_put_async_ms",
)
_CLIENT_WAIT_KEYS = (
    "flexkv.client.wait",
    "client_wait_ms",
    "sum_client_wait_ms",
)


def metric_short_name(hook_name: str) -> str:
    return hook_name.replace("flexkv.", "").replace(".", "_")


class IterationMetrics:
    """Per-offload-iteration FlexKV hook timings (filled during run_one_offload)."""

    def __init__(self) -> None:
        self._sum_ms: Dict[str, float] = defaultdict(float)
        self._count: Dict[str, int] = defaultdict(int)

    def record(self, name: str, elapsed_ms: float) -> None:
        self._sum_ms[name] += elapsed_ms
        self._count[name] += 1

    def get_sum_ms(self, name: str) -> float:
        return self._sum_ms[name]

    def get_count(self, name: str) -> int:
        return self._count[name]

    @property
    def hook_sums(self) -> Mapping[str, float]:
        return self._sum_ms

    def flexkv_offload_total_ms(self) -> float:
        """Raw sum of all hooks (nested overlap; debugging only)."""
        return sum(self._sum_ms[name] for name in FLEXKV_CSV_METRICS)

    def exclusive_breakdown(self) -> Dict[str, float]:
        return exclusive_breakdown_ms(self._sum_ms)

    def flexkv_offload_effective_total_ms(self) -> float:
        return sum(self.exclusive_breakdown().values())


def _first_ms(data: Mapping[str, float], keys: Tuple[str, ...]) -> float:
    for key in keys:
        if key in data:
            return float(data[key])
    return 0.0


def exclusive_breakdown_ms(raw: Mapping[str, float]) -> Dict[str, float]:
    launch = _first_ms(raw, _LAUNCH_KEYS)
    wait = _first_ms(raw, _WAIT_KEYS)
    put = _first_ms(raw, _PUT_KEYS)
    client_wait = _first_ms(raw, _CLIENT_WAIT_KEYS)
    poll_phase_ms = max(0.0, wait)
    return {
        "put_async": put,
        "launch_shell": max(0.0, launch - put),
        "try_wait": poll_phase_ms,
        "client_wait": client_wait,
    }


def exclusive_total_ms(raw: Mapping[str, float]) -> float:
    return sum(exclusive_breakdown_ms(raw).values())


def exclusive_total_ms_from_row(row: Mapping[str, str]) -> float:
    if "flexkv_offload_effective_total_ms" in row and row["flexkv_offload_effective_total_ms"]:
        return float(row["flexkv_offload_effective_total_ms"])
    return exclusive_total_ms(row)


def scenario_exclusive_sums(raw_totals: Mapping[str, float]) -> Dict[str, float]:
    return exclusive_breakdown_ms(raw_totals)


def scenario_exclusive_total(raw_totals: Mapping[str, float]) -> float:
    return exclusive_total_ms(raw_totals)


# --- origin_data CSV ---

def origin_csv_header() -> List[str]:
    cols = [
        "offload_batch_count",
        "iteration",
        "request_batch_size",
        "len_per_seq",
    ]
    for name in FLEXKV_CSV_METRICS:
        short = metric_short_name(name)
        cols.append(f"{short}_ms")
        cols.append(f"{short}_calls")
    cols.append("flexkv_offload_total_ms")
    cols.append("flexkv_offload_effective_total_ms")
    cols.append("launch_shell_ms")
    cols.append("offload_wait_poll_ms")
    return cols


def origin_csv_row(
    offload_batch_count: int,
    iteration: int,
    batch_size: int,
    len_per_seq: int,
    metrics: IterationMetrics,
) -> List:
    row: List = [offload_batch_count, iteration, batch_size, len_per_seq]
    for name in FLEXKV_CSV_METRICS:
        row.append(metrics.get_sum_ms(name))
        row.append(metrics.get_count(name))
    row.append(metrics.flexkv_offload_total_ms())
    excl = metrics.exclusive_breakdown()
    row.append(sum(excl.values()))
    row.append(excl["launch_shell"])
    row.append(excl["try_wait"])
    return row


# --- summarization CSV ---


def summarization_csv_header() -> List[str]:
    cols = [
        "offload_batch_count",
        "request_batch_size",
        "len_per_seq",
        "num_offloads_measured",
    ]
    for name in FLEXKV_CSV_METRICS:
        short = metric_short_name(name)
        cols.append(f"sum_{short}_ms")
        cols.append(f"sum_{short}_calls")
    cols.append("sum_flexkv_offload_total_ms")
    cols.append("sum_flexkv_offload_effective_total_ms")
    for key in EXCLUSIVE_SUM_KEYS:
        cols.append(f"sum_{key}_ms")
    cols.append("scenario_wall_ms")
    return cols


def summarization_csv_row(
    offload_batch_count: int,
    batch_size: int,
    len_per_seq: int,
    metrics_list: Sequence[IterationMetrics],
    scenario_wall_ms: float,
) -> List:
    n = len(metrics_list)
    if n == 0:
        raise ValueError("cannot summarize empty metrics list")
    row: List = [offload_batch_count, batch_size, len_per_seq, n]
    for name in FLEXKV_CSV_METRICS:
        row.append(sum(m.get_sum_ms(name) for m in metrics_list))
        row.append(sum(m.get_count(name) for m in metrics_list))
    row.append(sum(m.flexkv_offload_total_ms() for m in metrics_list))
    row.append(sum(m.flexkv_offload_effective_total_ms() for m in metrics_list))
    excl_totals = {key: 0.0 for key in EXCLUSIVE_SUM_KEYS}
    for m in metrics_list:
        for key, val in m.exclusive_breakdown().items():
            excl_totals[key] += val
    for key in EXCLUSIVE_SUM_KEYS:
        row.append(excl_totals[key])
    row.append(scenario_wall_ms)
    return row


def summary_row_sum_effective(summary_row: Sequence) -> float:
    """`sum_flexkv_offload_effective_total_ms` column from a summarization row."""
    # Columns after fixed prefix: 4 + 7*(ms+calls) + raw_total + effective
    return float(summary_row[-6])
