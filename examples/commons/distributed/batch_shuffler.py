import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.cuda.nvtx as nvtx
from commons.ops.collective_ops import gather_along_first_dim
from commons.perf_model.partitioner import karmarkar_karp
from commons.sequence_batch.batch import BaseBatch
from commons.utils.logger import debug_rank_0

from .batch_all2all import _build_dst_rank_local, pad_and_all2all_batch
from .batch_allgather import pad_and_allgather_batch

_PRINT_LOAD_BALANCE = os.environ.get("PRINT_LOAD_BALANCE", "0") == "1"
_PRINT_LOAD_BALANCE_START = int(os.environ.get("PRINT_LOAD_BALANCE_START", "0"))
_PRINT_LOAD_BALANCE_STOP = int(os.environ.get("PRINT_LOAD_BALANCE_STOP", "-1"))
_SHUFFLE_WITH_ALL2ALL = os.environ.get("SHUFFLE_WITH_ALL2ALL", "0") == "1"


class ShuffleHandle:
    """Handle for tracking async shuffle state across start_shuffle_async and finish_shuffle.

    This is an opaque identifier that should not be constructed directly by users.
    It is returned by start_shuffle_async() and passed to finish_shuffle().

    Using a handle type instead of a raw integer provides:
    - Type safety: prevents passing arbitrary integers
    - Clear API: makes it obvious this is an identifier/handle
    - Future extensibility: can add metadata without breaking API
    """

    __slots__ = ("_counter",)

    def __init__(self, counter: int) -> None:
        """Internal constructor. Users should not call this directly."""
        self._counter = counter

    def __int__(self) -> int:
        """Allow conversion to int for internal use."""
        return self._counter

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, ShuffleHandle):
            return self._counter == other._counter
        return NotImplemented

    def __hash__(self) -> int:
        """Hash support for use as dict key."""
        return hash(self._counter)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ShuffleHandle(counter={self._counter})"


def _sort_partitions_padding_last(
    partitions_indices: torch.Tensor,
    allgather_wl: torch.Tensor,
) -> torch.Tensor:
    """Sort each row of ``partitions_indices`` with real samples first, padding last.

    Within each group (real / padding) the indices are in ascending order.
    When there is no padding (complete batch) this reduces to a plain
    ``sort(dim=1)``.

    Args:
        partitions_indices: ``[W, B]`` **unsorted** global indices from KK.
        allgather_wl: ``[W*B]`` allgathered workloads; padding has ``wl == 0``.

    Returns:
        Sorted ``[W, B]`` tensor.
    """
    global_batch_size = partitions_indices.numel()
    is_padding = (allgather_wl[partitions_indices] == 0).long()
    sort_key = is_padding * (global_batch_size + 1) + partitions_indices
    return partitions_indices.gather(1, sort_key.argsort(dim=1, stable=True))


def _strip_dense_padding(batch: BaseBatch, actual_bs: int) -> BaseBatch:
    """Remove trailing padding rows from dense tensors, keep KJTs intact.

    Relies on ``_sort_partitions_padding_last`` having placed real samples
    before padding samples so that a simple ``[:actual_bs]`` slice suffices.

    Args:
        batch: batch whose dense tensors have ``batch.batch_size`` rows.
        actual_bs: number of real (non-padding) rows to keep.

    Returns:
        A new ``BaseBatch`` where each dense tensor has ``actual_bs`` rows
        and KJT fields are unchanged.
    """
    full_bs = batch.batch_size

    def _fn(t: Any) -> Any:
        if isinstance(t, torch.Tensor):
            return t.reshape(full_bs, -1)[:actual_bs].reshape(-1)
        return t

    return batch._apply_to_tensors_or_kjt(_fn, inplace=False)


def _log_load_balance(
    batch_idx: int,
    all_workloads: List[float],
    partitions_indices: torch.Tensor,
    local_batch_size: int,
    num_partitions: int,
) -> None:
    """Print per-rank math ops before/after load balancing and cross-rank spread.
    Called only on rank 0; all data is already globally consistent.

    Args:
        partitions_indices: 2-D int64 tensor ``[W, B]`` (may be on GPU).
    """
    wl = torch.tensor(all_workloads, dtype=torch.float64)
    pi = partitions_indices.cpu()
    before_loads = [
        wl[r * local_batch_size : (r + 1) * local_batch_size].sum().item()
        for r in range(num_partitions)
    ]
    after_loads = [wl[pi[r]].sum().item() for r in range(num_partitions)]
    before_max, before_min = max(before_loads), min(before_loads)
    after_max, after_min = max(after_loads), min(after_loads)
    before_imb = (before_max - before_min) / before_max * 100 if before_max > 0 else 0.0
    after_imb = (after_max - after_min) / after_max * 100 if after_max > 0 else 0.0

    debug_rank_0(
        f"[Load Balance] batch={batch_idx}\n"
        f"Before: all_ranks=[{', '.join(f'{x:.3e}' for x in before_loads)}]  "
        f"max-min={before_max - before_min:.3e}  imbalance={before_imb:.2f}%\n"
        f"After:  all_ranks=[{', '.join(f'{x:.3e}' for x in after_loads)}]  "
        f"max-min={after_max - after_min:.3e}  imbalance={after_imb:.2f}%",
    )


class BaseTaskBalancedBatchShuffler:
    _batch_counter: int = 0

    def __init__(self) -> None:
        # Single-worker thread pool used exclusively for the CPU-only
        # Karmarkar-Karp partitioning algorithm so that it can overlap with
        # GPU forward / backward on the main thread.
        self._kk_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kk")
        # Use a dict to track shuffle state per batch (for prefetch pipeline)
        # Key: ShuffleHandle, Value: dict with 'future' and 'meta'
        # This allows multiple batches to be in shuffle state simultaneously
        # (important for prefetch pipeline where 3 batches are in-flight)
        # Using ShuffleHandle instead of id(batch) is more reliable because:
        # 1. id(batch) may change if _to_device creates a new object
        # 2. Python object IDs can be reused, leading to collisions
        # 3. ShuffleHandle provides a stable, monotonically increasing identifier
        self._kk_states: Dict[ShuffleHandle, Dict[str, Any]] = {}

    def __del__(self) -> None:
        """Clean up ThreadPoolExecutor on object destruction."""
        if hasattr(self, "_kk_executor"):
            # Shutdown executor, wait=False to avoid blocking during cleanup
            # The executor will finish current tasks but won't accept new ones
            self._kk_executor.shutdown(wait=False)

    @abstractmethod
    def get_workloads(self, batch: BaseBatch, *args, **kwargs) -> Any:
        raise NotImplementedError

    def _should_print_load_balance(self) -> bool:
        """Check if load balance info should be printed for the current batch."""
        if not _PRINT_LOAD_BALANCE:
            return False
        idx = self._batch_counter
        if idx < _PRINT_LOAD_BALANCE_START:
            return False
        if _PRINT_LOAD_BALANCE_STOP >= 0 and idx >= _PRINT_LOAD_BALANCE_STOP:
            return False
        return True

    # ------------------------------------------------------------------
    # Two-phase async shuffle API
    #
    # Phase 1 (``start_shuffle_async``):
    #   Main thread: get workloads → AllGather workloads (NCCL)
    #   Background thread: run KK algorithm (pure CPU, no GPU/NCCL)
    #
    # Phase 2 (``finish_shuffle``):
    #   Main thread: wait for KK result → AllGather batch (NCCL)
    #               → index_select (GPU)
    #
    # The pipeline calls Phase 1 *before* forward so that KK overlaps
    # with forward / backward.  Phase 2 is called when the indices are
    # actually needed.
    # ------------------------------------------------------------------

    @staticmethod
    def _run_kk(
        allgather_workloads: List[int],
        num_partitions: int,
    ) -> List[List[int]]:
        """Pure-CPU Karmarkar-Karp partitioning — safe to run in a thread.

        ``allgather_workloads`` **must** be a plain Python list (not a GPU
        tensor) to avoid CUDA stream-synchronisation races when called from
        a background thread.
        """
        return karmarkar_karp(allgather_workloads, num_partitions, equal_size=True)

    def start_shuffle_async(
        self,
        batch: BaseBatch,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
        *args,
        **kwargs,
    ) -> ShuffleHandle:
        """Phase 1: AllGather workloads (NCCL, main thread) then submit KK
        to a background thread.

        Returns:
            ShuffleHandle: A handle for this batch's shuffle state.
            Pass this to :meth:`finish_shuffle` to retrieve the result.

        The KK future is stored internally per batch (using handle as key);
        call :meth:`finish_shuffle` with the returned handle to complete
        the data redistribution.
        """
        workloads = self.get_workloads(batch, *args, **kwargs)
        local_batch_size = batch.batch_size
        num_partitions = torch.distributed.get_world_size(pg_group)

        assert (
            workloads.shape[0] == local_batch_size
        ), "workloads should have the same length as local_batch_size"

        # NCCL collective — must stay on the main thread
        allgather_workloads = gather_along_first_dim(workloads, pg_group)

        # CRITICAL: Convert GPU tensor to CPU list while still on the main
        # thread (inside the _memcpy_stream context).  The AllGather result
        # lives on _memcpy_stream.  If we pass the raw GPU tensor to the
        # background thread, the thread's .tolist() would issue a D2H copy on
        # its own *default stream*, which does NOT wait for _memcpy_stream to
        # finish the AllGather — a classic stream-synchronisation race that
        # leads to reading incomplete / stale data and non-deterministic KK
        # partitions.  Converting here forces the D2H onto _memcpy_stream,
        # which is serialised after the AllGather.
        allgather_workloads_cpu = allgather_workloads.tolist()

        # Create handle for this batch's shuffle state
        # This allows multiple batches to be in shuffle state simultaneously
        # (important for prefetch pipeline where 3 batches are in-flight)
        # We use ShuffleHandle instead of id(batch) because:
        # 1. _to_device may create new objects, changing id(batch)
        # 2. Python object IDs can be reused, leading to collisions
        batch_counter = self._batch_counter
        self._batch_counter += 1
        handle = ShuffleHandle(batch_counter)

        # Submit KK (pure CPU) to background thread — receives a plain Python
        # list so no GPU access happens off the main thread.
        kk_future = self._kk_executor.submit(
            self._run_kk,
            allgather_workloads_cpu,
            num_partitions,
        )
        # Padding samples sit at the tail of each rank's local chunk, so
        # checking only the last position of each chunk is sufficient — O(W).
        has_padding = any(
            allgather_workloads_cpu[(i + 1) * local_batch_size - 1] == 0
            for i in range(num_partitions)
        )
        # Stash metadata needed by finish_shuffle
        self._kk_states[handle] = {
            "future": kk_future,
            "meta": {
                "allgather_workloads_cpu": allgather_workloads_cpu,
                "allgather_workloads_gpu": allgather_workloads,
                "has_padding": has_padding,
                "local_batch_size": local_batch_size,
                "num_partitions": num_partitions,
                "rank": torch.distributed.get_rank(pg_group),
                "device": workloads.device,
            },
        }
        return handle

    def finish_shuffle(
        self,
        batch: BaseBatch,
        handle: ShuffleHandle,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
        use_all2all: bool = False,
    ) -> BaseBatch:
        """Phase 2: Wait for KK result, then redistribute batch data
        (all on main thread — NCCL safe).

        Args:
            batch: The batch to shuffle (used for data redistribution).
            handle: The handle returned by :meth:`start_shuffle_async`.
            pg_group: Distributed process group.
            use_all2all: If True, use All2All communication (more efficient).
                        If False, use AllGather + index_select (default, more robust).

        Returns:
            The load-balanced ``BaseBatch`` for this rank.
        """
        assert handle in self._kk_states, (
            f"start_shuffle_async() must be called before finish_shuffle() "
            f"for handle {handle}"
        )

        state = self._kk_states[handle]
        nvtx.range_push("kk_future_wait")
        partitions_list: List[List[int]] = state["future"].result()
        nvtx.range_pop()
        meta = state["meta"]
        # Clean up state after use
        del self._kk_states[handle]

        has_padding = meta["has_padding"]
        allgather_wl = meta["allgather_workloads_gpu"]

        # KK partitions the *global* batch (including padding samples with
        # workload == 0), so ``partitions_indices`` — and consequently
        # ``indices_this_rank`` — may reference padding positions.
        partitions_indices = torch.tensor(
            partitions_list, dtype=torch.int64, device=meta["device"]
        )
        if has_padding:
            partitions_indices = _sort_partitions_padding_last(
                partitions_indices,
                allgather_wl,
            )

        # Optional logging (rank 0 only)
        if self._should_print_load_balance():
            _log_load_balance(
                int(handle),
                meta["allgather_workloads_cpu"],
                partitions_indices,
                meta["local_batch_size"],
                meta["num_partitions"],
            )

        # NOTE: ``indices_this_rank`` has ``local_batch_size`` elements and
        # may include padding indices (real samples first, padding last
        # after ``_sort_partitions_padding_last``).  All downstream
        # operations that consume the *full* indices (allgather path's
        # ``index_select``, all2all's ``recv_ids``) therefore produce
        # batches that still contain padding samples; the padding is
        # stripped from dense tensors at the end of this method.
        indices_this_rank = partitions_indices[meta["rank"]]
        actual_bs = (
            (allgather_wl[indices_this_rank] > 0).sum().item()
            if has_padding
            else indices_this_rank.numel()
        )

        if use_all2all or _SHUFFLE_WITH_ALL2ALL:
            dst_rank, recv_counts = _build_dst_rank_local(
                partitions_indices,
                meta["rank"],
                meta["local_batch_size"],
            )
            # ``indices_this_rank`` (including padding) is passed as
            # ``recv_ids``; the all2all exchange requires all
            # ``local_batch_size`` entries to maintain symmetric counts.
            new_batch = self.shuffle_batch_by_global_indices_all2all(
                batch,
                indices_this_rank,
                pg_group,
                dst_rank=dst_rank,
                recv_counts=recv_counts,
            )
            # all2all output arrives in ascending global-index order
            # because the sender uses ``dst_rank.argsort(stable=True)``
            # to group samples by destination, which preserves ascending
            # local-index (== global-index) order within each group.
            #
            # However ``indices_this_rank`` has been reordered by
            # ``_sort_partitions_padding_last`` so that real samples come
            # first and padding samples last.  When padding exists the
            # two orderings diverge, e.g.:
            #
            #   all2all output order (ascending):  [0, 3, 4, 7]
            #   indices_this_rank   (real-first):  [0, 4, 3, 7]
            #
            # ``perm`` maps ascending → real-first so that downstream
            # code can slice ``[0 : actual_bs]`` to get exactly the real
            # samples.
            if has_padding and actual_bs < indices_this_rank.numel():
                perm = indices_this_rank.sort().indices.argsort()
                new_batch = new_batch.index_select(perm)
        else:
            # ``indices_this_rank`` (including padding) is passed as
            # ``recv_ids``; the allgather path selects all
            # ``local_batch_size`` samples so that KJT lengths remain
            # padded.  Dense padding is stripped below.
            new_batch = self.shuffle_batch_by_global_indices_allgather(
                batch,
                indices_this_rank,
                pg_group,
            )

        new_batch.actual_batch_size = actual_bs

        if has_padding and actual_bs < new_batch.batch_size:
            new_batch = _strip_dense_padding(new_batch, actual_bs)

        return new_batch

    # ------------------------------------------------------------------
    # Original synchronous API (still available)
    # ------------------------------------------------------------------

    def compute_partition_indices(
        self,
        workloads: torch.Tensor,
        local_batch_size: int,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """AllGather workloads → KK partitioning → this-rank indices (synchronous).

        This is **batch-type agnostic** — it only operates on a 1-D workload
        tensor and returns the global indices assigned to this rank.

        Args:
            workloads: 1-D tensor of per-sample workloads (length = local_batch_size).
            local_batch_size: number of samples on this rank before allgather.
            pg_group: distributed process group.

        Returns:
            Tuple of (1-D int64 tensor of global-batch indices for this rank,
            2-D int64 tensor ``[W, B]`` of KK partitions for all ranks,
            1-D tensor ``[W*B]`` of allgathered workloads,
            bool indicating whether padding exists).
        """
        assert (
            workloads.shape[0] == local_batch_size
        ), "workloads should have the same length as local_batch_size"
        num_partitions = torch.distributed.get_world_size(pg_group)
        rank = torch.distributed.get_rank(pg_group)
        # 1. Allgather the workloads
        allgather_workloads = gather_along_first_dim(workloads, pg_group)
        # KK runs on CPU; .tolist() is needed regardless.
        allgather_wl_cpu = allgather_workloads.tolist()
        # Padding sits at the tail of each rank's chunk — O(W) check.
        has_padding = any(
            allgather_wl_cpu[(i + 1) * local_batch_size - 1] == 0
            for i in range(num_partitions)
        )
        # 2. Partition the workloads.
        # KK operates on the *global* batch (including padding samples with
        # workload == 0), so the returned indices may reference padding
        # positions.  After ``_sort_partitions_padding_last``, each row
        # has real samples first and padding indices last.
        partitions_list = karmarkar_karp(
            allgather_wl_cpu, num_partitions, equal_size=True
        )
        partitions_indices = torch.tensor(
            partitions_list, dtype=torch.int64, device=workloads.device
        )
        if has_padding:
            partitions_indices = _sort_partitions_padding_last(
                partitions_indices,
                allgather_workloads,
            )
        if self._should_print_load_balance():
            _log_load_balance(
                self._batch_counter,
                allgather_wl_cpu,
                partitions_indices,
                local_batch_size,
                num_partitions,
            )
        self._batch_counter += 1
        # NOTE: ``indices_this_rank`` may contain padding indices (trailing,
        # with workload == 0).  Callers must use ``has_padding`` and
        # ``allgather_workloads`` to determine the real sample count.
        indices_this_rank = partitions_indices[rank]
        return indices_this_rank, partitions_indices, allgather_workloads, has_padding

    @staticmethod
    def shuffle_tensor_by_global_indices(
        tensor: torch.Tensor,
        recv_ids: torch.Tensor,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
    ) -> torch.Tensor:
        """AllGather a dense tensor along dim-0, then index-select by recv_ids.

        Args:
            tensor: local dense tensor of shape ``[local_batch_size, ...]``.
            recv_ids: global-batch sample IDs this rank needs to receive
                (from :meth:`compute_partition_indices`).
            pg_group: distributed process group.

        Returns:
            Tensor of shape ``[len(recv_ids), ...]`` containing the
            rows assigned to this rank after load-balanced redistribution.
        """
        allgathered = gather_along_first_dim(tensor, pg_group)
        return allgathered[recv_ids]

    @staticmethod
    def shuffle_batch_by_global_indices_allgather(
        batch: BaseBatch,
        recv_ids: torch.Tensor,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
    ) -> BaseBatch:
        """Phase 2: AllGather the batch, then index-select by pre-computed indices.

        This is the data-redistribution step that depends on ``BaseBatch``.

        Args:
            batch: local batch to allgather.
            recv_ids: global-batch sample IDs this rank needs to receive
                (from :meth:`compute_partition_indices`).  May include
                padding indices (workload == 0); the caller is responsible
                for stripping dense padding from the result.
            pg_group: distributed process group.

        Returns:
            A new ``BaseBatch`` with ``len(recv_ids)`` samples (including
            any padding samples referenced by ``recv_ids``).
        """
        allgathered_batch = pad_and_allgather_batch(batch, pg_group)
        assert isinstance(allgathered_batch, BaseBatch)
        new_batch = allgathered_batch.index_select(recv_ids)
        return new_batch

    @staticmethod
    def shuffle_batch_by_global_indices_all2all(
        batch: BaseBatch,
        recv_ids: torch.Tensor,
        pg_group: torch.distributed.ProcessGroup,
        dst_rank: torch.Tensor,
        recv_counts: "List[int]",
    ) -> BaseBatch:
        """Phase 2: All2All the batch directly to target ranks, avoiding redundant communication.

        This version uses All2All communication instead of AllGather + index_select,
        reducing communication volume by only sending samples to ranks that need them.

        Args:
            batch: local batch to redistribute.
            recv_ids: **sorted** global-batch sample IDs this rank needs to receive
                (from :meth:`compute_partition_indices`).  May include
                padding indices (workload == 0); the caller is responsible
                for stripping dense padding from the result.
            pg_group: distributed process group.
            dst_rank: Destination rank for each local sample.  Shape
                ``(batch.batch_size,)``.
            recv_counts: Per-source-rank receive counts.

        Returns:
            A new ``BaseBatch`` with ``len(recv_ids)`` samples (including
            any padding samples referenced by ``recv_ids``).
        """
        return pad_and_all2all_batch(batch, recv_ids, pg_group, dst_rank, recv_counts)

    def shuffle(
        self,
        batch: BaseBatch,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
        return_indices: bool = False,  # indices within global batch
        return_workloads: bool = False,  # for debug
        use_all2all: bool = False,
        *args,
        **kwargs,
    ) -> Union[
        BaseBatch,
        Tuple[BaseBatch, torch.Tensor],
        Tuple[BaseBatch, torch.Tensor, torch.Tensor],
    ]:
        workloads = self.get_workloads(batch, *args, **kwargs)
        # ``indices_this_rank`` may contain padding indices — see
        # ``compute_partition_indices`` for details.
        (
            indices_this_rank,
            partitions_indices,
            allgather_wl,
            has_padding,
        ) = self.compute_partition_indices(workloads, batch.batch_size, pg_group)
        actual_bs = (
            (allgather_wl[indices_this_rank] > 0).sum().item()
            if has_padding
            else indices_this_rank.numel()
        )

        if use_all2all or _SHUFFLE_WITH_ALL2ALL:
            rank = torch.distributed.get_rank(pg_group)
            dst_rank, recv_counts = _build_dst_rank_local(
                partitions_indices,
                rank,
                batch.batch_size,
            )
            new_batch = self.shuffle_batch_by_global_indices_all2all(
                batch,
                indices_this_rank,
                pg_group,
                dst_rank=dst_rank,
                recv_counts=recv_counts,
            )
            if has_padding and actual_bs < indices_this_rank.numel():
                perm = indices_this_rank.sort().indices.argsort()
                new_batch = new_batch.index_select(perm)
        else:
            new_batch = self.shuffle_batch_by_global_indices_allgather(
                batch,
                indices_this_rank,
                pg_group,
            )

        new_batch.actual_batch_size = actual_bs

        if has_padding and actual_bs < new_batch.batch_size:
            new_batch = _strip_dense_padding(new_batch, actual_bs)

        ret = new_batch
        if return_indices:
            ret = (ret, indices_this_rank)
        if return_workloads:
            ret = (*ret, workloads) if isinstance(ret, tuple) else (ret, workloads)
        return ret

    def __call__(
        self,
        batch: BaseBatch,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
        *args,
        **kwargs,
    ) -> Union[
        BaseBatch,
        Tuple[BaseBatch, torch.Tensor],
        Tuple[BaseBatch, torch.Tensor, torch.Tensor],
    ]:
        return self.shuffle(batch, pg_group, *args, **kwargs)


class IdentityBalancedBatchShuffler(BaseTaskBalancedBatchShuffler):
    def __init__(self):
        super().__init__()

    def get_workloads(self, batch: BaseBatch, *args, **kwargs):
        return 0

    def shuffle(self, batch: BaseBatch, *args, **kwargs) -> BaseBatch:
        return batch
