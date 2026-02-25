import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Union

import torch
from commons.ops.collective_ops import gather_along_first_dim
from commons.perf_model.partitioner import karmarkar_karp
from commons.sequence_batch.batch import BaseBatch
from commons.utils.logger import debug_rank_0

from .batch_allgather import allgather_batch

_PRINT_LOAD_BALANCE = os.environ.get("PRINT_LOAD_BALANCE", "0") == "1"
_PRINT_LOAD_BALANCE_START = int(os.environ.get("PRINT_LOAD_BALANCE_START", "0"))
_PRINT_LOAD_BALANCE_STOP = int(os.environ.get("PRINT_LOAD_BALANCE_STOP", "-1"))


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


def _log_load_balance(
    batch_idx: int,
    all_workloads: List[float],
    partitions_indices: List[List[int]],
    local_batch_size: int,
    num_partitions: int,
) -> None:
    """Print per-rank math ops before/after load balancing and cross-rank spread.
    Called only on rank 0; all data is already globally consistent."""
    # Before balance: rank r originally owned indices [r*B, (r+1)*B)
    before_loads = [
        sum(all_workloads[r * local_batch_size : (r + 1) * local_batch_size])
        for r in range(num_partitions)
    ]
    # After balance: rank r owns partitions_indices[r]
    after_loads = [
        sum(all_workloads[idx] for idx in partitions_indices[r])
        for r in range(num_partitions)
    ]
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
        # Stash metadata needed by finish_shuffle
        self._kk_states[handle] = {
            "future": kk_future,
            "meta": {
                "allgather_workloads": allgather_workloads_cpu,
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
    ) -> BaseBatch:
        """Phase 2: Wait for KK result, then AllGather batch + index_select
        (all on main thread — NCCL safe).

        Args:
            batch: The batch to shuffle (used for data redistribution).
            handle: The handle returned by :meth:`start_shuffle_async`.

        Returns:
            The load-balanced ``BaseBatch`` for this rank.
        """
        assert handle in self._kk_states, (
            f"start_shuffle_async() must be called before finish_shuffle() "
            f"for handle {handle}"
        )

        state = self._kk_states[handle]
        partitions_indices: List[List[int]] = state["future"].result()
        meta = state["meta"]
        # Clean up state after use
        del self._kk_states[handle]

        # Optional logging (rank 0 only)
        if self._should_print_load_balance():
            aw = meta["allgather_workloads"]
            all_wl = aw.tolist() if isinstance(aw, torch.Tensor) else list(aw)
            _log_load_balance(
                int(handle),  # Convert to int for logging
                all_wl,
                partitions_indices,
                meta["local_batch_size"],
                meta["num_partitions"],
            )
        indices_this_rank = torch.tensor(
            partitions_indices[meta["rank"]],
            dtype=torch.int64,
            device=meta["device"],
        )
        indices_this_rank, _ = torch.sort(indices_this_rank)

        return self.shuffle_batch_by_global_indices(batch, indices_this_rank, pg_group)

    # ------------------------------------------------------------------
    # Original synchronous API (still available)
    # ------------------------------------------------------------------

    def compute_partition_indices(
        self,
        workloads: torch.Tensor,
        local_batch_size: int,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
    ) -> torch.Tensor:
        """AllGather workloads → KK partitioning → this-rank indices (synchronous).

        This is **batch-type agnostic** — it only operates on a 1-D workload
        tensor and returns the global indices assigned to this rank.

        Args:
            workloads: 1-D tensor of per-sample workloads (length = local_batch_size).
            local_batch_size: number of samples on this rank before allgather.
            pg_group: distributed process group.

        Returns:
            Sorted 1-D int64 tensor of global-batch indices for this rank.
        """
        assert (
            workloads.shape[0] == local_batch_size
        ), "workloads should have the same length as local_batch_size"
        num_partitions = torch.distributed.get_world_size(pg_group)
        rank = torch.distributed.get_rank(pg_group)
        # 1. Allgather the workloads
        allgather_workloads = gather_along_first_dim(workloads, pg_group)
        # 2. Partition the workloads
        partitions_indices = karmarkar_karp(
            allgather_workloads, num_partitions, equal_size=True
        )
        if self._should_print_load_balance():
            all_wl = (
                allgather_workloads.tolist()
                if isinstance(allgather_workloads, torch.Tensor)
                else list(allgather_workloads)
            )
            _log_load_balance(
                self._batch_counter,
                all_wl,
                partitions_indices,
                local_batch_size,
                num_partitions,
            )
        self._batch_counter += 1
        indices_this_rank = torch.tensor(
            partitions_indices[rank],
            dtype=torch.int64,
            device=workloads.device,
        )
        #! NOTE: This indices tensor always has a size equal to the full batch size,
        #! including padding indices for incomplete batches. Sorting ensures padding
        #! indices are stored contiguously at the tensor's end.
        indices_this_rank, _ = torch.sort(indices_this_rank)
        return indices_this_rank

    @staticmethod
    def shuffle_tensor_by_global_indices(
        tensor: torch.Tensor,
        global_indices: torch.Tensor,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
    ) -> torch.Tensor:
        """AllGather a dense tensor along dim-0, then index-select by global indices.

        Args:
            tensor: local dense tensor of shape ``[local_batch_size, ...]``.
            global_indices: global-batch indices for this rank
                (from :meth:`compute_partition_indices`).
            pg_group: distributed process group.

        Returns:
            Tensor of shape ``[len(global_indices), ...]`` containing the
            rows assigned to this rank after load-balanced redistribution.
        """
        allgathered = gather_along_first_dim(tensor, pg_group)
        return allgathered[global_indices]

    @staticmethod
    def shuffle_batch_by_global_indices(
        batch: BaseBatch,
        global_indices: torch.Tensor,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
    ) -> BaseBatch:
        """Phase 2: AllGather the batch, then index-select by pre-computed indices.

        This is the data-redistribution step that depends on ``BaseBatch``.

        Args:
            batch: local batch to allgather.
            global_indices: global-batch indices for this rank (from :meth:`compute_partition_indices`).
            pg_group: distributed process group.

        Returns:
            A new ``BaseBatch`` containing only the samples assigned to this rank.
        """
        allgathered_batch = allgather_batch(batch, pg_group)
        new_batch = allgathered_batch.index_select(global_indices)
        new_batch.batch_size = new_batch.batch_size // torch.distributed.get_world_size(
            pg_group
        )
        return new_batch

    def shuffle(
        self,
        batch: BaseBatch,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
        return_indices: bool = False,  # indices within global batch
        return_workloads: bool = False,  # for debug
        *args,
        **kwargs,
    ) -> Union[
        BaseBatch,
        Tuple[BaseBatch, torch.Tensor],
        Tuple[BaseBatch, torch.Tensor, torch.Tensor],
    ]:
        workloads = self.get_workloads(batch, *args, **kwargs)
        indices_this_rank = self.compute_partition_indices(
            workloads, batch.batch_size, pg_group
        )
        new_batch = self.shuffle_batch_by_global_indices(
            batch, indices_this_rank, pg_group
        )
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
