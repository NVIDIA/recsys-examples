import os
from abc import abstractmethod
from typing import Any, List, Tuple, Union

import torch
from commons.ops.collective_ops import gather_along_first_dim
from commons.perf_model.partitioner import karmarkar_karp
from commons.sequence_batch.batch import BaseBatch
from commons.utils.logger import debug_rank_0

from .batch_allgather import allgather_batch

_PRINT_LOAD_BALANCE = os.environ.get("PRINT_LOAD_BALANCE", "0") == "1"
_PRINT_LOAD_BALANCE_START = int(os.environ.get("PRINT_LOAD_BALANCE_START", "0"))
_PRINT_LOAD_BALANCE_STOP = int(os.environ.get("PRINT_LOAD_BALANCE_STOP", "-1"))


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
        assert (
            workloads.shape[0] == batch.batch_size
        ), "workloads should have the same shape as batch_size"
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
                else list[Any](allgather_workloads)
            )
            _log_load_balance(
                self._batch_counter,
                all_wl,
                partitions_indices,
                batch.batch_size,
                num_partitions,
            )
        self._batch_counter += 1
        indices_this_rank = torch.tensor(
            partitions_indices[rank],
            dtype=torch.int64,
            device=batch.features.lengths().device,
        )
        #! NOTE: This indices tensor always has a size equal to the full batch size,
        #! including padding indices for incomplete batches. Sorting ensures padding
        #! indices are stored contiguously at the tensor's end.
        indices_this_rank, _ = torch.sort(indices_this_rank)  #
        # 3. Allgather the batch, the batchsize is multiplied by the world size.
        allgathered_batch = allgather_batch(batch, pg_group)
        # 4. Select the batch
        new_batch = allgathered_batch.index_select(indices_this_rank)
        new_batch.batch_size = new_batch.batch_size // torch.distributed.get_world_size(
            pg_group
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
        pass

    def get_workloads(self, batch: BaseBatch, *args, **kwargs):
        return 0

    def shuffle(self, batch: BaseBatch, *args, **kwargs) -> BaseBatch:
        return batch
