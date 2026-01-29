from abc import abstractmethod
from typing import Any, Tuple, Union

import torch
from commons.ops.collective_ops import gather_along_first_dim
from commons.perf_model.partitioner import karmarkar_karp
from commons.sequence_batch.batch import BaseBatch

from .batch_allgather import allgather_batch


class BaseTaskBalancedBatchShuffler:
    @abstractmethod
    def get_workloads(self, batch: BaseBatch, *args, **kwargs) -> Any:
        raise NotImplementedError

    def shuffle(
        self,
        batch: BaseBatch,
        pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
        return_indices: bool = False,  # indices within global batch
        return_workloads: bool = False,  # for debug
        *args,
        **kwargs
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
        **kwargs
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
