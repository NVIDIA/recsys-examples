from typing import Union

import torch
from commons.ops.collective_ops import (
    gather_along_first_dim,
    keyed_jagged_tensor_allgather,
)
from commons.sequence_batch.batch import BaseBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def allgather_batch(
    batch: BaseBatch,
    pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
):
    """
    Allgather the batch across the process group.
    Note we will update the feature_to_max_seqlen in the batch.
    """

    def allgather_tensor_or_kjt(tensor_or_kjt: Union[torch.Tensor, KeyedJaggedTensor]):
        if isinstance(tensor_or_kjt, torch.Tensor):
            ag_object = gather_along_first_dim(tensor_or_kjt, pg_group)
            return ag_object
        elif isinstance(tensor_or_kjt, KeyedJaggedTensor):
            return keyed_jagged_tensor_allgather(tensor_or_kjt, pg_group)
        else:
            raise ValueError(f"Unsupported type: {type(tensor_or_kjt)}")

    new_batch = batch._apply_to_tensors_or_kjt(allgather_tensor_or_kjt, inplace=False)
    new_batch.batch_size = new_batch.batch_size * torch.distributed.get_world_size(
        pg_group
    )
    # use allreduce to sum up actual_batch_size on all processes
    actual_batch_size = torch.tensor(
        batch.actual_batch_size, device=batch.features.lengths().device
    )
    torch.distributed.all_reduce(
        actual_batch_size, op=torch.distributed.ReduceOp.SUM, group=pg_group
    )
    # this will block host until all processes have finished the allreduce.
    # TODO@junzhang, can we use a non-blocking allreduce?
    new_batch.actual_batch_size = actual_batch_size.item()
    # update feature_to_max_seqlen
    # This will cause a host sync
    # TODO@junzhang, can we avoid this?
    for feature_name in batch.features.keys():
        new_batch.feature_to_max_seqlen[feature_name] = torch.max(
            new_batch.features[feature_name].lengths()
        ).item()
    return new_batch


def allgather_batch_seqlen(
    batch: BaseBatch,
    pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
):
    """
    Allgather the batch across the process group.
    """
    seqlen = batch.features.lengths()
    seqlen_allgathered = gather_along_first_dim(seqlen, pg_group)
    return seqlen_allgathered
