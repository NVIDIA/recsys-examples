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
    # TODO@junzhang, for incomplete batch, we need do Allreduce!
    new_batch.actual_batch_size = new_batch.batch_size
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
