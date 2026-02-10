from dataclasses import fields
from typing import Dict, List, Union

import torch
from commons.ops.collective_ops import (
    gather_along_first_dim,
    gatherv_along_first_dim,
    keyed_jagged_tensor_list_allgather,
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

    All KJT fields are fused into a **single** AllGather call pair
    (1 for lengths, 1 for values) via :func:`keyed_jagged_tensor_list_allgather`.
    Dense tensor fields are gathered separately.
    """

    # TODO@junzhang, we should avoid coping with jagged padding...
    # use allreduce to sum up actual_batch_size on all processes
    # We need this for index_select!
    actual_batch_size = torch.tensor(
        batch.actual_batch_size, device=batch.features.lengths().device
    )
    torch.distributed.all_reduce(
        actual_batch_size, op=torch.distributed.ReduceOp.SUM, group=pg_group
    )
    world_size = torch.distributed.get_world_size(pg_group)
    global_batch_size = batch.batch_size * world_size

    # ---- Phase 1: collect KJT fields and fused AllGather them ----
    kjt_field_names: List[str] = []
    kjt_inputs: List[KeyedJaggedTensor] = []
    for f in fields(batch):
        val = getattr(batch, f.name)
        if isinstance(val, KeyedJaggedTensor):
            kjt_field_names.append(f.name)
            kjt_inputs.append(val)

    kjt_outputs = keyed_jagged_tensor_list_allgather(kjt_inputs, pg_group)
    kjt_result_map: Dict[str, KeyedJaggedTensor] = dict(
        zip(kjt_field_names, kjt_outputs)
    )

    # ---- Phase 2: gather dense tensors one by one ----
    def allgather_field(tensor_or_kjt: Union[torch.Tensor, KeyedJaggedTensor]):
        if isinstance(tensor_or_kjt, KeyedJaggedTensor):
            # Already gathered in Phase 1 – will be patched in below.
            return tensor_or_kjt  # placeholder, replaced after
        elif isinstance(tensor_or_kjt, torch.Tensor):
            if actual_batch_size != global_batch_size:
                ag_object = gatherv_along_first_dim(tensor_or_kjt, pg_group)
            else:
                ag_object = gather_along_first_dim(tensor_or_kjt, pg_group)
            return ag_object
        else:
            raise ValueError(f"Unsupported type: {type(tensor_or_kjt)}")

    new_batch = batch._apply_to_tensors_or_kjt(allgather_field, inplace=False)

    # Patch the KJT fields with the fused results
    for name, kjt_out in kjt_result_map.items():
        setattr(new_batch, name, kjt_out)

    new_batch.batch_size = new_batch.batch_size * world_size
    # this will block host until all processes have finished the allreduce.
    new_batch.actual_batch_size = actual_batch_size.item()
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
