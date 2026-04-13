import math
from dataclasses import fields
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from commons.ops.collective_ops import (
    gather_along_first_dim,
    keyed_jagged_tensor_list_allgather,
)
from commons.sequence_batch.batch import BaseBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _elems_per_sample(t: torch.Tensor, actual_batch_size: int) -> int:
    """Return the number of flat elements per sample in *t*.

    When *actual_batch_size* > 0 we can simply divide; when the batch is
    empty we fall back to the product of all dimensions after dim-0.
    """
    if actual_batch_size > 0:
        return t.numel() // actual_batch_size
    return math.prod(t.shape[1:]) if t.dim() > 1 else 1


def pad_and_allgather_batch(
    batch: BaseBatch,
    pg_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
    return_padding_flag: bool = False,
) -> Union[BaseBatch, Tuple[BaseBatch, torch.Tensor]]:
    """
    Allgather the batch across the process group.

    All KJT fields are fused into a **single** AllGather call pair
    (1 for lengths, 1 for values) via :func:`keyed_jagged_tensor_list_allgather`.
    Dense tensor fields are gathered separately.

    If ``actual_batch_size < batch_size`` on any rank, dense tensors
    are zero-padded to ``batch_size`` before gathering so that all
    ranks contribute the same dim-0 and global sample indices remain
    valid for the subsequent ``index_select``.

    **world_size == 1 fast-path**: When the process group contains only
    one rank, no collective communication is performed.  Dense tensors
    are still zero-padded to ``batch_size`` when ``actual_batch_size <
    batch_size`` (incomplete batch) and ``actual_batch_size`` is set to
    ``batch_size`` for consistency with the multi-rank code path.  KJT
    fields are returned as-is.

    Args:
        return_padding_flag: When True, an extra AllGather is performed
            to build a bool tensor of shape ``[global_batch_size]``
            indicating which positions are padding (True = padding).

    Returns:
        If *return_padding_flag* is False (default), returns the
        allgathered ``BaseBatch`` directly.  If True, returns a tuple
        of (allgathered_batch, is_padding) where ``is_padding`` is a
        bool tensor of shape ``[global_batch_size]``.
    """
    world_size = torch.distributed.get_world_size(pg_group)
    device = batch.features.values().device
    global_batch_size = batch.batch_size * world_size

    # ---- Fast path: world_size == 1 — only pad dense tensors, no collectives ----
    if world_size == 1:
        if batch.actual_batch_size < batch.batch_size:
            orig_actual_bs = batch.actual_batch_size

            def _pad_dense(
                tensor_or_kjt: Union[torch.Tensor, KeyedJaggedTensor],
            ) -> Union[torch.Tensor, KeyedJaggedTensor]:
                if isinstance(tensor_or_kjt, KeyedJaggedTensor):
                    return tensor_or_kjt
                elif isinstance(tensor_or_kjt, torch.Tensor):
                    t = tensor_or_kjt
                    eps = _elems_per_sample(t, orig_actual_bs)
                    pad_size = batch.batch_size * eps - t.numel()
                    return F.pad(t, (0, pad_size)) if pad_size > 0 else t
                else:
                    raise ValueError(f"Unsupported type: {type(tensor_or_kjt)}")

            new_batch = batch._apply_to_tensors_or_kjt(_pad_dense, inplace=False)
            new_batch.actual_batch_size = global_batch_size
        else:
            new_batch = batch

        if not return_padding_flag:
            return new_batch
        is_padding = (
            torch.arange(batch.batch_size, device=device) >= batch.actual_batch_size
        )
        return new_batch, is_padding

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

    # ---- Phase 2: gather dense tensors (pad if needed) ----
    pad_dense = batch.actual_batch_size < batch.batch_size

    def allgather_field(tensor_or_kjt: Union[torch.Tensor, KeyedJaggedTensor]):
        if isinstance(tensor_or_kjt, KeyedJaggedTensor):
            return tensor_or_kjt
        elif isinstance(tensor_or_kjt, torch.Tensor):
            if pad_dense:
                eps = _elems_per_sample(tensor_or_kjt, batch.actual_batch_size)
                pad_size = batch.batch_size * eps - tensor_or_kjt.numel()
                padded = F.pad(tensor_or_kjt, (0, pad_size))
                return gather_along_first_dim(padded, pg_group)
            return gather_along_first_dim(tensor_or_kjt, pg_group)
        else:
            raise ValueError(f"Unsupported type: {type(tensor_or_kjt)}")

    new_batch = batch._apply_to_tensors_or_kjt(allgather_field, inplace=False)

    for name, kjt_out in kjt_result_map.items():
        setattr(new_batch, name, kjt_out)

    new_batch.batch_size = global_batch_size
    # NOTE: actual_batch_size is set to global_batch_size (including padding
    # rows) because computing the true sum would require an extra collective.
    # Callers that need the correct actual count (e.g. finish_shuffle) must
    # recompute it from the partition indices.
    new_batch.actual_batch_size = global_batch_size

    if not return_padding_flag:
        return new_batch

    # ---- Phase 3: build global is_padding mask ----
    local_is_padding = (
        torch.arange(batch.batch_size, device=device) >= batch.actual_batch_size
    )
    global_is_padding = torch.empty(global_batch_size, dtype=torch.bool, device=device)
    torch.distributed.all_gather_into_tensor(
        global_is_padding, local_is_padding, group=pg_group
    )
    return new_batch, global_is_padding


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
