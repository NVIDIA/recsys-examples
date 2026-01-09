from typing import List

import torch
from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import (
    _get_main_grad_attr,
    _reshard_if_dtensor,
    _unshard_if_dtensor,
)
from megatron.core.distributed.finalize_model_grads import (
    finalize_model_grads as _finalize_model_grads,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

__all__ = ["finalize_model_grads"]


def _allreduce_data_parallel_embedding_grads(
    model: List[torch.nn.Module],
    all_reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
):
    if parallel_state.get_tensor_model_parallel_world_size() <= 1:
        return

    params = []
    grads = []

    for model_chunk in model:
        ddp_config = model_chunk.ddp_config
        for name, param in get_attr_wrapped_model(model_chunk, "named_parameters")():
            if param.requires_grad:
                if getattr(param, "need_tp_allreduce", False):
                    params.append(param)
                    grad_attr = _get_main_grad_attr(param, ddp_config.use_custom_fsdp)
                    grad = getattr(param, grad_attr)
                    grad = _unshard_if_dtensor(grad)
                    grads.append(grad.data)

    if grads:
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(
            coalesced,
            op=all_reduce_op,
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        for param, buf, synced in zip(
            params, grads, _unflatten_dense_tensors(coalesced, grads)
        ):
            buf.copy_(synced)
            grad_attr = _get_main_grad_attr(param, ddp_config.use_custom_fsdp)
            orig_grad = getattr(param, grad_attr)
            setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


def finalize_model_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    dp_embedding_reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
):
    _allreduce_data_parallel_embedding_grads(model, dp_embedding_reduce_op)
    _finalize_model_grads(model, config)
