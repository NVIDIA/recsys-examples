from typing import Dict, Optional, Union

import torch.distributed as dist
from dataset.utils import RankingBatch, RetrievalBatch
from megatron.core import parallel_state
from ops.collective_ops import (
    gather_along_first_dim,
    gatherv_along_first_dim,
    jagged_tensor_allgather,
    keyed_jagged_tensor_allgather,
)
from ops.grad_scaling import grad_scaling
from torchrec.sparse.jagged_tensor import JaggedTensor


def jt_dict_grad_scaling_and_allgather(
    jt: Dict[str, JaggedTensor],
    scaling_factor: Union[int, float] = 1,
    pg: Optional[dist.ProcessGroup] = None,
):
    if pg is None:
        pg = parallel_state.get_tensor_model_parallel_group()
    tp_size = dist.get_world_size(group=pg)

    if tp_size == 1:
        return jt

    output_jt_dict = {}
    # value is a JaggedTensor, value._values is a Tensor
    for key, value in jt.items():
        value._values = grad_scaling(value._values, scaling_factor)
        output_jt_dict[key] = jagged_tensor_allgather(value, pg)
    return output_jt_dict


# The features is a kjt, input to embedding module.
def dmp_batch_to_tp(
    batch: Union[RetrievalBatch, RankingBatch], exclude_features: bool = True
):
    tp_pg = parallel_state.get_tensor_model_parallel_group()
    tp_size = dist.get_world_size(group=tp_pg)
    batch_cls = type(batch)
    output_batch = batch_cls(**batch.__dict__)
    if tp_size == 1:
        return output_batch
    output_batch.batch_size = output_batch.batch_size * tp_size
    if not exclude_features:
        output_batch.features = keyed_jagged_tensor_allgather(batch.features, tp_pg)

    if batch.num_candidates is not None:
        output_batch.num_candidates = gather_along_first_dim(
            batch.num_candidates, tp_pg
        )

    if hasattr(batch, "labels") and batch.labels is not None:
        output_batch.labels = gatherv_along_first_dim(batch.labels, tp_pg)

    return output_batch
