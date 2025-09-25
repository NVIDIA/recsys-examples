# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import commons.utils.initialize as init
import fbgemm_gpu  # for jagged_to_padded_dense
import pytest
import torch
from megatron.core import parallel_state, tensor_parallel
from ops.collective_ops import (
    gather_along_first_dim,
    gatherv_along_first_dim,
    grouped_allgatherv_tensor_list,
)
from ops.length_to_offsets import length_to_complete_offsets


def get_source_and_ref_tensor(shape=(128, 1), dtype=torch.float):
    total_L = 1
    for L in shape:
        total_L *= L
    ret = torch.arange(0, total_L, dtype=dtype).cuda().view(shape)

    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_pg = parallel_state.get_model_parallel_group()

    # prepare for padded dense reference
    # 1. get max batch size
    varied_T_tensor = torch.tensor([shape[0]], dtype=torch.int64).cuda()
    torch.distributed.all_reduce(
        varied_T_tensor, op=torch.distributed.ReduceOp.MAX, group=tp_pg
    )
    max_batch_size = varied_T_tensor.item()
    batchsize_updated_shape = [max_batch_size, *shape[1:]]

    # 2. generate max arange
    aux_shape = (tp_size,) + (1,) * len(batchsize_updated_shape)
    aux_seq = torch.arange(0, tp_size, dtype=dtype).cuda().view(*aux_shape)
    updated_total_L = total_L / shape[0] * batchsize_updated_shape[0]
    ref_base = (
        torch.arange(0, updated_total_L, dtype=dtype)
        .cuda()
        .view(batchsize_updated_shape)
    )

    # 3. expand and reshape
    expand_size = (tp_size,) + (-1,) * len(batchsize_updated_shape)
    ref = ref_base.unsqueeze(0).expand(*expand_size) * aux_seq
    ref = ref.view(
        *(
            [
                tp_size * batchsize_updated_shape[0],
            ]
            + batchsize_updated_shape[1:]
        )
    )
    return ret * tp_rank, ref.detach().clone()


@pytest.mark.parametrize(
    "batchsize_per_gpu",
    [
        13,
    ],
)
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float, torch.int64])
@pytest.mark.parametrize("seqlen", [1, 4])
def test_allgather(batchsize_per_gpu, tp, dtype, seqlen):
    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if world_size < tp:
        print(f"no enough GPUs to run tp={tp}, will skip")
        return
    init.initialize_model_parallel(tp)
    init.set_random_seed(1234)
    world_size = torch.distributed.get_world_size()
    if world_size < tp:
        print(f"no enough GPUs to run tp={tp}")
        return
    tp_pg = parallel_state.get_model_parallel_group()
    input_shape = (batchsize_per_gpu, seqlen)
    tensor_to_gather, ref_tensor = get_source_and_ref_tensor(
        shape=input_shape, dtype=dtype
    )
    if dtype == torch.float:
        tensor_to_gather.requires_grad_()
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    tensor_out = gather_along_first_dim(tensor_to_gather, tp_pg)
    try:
        assert torch.all(ref_tensor == tensor_out)
        if dtype == torch.float:
            tensor_out.sum().backward()
            assert tensor_to_gather.grad is not None
    except:
        init.destroy_global_state()
        raise
    del (
        ref_tensor,
        tensor_out,
    )
    init.destroy_global_state()


@pytest.mark.parametrize("varied_T", [13, 128])
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("embdim", [0, 1, 2])  # dim=0 indicates using 1D tensors
def test_allgatherv(varied_T, tp, dtype, embdim):
    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if world_size < tp:
        print(f"no enough GPUs to run tp={tp}")
        return
    init.initialize_model_parallel(tp)
    tp_pg = parallel_state.get_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    # to ensure the batchsize per gpu is different
    with tensor_parallel.get_cuda_rng_tracker().fork():
        varied_T_tensor = torch.randint(low=1, high=varied_T, size=(1,), device="cuda")
    batch_size_gpu = varied_T_tensor.item()
    tensor_to_gather, ref_dense_out = get_source_and_ref_tensor(
        shape=(batch_size_gpu, embdim) if embdim != 0 else (batch_size_gpu,),
        dtype=dtype,
    )
    if dtype == torch.float:
        tensor_to_gather.requires_grad_()
    tensor_out = gatherv_along_first_dim(tensor_to_gather, tp_pg)
    batch_size_gathered = torch.empty(
        size=(tp_size,), dtype=varied_T_tensor.dtype, device=varied_T_tensor.device
    )
    torch.distributed.all_gather_into_tensor(
        batch_size_gathered, varied_T_tensor.contiguous(), group=tp_pg
    )
    batch_size_offsets = length_to_complete_offsets(batch_size_gathered)
    ref_dense_out = ref_dense_out.view(tp_size, -1, max(embdim, 1))
    ref_tensor = torch.ops.fbgemm.dense_to_jagged(ref_dense_out, [batch_size_offsets])[
        0
    ]
    if embdim == 0:
        ref_tensor = ref_tensor.squeeze(-1)
    try:
        assert torch.all(ref_tensor == tensor_out)
        if dtype == torch.float:
            tensor_out.sum().backward()
            assert tensor_to_gather.grad is not None
    except:
        init.destroy_global_state()
        raise
    del (
        ref_tensor,
        tensor_out,
    )
    init.destroy_global_state()


@pytest.mark.parametrize("batchsize_per_gpu", [2, 13])
@pytest.mark.parametrize("num_tensors", [1, 3, 6])
@pytest.mark.parametrize("tp", [1])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("max_seqlen", [4, 20])
def test_grouped_allgatherv(
    batchsize_per_gpu, max_seqlen, num_tensors, hidden_size, tp
):
    init.initialize_distributed()

    world_size = torch.distributed.get_world_size()
    torch.distributed.get_rank()
    if world_size < tp:
        print(f"no enough GPUs to run tp={tp}")
        return
    init.initialize_model_parallel(tp)
    init.set_random_seed(1234)
    pre_dtypes = [torch.float32, torch.int32, torch.long, torch.bfloat16]
    import numpy as np

    # this is on cpu and all ranks produce the same result
    tensor_dtypes = np.random.choice(pre_dtypes, size=(num_tensors,))
    emb_dims = np.random.choice([0, hidden_size], size=(num_tensors,))

    # make sure we have a 1D tensor
    emb_dims[0] = 0
    tensor_dtypes[0] = torch.float
    tp_pg = parallel_state.get_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    with tensor_parallel.get_cuda_rng_tracker().fork():
        seqlen = torch.randint(
            low=1, high=max_seqlen + 1, size=(batchsize_per_gpu,), device="cuda"
        )
        varied_T_tensor = seqlen.sum()
    varied_T = varied_T_tensor.item()
    value_list = []
    dense_ref_list = []
    for i in range(num_tensors):
        embdim = emb_dims[i]
        dtype = tensor_dtypes[i]
        tensor_to_gather, ref_dense_out = get_source_and_ref_tensor(
            shape=(varied_T, embdim) if embdim != 0 else (varied_T,), dtype=dtype
        )
        if dtype == torch.float or dtype == torch.bfloat16:
            tensor_to_gather.requires_grad_()
        value_list.append(tensor_to_gather)
        dense_ref_list.append(ref_dense_out)

    tensor_out_list = grouped_allgatherv_tensor_list(value_list, tp_pg)

    varied_T_gathered = torch.empty(
        size=(tp_size,), dtype=varied_T_tensor.dtype, device=varied_T_tensor.device
    )
    torch.distributed.all_gather_into_tensor(
        varied_T_gathered, varied_T_tensor.contiguous(), group=tp_pg
    )
    varied_T_offsets = length_to_complete_offsets(varied_T_gathered)

    for i in range(num_tensors):
        embdim = emb_dims[i]
        dtype = tensor_dtypes[i]
        ref_dense_out = dense_ref_list[i].view(tp_size, -1, max(embdim, 1))
        ref_tensor = torch.ops.fbgemm.dense_to_jagged(
            ref_dense_out, [varied_T_offsets]
        )[0]
        if embdim == 0:
            ref_tensor = ref_tensor.squeeze(-1)
        try:
            assert torch.all(ref_tensor == tensor_out_list[i])
            if dtype == torch.float or dtype == torch.bfloat16:
                sum_ = tensor_out_list[i].sum() * 3.0
                assert (
                    getattr(sum_, "grad_fn", None) is not None
                ), "gathered tensor should be differentiable"
                sum_.backward()
                assert torch.all(value_list[i].grad == 3.0)
        except:
            init.destroy_global_state()
            raise
    init.destroy_global_state()
