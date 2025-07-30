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
from typing import Optional

import torch


def collective_assert(
    flag: bool, err_msg: str = "", group: torch.distributed.ProcessGroup = None
):
    flag_tensor = torch.tensor(flag, dtype=torch.bool).cuda()
    torch.distributed.all_reduce(
        flag_tensor, op=torch.distributed.ReduceOp.MIN, group=group
    )
    assert flag_tensor.item(), err_msg


def collective_assert_tensor(
    tensor: torch.Tensor,
    compare_type: str = "equal",
    pg: Optional[torch.distributed.ProcessGroup] = None,
    msg: str = "",
):
    cur_rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_tensors, tensor.contiguous(), group=pg)
    torch.distributed.barrier(group=pg)

    for i in range(world_size):
        if i == cur_rank:
            continue
        original_diff = torch.abs(tensor.reshape(-1) - gathered_tensors[i].reshape(-1))
        original_diff_val, original_diff_idx = torch.sort(
            original_diff, dim=0, descending=True
        )
        if compare_type == "equal":
            if not torch.equal(tensor, gathered_tensors[i]):
                left_bits = tensor.reshape(-1).view(torch.int32)
                right_bits = gathered_tensors[i].reshape(-1).view(torch.int32)
                diff = torch.bitwise_xor(left_bits, right_bits)
                diff_val, diff_idx = torch.sort(diff, dim=0, descending=True)

                assert torch.equal(
                    tensor, gathered_tensors[i]
                ), f"{msg} rank {cur_rank} and rank {i} tensor are not equal, selected diff bits {diff_val[0:10]}; \nmost numerically diff {original_diff_val[0:10]}"
        elif compare_type == "not_equal":
            assert not torch.equal(
                tensor, gathered_tensors[i]
            ), f"{msg}rank {cur_rank} and rank {i} tensor are equal"
        elif compare_type == "close":
            assert torch.allclose(
                tensor,
                gathered_tensors[i],
            ), f"{msg} rank {cur_rank} and rank {i} tensor are not close, diff top10 {original_diff_val[0:10]}"
        elif compare_type == "not_close":
            assert not torch.allclose(
                tensor, gathered_tensors[i]
            ), f"{msg} rank {cur_rank} and rank {i} tensor are close"
        else:
            raise ValueError(f"compare_type {compare_type} is not supported")


def grad_collective_equal_assert_hook(
    grad, pg: Optional[torch.distributed.ProcessGroup] = None, msg: str = ""
):
    grad = grad.detach()
    collective_assert_tensor(
        grad, compare_type="equal", msg=msg + f" {grad.dtype}", pg=pg
    )
    return grad
