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
from typing import Tuple

import torch


def hstu_close(
    actual, out_ref, fp32_ref, try_allclose: bool = False, multiplier: int = 2
) -> bool:
    """
    compare the maximum absolute error between actual and fp32_ref
    with the maximum absolute error between out_ref and fp32_ref
    """
    actual = actual.reshape(-1)
    out_ref = out_ref.reshape(-1)
    fp32_ref = fp32_ref.reshape(-1)
    assert fp32_ref.dtype == torch.float32, "fp32_ref should be float32"

    try_allclose = torch.allclose(actual, out_ref) and try_allclose

    left_abs_max = (actual - fp32_ref).abs().max().item()
    right_abs_max = (out_ref - fp32_ref).abs().max().item()
    return (left_abs_max <= multiplier * right_abs_max) or (try_allclose)


def assert_hstu_close(
    actual, out_ref, fp32_ref, try_allclose: bool = False, fwd: bool = True
):
    """
    compare the maximum absolute error between actual and fp32_ref
    with the maximum absolute error between out_ref and fp32_ref
    """
    close_flag = hstu_close(
        actual, out_ref, fp32_ref, try_allclose=try_allclose, multiplier=2 if fwd else 5
    )
    assert close_flag


def max_abs_diff(actual, out_ref, fp32_ref) -> Tuple[float, float, float]:
    L = actual.size(0)
    actual = actual.view(L, -1)
    out_ref = out_ref.view(L, -1)
    fp32_ref = fp32_ref.view(L, -1)
    assert fp32_ref.dtype == torch.float32, "fp32_ref should be float32"

    return (
        (actual - out_ref).abs().max().item(),
        (actual - fp32_ref).abs().max().item(),
        (out_ref - fp32_ref).abs().max().item(),
    )


# diff_pair:
# - 01 : arg0 - arg1
# - 02 : arg0 - arg2
# - 12 : arg1 - arg2
def top_k_by_diff_abs(
    actual, out_ref, fp32_ref, k: int = 10, diff_pair="01"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actual = actual.reshape(-1)
    out_ref = out_ref.reshape(-1)
    fp32_ref = fp32_ref.reshape(-1)
    assert diff_pair in ["01", "02", "12"], "diff_pair should be 01, 02, or 12"
    assert fp32_ref.dtype == torch.float32, "fp32_ref should be float32"

    if diff_pair == "01":
        diff = actual - out_ref
    elif diff_pair == "02":
        diff = actual - fp32_ref
    elif diff_pair == "12":
        diff = out_ref - fp32_ref

    diff_abs = diff.abs()
    k = min(k, diff_abs.size(0))
    _, top_k_indices = diff_abs.topk(k, dim=0)
    top_k_indices = top_k_indices

    top_k_actual = actual[top_k_indices]
    top_k_out_ref = out_ref[top_k_indices]
    top_k_fp32_ref = fp32_ref[top_k_indices]
    return top_k_actual, top_k_out_ref, top_k_fp32_ref
