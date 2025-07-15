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
import torch


def assert_hstu_close(actual, out_ref, fp32_ref, fwd: bool = True):
    """
    compare the maximum absolute error between actual and fp32_ref
    with the maximum absolute error between out_ref and fp32_ref
    """
    L = actual.size(0)
    actual = actual.view(L, -1)
    out_ref = out_ref.view(L, -1)
    fp32_ref = fp32_ref.view(L, -1)
    assert fp32_ref.dtype == torch.float32, "fp32_ref should be float32"

    multiplier = 2 if fwd else 5
    left_abs_max = (actual - fp32_ref).abs().max().item()
    right_abs_max = (out_ref - fp32_ref).abs().max().item()

    assert left_abs_max <= multiplier * right_abs_max, (
        f"actual - fp32_ref: {left_abs_max}, " f"out_ref - fp32_ref: {right_abs_max}"
    )


def hstu_close(actual, out_ref, fp32_ref, multiplier: int = 2):
    """
    compare the maximum absolute error between actual and fp32_ref
    with the maximum absolute error between out_ref and fp32_ref
    """
    L = actual.size(0)
    actual = actual.view(L, -1)
    out_ref = out_ref.view(L, -1)
    fp32_ref = fp32_ref.view(L, -1)
    assert fp32_ref.dtype == torch.float32, "fp32_ref should be float32"

    left_abs_max = (actual - fp32_ref).abs().max().item()
    right_abs_max = (out_ref - fp32_ref).abs().max().item()

    return left_abs_max <= multiplier * right_abs_max
