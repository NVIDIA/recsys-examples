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

"""The fake (meta) kernel of ``INFERENCE_EMB::table_lookup``.

The dispatcher hands a fake kernel every argument of the op's schema, so the
Python signature has to accept exactly what ``lookup_torch_binding.cu``
declares -- including ``table_global_capacities``, the optional per-table
global capacity of the ``hash_bucket`` layout. CPU-only: no extension needed.
"""

import pytest
import torch
from dynamicemb.lookup_meta import _table_lookup_fake

SCHEMA_ARGS = (
    "table_storage",
    "table_bucket_offsets",
    "bucket_capacity",
    "keys",
    "table_ids",
    "score_input",
    "policy_type",
    "ovf_storage",
    "ovf_bucket_capacity",
    "ovf_output_offsets",
    "table_global_capacities",
)


def _args(n: int = 7):
    return dict(
        table_storage=torch.empty(1024, dtype=torch.uint8),
        table_bucket_offsets=torch.tensor([0, 4]),
        bucket_capacity=128,
        keys=torch.arange(n),
        table_ids=torch.zeros(n, dtype=torch.int64),
        score_input=None,
        policy_type=0,
        ovf_storage=None,
        ovf_bucket_capacity=0,
        ovf_output_offsets=None,
        table_global_capacities=None,
    )


def test_fake_accepts_every_schema_argument_positionally():
    args = _args()
    assert tuple(args) == SCHEMA_ARGS
    score_out, founds, indices = _table_lookup_fake(*args.values())
    assert score_out.shape == founds.shape == indices.shape == (7,)
    assert score_out.dtype == indices.dtype == torch.int64
    assert founds.dtype == torch.bool


@pytest.mark.parametrize("global_caps", [None, torch.tensor([7813 * 128])])
def test_fake_takes_the_global_capacities(global_caps):
    args = _args()
    args["table_global_capacities"] = global_caps
    score_out, _, _ = _table_lookup_fake(**args)
    assert score_out.numel() == 7


def test_fake_still_works_without_the_new_argument():
    args = _args()
    del args["table_global_capacities"]
    _table_lookup_fake(**args)


def test_fake_rejects_a_2d_global_capacities():
    args = _args()
    args["table_global_capacities"] = torch.zeros(2, 2, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="1D"):
        _table_lookup_fake(**args)
