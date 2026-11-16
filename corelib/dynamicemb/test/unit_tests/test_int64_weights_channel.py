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

"""An int64 KJT ``weights`` tensor rides the input-dist path bit-for-bit.

Per-key scores are carried in the KJT weights channel as int64. These tests
check the pieces of that path that act on weights: the RW bucketizer, and the
KJT permute / split that surround it.
"""

import pytest
import torch
from dynamicemb.shard.input_dist import bucketize_kjt_before_all2all, dist_type_codes
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _make_kjt(weights_dtype, num_features=3, batch=8, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lengths = torch.randint(0, 5, (num_features * batch,), generator=g)
    n = int(lengths.sum())
    values = torch.randint(0, 1000, (n,), generator=g, dtype=torch.int64)
    if weights_dtype == torch.int64:
        # Large values, above 2**53, so a float round trip would lose bits.
        weights = torch.randint(2**60, 2**62, (n,), generator=g, dtype=torch.int64)
    else:
        weights = torch.rand(n, generator=g).to(weights_dtype)
    return KeyedJaggedTensor.from_lengths_sync(
        keys=[f"f{i}" for i in range(num_features)],
        values=values.cuda(),
        weights=weights.cuda(),
        lengths=lengths.cuda(),
    )


@pytest.mark.parametrize("dist_type", ["continuous", "hash_roundrobin"])
@pytest.mark.parametrize("num_buckets", [2, 4])
def test_bucketize_carries_int64_weights(dist_type, num_buckets):
    kjt = _make_kjt(torch.int64)
    block_sizes = torch.full((len(kjt.keys()),), 250, dtype=torch.int64, device="cuda")
    dist_types = dist_type_codes([dist_type] * len(kjt.keys()), device="cuda")
    out, _ = bucketize_kjt_before_all2all(
        kjt,
        num_buckets=num_buckets,
        block_sizes=block_sizes,
        dist_type_per_feature=dist_types,
    )
    assert out.weights().dtype == torch.int64

    # The (index, weight) pairs of a feature are the same multiset after the
    # bucketizer; weights must still be attached to their own key. Compare
    # against the float32 run, whose key order is identical.
    kjt_f = KeyedJaggedTensor.from_lengths_sync(
        keys=kjt.keys(),
        values=kjt.values(),
        weights=torch.arange(kjt.values().numel(), device="cuda").float(),
        lengths=kjt.lengths(),
    )
    out_f, _ = bucketize_kjt_before_all2all(
        kjt_f,
        num_buckets=num_buckets,
        block_sizes=block_sizes,
        dist_type_per_feature=dist_types,
    )
    src_pos = out_f.weights().long()
    torch.testing.assert_close(out.weights(), kjt.weights()[src_pos])


def test_permute_and_split_keep_int64_weights():
    kjt = _make_kjt(torch.int64)
    permuted = kjt.permute([2, 0, 1])
    assert permuted.weights().dtype == torch.int64
    by_key = kjt.to_dict()
    for key, jt in permuted.to_dict().items():
        torch.testing.assert_close(jt.weights(), by_key[key].weights())

    parts = kjt.split([1, 2])
    assert [p.weights().dtype for p in parts] == [torch.int64, torch.int64]
    torch.testing.assert_close(torch.cat([p.weights() for p in parts]), kjt.weights())
