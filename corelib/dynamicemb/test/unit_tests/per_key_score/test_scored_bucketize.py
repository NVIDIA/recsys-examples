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

"""The scores follow their keys through the bucketize.

A score is sent beside its key, so after the bucketize each score must stand
where its key does. The scores are moved by ``unbucketize_permute``, which says
where the bucketize put every input key; these tests check that the score then
at each bucketized key is the score its original key had.
"""

import pytest
import torch
from dynamicemb import ScoredKeyedJaggedTensor
from dynamicemb.shard.input_dist import (
    bucketize_kjt_before_all2all,
    dist_type_codes,
    permute_scores_like_keys,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

NUM_FEATURES = 3
BATCH = 16
HASH_SIZE = 1000
TAG = 2**40  # a score above 2**32, so an int32 slip would show


def _scored(seed: int) -> ScoredKeyedJaggedTensor:
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, 6, (NUM_FEATURES * BATCH,), generator=generator)
    # Keys repeat, so the scores can not be told apart by their keys: each score
    # is TAG plus the position of its key, which says which input key it is.
    values = torch.randint(0, HASH_SIZE, (int(lengths.sum()),), generator=generator)
    scores = TAG + torch.arange(values.numel())
    device = torch.device("cuda")
    return ScoredKeyedJaggedTensor(
        keys=[f"f{i}" for i in range(NUM_FEATURES)],
        values=values.to(device),
        lengths=lengths.to(device),
        scores=scores.to(device),
    )


def _bucketize(kjt, num_buckets, dist_type):
    block_size = (HASH_SIZE + num_buckets - 1) // num_buckets
    bucketized, permute = bucketize_kjt_before_all2all(
        kjt,
        num_buckets=num_buckets,
        block_sizes=torch.full(
            (NUM_FEATURES,), block_size, dtype=torch.int64, device=kjt.device()
        ),
        output_permute=True,
        dist_type_per_feature=dist_type_codes(
            [dist_type] * NUM_FEATURES, device=kjt.device()
        ),
    )
    return bucketized, permute, block_size


@pytest.mark.parametrize("dist_type", ["continuous", "roundrobin", "hash_roundrobin"])
@pytest.mark.parametrize("num_buckets", [2, 4])
def test_a_score_stands_where_its_key_does(dist_type, num_buckets):
    kjt = _scored(seed=num_buckets)
    bucketized, permute, block_size = _bucketize(kjt, num_buckets, dist_type)
    scores = permute_scores_like_keys(kjt.scores(), permute)

    assert scores.dtype == torch.int64
    # Every score arrived exactly once.
    assert torch.equal(scores.sort().values, kjt.scores())

    # The score at each bucketized key names the input key it came from.
    origin = kjt.values()[scores - TAG]
    if dist_type == "continuous":
        # This rule rewrites a key to its offset in the block.
        expected = origin % block_size
    else:
        expected = origin
    assert torch.equal(bucketized.values(), expected)


@pytest.mark.parametrize("num_buckets", [2, 4])
def test_a_score_is_in_the_bucket_of_its_key(num_buckets):
    kjt = _scored(seed=num_buckets)
    bucketized, permute, _ = _bucketize(kjt, num_buckets, "roundrobin")
    scores = permute_scores_like_keys(kjt.scores(), permute)

    # The bucketized lengths are laid out bucket after bucket, and the keys in
    # the same order, so the keys of bucket ``p`` are a run of that many values.
    per_bucket = bucketized.lengths().view(num_buckets, -1).sum(dim=1)
    bucket = torch.repeat_interleave(
        torch.arange(num_buckets, device=scores.device), per_bucket
    )
    assert torch.equal(kjt.values()[scores - TAG] % num_buckets, bucket)


def test_the_scores_are_the_same_every_run():
    kjt = _scored(seed=0)
    bucketized, permute, _ = _bucketize(kjt, 4, "roundrobin")
    first = permute_scores_like_keys(kjt.scores(), permute)
    for _ in range(20):
        again, permute, _ = _bucketize(kjt, 4, "roundrobin")
        assert torch.equal(permute_scores_like_keys(kjt.scores(), permute), first)
        assert torch.equal(again.values(), bucketized.values())
