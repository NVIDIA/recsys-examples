# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The index dedup of an embedding collection reads no count.

``ShardedDynamicEmbeddingCollection._dedup_indices`` used to read the number of unique
keys (``num_uniques.item()``) to cut its arrays to length, a wait for the device once per
sharding per step. For a row-wise sharding it now keeps the arrays at the length of its
input, the real keys first, and ``RwSparseFeaturesDist.forward`` cuts them to what the
lengths add up to, with the read of the split sizes that the all2all makes anyway.

What is checked here, on one GPU (a process group of one rank; the two-rank run is in
per_key_score/test_per_key_score_dist.py):

* ``_dedup_indices`` waits for nothing when the distributor of the sharding takes padded
  arrays, and still reads the count when it is a distributor of TorchRec's, which does
  not;
* the cut itself: exact arrays are left alone, padded ones are cut, whichever of the keys,
  weights, permute and scores are padded;
* ``forward`` hands the all2all, the permute of the scores and the output distribution
  exactly what it would have got from exact arrays, also when the memory past the real
  keys holds garbage (which is what the bucketize leaves there: it does not write it), and
  when the padded input was rebuilt on the way (nothing marks it).
"""
import types

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)
try:
    import dynamicemb  # noqa: F401  (loads dynamicemb_extensions)
    from dynamicemb import ScoredKeyedJaggedTensor
    from dynamicemb.shard import input_dist as input_dist_module
    from dynamicemb.shard.embedding import (
        DynamicEmbeddingCollectionContext,
        ShardedDynamicEmbeddingCollection,
    )
    from dynamicemb.shard.input_dist import RwSparseFeaturesDist, cut_to_length_per_key
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
except ImportError as e:
    pytest.skip("needs dynamicemb and torchrec: %s" % (e,), allow_module_level=True)

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
DEVICES = ["cpu"] + (["cuda"] if HAS_CUDA else [])

NUM_FEATURES = 3
BATCH = 4
POISON_KEY = 2**62 + 12345
POISON_INDEX = 2**40
KEYS = [f"f{i}" for i in range(NUM_FEATURES)]


def _real_arrays(device, seed=0, with_weights=False):
    """The keys, lengths, scores and weights of what a dedup found: distinct keys, so the
    scores (derived from the key) can be told apart."""
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, 5, (NUM_FEATURES * BATCH,), generator=generator)
    lengths[0] = 2  # something to send whatever the draw
    count = int(lengths.sum())
    values = torch.randperm(100_000, generator=generator)[:count]
    scores = values * 10 + 1
    weights = (values % 7 + 1).float() if with_weights else None
    return (
        values.to(device),
        lengths.to(device),
        scores.to(device),
        None if weights is None else weights.to(device),
    )


def _padded(values, scores, weights, extra, poison):
    """The same arrays at the length of the input: ``extra`` entries more. The dedup leaves
    the key EMPTY_KEY, the score 0 and the frequency 0 there; ``poison`` leaves junk."""
    device = values.device

    def tail(like, junk, inert):
        return torch.full((extra,), junk if poison else inert, dtype=like.dtype)

    return (
        torch.cat([values, tail(values, POISON_KEY, -1).to(device)]),
        torch.cat([scores, tail(scores, -POISON_KEY, 0).to(device)]),
        None
        if weights is None
        else torch.cat([weights, tail(weights, float("nan"), 0.0).to(device)]),
    )


def _scored(keys, lengths, values, scores, weights):
    return ScoredKeyedJaggedTensor(
        keys=keys, values=values, lengths=lengths, scores=scores, weights=weights
    )


# --------------------------------------------------------------------------
# the cut
# --------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_exact_arrays_are_left_alone(device):
    values, lengths, scores, weights = _real_arrays(device, with_weights=True)
    kjt = KeyedJaggedTensor(keys=KEYS, values=values, lengths=lengths, weights=weights)
    permute = torch.arange(values.numel(), device=device)
    cut_permute, cut_scores = cut_to_length_per_key(kjt, permute, scores)
    assert kjt.values() is values and kjt.weights() is weights
    assert cut_permute is permute and cut_scores is scores


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("extra", [1, 17])
def test_padded_arrays_are_cut_to_what_the_lengths_add_up_to(device, extra):
    values, lengths, scores, weights = _real_arrays(device, with_weights=True)
    count = values.numel()
    padded_values, padded_scores, padded_weights = _padded(
        values, scores, weights, extra, poison=True
    )
    permute = torch.cat(
        [
            torch.arange(count - 1, -1, -1),
            torch.full((extra,), POISON_INDEX),
        ]
    ).to(device)
    kjt = KeyedJaggedTensor(
        keys=KEYS, values=padded_values, lengths=lengths, weights=padded_weights
    )
    cut_permute, cut_scores = cut_to_length_per_key(kjt, permute, padded_scores)
    assert torch.equal(kjt.values(), values)
    assert torch.equal(kjt.weights(), weights)
    assert torch.equal(cut_scores, scores)
    assert torch.equal(cut_permute, torch.arange(count - 1, -1, -1, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_only_the_arrays_that_are_too_long_are_cut(device):
    values, lengths, scores, _ = _real_arrays(device)
    _, padded_scores, _ = _padded(values, scores, None, 5, poison=True)
    kjt = KeyedJaggedTensor(keys=KEYS, values=values, lengths=lengths)
    cut_permute, cut_scores = cut_to_length_per_key(kjt, None, padded_scores)
    assert cut_permute is None and torch.equal(cut_scores, scores)
    assert kjt.values() is values


# --------------------------------------------------------------------------
# _dedup_indices reads no count for a distributor that takes padded arrays
# --------------------------------------------------------------------------
class _Collection:
    """What ``_dedup_indices`` reads of a ShardedDynamicEmbeddingCollection: the
    distributors, the device and the three buffers of each sharding."""

    def __init__(self, input_dists, device, features_per_table=(2, 1)):
        self._input_dists = input_dists
        self._device = device
        self._sharding_is_lfu = [False] * len(input_dists)
        self._sharding_has_admit = [False] * len(input_dists)
        table_offsets = [0]
        for n in features_per_table:
            table_offsets.append(table_offsets[-1] + n)
        self._buffers = {}
        for i in range(len(input_dists)):
            self._buffers[f"_hash_size_offset_tensor_{i}"] = torch.zeros(
                table_offsets[-1] + 1, dtype=torch.int64, device=device
            )
            self._buffers[f"_nonfuse_table_feature_offsets_device_{i}"] = torch.tensor(
                table_offsets, dtype=torch.int64, device=device
            )

    def get_buffer(self, name):
        return self._buffers[name]


def _batch_with_duplicates(device):
    generator = torch.Generator().manual_seed(7)
    lengths = torch.randint(1, 6, (NUM_FEATURES * BATCH,), generator=generator)
    values = torch.randint(0, 20, (int(lengths.sum()),), generator=generator)
    return KeyedJaggedTensor(
        keys=KEYS, values=values.to(device), lengths=lengths.to(device)
    )


@cuda_only
def test_the_dedup_of_a_distributor_that_takes_padded_arrays_reads_nothing(no_sync):
    device = torch.device("cuda", torch.cuda.current_device())
    collection = _Collection([types.SimpleNamespace(accepts_padded_dedup=True)], device)
    features = _batch_with_duplicates(device)
    context = DynamicEmbeddingCollectionContext()
    with torch.no_grad(), no_sync():
        (deduped,) = ShardedDynamicEmbeddingCollection._dedup_indices(
            collection, context, [features]
        )
    # The arrays keep the length of the input; what is past the real keys is inert.
    count = int(deduped.lengths().sum())
    assert 0 < count < features.values().numel()
    assert deduped.values().numel() == features.values().numel()
    assert (deduped.values()[count:] == -1).all()


@cuda_only
def test_the_dedup_of_a_distributor_of_torchrec_still_reads_the_count(no_sync):
    device = torch.device("cuda", torch.cuda.current_device())
    collection = _Collection([types.SimpleNamespace()], device)
    features = _batch_with_duplicates(device)
    context = DynamicEmbeddingCollectionContext()
    with torch.no_grad(), no_sync():
        with pytest.raises(AssertionError, match="host sync outside sf.read_now"):
            ShardedDynamicEmbeddingCollection._dedup_indices(
                collection, context, [features]
            )
    # outside the guard it gives the exact arrays that such a distributor needs
    with torch.no_grad():
        (deduped,) = ShardedDynamicEmbeddingCollection._dedup_indices(
            collection, DynamicEmbeddingCollectionContext(), [features]
        )
    assert deduped.values().numel() == int(deduped.lengths().sum())


def test_only_the_dynamicemb_distributor_takes_padded_arrays():
    from torchrec.distributed.sharding.rw_sharding import (
        RwSparseFeaturesDist as TorchRecRwSparseFeaturesDist,
    )

    assert RwSparseFeaturesDist.accepts_padded_dedup is True
    assert not hasattr(TorchRecRwSparseFeaturesDist, "accepts_padded_dedup")


# --------------------------------------------------------------------------
# forward: padded input gives what exact input gives
# --------------------------------------------------------------------------
@pytest.fixture
def one_rank():
    import torch.distributed as dist

    created = not dist.is_initialized()
    if created:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield dist.group.WORLD
    if created:
        dist.destroy_process_group()


def _distributor(group, per_key_score):
    device = torch.device("cuda", torch.cuda.current_device())
    return RwSparseFeaturesDist(
        pg=group,
        num_features=NUM_FEATURES,
        feature_hash_sizes=[1000] * NUM_FEATURES,
        device=device,
        is_sequence=True,
        dist_types=["roundrobin"] * NUM_FEATURES,
        per_key_score=per_key_score,
    )


def _poison_the_bucketize(monkeypatch):
    """The bucketize allocates its outputs at the length of its input and writes a prefix:
    put junk in the rest, which is what that memory can hold."""
    real = input_dist_module.bucketize_kjt_before_all2all

    def poisoned(*args, **kwargs):
        kjt, permute = real(*args, **kwargs)
        total = int(kjt.lengths().sum())
        kjt._values[total:] = POISON_KEY
        if kjt._weights is not None:
            kjt._weights[total:] = float("nan")
        if permute is not None:
            permute[total:] = POISON_INDEX
        return kjt, permute

    monkeypatch.setattr(input_dist_module, "bucketize_kjt_before_all2all", poisoned)


def _distribute(distributor, features):
    received = distributor(features).wait().wait()
    scored = isinstance(received, ScoredKeyedJaggedTensor)
    out = {
        "values": received.values().clone(),
        "lengths": received.lengths().clone(),
        "scores": received.scores().clone() if scored else None,
        "weights": (
            None
            if received.weights_or_none() is None
            else received.weights_or_none().clone()
        ),
        "permute": distributor.unbucketize_permute_tensor.clone(),
    }
    torch.cuda.synchronize()
    return out


def _same(got, want):
    for name, expected in want.items():
        if expected is None:
            assert got[name] is None, name
        else:
            assert got[name] is not None, name
            assert got[name].shape == expected.shape, name
            assert torch.equal(got[name], expected), name


@cuda_only
@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize("poison", [False, True])
def test_padded_input_distributes_like_exact_input(
    one_rank, monkeypatch, with_weights, poison
):
    device = torch.device("cuda", torch.cuda.current_device())
    distributor = _distributor(one_rank, per_key_score=True)
    values, lengths, scores, weights = _real_arrays(device, with_weights=with_weights)

    exact = _distribute(distributor, _scored(KEYS, lengths, values, scores, weights))
    # one rank: every key goes to its only bucket, in its place
    assert torch.equal(exact["values"], values)
    assert torch.equal(exact["scores"], scores)
    assert torch.equal(exact["permute"], torch.arange(values.numel(), device=device))

    if poison:
        _poison_the_bucketize(monkeypatch)
    padded_values, padded_scores, padded_weights = _padded(
        values, scores, weights, extra=19, poison=poison
    )
    # a brand-new object: nothing but its fields says that it is padded
    padded = _scored(KEYS, lengths, padded_values, padded_scores, padded_weights)
    _same(_distribute(distributor, padded), exact)


@cuda_only
@pytest.mark.parametrize("poison", [False, True])
def test_padded_input_without_scores_distributes_like_exact_input(
    one_rank, monkeypatch, poison
):
    device = torch.device("cuda", torch.cuda.current_device())
    distributor = _distributor(one_rank, per_key_score=False)
    values, lengths, scores, weights = _real_arrays(device, with_weights=True)
    exact = _distribute(
        distributor,
        KeyedJaggedTensor(keys=KEYS, values=values, lengths=lengths, weights=weights),
    )
    if poison:
        _poison_the_bucketize(monkeypatch)
    padded_values, _, padded_weights = _padded(
        values, scores, weights, extra=11, poison=poison
    )
    padded = KeyedJaggedTensor(
        keys=KEYS, values=padded_values, lengths=lengths, weights=padded_weights
    )
    _same(_distribute(distributor, padded), exact)


@cuda_only
def test_a_distribution_of_exact_arrays_is_not_cut(one_rank):
    """Exact arrays are not shortened: the permute handed to the output distribution has
    one place for every key."""
    device = torch.device("cuda", torch.cuda.current_device())
    distributor = _distributor(one_rank, per_key_score=True)
    values, lengths, scores, _ = _real_arrays(device)
    distributor(_scored(KEYS, lengths, values, scores, None)).wait().wait()
    assert distributor.unbucketize_permute_tensor.numel() == values.numel()
