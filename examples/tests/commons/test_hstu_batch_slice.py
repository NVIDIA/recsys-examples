import sys
from pathlib import Path
from typing import List

import pytest
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from commons.datasets.hstu_batch import FeatureConfig, HSTUBatch, is_batch_valid
from commons.datasets.hstu_random_dataset import HSTURandomDataset


def _make_values(lengths: torch.Tensor, base: int) -> torch.Tensor:
    return torch.arange(int(lengths.sum().item()), dtype=torch.long) + base


def _make_batch(actual_batch_size: int = 6) -> HSTUBatch:
    item_lengths = torch.tensor([2, 0, 3, 1, 4, 2], dtype=torch.long)
    action_lengths = item_lengths.clone()
    user_lengths = torch.ones_like(item_lengths)
    num_candidates = torch.minimum(item_lengths, torch.full_like(item_lengths, 2))

    features = KeyedJaggedTensor.from_lengths_sync(
        keys=["item", "action", "user"],
        values=torch.cat(
            [
                _make_values(item_lengths, 0),
                _make_values(action_lengths, 100),
                _make_values(user_lengths, 200),
            ]
        ),
        lengths=torch.cat([item_lengths, action_lengths, user_lengths]),
    )
    labels = KeyedJaggedTensor.from_lengths_sync(
        keys=["label"],
        values=_make_values(num_candidates, 300),
        lengths=num_candidates,
    )
    return HSTUBatch(
        features=features,
        batch_size=item_lengths.numel(),
        feature_to_max_seqlen={"item": 4, "action": 4, "user": 1},
        contextual_feature_names=["user"],
        labels=labels,
        actual_batch_size=actual_batch_size,
        item_feature_name="item",
        action_feature_name="action",
        max_num_candidates=2,
        num_candidates=num_candidates,
    )


def _expected_kjt_values(
    kjt: KeyedJaggedTensor,
    key: str,
    start: int,
    end: int,
) -> torch.Tensor:
    feature = kjt[key]
    offsets = feature.offsets()
    return feature.values()[offsets[start].item() : offsets[end].item()]


def _assert_kjt_slice(
    sliced: KeyedJaggedTensor,
    source: KeyedJaggedTensor,
    start: int,
    end: int,
    output_batch_size: int,
):
    pad_size = output_batch_size - (end - start)
    assert list(sliced.keys()) == list(source.keys())
    for key in source.keys():
        expected_lengths = source[key].lengths()[start:end]
        if pad_size > 0:
            expected_lengths = torch.cat(
                [expected_lengths, torch.zeros(pad_size, dtype=expected_lengths.dtype)]
            )
        assert torch.equal(sliced[key].lengths(), expected_lengths)
        assert torch.equal(
            sliced[key].values(),
            _expected_kjt_values(source, key, start, end),
        )


def test_hstu_batch_slice_contiguous_rows():
    batch = _make_batch()
    sliced = batch.slice(2, 5)

    is_batch_valid(sliced)
    assert sliced.batch_size == 3
    assert sliced.actual_batch_size == 3
    assert sliced.feature_to_max_seqlen == batch.feature_to_max_seqlen
    assert sliced.contextual_feature_names == batch.contextual_feature_names
    assert sliced.item_feature_name == batch.item_feature_name
    assert sliced.action_feature_name == batch.action_feature_name
    assert sliced.max_num_candidates == batch.max_num_candidates
    assert torch.equal(sliced.num_candidates, batch.num_candidates[2:5])
    _assert_kjt_slice(sliced.features, batch.features, 2, 5, output_batch_size=3)
    assert batch.labels is not None and sliced.labels is not None
    _assert_kjt_slice(sliced.labels, batch.labels, 2, 5, output_batch_size=3)


def test_hstu_batch_slice_can_pad_output_batch():
    batch = _make_batch(actual_batch_size=6)
    sliced = batch.slice(4, 6, batch_size=4)

    is_batch_valid(sliced)
    assert sliced.batch_size == 4
    assert sliced.actual_batch_size == 2
    assert torch.equal(
        sliced.num_candidates,
        torch.tensor([2, 2, 0, 0], dtype=batch.num_candidates.dtype),
    )
    _assert_kjt_slice(sliced.features, batch.features, 4, 6, output_batch_size=4)
    assert batch.labels is not None and sliced.labels is not None
    _assert_kjt_slice(sliced.labels, batch.labels, 4, 6, output_batch_size=4)


def test_hstu_batch_slice_rejects_invalid_ranges():
    batch = _make_batch()

    with pytest.raises(ValueError):
        batch.slice(1, 1)
    with pytest.raises(ValueError):
        batch.slice(1, 3, batch_size=1)


class _RecordingDistribution:
    def __init__(self, values: torch.Tensor):
        self.values = values
        self.calls: List[int] = []
        self.offset = 0

    def sample(self, size: int, device: torch.device) -> torch.Tensor:
        self.calls.append(size)
        out = self.values[self.offset : self.offset + size].to(device)
        self.offset += size
        return out


def test_hstu_random_dataset_generates_once_then_slices():
    batch_size = 2
    num_generated_batches = 3
    seqlen_values = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    value_count = int(seqlen_values.sum().item())
    seqlen_dist = _RecordingDistribution(seqlen_values)
    item_dist = _RecordingDistribution(torch.arange(value_count, dtype=torch.long))
    action_dist = _RecordingDistribution(
        torch.arange(value_count, dtype=torch.long) + 100
    )

    dataset = HSTURandomDataset(
        batch_size=batch_size,
        feature_configs=[
            FeatureConfig(
                feature_names=["item", "action"],
                max_item_ids=[1000, 1000],
                max_sequence_length=6,
                is_jagged=True,
                seqlen_dist=seqlen_dist,
                value_dists={"item": item_dist, "action": action_dist},
            )
        ],
        item_feature_name="item",
        action_feature_name="action",
        max_num_candidates=0,
        num_generated_batches=num_generated_batches,
        num_tasks=1,
        num_batches=num_generated_batches,
    )

    assert seqlen_dist.calls == [batch_size * num_generated_batches]
    assert item_dist.calls == [value_count]
    assert action_dist.calls == [value_count]

    batches = list(iter(dataset))
    for i, batch in enumerate(batches):
        start = i * batch_size
        end = start + batch_size
        expected_lengths = seqlen_values[start:end]
        expected_value_start = int(seqlen_values[:start].sum().item())
        expected_value_end = int(seqlen_values[:end].sum().item())

        is_batch_valid(batch)
        assert torch.equal(batch.features["item"].lengths(), expected_lengths)
        assert torch.equal(batch.features["action"].lengths(), expected_lengths)
        assert torch.equal(
            batch.features["item"].values(),
            torch.arange(expected_value_start, expected_value_end),
        )
        assert torch.equal(
            batch.features["action"].values(),
            torch.arange(expected_value_start, expected_value_end) + 100,
        )
