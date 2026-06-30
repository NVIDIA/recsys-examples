import sys
from pathlib import Path

import pytest
import torch

HSTU_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = HSTU_DIR.parent
for path in [EXAMPLES_DIR, HSTU_DIR, HSTU_DIR / "model"]:
    sys.path.insert(0, str(path))

import inference_ranking_gr


def _reference_strip_counts(lengths, origin_num_cached, feature_order, batch_size):
    remaining_num_cached = origin_num_cached.clone()
    strip_counts = torch.zeros_like(lengths)

    for feature_idx in feature_order[:-2]:
        for batch_idx in range(batch_size):
            row_idx = feature_idx * batch_size + batch_idx
            row_length = int(lengths[row_idx].item())
            remaining_cached = int(remaining_num_cached[batch_idx].item())
            strip_count = min(remaining_cached, row_length)
            remaining_num_cached[batch_idx] -= strip_count
            strip_counts[row_idx] = strip_count

    item_feature_idx = feature_order[-2]
    action_feature_idx = feature_order[-1]
    for batch_idx in range(batch_size):
        remaining_cached = int(remaining_num_cached[batch_idx].item())
        item_strip_count = (remaining_cached + 1) // 2
        action_strip_count = remaining_cached // 2

        item_row_idx = item_feature_idx * batch_size + batch_idx
        action_row_idx = action_feature_idx * batch_size + batch_idx
        strip_counts[item_row_idx] = min(item_strip_count, int(lengths[item_row_idx].item()))
        strip_counts[action_row_idx] = min(action_strip_count, int(lengths[action_row_idx].item()))

    return strip_counts


def _feature_label(feature_idx, feature_order):
    if feature_idx == feature_order[-2]:
        return "item"
    if feature_idx == feature_order[-1]:
        return "action"
    return "context"


def _print_strip_debug(values, lengths, actual_values, actual_lengths, origin_num_cached, feature_order, batch_size):
    old_offsets = torch.cat([lengths.new_zeros((1,)), torch.cumsum(lengths, dim=0)])
    actual_offsets = torch.cat([actual_lengths.new_zeros((1,)), torch.cumsum(actual_lengths, dim=0)])
    strip_counts = _reference_strip_counts(lengths, origin_num_cached, feature_order, batch_size)

    print("per-feature per-sequence strip detail:")
    for output_feature_idx, feature_idx in enumerate(feature_order):
        for batch_idx in range(batch_size):
            input_row_idx = feature_idx * batch_size + batch_idx
            output_row_idx = output_feature_idx * batch_size + batch_idx
            strip_count = int(strip_counts[input_row_idx].item())

            input_start = int(old_offsets[input_row_idx].item())
            input_end = int(old_offsets[input_row_idx + 1].item())
            output_start = int(actual_offsets[output_row_idx].item())
            output_end = int(actual_offsets[output_row_idx + 1].item())

            original_tokens = values[input_start:input_end].tolist()
            print(
                f"  feature={_feature_label(feature_idx, feature_order)}[{feature_idx}] "
                f"sequence={batch_idx} strip_count={strip_count} "
                f"stripped={original_tokens[:strip_count]} "
                f"expected_kept={original_tokens[strip_count:]} "
                f"actual_kept={actual_values[output_start:output_end].tolist()}"
            )


def _reference_strip_cached_tokens(values, lengths, origin_num_cached, feature_order, batch_size):
    old_offsets = torch.cat([lengths.new_zeros((1,)), torch.cumsum(lengths, dim=0)])
    new_values = []
    new_lengths = []

    strip_counts = _reference_strip_counts(lengths, origin_num_cached, feature_order, batch_size)

    for feature_idx in feature_order:
        for batch_idx in range(batch_size):
            row_idx = feature_idx * batch_size + batch_idx
            row_length = int(lengths[row_idx].item())
            strip_count = int(strip_counts[row_idx].item())
            start = int(old_offsets[row_idx].item()) + strip_count
            end = int(old_offsets[row_idx + 1].item())
            new_values.append(values[start:end])
            new_lengths.append(row_length - strip_count)

    return torch.cat(new_values), torch.tensor(new_lengths, dtype=lengths.dtype)


@pytest.mark.parametrize(
    "origin_num_cached",
    [
        torch.tensor([0, 0]),
        torch.tensor([2, 5]),
        torch.tensor([9, 12]),
    ],
)
def test_strip_cached_tokens_cpp_op(origin_num_cached):
    assert inference_ranking_gr._HAS_STRIP_CACHED_TOKENS_OP

    batch_size = 2
    num_context = 2
    lengths = torch.tensor([1, 2, 3, 1, 4, 2, 3, 5], dtype=torch.int64)
    values = torch.arange(int(lengths.sum().item()), dtype=torch.int64)

    print(f"origin_num_cached: {origin_num_cached.tolist()}")
    print(f"original values: {values.tolist()}")
    print(f"original lengths: {lengths.tolist()}, seqlen: {lengths.view(num_context + 2, batch_size).sum(dim=0).tolist()}")

    actual_values, actual_lengths = torch.ops.hstu_cuda_ops.strip_cached_tokens(
        values,
        lengths,
        torch.cat([lengths.new_zeros((1,)), torch.cumsum(lengths, dim=0)]),
        origin_num_cached,
        list(range(num_context + 2)),
    )
    print(f"stripped values: {actual_values.tolist()}")
    print(f"stripped lengths: {actual_lengths.tolist()}, , seqlen: {actual_lengths.view(num_context + 2, batch_size).sum(dim=0).tolist()}")

    expected_values, expected_lengths = _reference_strip_cached_tokens(
        values,
        lengths,
        origin_num_cached,
        list(range(num_context + 2)),
        batch_size,
    )
    print(f"expected values: {expected_values.tolist()}")
    print(f"expected lengths: {expected_lengths.tolist()}, seqlen: {expected_lengths.view(num_context + 2, batch_size).sum(dim=0).tolist()}")
    _print_strip_debug(
        values,
        lengths,
        actual_values,
        actual_lengths,
        origin_num_cached,
        list(range(num_context + 2)),
        batch_size,
    )

    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_lengths, expected_lengths)
    torch.testing.assert_close(
        actual_lengths.view(num_context + 2, batch_size).sum(dim=0),
        expected_lengths.view(num_context + 2, batch_size).sum(dim=0),
    )


def test_strip_cached_tokens_cpp_op_respects_feature_order():
    assert inference_ranking_gr._HAS_STRIP_CACHED_TOKENS_OP

    batch_size = 2
    lengths = torch.tensor([1, 2, 3, 1, 4, 2, 3, 5], dtype=torch.int64)
    values = torch.arange(int(lengths.sum().item()), dtype=torch.int64)
    origin_num_cached = torch.tensor([1, 3], dtype=torch.int64)
    feature_order = [2, 0, 3, 1]

    actual_values, actual_lengths = torch.ops.hstu_cuda_ops.strip_cached_tokens(
        values,
        lengths,
        torch.cat([lengths.new_zeros((1,)), torch.cumsum(lengths, dim=0)]),
        origin_num_cached,
        feature_order,
    )
    expected_values, expected_lengths = _reference_strip_cached_tokens(
        values,
        lengths,
        origin_num_cached,
        feature_order,
        batch_size,
    )
    print(f"feature_order: {feature_order}")
    print(f"actual values: {actual_values.tolist()}")
    print(f"expected values: {expected_values.tolist()}")
    print(f"actual lengths: {actual_lengths.tolist()}")
    print(f"expected lengths: {expected_lengths.tolist()}")
    _print_strip_debug(
        values,
        lengths,
        actual_values,
        actual_lengths,
        origin_num_cached,
        feature_order,
        batch_size,
    )

    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_lengths, expected_lengths)


def test_strip_cached_tokens_cpp_op_item_action_split_details():
    assert inference_ranking_gr._HAS_STRIP_CACHED_TOKENS_OP

    batch_size = 2
    num_context = 3
    feature_order = list(range(num_context + 2))
    lengths = torch.tensor(
        [
            1, 1,
            1, 1,
            1, 1,
            5, 5,
            5, 5,
        ],
        dtype=torch.int64,
    )
    values = torch.arange(int(lengths.sum().item()), dtype=torch.int64)
    origin_num_cached = torch.tensor([7, 8], dtype=torch.int64)

    actual_values, actual_lengths = torch.ops.hstu_cuda_ops.strip_cached_tokens(
        values,
        lengths,
        torch.cat([lengths.new_zeros((1,)), torch.cumsum(lengths, dim=0)]),
        origin_num_cached,
        feature_order,
    )
    expected_values, expected_lengths = _reference_strip_cached_tokens(
        values,
        lengths,
        origin_num_cached,
        feature_order,
        batch_size,
    )
    strip_counts = _reference_strip_counts(lengths, origin_num_cached, feature_order, batch_size)

    item_feature_idx = feature_order[-2]
    action_feature_idx = feature_order[-1]
    item_strip_counts = strip_counts[item_feature_idx * batch_size : (item_feature_idx + 1) * batch_size]
    action_strip_counts = strip_counts[action_feature_idx * batch_size : (action_feature_idx + 1) * batch_size]

    print(f"origin_num_cached: {origin_num_cached.tolist()}")
    print(f"feature_order: {feature_order}")
    print(f"original values: {values.tolist()}")
    print(f"original lengths: {lengths.tolist()}, seqlen: {lengths.view(num_context + 2, batch_size).sum(dim=0).tolist()}")
    print(f"item_strip_counts: {item_strip_counts.tolist()}")
    print(f"action_strip_counts: {action_strip_counts.tolist()}")
    print(f"actual values: {actual_values.tolist()}")
    print(f"expected values: {expected_values.tolist()}")
    print(f"actual lengths: {actual_lengths.tolist()}")
    print(f"expected lengths: {expected_lengths.tolist()}")
    _print_strip_debug(
        values,
        lengths,
        actual_values,
        actual_lengths,
        origin_num_cached,
        feature_order,
        batch_size,
    )

    assert item_strip_counts[0].item() == action_strip_counts[0].item()
    assert item_strip_counts[1].item() == action_strip_counts[1].item() + 1
    assert torch.all(item_strip_counts > 0)
    assert torch.all(action_strip_counts > 0)
    assert torch.all(item_strip_counts < lengths[item_feature_idx * batch_size : (item_feature_idx + 1) * batch_size])
    assert torch.all(action_strip_counts < lengths[action_feature_idx * batch_size : (action_feature_idx + 1) * batch_size])
    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_lengths, expected_lengths)


def test_strip_cached_tokens_cpp_op_all_stripped():
    assert inference_ranking_gr._HAS_STRIP_CACHED_TOKENS_OP

    batch_size = 3
    num_context = 3
    feature_order = list(range(num_context + 2))
    lengths = torch.tensor(
        [
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        ],
        dtype=torch.int64,
    )
    values = torch.arange(int(lengths.sum().item()), dtype=torch.int64)
    origin_num_cached = torch.tensor([10, 10, 10], dtype=torch.int64)

    actual_values, actual_lengths = torch.ops.hstu_cuda_ops.strip_cached_tokens(
        values,
        lengths,
        torch.cat([lengths.new_zeros((1,)), torch.cumsum(lengths, dim=0)]),
        origin_num_cached,
        feature_order,
    )
    print(f"all stripped actual values: {actual_values.tolist()}")
    print(f"all stripped actual lengths: {actual_lengths.tolist()}")
    _print_strip_debug(
        values,
        lengths,
        actual_values,
        actual_lengths,
        origin_num_cached,
        feature_order,
        batch_size,
    )
    strip_counts = _reference_strip_counts(lengths, origin_num_cached, feature_order, batch_size)
    item_feature_idx = feature_order[-2]
    action_feature_idx = feature_order[-1]
    item_strip_counts = strip_counts[item_feature_idx * batch_size : (item_feature_idx + 1) * batch_size]
    action_strip_counts = strip_counts[action_feature_idx * batch_size : (action_feature_idx + 1) * batch_size]
    item_lengths = lengths[item_feature_idx * batch_size : (item_feature_idx + 1) * batch_size]
    action_lengths = lengths[action_feature_idx * batch_size : (action_feature_idx + 1) * batch_size]

    torch.testing.assert_close(item_strip_counts, item_lengths)
    torch.testing.assert_close(action_strip_counts, action_lengths)

    assert actual_values.numel() == 0
    torch.testing.assert_close(actual_lengths, torch.zeros_like(lengths))