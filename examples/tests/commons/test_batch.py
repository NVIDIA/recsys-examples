import sys

import torch

sys.path.append("../../examples")
from typing import Tuple

import commons.utils.initialize as init
import pytest
from commons.distributed.batch_allgather import allgather_batch
from commons.sequence_batch.batch import BaseBatch
from megatron.core import parallel_state
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def select_csr_rows_vectorized(
    row_offsets: torch.Tensor, values: torch.Tensor, indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select specific rows from a CSR structure (vectorized version, more efficient)

    Args:
        row_offsets: Row offsets tensor, shape [num_rows + 1]
        values: Values tensor, shape [nnz]
        indices: Indices of rows to select, shape [num_selected_rows]

    Returns:
        new_row_offsets: New row offsets, shape [num_selected_rows + 1]
        new_values: New values tensor, shape [new_nnz]
    """
    device = row_offsets.device

    start_offsets = row_offsets[indices]
    end_offsets = row_offsets[indices + 1]
    row_lengths = end_offsets - start_offsets

    new_row_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=row_offsets.dtype),
            torch.cumsum(row_lengths, dim=0),
        ]
    )

    total_new_nnz = new_row_offsets[-1].item()

    if total_new_nnz == 0:
        return new_row_offsets, torch.empty(0, dtype=values.dtype, device=device)

    output_positions = torch.arange(total_new_nnz, device=device)

    output_row_ids = (
        torch.searchsorted(new_row_offsets, output_positions, side="right") - 1
    )
    positions_in_row = output_positions - new_row_offsets[output_row_ids]
    input_positions = start_offsets[output_row_ids] + positions_in_row

    new_values = values[input_positions]

    return new_row_offsets, new_values


def generate_batch(
    batch_size,
    max_sequence_length,
    num_features,
    dense_label,
):
    feature_names = [f"feature{i}" for i in range(num_features)]
    feature_lengths = torch.randint(
        1, max_sequence_length, (batch_size * num_features,)
    ).cuda()
    feature_values = torch.randint(0, 100000, (feature_lengths.sum().item(),)).cuda()
    if dense_label:
        labels = (
            torch.arange(batch_size * num_features, device=torch.device("cuda")).view(
                -1
            )
            // num_features
        )
    else:
        label_lengths = torch.randint(1, 20, (batch_size,)).cuda()
        label_values = torch.arange(
            label_lengths.sum().item(), device=torch.device("cuda")
        )
        labels = KeyedJaggedTensor.from_lengths_sync(
            keys=["label"],
            values=label_values,
            lengths=label_lengths,
        )
    features = KeyedJaggedTensor.from_lengths_sync(
        keys=feature_names,
        values=feature_values,
        lengths=feature_lengths.view(-1),
    )
    return BaseBatch(
        features=features,
        batch_size=batch_size,
        feature_to_max_seqlen={
            feature_name: max_sequence_length for feature_name in feature_names
        },
        labels=labels,
    )


def kjt_equal(kjt1: KeyedJaggedTensor, kjt2: KeyedJaggedTensor):
    return (
        torch.equal(kjt1.values(), kjt2.values())
        & torch.equal(kjt1.offsets(), kjt2.offsets())
        & torch.equal(kjt1.lengths(), kjt2.lengths())
    )


@pytest.mark.parametrize("batch_size", [10, 20, 30])
@pytest.mark.parametrize("max_sequence_length", [10, 20, 30])
@pytest.mark.parametrize("num_features", [3, 1, 2])
@pytest.mark.parametrize("dense_label", [True, False])
def test_batch_index_select(
    batch_size,
    max_sequence_length,
    num_features,
    dense_label,
):
    batch = generate_batch(batch_size, max_sequence_length, num_features, dense_label)
    for i in range(20):
        num_selected = torch.randint(1, batch_size, (1,)).item()
        indices = torch.randperm(batch_size)[:num_selected].cuda()
        selected_batch = batch.index_select(indices)

        for kjt_name in batch.features.keys():
            ref_offsets, ref_values = select_csr_rows_vectorized(
                batch.features[kjt_name].offsets(),
                batch.features[kjt_name].values(),
                indices,
            )
            assert torch.equal(selected_batch.features[kjt_name].values(), ref_values)
            assert torch.equal(selected_batch.features[kjt_name].offsets(), ref_offsets)

        if dense_label:
            selected_labels = selected_batch.labels.view(
                selected_batch.actual_batch_size, -1
            )
            ref_labels = indices.unsqueeze(-1).expand_as(selected_labels)
            assert torch.equal(selected_labels, ref_labels)
        else:
            selected_labels = selected_batch.labels
            ref_offsets, ref_labels = select_csr_rows_vectorized(
                batch.labels.offsets(), batch.labels.values(), indices
            )
            assert torch.equal(selected_labels.values(), ref_labels)
            assert torch.equal(selected_labels.offsets(), ref_offsets)
        assert selected_batch.actual_batch_size == num_selected


@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("max_sequence_length", [10, 20, 30])
@pytest.mark.parametrize("num_features", [3, 1, 2])
@pytest.mark.parametrize("dense_label", [True, False])
def test_batch_allgather(
    batch_size,
    max_sequence_length,
    num_features,
    dense_label,
):
    init.initialize_distributed()

    with init.auto_destroy_global_state():
        init.initialize_model_parallel(1)
        init.set_random_seed(1234)
        dp_rank = parallel_state.get_data_parallel_rank()
        parallel_state.get_data_parallel_world_size()
        batch = generate_batch(
            batch_size, max_sequence_length, num_features, dense_label
        )
        allgathered_batch = allgather_batch(
            batch, pg_group=parallel_state.get_data_parallel_group()
        )

        slice_indices = (
            torch.arange(batch_size, device=torch.device("cuda")) + dp_rank * batch_size
        )
        sliced_batch = allgathered_batch.index_select(slice_indices)

        assert kjt_equal(sliced_batch.features, batch.features)
        if dense_label:
            assert torch.equal(sliced_batch.labels, batch.labels)
        else:
            assert kjt_equal(sliced_batch.labels, batch.labels)
