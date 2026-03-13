import sys

import pytest
import torch
import torch.distributed as dist

sys.path.append("../../examples")

import commons.utils.initialize as init
from commons.distributed.batch_all2all import (
    _build_dst_rank_local,
    pad_and_all2all_batch,
)
from commons.distributed.batch_allgather import pad_and_allgather_batch
from commons.distributed.batch_shuffler import _strip_dense_padding
from commons.sequence_batch.batch import BaseBatch
from megatron.core import parallel_state
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# TODO, consolidate with test_batch.py
def generate_batch(
    batch_size,
    max_sequence_length,
    num_features,
    dense_label,
    actual_batch_size=None,
):
    if actual_batch_size is None:
        actual_batch_size = batch_size

    feature_names = [f"feature{i}" for i in range(num_features)]
    feature_lengths = torch.randint(
        1, max_sequence_length, (batch_size * num_features,)
    ).cuda()

    if actual_batch_size < batch_size:
        lengths_2d = feature_lengths.view(num_features, batch_size)
        lengths_2d[:, actual_batch_size:] = 0
        feature_lengths = lengths_2d.view(-1)

    feature_values = torch.randint(0, 100000, (feature_lengths.sum().item(),)).cuda()
    if dense_label:
        labels = (
            torch.arange(
                actual_batch_size * num_features, device=torch.device("cuda")
            ).view(-1)
            // num_features
        )
    else:
        label_lengths = torch.randint(1, 20, (batch_size,)).cuda()
        if actual_batch_size < batch_size:
            label_lengths[actual_batch_size:] = 0
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
        actual_batch_size=actual_batch_size,
    )


def kjt_equal(kjt1: KeyedJaggedTensor, kjt2: KeyedJaggedTensor):
    return (
        torch.equal(kjt1.values(), kjt2.values())
        & torch.equal(kjt1.offsets(), kjt2.offsets())
        & torch.equal(kjt1.lengths(), kjt2.lengths())
    )


@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("max_sequence_length", [10, 20, 30])
@pytest.mark.parametrize("num_features", [3, 1, 2])
@pytest.mark.parametrize("dense_label", [True, False])
@pytest.mark.parametrize("incomplete_batch", [True, False])
def test_batch_allgather(
    batch_size,
    max_sequence_length,
    num_features,
    dense_label,
    incomplete_batch,
):
    init.initialize_distributed()

    with init.auto_destroy_global_state():
        init.initialize_model_parallel(1)
        init.set_random_seed(1234)
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_world_size = parallel_state.get_data_parallel_world_size()

        if incomplete_batch:
            actual_bs = (
                max(1, batch_size // 2) if dp_rank == dp_world_size - 1 else batch_size
            )
        else:
            actual_bs = batch_size

        batch = generate_batch(
            batch_size,
            max_sequence_length,
            num_features,
            dense_label,
            actual_batch_size=actual_bs,
        )
        allgathered_batch = pad_and_allgather_batch(
            batch,
            pg_group=parallel_state.get_data_parallel_group(),
        )

        slice_indices = (
            torch.arange(batch_size, device=torch.device("cuda")) + dp_rank * batch_size
        )
        sliced_batch = allgathered_batch.index_select(slice_indices)

        real_indices = torch.arange(actual_bs, device=torch.device("cuda"))
        original_real = batch.index_select(real_indices)
        sliced_real = sliced_batch.index_select(real_indices)

        assert kjt_equal(sliced_real.features, original_real.features)
        if dense_label:
            assert torch.equal(sliced_real.labels, original_real.labels)
        else:
            assert kjt_equal(sliced_real.labels, original_real.labels)

        # --- Verify _strip_dense_padding correctness ---
        # After allgather + index_select back to this rank's slice, dense
        # has batch_size rows with padding trailing (allgather preserves
        # each rank's ordering; generate_batch places padding at the end).
        if incomplete_batch and actual_bs < batch_size:
            stripped = _strip_dense_padding(sliced_batch, actual_bs)
            stripped.actual_batch_size = actual_bs

            assert (
                stripped.batch_size == batch_size
            ), f"batch_size should stay {batch_size}, got {stripped.batch_size}"
            assert (
                stripped.actual_batch_size == actual_bs
            ), f"actual_batch_size should be {actual_bs}, got {stripped.actual_batch_size}"
            assert stripped.features.lengths().numel() == batch_size * num_features, (
                f"KJT lengths count should be {batch_size * num_features}, "
                f"got {stripped.features.lengths().numel()}"
            )
            if dense_label:
                assert stripped.labels.numel() == actual_bs * num_features, (
                    f"Dense labels numel should be {actual_bs * num_features}, "
                    f"got {stripped.labels.numel()}"
                )
                assert torch.equal(
                    stripped.labels, batch.labels
                ), "Stripped dense labels should match original unpadded labels"


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("max_sequence_length", [10, 127])
@pytest.mark.parametrize("num_features", [2, 3])
@pytest.mark.parametrize("dense_label", [True, False])
@pytest.mark.parametrize("incomplete_batch", [True, False])
def test_all2all_vs_allgather(
    batch_size,
    max_sequence_length,
    num_features,
    dense_label,
    incomplete_batch,
):
    """Test that pad_and_all2all_batch produces the same results as allgather + index_select.

    When *incomplete_batch* is True each rank gets a different random
    actual_batch_size in [1, batch_size), simulating heterogeneous
    incomplete batches.  Also verifies _strip_dense_padding correctness
    for the incomplete case.
    """
    init.initialize_distributed()

    with init.auto_destroy_global_state():
        init.initialize_model_parallel(1)
        init.set_random_seed(1234)
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_world_size = parallel_state.get_data_parallel_world_size()
        dp_group = parallel_state.get_data_parallel_group()

        if incomplete_batch:
            actual_bs = torch.randint(1, batch_size, (1,)).item()
        else:
            actual_bs = batch_size

        batch = generate_batch(
            batch_size,
            max_sequence_length,
            num_features,
            dense_label,
            actual_batch_size=actual_bs,
        )

        # Build recv_ids as a partition of [0, global_batch_size).
        global_batch_size = batch_size * dp_world_size
        if dp_rank == 0:
            perm = torch.randperm(global_batch_size, device=torch.device("cuda"))
        else:
            perm = torch.empty(
                global_batch_size, dtype=torch.int64, device=torch.device("cuda")
            )
        dist.broadcast(perm, src=0, group=dp_group)

        # Each rank sorts its own chunk; all ranks can reconstruct the
        # full partitions_indices locally since perm is broadcast.
        partitions_indices = torch.stack(
            [
                perm[r * batch_size : (r + 1) * batch_size].sort()[0]
                for r in range(dp_world_size)
            ]
        )
        recv_ids = partitions_indices[dp_rank]

        dst_rank, recv_counts = _build_dst_rank_local(
            partitions_indices,
            dp_rank,
            batch_size,
        )

        # Method 1: AllGather + index_select (reference implementation)
        allgathered_batch, is_padding = pad_and_allgather_batch(
            batch,
            pg_group=dp_group,
            return_padding_flag=True,
        )
        allgather_result = allgathered_batch.index_select(recv_ids)

        # Method 2: All2All (tested implementation)
        all2all_result = pad_and_all2all_batch(
            batch,
            recv_ids,
            dp_group,
            dst_rank,
            recv_counts,
        )

        # 1. Compare features (KJT)
        assert kjt_equal(allgather_result.features, all2all_result.features), (
            f"Features mismatch on rank {dp_rank}:\n"
            f"AllGather values: {allgather_result.features.values()}\n"
            f"All2All values: {all2all_result.features.values()}\n"
            f"AllGather lengths: {allgather_result.features.lengths()}\n"
            f"All2All lengths: {all2all_result.features.lengths()}"
        )

        # 2. Compare labels
        if dense_label:
            assert torch.equal(allgather_result.labels, all2all_result.labels), (
                f"Labels mismatch on rank {dp_rank}:\n"
                f"AllGather: {allgather_result.labels}\n"
                f"All2All: {all2all_result.labels}"
            )
        else:
            assert kjt_equal(allgather_result.labels, all2all_result.labels), (
                f"Labels mismatch on rank {dp_rank}:\n"
                f"AllGather values: {allgather_result.labels.values()}\n"
                f"All2All values: {all2all_result.labels.values()}\n"
                f"AllGather lengths: {allgather_result.labels.lengths()}\n"
                f"All2All lengths: {all2all_result.labels.lengths()}"
            )

        # 3. Compare batch sizes
        assert allgather_result.batch_size == all2all_result.batch_size, (
            f"Batch size mismatch on rank {dp_rank}: "
            f"AllGather={allgather_result.batch_size}, All2All={all2all_result.batch_size}"
        )

        # 4. Compare feature_to_max_seqlen
        assert (
            allgather_result.feature_to_max_seqlen
            == all2all_result.feature_to_max_seqlen
        ), f"feature_to_max_seqlen mismatch on rank {dp_rank}"

        # --- Verify _strip_dense_padding correctness ---
        recv_padding = is_padding[recv_ids]
        actual_bs_recv = int((~recv_padding).sum().item())

        if actual_bs_recv < batch_size:
            # Reorder so real samples precede padding (_strip_dense_padding
            # assumes padding is trailing).
            sorted_order = torch.argsort(recv_padding.long(), stable=True)
            ag_reordered = allgather_result.index_select(sorted_order)
            a2a_reordered = all2all_result.index_select(sorted_order)

            ag_stripped = _strip_dense_padding(ag_reordered, actual_bs_recv)
            ag_stripped.actual_batch_size = actual_bs_recv
            a2a_stripped = _strip_dense_padding(a2a_reordered, actual_bs_recv)
            a2a_stripped.actual_batch_size = actual_bs_recv

            for tag, b in [("AllGather", ag_stripped), ("All2All", a2a_stripped)]:
                assert (
                    b.batch_size == batch_size
                ), f"{tag} batch_size should be {batch_size}, got {b.batch_size}"
                assert b.actual_batch_size == actual_bs_recv, (
                    f"{tag} actual_batch_size should be {actual_bs_recv}, "
                    f"got {b.actual_batch_size}"
                )
                assert b.features.lengths().numel() == batch_size * num_features, (
                    f"{tag} KJT lengths count should be {batch_size * num_features}, "
                    f"got {b.features.lengths().numel()}"
                )

            if dense_label:
                assert ag_stripped.labels.numel() == actual_bs_recv * num_features
                assert a2a_stripped.labels.numel() == actual_bs_recv * num_features
                assert torch.equal(
                    ag_stripped.labels, a2a_stripped.labels
                ), f"Stripped dense labels mismatch on rank {dp_rank}"
            else:
                assert kjt_equal(
                    ag_stripped.labels, a2a_stripped.labels
                ), f"Stripped KJT labels mismatch on rank {dp_rank}"
            assert kjt_equal(
                ag_stripped.features, a2a_stripped.features
            ), f"Stripped features mismatch on rank {dp_rank}"
