import sys

import pytest
import torch
import torch.distributed as dist

sys.path.append("../../examples")

import commons.utils.initialize as init
from commons.distributed.batch_all2all import all2all_batch
from commons.distributed.batch_allgather import allgather_batch
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
            torch.arange(batch_size * num_features, device=torch.device("cuda")).view(
                -1
            )
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


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("max_sequence_length", [10])
@pytest.mark.parametrize("num_features", [2])
@pytest.mark.parametrize("dense_label", [False])
def test_all2all_vs_allgather(
    batch_size,
    max_sequence_length,
    num_features,
    dense_label,
):
    """Test that all2all_batch produces the same results as allgather + index_select.

    This test verifies the correctness of all2all_batch by comparing it with
    the allgather-based approach. Both methods should produce identical results
    when given the same recv_ids.
    """
    init.initialize_distributed()

    with init.auto_destroy_global_state():
        init.initialize_model_parallel(1)
        init.set_random_seed(1234)
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_world_size = parallel_state.get_data_parallel_world_size()
        dp_group = parallel_state.get_data_parallel_group()

        # Generate local batch
        batch = generate_batch(
            batch_size, max_sequence_length, num_features, dense_label
        )

        # Build recv_ids as a partition of [0, global_batch_size).
        # Rank 0 generates the full permutation and broadcasts to all ranks.
        global_batch_size = batch_size * dp_world_size
        if dp_rank == 0:
            perm = torch.randperm(global_batch_size, device=torch.device("cuda"))
        else:
            perm = torch.empty(
                global_batch_size, dtype=torch.int64, device=torch.device("cuda")
            )
        dist.broadcast(perm, src=0, group=dp_group)

        # Each rank takes a contiguous chunk of the permutation
        recv_ids = perm[dp_rank * batch_size : (dp_rank + 1) * batch_size]
        recv_ids, _ = torch.sort(recv_ids)

        # Method 1: AllGather + index_select (reference implementation)
        allgathered_batch = allgather_batch(batch, pg_group=dp_group)
        allgather_result = allgathered_batch.index_select(recv_ids)
        allgather_result.batch_size = allgather_result.batch_size // dp_world_size

        # Method 2: All2All (tested implementation)
        all2all_result = all2all_batch(batch, recv_ids, pg_group=dp_group)

        # Compare results
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
        assert allgather_result.actual_batch_size == all2all_result.actual_batch_size, (
            f"Actual batch size mismatch on rank {dp_rank}: "
            f"AllGather={allgather_result.actual_batch_size}, All2All={all2all_result.actual_batch_size}"
        )

        # 4. Compare feature_to_max_seqlen
        assert (
            allgather_result.feature_to_max_seqlen
            == all2all_result.feature_to_max_seqlen
        ), f"feature_to_max_seqlen mismatch on rank {dp_rank}"


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("max_sequence_length", [10, 127])
@pytest.mark.parametrize("num_features", [2, 3])
@pytest.mark.parametrize("dense_label", [True, False])
def test_all2all_vs_allgather_incomplete_batch(
    batch_size,
    max_sequence_length,
    num_features,
    dense_label,
):
    """Test that all2all_batch handles incomplete batches correctly.

    Each rank gets a different random actual_batch_size in [1, batch_size],
    simulating heterogeneous incomplete batches.  Verifies that the
    redistributed data content matches between the AllGather and All2All
    paths.
    """
    init.initialize_distributed()

    with init.auto_destroy_global_state():
        init.initialize_model_parallel(1)
        init.set_random_seed(1234)
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_world_size = parallel_state.get_data_parallel_world_size()
        dp_group = parallel_state.get_data_parallel_group()

        # Each rank gets a different random actual_batch_size in [1, batch_size).
        # init.set_random_seed guarantees different random state per rank.
        actual_batch_size = torch.randint(1, batch_size, (1,)).item()

        batch = generate_batch(
            batch_size,
            max_sequence_length,
            num_features,
            dense_label,
            actual_batch_size=actual_batch_size,
        )

        # Build recv_ids as a partition of [0, global_batch_size)
        global_batch_size = batch_size * dp_world_size
        if dp_rank == 0:
            perm = torch.randperm(global_batch_size, device=torch.device("cuda"))
        else:
            perm = torch.empty(
                global_batch_size, dtype=torch.int64, device=torch.device("cuda")
            )
        dist.broadcast(perm, src=0, group=dp_group)

        recv_ids = perm[dp_rank * batch_size : (dp_rank + 1) * batch_size]
        recv_ids, _ = torch.sort(recv_ids)

        # Method 1: AllGather + index_select (reference)
        allgathered_batch = allgather_batch(batch, pg_group=dp_group)
        allgather_result = allgathered_batch.index_select(recv_ids)
        allgather_result.batch_size = allgather_result.batch_size // dp_world_size

        # Method 2: All2All (tested implementation)
        all2all_result = all2all_batch(batch, recv_ids, pg_group=dp_group)

        # 1. batch_size must match
        assert allgather_result.batch_size == all2all_result.batch_size, (
            f"Batch size mismatch on rank {dp_rank}: "
            f"AllGather={allgather_result.batch_size}, "
            f"All2All={all2all_result.batch_size}"
        )

        # 2. All2All sets actual_batch_size = recv_ids.numel() (= batch_size),
        #    treating all received samples as actual.  The AllGather path may
        #    report a smaller value because index_select filters padding indices.
        assert all2all_result.actual_batch_size == recv_ids.numel(), (
            f"All2All actual_batch_size should equal recv_ids count on rank {dp_rank}: "
            f"got {all2all_result.actual_batch_size}, expected {recv_ids.numel()}"
        )

        # 3. Data content must match
        assert kjt_equal(
            allgather_result.features, all2all_result.features
        ), f"Features mismatch on rank {dp_rank}"
        if dense_label:
            assert torch.equal(
                allgather_result.labels, all2all_result.labels
            ), f"Labels mismatch on rank {dp_rank}"
        else:
            assert kjt_equal(
                allgather_result.labels, all2all_result.labels
            ), f"Labels mismatch on rank {dp_rank}"
