from dataclasses import fields
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from commons.sequence_batch.batch import BaseBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _build_dst_rank_local(
    partitions_indices: torch.Tensor,
    rank: int,
    local_batch_size: int,
) -> tuple:
    """Build ``dst_rank`` and ``recv_counts`` locally from KK partitions.

    When every rank holds the same ``partitions_indices`` (produced by KK
    on AllGathered workloads with ``equal_size=True``), ``dst_rank`` can be
    derived without any communication.

    Args:
        partitions_indices: 2-D int64 tensor of shape ``[W, B]`` where
            ``partitions_indices[r]`` contains the global indices assigned
            to rank ``r``.
        rank: This rank's index.
        local_batch_size: ``B`` — samples per rank before shuffling.

    Returns:
        ``(dst_rank, recv_counts)`` where

        * ``dst_rank`` – ``LongTensor(local_batch_size,)`` mapping each
          local sample to its destination rank.
        * ``recv_counts`` – ``List[int]`` of length ``W``;
          ``recv_counts[r]`` = number of samples this rank receives from
          source rank ``r``.
    """
    num_partitions = partitions_indices.shape[0]
    device = partitions_indices.device
    global_batch_size = local_batch_size * num_partitions

    rank_ids = (
        torch.arange(num_partitions, device=device)
        .unsqueeze(1)
        .expand_as(partitions_indices)
    )
    global_to_rank = torch.empty(global_batch_size, dtype=torch.long, device=device)
    global_to_rank[partitions_indices.reshape(-1)] = rank_ids.reshape(-1)

    my_offset = rank * local_batch_size
    dst_rank = global_to_rank[my_offset : my_offset + local_batch_size].contiguous()

    source_ranks = partitions_indices[rank] // local_batch_size
    recv_counts = torch.bincount(source_ranks, minlength=num_partitions).tolist()

    return dst_rank, recv_counts


def _all2all_dense_tensor(
    tensor: torch.Tensor,
    dst_rank: torch.Tensor,
    recv_counts: List[int],
    local_batch_size: int,
    world_size: int,
    pg_group: dist.ProcessGroup,
) -> torch.Tensor:
    """All2All a dense tensor using a per-sample rank assignment.

    Because ``recv_ids`` is sorted, the sender's stable argsort on
    ``dst_rank`` produces samples in ascending global-index order within
    each destination group.  The receiver therefore gets data already
    aligned with its sorted ``recv_ids`` — no post-reordering needed.

    Args:
        tensor: Local tensor of shape ``[local_batch_size, ...]``.
        dst_rank: 1D tensor of shape ``(local_batch_size,)`` where
            ``dst_rank[s]`` is the destination rank for sample ``s``.
        recv_counts: Number of samples to receive from each rank.
        local_batch_size: Local batch size.
        world_size: Total number of ranks.
        pg_group: Process group.

    Returns:
        Tensor of shape ``[sum(recv_counts), ...]`` flattened to 1-D,
        ordered to match the caller's sorted ``recv_ids``.
    """
    # Reshape to [local_batch_size, -1] so each row is one sample.
    # Dense fields may be stored as 1-D (e.g. labels with shape (B*F,)),
    # so we must use local_batch_size, NOT tensor.shape[0].
    tensor_2d = tensor.reshape(local_batch_size, -1)

    # Sort samples by destination rank; stable argsort keeps within-rank order
    sorted_indices = dst_rank.argsort(stable=True)
    send_counts = torch.bincount(dst_rank, minlength=world_size).tolist()

    # Prepare send tensors: index then split by per-rank counts
    sorted_tensor = tensor_2d[sorted_indices].contiguous()
    send_tensors = list(sorted_tensor.split(send_counts, dim=0))

    # Prepare receive buffers
    recv_tensors = [
        torch.empty(
            (count, tensor_2d.shape[1]), dtype=tensor.dtype, device=tensor.device
        )
        for count in recv_counts
    ]

    # Perform all2all
    dist.all_to_all(recv_tensors, send_tensors, group=pg_group)

    # Received data is already in sorted recv_ids order — just concat.
    result = torch.cat(recv_tensors, dim=0)

    # Flatten back to 1-D (consistent with BaseBatch.index_select behaviour)
    return result.reshape(-1)


def _all2all_kjt(
    kjt: KeyedJaggedTensor,
    dst_rank: torch.Tensor,
    recv_counts: List[int],
    world_size: int,
    pg_group: dist.ProcessGroup,
) -> KeyedJaggedTensor:
    """All-to-all a KJT based on per-sample rank assignment.

    Directly performs all-to-all on lengths, values, and weights of the KJT,
    without depending on TorchRec's ``KJTAllToAll``.

    Steps:
        1. **Sort** local samples by destination rank using
           ``keyed_jagged_index_select_dim1``.  Stable argsort on
           ``dst_rank.repeat(num_keys)`` produces (rank, key, sample) ordering.
        2. **All-to-all** lengths, values, and weights separately
           (3 or 2 NCCL calls).
        3. **Transpose** received data from
           ``(source_rank, key, sample)`` → ``(key, source_rank, sample)``
           via a vectorized block-transpose permutation.

    When ``recv_ids`` is sorted on every rank (required by
    ``pad_and_all2all_batch``), the output samples within each key are already
    in ascending global-index order — no extra reordering is needed.

    Args:
        kjt: Local KJT with ``num_keys`` keys and ``batch_size`` samples per key.
        dst_rank: Shape ``(batch_size,)``. ``dst_rank[s]`` = destination rank for
            sample ``s``.
        recv_counts: ``recv_counts[r]`` = number of *samples* to receive from
            rank ``r``.
        world_size: Total number of ranks.
        pg_group: Process group.

    Returns:
        KJT with ``num_keys`` keys and ``sum(recv_counts)`` samples per key.
    """
    device = kjt.values().device
    num_keys = len(kjt.keys())
    batch_size = kjt.lengths().numel() // num_keys
    has_weights = kjt.weights_or_none() is not None
    recv_counts_t = torch.tensor(recv_counts, dtype=torch.long, device=device)

    # ---- Step 1: Reorder by destination rank ----
    # Stable argsort of dst_rank.repeat(num_keys) gives (rank, key, sample) order.
    sorted_indices = dst_rank.repeat(num_keys).argsort(stable=True)

    select_out = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
        kjt.values(),
        kjt.lengths(),
        kjt.offsets(),
        sorted_indices,
        num_keys * batch_size,
        kjt.weights_or_none(),
    )
    sorted_values, sorted_lengths = select_out[0], select_out[1]
    sorted_weights = select_out[2] if has_weights else None

    # ---- Step 2: All-to-all on lengths ----
    send_lc = torch.bincount(dst_rank, minlength=world_size) * num_keys
    recv_lc = recv_counts_t * num_keys

    send_lengths_list = list(sorted_lengths.split(send_lc.tolist()))
    recv_lengths_list = [
        torch.empty(c, dtype=sorted_lengths.dtype, device=device)
        for c in recv_lc.tolist()
    ]
    dist.all_to_all(recv_lengths_list, send_lengths_list, group=pg_group)
    recv_lengths_flat = torch.cat(recv_lengths_list)

    # ---- Step 3: All-to-all on values (and optionally weights) ----
    # Vectorized segment sums via scatter_add (1 GPU→CPU sync instead of W)
    def _seg_sum(data: torch.Tensor, seg_sizes: torch.Tensor) -> List[int]:
        """Sum *data* within contiguous segments of sizes *seg_sizes*."""
        ids = torch.arange(world_size, device=device).repeat_interleave(seg_sizes)
        out = data.new_zeros(world_size, dtype=torch.long)
        out.scatter_add_(0, ids, data.to(torch.long))
        return out.tolist()

    send_vc = _seg_sum(sorted_lengths, send_lc)
    recv_vc = _seg_sum(recv_lengths_flat, recv_lc)

    send_values_list = list(sorted_values.split(send_vc))
    recv_values_list = [
        torch.empty(c, dtype=sorted_values.dtype, device=device) for c in recv_vc
    ]
    dist.all_to_all(recv_values_list, send_values_list, group=pg_group)
    recv_values_flat = torch.cat(recv_values_list)

    recv_weights_flat: Optional[torch.Tensor] = None
    if has_weights:
        send_weights_list = list(sorted_weights.split(send_vc))
        recv_weights_list = [
            torch.empty(c, dtype=sorted_weights.dtype, device=device) for c in recv_vc
        ]
        dist.all_to_all(recv_weights_list, send_weights_list, group=pg_group)
        recv_weights_flat = torch.cat(recv_weights_list)

    # ---- Step 4: Vectorized block-transpose ----
    # Received: (source_rank, key, sample) → Target: (key, source_rank, sample)
    #   source flat index (r, k, s) = prefix[r] + k * n_r + s
    #   target iterates k=0..K-1 (outer), r=0..W-1 (inner), s=0..n_r-1
    total = recv_lengths_flat.numel()
    if total == 0:
        return KeyedJaggedTensor(
            keys=kjt.keys(),
            values=torch.empty(0, dtype=kjt.values().dtype, device=device),
            lengths=torch.empty(0, dtype=kjt.lengths().dtype, device=device),
        )

    # prefix[r] = flat offset where source rank r's block starts
    prefix = torch.zeros(world_size, dtype=torch.long, device=device)
    if world_size > 1:
        prefix[1:] = (recv_counts_t[:-1] * num_keys).cumsum(0)

    # Target block (k, r) has recv_counts[r] entries.
    k_idx = torch.arange(num_keys, device=device).repeat_interleave(world_size)
    r_idx = torch.arange(world_size, device=device).repeat(num_keys)
    blk_sz = recv_counts_t.repeat(num_keys)

    # Source start for each target block
    src_starts = prefix[r_idx] + k_idx * recv_counts_t[r_idx]

    # Expand to per-entry and add within-block offset [0, 1, …, n_r-1]
    expanded = src_starts.repeat_interleave(blk_sz)
    blk_cum = blk_sz.cumsum(0)
    blk_off = torch.zeros_like(blk_cum)
    blk_off[1:] = blk_cum[:-1]
    within = torch.arange(
        total, device=device, dtype=torch.long
    ) - blk_off.repeat_interleave(blk_sz)
    perm = expanded + within

    recv_offsets = torch.zeros(total + 1, dtype=torch.long, device=device)
    recv_offsets[1:] = recv_lengths_flat.to(torch.long).cumsum(0)

    out = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
        recv_values_flat,
        recv_lengths_flat,
        recv_offsets,
        perm,
        total,
        recv_weights_flat,
    )

    return KeyedJaggedTensor(
        keys=kjt.keys(),
        values=out[0],
        weights=out[2] if has_weights else None,
        lengths=out[1],
    )


def _all2all_kjt_list(
    kjt_list: List[KeyedJaggedTensor],
    dst_rank: torch.Tensor,
    recv_counts: List[int],
    world_size: int,
    pg_group: dist.ProcessGroup,
) -> List[KeyedJaggedTensor]:
    """Fused all-to-all for a **list** of KeyedJaggedTensors.

    All KJTs are concatenated into a single mega-KJT before communication so
    that the entire list is exchanged with only **2–3 NCCL calls** (lengths,
    values, [weights]) regardless of how many KJTs there are.  After
    receiving, the result is split back into individual KJTs.

    Requirements:
      - All KJTs must share the same ``batch_size`` (stride).
      - All KJTs must share the same ``values`` dtype.

    Args:
        kjt_list: List of KJTs to redistribute.
        dst_rank: Shape ``(batch_size,)``.  ``dst_rank[s]`` = destination rank.
        recv_counts: ``recv_counts[r]`` = samples to receive from rank ``r``.
        world_size: Total number of ranks.
        pg_group: Process group.

    Returns:
        List of redistributed KJTs, one per input KJT.
    """
    if not kjt_list:
        return []
    if len(kjt_list) == 1:
        return [_all2all_kjt(kjt_list[0], dst_rank, recv_counts, world_size, pg_group)]

    # Check weight configuration — fall back to per-KJT calls if mixed
    has_weights = [kjt.weights_or_none() is not None for kjt in kjt_list]
    if any(has_weights) and not all(has_weights):
        return [
            _all2all_kjt(kjt, dst_rank, recv_counts, world_size, pg_group)
            for kjt in kjt_list
        ]

    keys_list = [list(kjt.keys()) for kjt in kjt_list]
    K_list = [len(keys) for keys in keys_list]
    all_keys = [key for keys in keys_list for key in keys]

    # Merge into a single mega-KJT
    all_lengths = torch.cat([kjt.lengths() for kjt in kjt_list])
    all_values = torch.cat([kjt.values() for kjt in kjt_list])
    all_weights = (
        torch.cat([kjt.weights() for kjt in kjt_list]) if all(has_weights) else None
    )
    merged_kjt = KeyedJaggedTensor(
        keys=all_keys,
        values=all_values,
        lengths=all_lengths,
        weights=all_weights,
    )

    # Fused all-to-all (2–3 NCCL calls instead of K × (2–3))
    merged_result = _all2all_kjt(
        merged_kjt, dst_rank, recv_counts, world_size, pg_group
    )

    # Split back into individual KJTs
    K_total = sum(K_list)
    recv_total = sum(recv_counts)
    result_lengths = merged_result.lengths()
    result_values = merged_result.values()
    result_weights = merged_result.weights_or_none()

    per_key_value_counts = (
        result_lengths.view(K_total, recv_total).to(torch.long).sum(dim=1)
    )
    kjt_value_counts = []
    key_offset = 0
    for K in K_list:
        kjt_value_counts.append(per_key_value_counts[key_offset : key_offset + K].sum())
        key_offset += K
    kjt_value_counts_cpu = torch.stack(kjt_value_counts).cpu()

    result_kjts: List[KeyedJaggedTensor] = []
    lengths_offset = 0
    values_offset = 0
    weights_offset = 0
    for i, (keys, K) in enumerate(zip(keys_list, K_list)):
        kjt_num_lengths = K * recv_total
        kjt_lengths = result_lengths[lengths_offset : lengths_offset + kjt_num_lengths]
        kjt_num_values = kjt_value_counts_cpu[i].item()
        kjt_values = result_values[values_offset : values_offset + kjt_num_values]

        kjt_weights_i = None
        if result_weights is not None:
            kjt_weights_i = result_weights[
                weights_offset : weights_offset + kjt_num_values
            ]
            weights_offset += kjt_num_values

        result_kjts.append(
            KeyedJaggedTensor(
                keys=keys,
                values=kjt_values,
                lengths=kjt_lengths,
                weights=kjt_weights_i,
            )
        )
        lengths_offset += kjt_num_lengths
        values_offset += kjt_num_values

    return result_kjts


def pad_and_all2all_batch(
    batch: BaseBatch,
    recv_ids: torch.Tensor,
    pg_group: dist.ProcessGroup,
    dst_rank: torch.Tensor,
    recv_counts: List[int],
) -> BaseBatch:
    """Redistribute a batch across ranks via all-to-all based on global indices.

    Each rank specifies which global sample indices it needs via
    ``recv_ids``.  The function figures out which local samples must be
    sent to which rank, performs the all-to-all exchange for both KJT and
    dense-tensor fields, and returns a new batch whose samples match
    ``recv_ids``.

    Compared to AllGather + index_select:
      * Communicates only the needed samples (O(B) instead of O(W·B)).
      * More efficient when each rank needs a small subset of global samples.

    Communication cost:
      * All KJT fields fused: 2 ``all_to_all`` calls total (lengths, values)
      * Per dense field: 1 ``all_to_all`` call

    **world_size == 1 fast-path**: When the process group contains only
    one rank, no collective communication is performed.  Dense tensors
    are still zero-padded to ``batch_size`` when ``actual_batch_size <
    batch_size`` (incomplete batch) and ``actual_batch_size`` is set to
    ``recv_ids.numel()`` for consistency with the multi-rank code path.
    KJT fields are returned as-is.

    Args:
        batch: Local batch to redistribute.
        recv_ids: **Sorted** global batch indices this rank needs.
            All ranks' ``recv_ids`` must form a partition of
            ``[0, global_batch_size)``.
        pg_group: Process group for distributed operations.
        dst_rank: Destination rank for each local sample.  Shape
            ``(batch.batch_size,)``.  Typically produced by
            :func:`_build_dst_rank_local`.
        recv_counts: Per-source-rank receive counts.  ``recv_counts[r]``
            = number of samples this rank receives from rank ``r``.

    Returns:
        A new ``BaseBatch`` containing only the samples specified by
        ``recv_ids``, in that order.
    """
    world_size = dist.get_world_size(pg_group)
    local_batch_size = batch.batch_size

    if world_size == 1:
        # Under the unified convention, dense tensors already have
        # dim-0 == batch_size, so no padding is needed.
        return batch

    # ---- Phase 1: KJT fields — fused all-to-all via _all2all_kjt_list ----
    kjt_field_names: List[str] = []
    kjt_inputs: List[KeyedJaggedTensor] = []
    for f in fields(batch):
        val = getattr(batch, f.name)
        if isinstance(val, KeyedJaggedTensor):
            kjt_field_names.append(f.name)
            kjt_inputs.append(val)

    kjt_outputs = _all2all_kjt_list(
        kjt_inputs, dst_rank, recv_counts, world_size, pg_group
    )

    kjt_result_map: Dict[str, KeyedJaggedTensor] = dict(
        zip(kjt_field_names, kjt_outputs)
    )

    # ---- Phase 2: Dense tensor fields — all-to-all via _all2all_dense_tensor ----
    # Under the unified convention, dense tensors already have dim-0 ==
    # batch_size (== local_batch_size), so no padding is needed.
    def all2all_field(
        tensor_or_kjt: Union[torch.Tensor, KeyedJaggedTensor],
    ) -> Union[torch.Tensor, KeyedJaggedTensor]:
        if isinstance(tensor_or_kjt, KeyedJaggedTensor):
            return tensor_or_kjt  # already handled in Phase 1
        elif isinstance(tensor_or_kjt, torch.Tensor):
            return _all2all_dense_tensor(
                tensor_or_kjt,
                dst_rank,
                recv_counts,
                local_batch_size,
                world_size,
                pg_group,
            )
        else:
            raise ValueError(f"Unsupported type: {type(tensor_or_kjt)}")

    new_batch = batch._apply_to_tensors_or_kjt(all2all_field, inplace=False)

    # Patch KJT fields (overwrite placeholders from Phase 2)
    for name, kjt_out in kjt_result_map.items():
        setattr(new_batch, name, kjt_out)

    new_batch.actual_batch_size = recv_ids.numel()

    return new_batch
