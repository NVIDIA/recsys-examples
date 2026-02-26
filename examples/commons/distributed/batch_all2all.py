from dataclasses import fields
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from commons.sequence_batch.batch import BaseBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _build_dst_rank(
    recv_ids: torch.Tensor,
    local_batch_size: int,
    world_size: int,
    pg_group: dist.ProcessGroup,
) -> tuple:
    """Build ``dst_rank`` and ``recv_counts`` via all-to-all of requested indices.

    1. **Count exchange** – each rank tells every source rank how many
       samples it needs (``all_to_all_single`` on counts).
    2. **Index exchange** – each rank sends the specific local-sample
       indices it needs to the source rank; the source rank receives
       which of its local samples each destination rank wants.

    From the received indices, ``dst_rank[s] = destination_rank`` is trivially built.

    Args:
        recv_ids: **Sorted** global batch indices this rank needs.
        local_batch_size: Uniform local batch size on every rank.
        world_size: Total number of ranks.
        pg_group: Process group.

    Returns:
        ``(dst_rank, recv_counts)`` where

        * ``dst_rank`` – ``LongTensor(local_batch_size,)`` mapping each
          local sample to its destination rank.
        * ``recv_counts`` – ``List[int]`` of length ``world_size``;
          ``recv_counts[r]`` = number of samples this rank receives from
          source rank ``r`` in the data all-to-all.
    """
    device = recv_ids.device

    # recv_counts[r] = how many samples this rank needs from source rank r
    source_ranks = recv_ids // local_batch_size
    recv_counts = torch.bincount(source_ranks, minlength=world_size).tolist()

    # For each source rank, figure out WHICH local indices we need
    local_indices_needed = (recv_ids % local_batch_size).long()
    # recv_ids is sorted ⇒ source_ranks is non-decreasing ⇒ split is correct
    send_idx_list = list(local_indices_needed.split(recv_counts))

    # 1) Exchange counts: our recv_counts (data) = send counts for the INDEX a2a
    send_counts_t = torch.tensor(recv_counts, dtype=torch.long, device=device)
    recv_counts_t = torch.empty(world_size, dtype=torch.long, device=device)
    dist.all_to_all_single(recv_counts_t, send_counts_t, group=pg_group)

    # 2) Exchange indices: tell each source rank which local samples we need
    recv_idx_list = [
        torch.empty(int(recv_counts_t[r].item()), dtype=torch.long, device=device)
        for r in range(world_size)
    ]
    dist.all_to_all(recv_idx_list, send_idx_list, group=pg_group)

    # Build dst_rank from received indices
    dst_rank = torch.empty(local_batch_size, dtype=torch.long, device=device)
    for r in range(world_size):
        if recv_idx_list[r].numel() > 0:
            dst_rank[recv_idx_list[r]] = r

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
    ``all2all_batch``), the output samples within each key are already
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


def all2all_batch(
    batch: BaseBatch,
    recv_ids: torch.Tensor,
    pg_group: dist.ProcessGroup = dist.group.WORLD,
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
      * 2 ``all_to_all`` calls to build ``dst_rank`` (counts + indices)
      * Per KJT field: 2–3 ``all_to_all`` calls (lengths, values, [weights])
      * Per dense field: 1 ``all_to_all`` call

    Args:
        batch: Local batch to redistribute.
        recv_ids: **Sorted** global batch indices this rank needs.
            All ranks' ``recv_ids`` must form a partition of
            ``[0, global_batch_size)``.
        pg_group: Process group for distributed operations.

    Returns:
        A new ``BaseBatch`` containing only the samples specified by
        ``recv_ids``, in that order.
    """
    world_size = dist.get_world_size(pg_group)
    local_batch_size = batch.batch_size

    if world_size == 1:
        return batch

    # ---- Phase 0: Build dst_rank and recv_counts via all-to-all ----
    dst_rank, recv_counts = _build_dst_rank(
        recv_ids, local_batch_size, world_size, pg_group
    )

    # ---- Phase 1: KJT fields — all-to-all via _all2all_kjt ----
    kjt_field_names: List[str] = []
    kjt_inputs: List[KeyedJaggedTensor] = []
    for f in fields(batch):
        val = getattr(batch, f.name)
        if isinstance(val, KeyedJaggedTensor):
            kjt_field_names.append(f.name)
            kjt_inputs.append(val)

    kjt_outputs: List[KeyedJaggedTensor] = []
    for kjt in kjt_inputs:
        kjt_outputs.append(
            _all2all_kjt(kjt, dst_rank, recv_counts, world_size, pg_group)
        )

    kjt_result_map: Dict[str, KeyedJaggedTensor] = dict(
        zip(kjt_field_names, kjt_outputs)
    )

    # ---- Phase 2: Dense tensor fields — all-to-all via _all2all_dense_tensor ----
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

    new_batch.batch_size = recv_ids.numel()
    new_batch.actual_batch_size = new_batch.batch_size

    return new_batch
