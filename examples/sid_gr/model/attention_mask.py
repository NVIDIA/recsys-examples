import torch


@torch.fx.wrap
def padded_causal_mask_with_optional_bos(
    input_offsets: torch.Tensor,
    input_max_seqlen: int,
    add_bos_to_history: bool = False,
    bos_interval: int = 0,
) -> torch.Tensor:
    B = input_offsets.size(0) - 1
    S = input_max_seqlen

    # bs, num_head, seq, seq
    lower_triangle_mask = torch.tril(
        torch.ones(
            (B, 1, S, S),
            dtype=torch.bool,
            device=torch.cuda.current_device(),
        )
    )
    if add_bos_to_history:
        num_hierarchies_with_bos = bos_interval + 1
        # [[{s0,s1,s2| bos, s3,s4,s5| bos, s6,s7,s8| bos, ..., s_{3N-1}}, {bos, c0,c1,c2}], [{s3,s4,s5| bos, s6,s7,s8| bos, ..., s_{3M-1}}, {bos, c4,c5,c6}]]
        assert (
            S + 1
        ) % num_hierarchies_with_bos == 0, (
            "input_max_seqlen + 1 should be divisible by bos_interval + 1"
        )

        # later history tokens can't attend to previous bos tokens
        bos_row_ids = torch.arange(
            0, S, device=input_offsets.device, dtype=input_offsets.dtype
        ).view(-1, 1)
        bos_col_ids = torch.arange(
            0, S, device=input_offsets.device, dtype=input_offsets.dtype
        ).view(1, -1)
        bos_col_mask = (bos_col_ids + 1) % num_hierarchies_with_bos == 0
        bos_col_mask = bos_col_mask & (
            bos_row_ids >= bos_col_ids + num_hierarchies_with_bos
        )
        lower_triangle_mask = lower_triangle_mask & ~bos_col_mask
        # bos_row_ids = bos_row_ids[bos_row_ids % (num_hierarchies + 1) == 0] * (num_hierarchies + 1)
    else:
        # [[{item0, item1, item2, ..., itemN}, {bos}], [{item3, item4, item5, ..., itemM}, {bos}]]
        # it's causal
        pass
    # we set the bos
    # broadcast num_head, s_kv
    mask = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=torch.ones(size=(input_offsets[-1],)).cuda(),
            offsets=[input_offsets],
            max_lengths=[input_max_seqlen],
        )
        .unsqueeze(1)
        .unsqueeze(-1)
    )
    jagged_causal_mask = torch.logical_and(
        lower_triangle_mask,
        mask,
    )
    # note that we return the inverse of the mask to match the attention mask format.
    return ~jagged_causal_mask


# refer to hstu https://github.com/jiayus-nvidia/FBGEMM/blob/main/fbgemm_gpu/experimental/hstu/img/context_causal_target.png
def padded_target_aware_causal_mask(
    batchsize,
    history_seqlen: int,
    max_history_seqlen: int,
    num_target_region: int,
    target_max_seqlen_per_region: int,
    device: torch.device,
) -> torch.Tensor:
    """
    input sequence is : [history, target_region_0, target_region_1, ... padding_0, padding_1, ...],
                        where history length is history_seqlen, each target region length is target_max_seqlen_per_region,
                        and padding length is (max_history_seqlen - history_seqlen).
    intra region: causal ; inter region: invisible.
    each target needs to attend to the history

    """
    total_seqlen = max_history_seqlen + num_target_region * target_max_seqlen_per_region
    valid_seqlen = history_seqlen + target_max_seqlen_per_region * num_target_region
    # create row and col indices [total_seqlen, total_seqlen]
    row_indices = torch.arange(total_seqlen, device=device).unsqueeze(
        1
    )  # [total_seqlen, 1]
    col_indices = torch.arange(total_seqlen, device=device).unsqueeze(
        0
    )  # [1, total_seqlen]

    # history region: causal (row < max_history_seqlen AND col < max_history_seqlen AND row >= col)
    is_history_row = row_indices < history_seqlen
    is_history_col = col_indices < history_seqlen
    history_causal = is_history_row & is_history_col & (row_indices >= col_indices)

    # target to history (row >= max_history_seqlen AND col < max_history_seqlen)
    is_target_row = (row_indices >= history_seqlen) & (row_indices < valid_seqlen)
    target_attend_history = is_target_row & is_history_col

    # intra region: causal
    target_row_idx = row_indices - history_seqlen  # target region row index
    target_col_idx = col_indices - history_seqlen  # target region col index
    row_region_id = target_row_idx // target_max_seqlen_per_region
    row_offset = target_row_idx % target_max_seqlen_per_region

    col_region_id = target_col_idx // target_max_seqlen_per_region
    col_offset = target_col_idx % target_max_seqlen_per_region

    # intra target region: causal
    is_target_col = (col_indices >= history_seqlen) & (col_indices < valid_seqlen)
    same_region = row_region_id == col_region_id
    causal_within_region = row_offset >= col_offset
    target_internal_mask = (
        is_target_row & is_target_col & same_region & causal_within_region
    )

    mask = history_causal | target_attend_history | target_internal_mask

    # expand batch dimension: [batchsize, 1, total_seqlen, total_seqlen]
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batchsize, 1, -1, -1)

    # note that we return the inverse of the mask to match the attention mask format.
    return ~mask
