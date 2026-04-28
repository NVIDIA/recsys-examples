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


@torch.fx.wrap
def padded_history_mask_with_causal_target(
    history_seqlen: torch.Tensor,
    history_max_seqlen: int,
    target_seqlen: int,
    history_causal: bool = True,
) -> torch.Tensor:
    """
    generate a mask for the history and the causal target.
    For history, we pretend it's an encoder, while for target, we pretend it's a decoder.
    Args:
        history_offsets: [batchsize + 1]
        history_max_seqlen: int
        target_offsets: [batchsize + 1]
        target_max_seqlen: int
    Returns:
        mask: [batchsize, 1, history_max_seqlen, history_max_seqlen + target_max_seqlen]


    Example:

       history_max_seqlen = 4
       target_seqlen = 3
       history_offsets = [0, 4, 6]
       [[a0,a1,a2,a3,c0,c1,c2], [a4,a5,c3,c4,c5]]
    mask:
        [
         [ [1,1,1,1,0,0,0],
           [1,1,1,1,0,0,0],
           [1,1,1,1,0,0,0],
           [1,1,1,1,0,0,0]]
           [1,1,1,1,1,0,0]]
           [1,1,1,1,1,1,0]]
           [1,1,1,1,1,1,1]],

         [ [1,1,0,0,0,0,0],
           [1,1,0,0,0,0,0],
           [1,1,1,0,0,0,0],
           [1,1,1,1,0,0,0],
           [1,1,1,1,1,0,0],
           [0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0],
         ]
    """
    device = history_seqlen.device
    # [B,1,1]
    valid_lengths = (history_seqlen + target_seqlen).view(-1, 1, 1)
    N = history_max_seqlen + target_seqlen
    ids = torch.arange(0, N, device=device).view(1, N)
    # [1,N,N]
    row_ids = ids.unsqueeze(-1).expand(-1, N, N)
    col_ids = row_ids.transpose(1, 2)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    causal_mask = torch.logical_or(row_col_dist > 0, valid_attn_mask)
    history_and_target_mask = torch.logical_and(
        row_ids < valid_lengths.view(-1, 1, 1), col_ids < valid_lengths.view(-1, 1, 1)
    )
    if not history_causal:
        history_mask = torch.logical_and(
            row_ids < history_seqlen.view(-1, 1, 1),
            col_ids < history_seqlen.view(-1, 1, 1),
        )
        history_upper_mask = torch.logical_and(history_mask, row_ids < col_ids)
        causal_mask = causal_mask | history_upper_mask
    valid_attn_mask = history_and_target_mask & causal_mask
    # [B, 1, N, N] for num_head attention
    valid_attn_mask = valid_attn_mask.unsqueeze(1)
    return ~valid_attn_mask


# refer to hstu https://github.com/jiayus-nvidia/FBGEMM/blob/main/fbgemm_gpu/experimental/hstu/img/context_causal_target.png
def padded_target_aware_causal_mask(
    history_seqlen: torch.Tensor,
    max_history_seqlen: int,
    num_target_region: int,
    target_max_seqlen_per_region: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Used for the beam search where there are multiple beams (targets) for each history.
    input sequence is : [history, target_region_0, target_region_1, ... padding_0, padding_1, ...],
                        where history length is history_seqlen, each target region length is target_max_seqlen_per_region,
                        and padding length is (max_history_seqlen - history_seqlen).
    intra region: causal ; inter region: invisible.
    each target needs to attend to the history

    """
    device = history_seqlen.device
    target_lengths = target_max_seqlen_per_region * num_target_region
    N = max_history_seqlen + target_lengths
    valid_lengths = (history_seqlen + target_lengths).view(-1, 1, 1)

    ids = torch.arange(0, N, device=device).view(1, N)
    # [B,1,1]
    row_ids = ids.unsqueeze(-1).expand(-1, N, N)
    col_ids = row_ids.transpose(1, 2)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)

    valid_region_mask = torch.logical_and(
        row_ids < valid_lengths.view(-1, 1, 1), col_ids < valid_lengths.view(-1, 1, 1)
    )
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if num_target_region > 0:
        target_group_row_ids = (
            torch.clamp(row_ids - valid_lengths + target_lengths, min=-1)
            // target_max_seqlen_per_region
        )
        target_group_col_ids = target_group_row_ids.transpose(1, 2)
        target_dist = target_group_row_ids - target_group_col_ids

        target_group_mask = torch.logical_or(
            target_dist == 0, (target_group_row_ids < 0) + (target_group_col_ids < 0)
        )
        # preserve the intra-target-group attention and purge the inter-target-group attention
        valid_attn_mask = torch.logical_and(valid_attn_mask, target_group_mask)

    # [B, N, N]
    valid_attn_mask = valid_attn_mask & valid_region_mask

    # [B, 1, N, N] for num_head attention
    valid_attn_mask = valid_attn_mask.unsqueeze(1)
    # note that we return the inverse of the mask to match the attention mask format.
    return ~valid_attn_mask


def dense_mask_to_arbitrary_func(
    valid_mask: torch.Tensor,
    seqlen: int,
    padding: int = 256,
) -> torch.Tensor:
    """
    Convert a dense bool attention mask to flash_attn's interval-based
    arbitrary_func tensor.

    For each query position q, the arbitrary_func encodes visible key
    positions as a union of intervals:
        visible(q) = [0, F0) ∪ [F1, F2) ∪ [F3, F4) ∪ ...

    Args:
        valid_mask: [B, N, N] or [B, 1, N, N] bool tensor (True = can attend).
        seqlen: sequence length N.
        padding: extra padding on last dim (FA convention, default 256).

    Returns:
        arbitrary_func: [B, 1, n_func, seqlen + padding] int32 tensor.
    """
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.squeeze(1)
    assert valid_mask.dim() == 3, f"Expected [B, N, N], got {valid_mask.shape}"

    B, N, _ = valid_mask.shape
    device = valid_mask.device

    # Detect interval boundaries via transitions
    shifted = torch.zeros_like(valid_mask)
    shifted[:, :, 1:] = valid_mask[:, :, :-1]
    starts = valid_mask & ~shifted  # start of each True run
    max_intervals = int(starts.sum(dim=-1).max().item())
    n_func = max(2 * max_intervals - 1, 1)
    if n_func % 2 == 0:
        n_func += 1

    # When first interval doesn't start at 0, it needs an extra slot.
    # Recount: base interval [0, F0) is free only if first run starts at 0.
    # Worst case: all intervals need explicit [F_start, F_end) pairs.
    # n_func = 2*max_intervals + 1 covers all cases.
    n_func = 2 * max_intervals + 1
    if n_func % 2 == 0:
        n_func += 1

    af = torch.zeros(B, 1, n_func, seqlen + padding, dtype=torch.int32, device=device)

    ends_shifted = torch.zeros_like(valid_mask)
    ends_shifted[:, :, :-1] = valid_mask[:, :, 1:]
    ends = valid_mask & ~ends_shifted  # last True position in each run

    for b in range(B):
        for q in range(N):
            row = valid_mask[b, q]
            if not row.any():
                continue
            start_pos = starts[b, q].nonzero(as_tuple=False).squeeze(-1)
            end_pos = ends[b, q].nonzero(as_tuple=False).squeeze(-1) + 1

            # F0 encodes [0, F0). If first interval starts at 0, use F0.
            # Otherwise F0 stays 0 (empty base interval) and all intervals
            # go into the extra slots.
            extra_idx = 0
            for iv in range(len(start_pos)):
                s, e = start_pos[iv].item(), end_pos[iv].item()
                if iv == 0 and s == 0:
                    af[b, 0, 0, q] = e
                else:
                    af[b, 0, 2 * extra_idx + 1, q] = s
                    af[b, 0, 2 * extra_idx + 2, q] = e
                    extra_idx += 1

    return af


def build_jagged_causal_arbitrary_func(
    offsets: torch.Tensor,
    total_tokens: int,
    padding: int = 256,
) -> torch.Tensor:
    """
    Build arbitrary_func for flattened jagged causal attention (B=1).

    All batch sequences are concatenated into a single sequence of length
    *total_tokens*.  Each query at global position *q* in batch element *b*
    can attend to keys in ``[offset[b], q+1)`` — standard causal within its
    own sequence, invisible to other sequences.

    Args:
        offsets: [B+1] cumulative sequence-length offsets.
        total_tokens: ``offsets[-1].item()`` — total number of tokens.
        padding: FA convention padding on the last dim (default 256).

    Returns:
        arbitrary_func: [1, 1, 3, total_tokens + padding] int32 tensor.
    """
    device = offsets.device
    n_func = 3  # F0=0, single interval [F1, F2)

    af = torch.zeros(
        1, 1, n_func, total_tokens + padding, dtype=torch.int32, device=device
    )

    positions = torch.arange(total_tokens, device=device)
    batch_ids = torch.searchsorted(offsets[1:], positions, right=True)
    batch_starts = offsets[batch_ids]

    # visible(q) = [0, 0) ∪ [batch_start, q+1)
    af[0, 0, 1, :total_tokens] = batch_starts.to(torch.int32)
    af[0, 0, 2, :total_tokens] = (positions + 1).to(torch.int32)

    return af


if __name__ == "__main__":
    history_seqlen = torch.tensor([4, 3]).cuda()
    max_history_seqlen = 6
    num_target_region = 3
    target_max_seqlen_per_region = 3
    device = torch.device("cuda")
    history_causal = False
    mask = padded_target_aware_causal_mask(
        history_seqlen,
        max_history_seqlen,
        num_target_region,
        target_max_seqlen_per_region,
        history_causal,
    )
    valid_mask = ~mask

    mask = padded_history_mask_with_causal_target(
        history_seqlen,
        max_history_seqlen,
        target_max_seqlen_per_region,
        history_causal,
    )
