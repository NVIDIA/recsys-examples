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
from typing import List

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


def build_incremental_append_arbitrary_func(
    history_seqlens: torch.Tensor,
    max_history_seqlen: int,
    current_step: int,
    beam_widths: List[int],
    ancestor_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Build an arbitrary_func tensor for the Method A (Incremental Append) beam
    search layout, compatible with flash_attn's interval-based mask encoding.

    Method A sequence layout (at the time of predicting step ``current_step``):
      [history..., step0_beams, step1_beams, ..., step_{current_step-1}_beams]

    Mask semantics:
      - History tokens: causal within history (token i sees [0, i+1))
      - Generated token at (step s, beam b): sees
          [0, hist_len)                       (all history)
        plus its ancestor chain encoded via ancestor_positions.
      - The last step's tokens (step current_step-1) use precise tree mask.
      - Earlier steps' tokens use a causal-like approximation.

    Args:
        history_seqlens: [B] per-sample history lengths.
        max_history_seqlen: max history length (for padding).
        current_step: the hierarchy step being predicted (0-based).
            The sequence contains tokens from steps 0..current_step-1.
        beam_widths: list of beam widths per step.
        ancestor_positions: [B, topk_current, current_step] physical positions
            of each beam's ancestors at steps 0..current_step-1.
            None if current_step == 0.

    Returns:
        arbitrary_func: [B, 1, n_func, total_seqlen + 256] int32 tensor.
    """
    device = history_seqlens.device
    B = history_seqlens.shape[0]
    total_generated = sum(beam_widths[:current_step])
    total_seqlen = max_history_seqlen + total_generated

    if current_step == 0:
        # Pure causal mask over history, n_func=1
        af = torch.zeros(
            B, 1, 1, total_seqlen + 256, dtype=torch.int32, device=device
        )
        hist_positions = torch.arange(max_history_seqlen, device=device)
        hist_f0 = (hist_positions + 1).unsqueeze(0).expand(B, -1)
        hist_f0 = torch.min(
            hist_f0, history_seqlens.unsqueeze(1).expand_as(hist_f0)
        )
        af[:, 0, 0, :max_history_seqlen] = hist_f0.to(torch.int32)
        return af

    topk_current = beam_widths[current_step - 1]

    # n_func must be odd. The last-step tokens need at most
    # (current_step + 1) intervals: 1 (history) + current_step (ancestors incl self).
    max_intervals = current_step + 1
    n_func = max(2 * max_intervals - 1, 3)

    af = torch.zeros(
        B, 1, n_func, total_seqlen + 256, dtype=torch.int32, device=device
    )

    # --- History tokens: causal mask, F0 = pos + 1 ---
    hist_positions = torch.arange(max_history_seqlen, device=device)
    hist_f0 = (hist_positions + 1).unsqueeze(0).expand(B, -1)
    hist_f0 = torch.min(
        hist_f0, history_seqlens.unsqueeze(1).expand_as(hist_f0)
    )
    af[:, 0, 0, :max_history_seqlen] = hist_f0.to(torch.int32)

    # --- Generated tokens ---
    gen_offset = 0
    for s in range(current_step):
        bw_s = beam_widths[s]
        gen_start = max_history_seqlen + gen_offset
        is_last_step = s == current_step - 1

        for b in range(bw_s):
            pos = gen_start + b

            # F0: base interval = [0, hist_len) — see all history
            af[:, 0, 0, pos] = history_seqlens.to(torch.int32)

            if is_last_step and ancestor_positions is not None:
                # Tree mask: precise ancestor intervals from ancestor_positions.
                # ancestor_positions[:, b, :] has positions at steps 0..current_step-1.
                # Steps 0..current_step-2 are true ancestors; step current_step-1 is self.
                for anc_idx in range(current_step):
                    anc_pos = ancestor_positions[:, b, anc_idx]  # [B]
                    af[:, 0, 2 * anc_idx + 1, pos] = anc_pos.to(torch.int32)
                    af[:, 0, 2 * anc_idx + 2, pos] = (anc_pos + 1).to(
                        torch.int32
                    )
            else:
                # Earlier steps' tokens: causal over [hist_len, pos+1)
                af[:, 0, 1, pos] = max_history_seqlen
                af[:, 0, 2, pos] = pos + 1

        gen_offset += bw_s

    return af


def build_incremental_append_dense_mask(
    history_seqlens: torch.Tensor,
    max_history_seqlen: int,
    current_step: int,
    beam_widths: List[int],
    ancestor_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Build a dense [B, N, N] bool mask for Method A layout — for testing and
    validation against arbitrary_func encoding.

    Args:
        current_step: the hierarchy step being predicted (0-based).
            The sequence contains tokens from steps 0..current_step-1.
        ancestor_positions: [B, topk, current_step] or None.
            Covers steps 0..current_step-1. The last entry is the self position.

    Returns:
        mask: [B, N, N] bool tensor where True = can attend.
    """
    device = history_seqlens.device
    B = history_seqlens.shape[0]
    total_generated = sum(beam_widths[:current_step])
    N = max_history_seqlen + total_generated

    mask = torch.zeros(B, N, N, dtype=torch.bool, device=device)

    # History region: causal
    for b_idx in range(B):
        hl = history_seqlens[b_idx].item()
        for i in range(hl):
            mask[b_idx, i, : i + 1] = True

    if current_step == 0:
        return mask

    topk_current = beam_widths[current_step - 1]

    # Generated tokens from earlier steps (steps 0..current_step-2): causal
    gen_offset = 0
    for s in range(current_step - 1):
        bw_s = beam_widths[s]
        gen_start = max_history_seqlen + gen_offset
        for b in range(bw_s):
            pos = gen_start + b
            for b_idx in range(B):
                hl = history_seqlens[b_idx].item()
                mask[b_idx, pos, :hl] = True
                mask[b_idx, pos, max_history_seqlen : pos + 1] = True
        gen_offset += bw_s

    # Last step tokens (step current_step-1): precise tree mask
    gen_start = max_history_seqlen + gen_offset
    for b in range(topk_current):
        pos = gen_start + b
        for b_idx in range(B):
            hl = history_seqlens[b_idx].item()
            mask[b_idx, pos, :hl] = True  # history
            if ancestor_positions is not None:
                for anc_step in range(current_step):
                    anc_pos = ancestor_positions[b_idx, b, anc_step].item()
                    mask[b_idx, pos, anc_pos] = True
            else:
                mask[b_idx, pos, pos] = True  # self only

    return mask


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
