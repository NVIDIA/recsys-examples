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
import itertools
import sys
from typing import List, Optional

sys.path.append("../commons/utils")
import flashinfer
import pytest
import torch
import torch.nn.functional as F
from configs import (
    InferenceHSTUConfig,
    KVCacheMetadata,
    copy_kvcache_metadata,
    get_kvcache_config,
    get_kvcache_metadata_buffer,
)
from hstu_assert_close import assert_hstu_close
from modules.hstu_block_inference import HSTUBlockInference
from modules.jagged_data import JaggedData
from test_paged_hstu_attn_kernel import _hstu_attention_maybe_from_cache


def get_jagged_metadata_buffer(max_batchsize, max_len_per_seq):
    int_dtype = torch.int32
    device = torch.cuda.current_device()

    default_num_candidates = max_len_per_seq // 2

    return JaggedData(
        values=None,
        # hidden states
        max_seqlen=max_len_per_seq,
        seqlen=torch.full(
            (max_batchsize,), max_len_per_seq, dtype=int_dtype, device=device
        ),
        seqlen_offsets=torch.arange(
            end=max_batchsize + 1, dtype=int_dtype, device=device
        )
        * max_len_per_seq,
        # candidates (included in hidden states)
        max_num_candidates=default_num_candidates,
        num_candidates=torch.full(
            (max_batchsize,), default_num_candidates, dtype=int_dtype, device=device
        ),
        num_candidates_offsets=torch.arange(
            end=max_batchsize + 1, dtype=int_dtype, device=device
        )
        * default_num_candidates,
        # contextual features
        contextual_max_seqlen=0,
        contextual_seqlen=None,
        contextual_seqlen_offsets=None,
        has_interleaved_action=True,
    )


def copy_jagged_metadata(dst_metadata, src_metata):
    bs = src_metata.seqlen.shape[0]
    dst_metadata.max_seqlen = src_metata.max_seqlen
    dst_metadata.seqlen[:bs].copy_(src_metata.seqlen[:bs], non_blocking=True)
    dst_metadata.seqlen_offsets[: bs + 1].copy_(
        src_metata.seqlen_offsets[: bs + 1], non_blocking=True
    )
    dst_metadata.max_num_candidates = src_metata.max_num_candidates
    dst_metadata.num_candidates[:bs].copy_(
        src_metata.num_candidates[:bs], non_blocking=True
    )
    dst_metadata.num_candidates_offsets[: bs + 1].copy_(
        src_metata.num_candidates_offsets[: bs + 1], non_blocking=True
    )


def get_offsets_from_lengths(lengths):
    offsets = torch.zeros(
        (lengths.shape[0] + 1,), dtype=lengths.dtype, device=lengths.device
    )
    torch.cumsum(lengths, 0, out=offsets[1:])
    return offsets


def setup_kvcache_testcase(
    kvcache_page_size: int,
    kvcache_num_heads: int,
    kvcache_head_dim: int,
    kvcache_dtype: torch.dtype,
    new_history_kv_data: torch.Tensor,  # [num_layers, 2, num_tokens, num_heads, head_dim]
    new_history_kv_lengths: torch.Tensor,
    kvcache_table: Optional[torch.Tensor] = None,
):
    batch_size = new_history_kv_lengths.shape[0]

    new_history_kv_length_offsets = get_offsets_from_lengths(new_history_kv_lengths)

    page_size = kvcache_page_size

    kv_page_indices = list()
    kv_page_indptr = list([0])
    kv_last_page_len = list()

    if kvcache_table is None:
        kvcache_table = torch.zeros(
            (2048, 2, page_size, kvcache_num_heads, kvcache_head_dim),
            dtype=kvcache_dtype,
            device=torch.cuda.current_device(),
        )

    acc_num_pages = 0
    for seq_idx in range(batch_size):
        new_history_length = new_history_kv_lengths[seq_idx].item()

        user_kv_data = new_history_kv_data[
            :,
            new_history_kv_length_offsets[seq_idx] : new_history_kv_length_offsets[
                seq_idx + 1
            ],
            ...,
        ]

        # Allocation
        num_pages = (new_history_length + page_size - 1) // page_size
        page_ids = list(range(acc_num_pages, acc_num_pages + num_pages))
        acc_num_pages += num_pages

        # Copy data
        last_page_size = new_history_length % page_size
        for page_idx in range(0, (new_history_length - last_page_size) // page_size):
            page_id = page_ids[page_idx]
            token_begin = page_idx * page_size
            token_end = (page_idx + 1) * page_size
            kvcache_table[page_id, ...].copy_(
                user_kv_data[:, token_begin:token_end, ...], non_blocking=True
            )
        if last_page_size > 0:
            page_idx = (new_history_length - last_page_size) // page_size
            page_id = page_ids[page_idx]
            token_begin = page_idx * page_size
            token_end = token_begin + last_page_size
            kvcache_table[page_id, ...] *= 0.0
            kvcache_table[page_id, :, :last_page_size, ...].copy_(
                user_kv_data[:, token_begin:token_end, ...], non_blocking=True
            )

        kv_page_indices.append(page_ids)
        kv_page_indptr.append(acc_num_pages)
        kv_last_page_len.append(last_page_size if last_page_size > 0 else page_size)

    torch.cuda.synchronize()

    return kvcache_table, (kv_page_indices, kv_page_indptr, kv_last_page_len)


def gather_kvdata(
    host_kv_data,
    host_kv_offsets,
    gpu_kv_data,
    gpu_kv_offsets,
    input_k,
    input_v,
    seqlen_offsets,
):
    num_heads = host_kv_data.shape[-2]
    head_dim = host_kv_data.shape[-1]

    host_k_data, host_v_data = torch.unbind(host_kv_data, dim=1)
    host_k_data = host_k_data.reshape(-1, num_heads * head_dim)
    host_v_data = host_v_data.reshape(-1, num_heads * head_dim)

    gpu_k_data, gpu_v_data = torch.unbind(gpu_kv_data, dim=0)
    gpu_k_data = gpu_k_data.view(-1, num_heads * head_dim)
    gpu_v_data = gpu_v_data.view(-1, num_heads * head_dim)

    batchsize = seqlen_offsets.shape[0] - 1
    k_con = torch.empty(
        (0, num_heads * head_dim), dtype=input_k.dtype, device=input_k.device
    )
    v_con = torch.empty(
        (0, num_heads * head_dim), dtype=input_v.dtype, device=input_v.device
    )
    for seq_idx in range(batchsize):
        k_con = torch.cat(
            [
                k_con,
                host_k_data[
                    host_kv_offsets[seq_idx] : host_kv_offsets[seq_idx + 1]
                ].cuda(),
                gpu_k_data[gpu_kv_offsets[seq_idx] : gpu_kv_offsets[seq_idx + 1]],
                input_k[seqlen_offsets[seq_idx] : seqlen_offsets[seq_idx + 1]],
            ],
            dim=0,
        ).contiguous()
        v_con = torch.cat(
            [
                v_con,
                host_v_data[
                    host_kv_offsets[seq_idx] : host_kv_offsets[seq_idx + 1]
                ].cuda(),
                gpu_v_data[gpu_kv_offsets[seq_idx] : gpu_kv_offsets[seq_idx + 1]],
                input_v[seqlen_offsets[seq_idx] : seqlen_offsets[seq_idx + 1]],
            ],
            dim=0,
        ).contiguous()

    kv_seqlen_offsets = host_kv_offsets + gpu_kv_offsets + seqlen_offsets.cpu()
    torch.cuda.synchronize()
    return k_con, v_con, kv_seqlen_offsets


def reference_hstu_layer(
    paged_hstu_layer,
    x: torch.Tensor,
    jd: JaggedData,
    host_kv_data: torch.Tensor,
    host_kv_offsets: torch.Tensor,
    gpu_kv_data: torch.Tensor,
    gpu_kv_offsets: torch.Tensor,
    use_fp32: bool,
):
    def get_causal_mask(kv_seqlen_offsets):
        batchsize = jd.seqlen.shape[0]
        num_heads = paged_hstu_layer._num_heads

        mask = torch.zeros(
            (batchsize, num_heads, 4096, 4096), dtype=x.dtype, device=x.device
        )
        for seq_idx in range(batchsize):
            qlen = jd.seqlen[seq_idx].item()
            klen = (
                kv_seqlen_offsets[seq_idx + 1].item()
                - kv_seqlen_offsets[seq_idx].item()
            )
            num_cand = jd.num_candidates[seq_idx].item()

            seq_mask = torch.cat(
                [
                    torch.tril(
                        torch.ones((qlen, klen - num_cand), dtype=torch.int32),
                        diagonal=klen - qlen,
                    ),
                    torch.cat(
                        [
                            torch.zeros((qlen - num_cand, num_cand), dtype=torch.int32),
                            torch.eye(num_cand, dtype=torch.int32),
                        ],
                        dim=0,
                    ),
                ],
                dim=1,
            )
            mask[seq_idx, :, :qlen, :klen].copy_(seq_mask.type(x.dtype))
        return mask

    def get_u_q_k_v(_mixed_uvqk):
        (_user, _value, _query, _key) = torch.split(
            _mixed_uvqk, paged_hstu_layer._split_arg_list, dim=-1
        )
        _user_c = _user.contiguous()
        _value_c = _value.view(
            -1, paged_hstu_layer._num_heads * paged_hstu_layer._linear_dim_per_head
        ).contiguous()
        _query_c = _query.view(
            -1, paged_hstu_layer._num_heads * paged_hstu_layer._attention_dim_per_head
        ).contiguous()
        _key_c = _key.view(
            -1, paged_hstu_layer._num_heads * paged_hstu_layer._attention_dim_per_head
        ).contiguous()
        return _user_c, _value_c, _query_c, _key_c

    normed_x = F.layer_norm(
        x,
        normalized_shape=[paged_hstu_layer._embedding_dim],
        weight=paged_hstu_layer._input_layernorm_weight,
        bias=paged_hstu_layer._input_layernorm_bias,
        eps=paged_hstu_layer._eps,
    )

    mixed_uvqk = F.silu(paged_hstu_layer._linear_uvqk(normed_x))
    (user, value, query, key) = get_u_q_k_v(mixed_uvqk)
    k_con, v_con, kv_seqlen_offsets = gather_kvdata(
        host_kv_data,
        host_kv_offsets,
        gpu_kv_data,
        gpu_kv_offsets,
        key,
        value,
        jd.seqlen_offsets,
    )
    # elevate to fp32 for higher precision
    attn_output = _hstu_attention_maybe_from_cache(
        num_heads=paged_hstu_layer._num_heads,
        attention_dim=paged_hstu_layer._attention_dim_per_head,
        linear_dim=paged_hstu_layer._linear_dim_per_head,
        seqlen_q=4096,
        seqlen_k=4096,
        q=query,
        k=k_con,
        v=v_con,
        q_offsets=jd.seqlen_offsets,
        k_offsets=kv_seqlen_offsets,
        rab=None,
        invalid_attn_mask=get_causal_mask(kv_seqlen_offsets),
        alpha=paged_hstu_layer._alpha,
        upcast=use_fp32,
        is_delta_q=False,
    )
    parallel_input = user * F.layer_norm(
        attn_output.to(user.dtype),
        normalized_shape=(attn_output.shape[-1],),
        weight=paged_hstu_layer._output_layernorm_weight,
        bias=paged_hstu_layer._output_layernorm_bias,
        eps=paged_hstu_layer._eps,
    )
    output = paged_hstu_layer._linear_proj(parallel_input)
    if paged_hstu_layer._residual:
        output = output + x
    return output


def generate_test_input_data(
    batchsize: int,
    max_seq_len: int,
    max_num_candidates: int,
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_gpu_kv_length: int,
    page_size: int,
    host_kv_chunksize: int,
    kvcache_table: List[torch.Tensor],
    kvcache_table_reserve_idx: int,
    dtype: torch.dtype,
):
    device = torch.cuda.current_device()

    max_new_history_len = max_seq_len - max_num_candidates
    new_history_lengths = torch.randint(
        low=max_new_history_len // 2,
        high=max_new_history_len,
        size=(batchsize,),
        dtype=torch.int32,
    )
    new_history_offsets = torch.cat(
        [torch.zeros((1,)), torch.cumsum(new_history_lengths, 0)], dim=0
    ).to(torch.int32)

    num_candidates = torch.randint(
        low=max_num_candidates // 2,
        high=max_num_candidates,
        size=(batchsize,),
        dtype=torch.int32,
    )
    num_candidates_offsets = torch.cat(
        [torch.zeros((1,)), torch.cumsum(num_candidates, 0)], dim=0
    ).to(torch.int32)

    seq_lengths = new_history_lengths + num_candidates
    seqlen_offsets = new_history_offsets + num_candidates_offsets

    input_num_tokens = seqlen_offsets[-1].item()
    input_hidden_states = torch.randn(
        (input_num_tokens, embedding_dim), dtype=dtype, device=device
    )

    assert host_kv_chunksize % page_size == 0

    # initialize old history KVs
    host_kv_lengths = (
        torch.randint(low=1, high=3, size=(batchsize,), dtype=torch.int32)
        * host_kv_chunksize
    )
    host_kv_offsets = torch.cat(
        [torch.zeros((1,)), torch.cumsum(host_kv_lengths, 0)], dim=0
    ).to(torch.int32)
    host_kv_len = host_kv_offsets[-1].item()
    host_kv_data = torch.empty(
        (num_layers, (host_kv_len // page_size), 2, page_size, num_heads, head_dim),
        dtype=dtype,
        pin_memory=True,
    ).uniform_(-0.1, 0.1)

    gpu_kv_lengths = torch.randint(low=0, high=max_gpu_kv_length, size=(batchsize,))
    gpu_kv_offsets = torch.cat(
        [torch.zeros((1,)), torch.cumsum(gpu_kv_lengths, 0)], dim=0
    ).to(torch.int32)
    gpu_kv_len = gpu_kv_offsets[-1].item()
    gpu_kv_data = torch.empty(
        (num_layers, 2, gpu_kv_len, num_heads, head_dim), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)

    # Setup gpu kvcache init state
    for layer_idx in range(num_layers):
        _, (kv_page_indices, kv_page_indptr, kv_last_page_len) = setup_kvcache_testcase(
            page_size,
            num_heads,
            head_dim,
            dtype,
            gpu_kv_data[layer_idx, ...],
            gpu_kv_lengths,
            kvcache_table[layer_idx],
        )

    # Allocate new history kvcache
    acc_num_page = 0
    acc_page_id = kv_page_indptr[-1]
    for seq_idx in range(batchsize):
        new_history_length = new_history_lengths[seq_idx].item()
        last_page_lens = kv_last_page_len[seq_idx]

        num_new_pages = (new_history_length + last_page_lens - 1) // page_size
        last_page_lens = (new_history_length + last_page_lens) % page_size
        kv_last_page_len[seq_idx] = page_size if last_page_lens == 0 else last_page_lens
        kv_page_indices[seq_idx].extend(
            list(range(acc_page_id, acc_page_id + num_new_pages))
        )
        acc_page_id += num_new_pages
        acc_num_page += num_new_pages
        kv_page_indptr[seq_idx + 1] += acc_num_page

    # Get mapped host kvdata metadata
    host_kv_page_num = (host_kv_lengths / page_size).to(torch.int32)
    host_kv_page_indptr = torch.cat(
        [torch.zeros((1,)), torch.cumsum(host_kv_page_num, 0)], dim=0
    ).to(torch.int32)
    host_kv_page_ids = [
        list(
            range(
                kvcache_table_reserve_idx + host_kv_page_indptr[seq_idx],
                kvcache_table_reserve_idx + host_kv_page_indptr[seq_idx + 1],
            )
        )
        for seq_idx in range(batchsize)
    ]
    kv_page_indices = list(
        itertools.chain(
            *[
                host_kv_page_ids[seq_idx] + kv_page_indices[seq_idx]
                for seq_idx in range(batchsize)
            ]
        )
    )
    kv_page_indptr = (
        host_kv_page_indptr + torch.tensor(kv_page_indptr, dtype=torch.int32)
    ).to(device=device)

    total_history_lengths = host_kv_lengths + gpu_kv_lengths + new_history_lengths
    batch_indices, position = flashinfer.page.get_batch_indices_positions(
        new_history_offsets.cuda(),
        total_history_lengths.cuda(),
        new_history_offsets[-1].item(),
    )
    total_history_offsets = get_offsets_from_lengths(total_history_lengths)

    jd = JaggedData(
        values=None,
        max_seqlen=max_seq_len,
        seqlen=seq_lengths.cuda(),
        seqlen_offsets=seqlen_offsets.cuda(),
        max_num_candidates=max_num_candidates,
        num_candidates=num_candidates.cuda(),
        num_candidates_offsets=num_candidates_offsets.cuda(),
        contextual_max_seqlen=0,
        contextual_seqlen=None,
        contextual_seqlen_offsets=None,
        has_interleaved_action=True,
    )

    kvcache_metadata = KVCacheMetadata(
        # paged cache metadata
        kv_indices=torch.tensor(kv_page_indices, dtype=torch.int32, device=device),
        kv_indptr=kv_page_indptr,
        kv_last_page_len=torch.tensor(
            kv_last_page_len, dtype=torch.int32, device=device
        ),
        total_history_lengths=None,
        total_history_offsets=total_history_offsets.cuda(),
        # appending metadata
        batch_indices=batch_indices,
        position=position,
        new_history_nnz=new_history_offsets[-1].item(),
        new_history_nnz_cuda=torch.tensor(
            [new_history_offsets[-1].item()], dtype=torch.int32, device=device
        ),
    )

    return (
        input_hidden_states,
        jd,
        kvcache_metadata,
        host_kv_lengths,
        host_kv_offsets,
        host_kv_data,
        gpu_kv_lengths,
        gpu_kv_offsets,
        gpu_kv_data,
    )


class TestModule:
    def __init__(self):
        hstu_config = InferenceHSTUConfig(
            hidden_size=1024,
            num_layers=4,
            num_heads=4,
            head_dim=128,
            bf16=True,
        )
        self.kvcache_config = get_kvcache_config(
            blocks_in_primary_pool=10240,
            page_size=32,
            offload_chunksize=1024,
            max_batch_size=16,
            max_seq_len=4096,
        )
        device = torch.cuda.current_device()

        self.embedding_dim = hstu_config.hidden_size
        self.num_layers = hstu_config.num_layers
        self.num_heads = hstu_config.num_heads
        self.head_dim = hstu_config.head_dim
        self.dtype = torch.bfloat16
        self.max_batchsize = self.kvcache_config.max_batch_size
        self.max_len_per_seq = self.kvcache_config.max_seq_len

        self.hstu_block_inference = HSTUBlockInference(
            hstu_config, kvcache_config=self.kvcache_config
        )
        self.hstu_block_inference.bfloat16()

        self.page_size = self.kvcache_config.page_size
        self.reserved_pages = (
            self.max_batchsize * self.max_len_per_seq // self.page_size
        )
        self.total_blocks = (
            self.kvcache_config.blocks_in_primary_pool + self.reserved_pages
        )
        self.kvcache_tables = [
            torch.empty(
                (self.total_blocks, 2, self.page_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=device,
            )
            for _ in range(self.num_layers)
        ]
        self.side_stream = torch.cuda.Stream()

        self.static_input_buffer = torch.randn(
            (self.max_batchsize * self.max_len_per_seq, self.embedding_dim),
            dtype=self.dtype,
            device=device,
        )
        self.static_jagged_metadata = get_jagged_metadata_buffer(
            self.max_batchsize, self.max_len_per_seq
        )
        self.static_kvcache_metadata = get_kvcache_metadata_buffer(
            hstu_config, self.kvcache_config
        )
        self.static_kvcache_metadata.onload_history_kv_buffer = [
            self.kvcache_tables[layer_idx][
                self.kvcache_config.blocks_in_primary_pool :, ...
            ]
            for layer_idx in range(self.num_layers)
        ]
        self.static_kvcache_metadata.onload_history_kv_events = [
            torch.cuda.Event() for _ in range(self.num_layers)
        ]
        self.static_kvcache_metadata.kv_cache_table = [
            self.kvcache_tables[layer_idx] for layer_idx in range(self.num_layers)
        ]

        self.hstu_block_inference.set_cudagraph(
            self.max_batchsize,
            self.max_len_per_seq,
            self.static_input_buffer,
            self.static_jagged_metadata,
            self.static_kvcache_metadata,
        )

    def onboard(
        self,
        host_kv_data: torch.Tensor,
        onload_num_pages: int,
        kvcache_metadata: KVCacheMetadata,
    ):
        if onload_num_pages == 0:
            return
        with torch.cuda.stream(self.side_stream):
            for layer_idx in range(self.num_layers):
                kvcache_metadata.onload_history_kv_buffer[layer_idx][
                    :onload_num_pages, ...
                ].copy_(
                    host_kv_data[layer_idx, :onload_num_pages, ...], non_blocking=True
                )
                kvcache_metadata.onload_history_kv_events[layer_idx].record(
                    self.side_stream
                )

    def get_paged_hstu_output_with_kvcache(
        self,
        batchsize: int,
        num_tokens: int,
        hidden_states: torch.Tensor,
        jd: JaggedData,
        kvcache_metadata: KVCacheMetadata,
        host_kv_data: torch.Tensor,
        use_cudagraph: bool,
    ):
        self.onboard(host_kv_data, host_kv_data.shape[1], kvcache_metadata)
        self.static_input_buffer[:num_tokens].copy_(hidden_states, non_blocking=True)

        output = self.hstu_block_inference.predict(
            batchsize,
            num_tokens,
            hidden_states,
            jd,
            kvcache_metadata,
            use_cudagraph=use_cudagraph,
        )
        return output

    def get_reference_hstu_output(
        self,
        hidden_states: torch.Tensor,
        jd: JaggedData,
        host_kv_data: torch.Tensor,
        host_kv_offsets: torch.Tensor,
        gpu_kv_data: torch.Tensor,
        gpu_kv_offsets: torch.Tensor,
        use_fp32: bool,
    ):
        hidden_data = hidden_states
        for layer_idx, hstu_layer in enumerate(
            self.hstu_block_inference._attention_layers
        ):
            hidden_data = reference_hstu_layer(
                hstu_layer,
                hidden_data,
                jd,
                host_kv_data[layer_idx],
                host_kv_offsets,
                gpu_kv_data[layer_idx],
                gpu_kv_offsets,
                use_fp32,
            )
        if use_fp32:
            return hidden_data.to(torch.float32)
        return hidden_data

    def run_test(
        self,
        batchsize,
        max_seqlen,
        max_num_candidates,
        max_gpu_kv_length,
        using_cudagraph,
    ):
        (
            input_hidden_states,
            jagged_metadata,
            kvcache_metadata,
            host_kv_lengths,
            host_kv_offsets,
            host_kv_data,
            gpu_kv_lengths,
            gpu_kv_offsets,
            gpu_kv_data,
        ) = generate_test_input_data(
            batchsize,
            max_seqlen,
            max_num_candidates,
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.head_dim,
            max_gpu_kv_length,
            self.page_size,
            self.kvcache_config.offload_chunksize,
            self.kvcache_tables,
            self.kvcache_config.blocks_in_primary_pool,
            self.dtype,
        )

        kvcache_metadata.total_history_offsets += jagged_metadata.num_candidates_offsets
        copy_jagged_metadata(self.static_jagged_metadata, jagged_metadata)
        copy_kvcache_metadata(self.static_kvcache_metadata, kvcache_metadata)
        output = self.get_paged_hstu_output_with_kvcache(
            batchsize,
            input_hidden_states.shape[0],
            input_hidden_states,
            self.static_jagged_metadata,
            self.static_kvcache_metadata,
            host_kv_data,
            use_cudagraph=True,
        )
        torch.cuda.synchronize()

        nograph_output = self.get_paged_hstu_output_with_kvcache(
            batchsize,
            input_hidden_states.shape[0],
            input_hidden_states,
            jagged_metadata,
            self.static_kvcache_metadata,
            host_kv_data,
            use_cudagraph=False,
        )
        torch.cuda.synchronize()

        ref_output = self.get_reference_hstu_output(
            input_hidden_states,
            jagged_metadata,
            host_kv_data,
            host_kv_offsets,
            gpu_kv_data,
            gpu_kv_offsets,
            use_fp32=False,
        )
        torch.cuda.synchronize()

        fp32_ref_output = self.get_reference_hstu_output(
            input_hidden_states,
            jagged_metadata,
            host_kv_data,
            host_kv_offsets,
            gpu_kv_data,
            gpu_kv_offsets,
            use_fp32=True,
        )
        torch.cuda.synchronize()

        assert_hstu_close(output, ref_output, fp32_ref_output, fwd=True)
        assert_hstu_close(nograph_output, ref_output, fp32_ref_output, fwd=True)


@pytest.mark.parametrize("using_cudagraph", [True])
def test_kvcache_onload_with_predict(using_cudagraph):
    batchsize_case = [1, 2, 4, 8]
    length_case = [
        # (max_seqlen, max_num_candidates)
        (128, 100),
        (256, 128),
        (512, 128),
        (1024, 256),
    ]
    max_gpu_kv_length_case = [512]

    test_cases = [batchsize_case, length_case, max_gpu_kv_length_case]

    with torch.inference_mode():
        # Use a module for tests to share cudagraph
        # Run all tests sequentially to avoid data racing
        test_module = TestModule()

        for (
            batchsize,
            (max_seqlen, max_num_candidates),
            max_gpu_kv_length,
        ) in itertools.product(*test_cases):
            test_module.run_test(
                batchsize,
                max_seqlen,
                max_num_candidates,
                max_gpu_kv_length,
                using_cudagraph,
            )
