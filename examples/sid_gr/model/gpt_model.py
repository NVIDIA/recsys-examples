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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from beam_search.beam_search import BeamSearch
from commons.datasets.gpt_sid_batch import GPTSIDBatch
from commons.modules.embedding import ShardedEmbedding, ShardedEmbeddingConfig
from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
from commons.ops.length_to_offsets import length_to_complete_offsets
from commons.ops.triton_ops.triton_jagged import triton_split_2D_jagged
from configs.gpt_config import BOSMode
from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.common.embeddings.relative_pos_embedding import (
    RelativePositionEmbedding,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from modules.eval_metrics import SIDRetrievalEvaluator
from modules.gpt_loss_module import GPTSIDLossModule
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

from .attention_mask import (
    build_jagged_causal_arbitrary_func,
    build_jagged_target_aware_arbitrary_func,
    dense_mask_to_jagged_arbitrary_func,
    padded_causal_mask_with_optional_bos,
    padded_target_aware_causal_mask,
)
from .jagged_flash_attn_block import JaggedTransformerBlock


def _padding_to_dense_and_transpose(
    jagged_input_hidden_states: torch.Tensor,
    input_offsets: torch.Tensor,
    input_max_seqlen: int,
) -> torch.Tensor:
    """
    Padding the jagged input hidden states to dense.
    input is Batch major, output is Sequence major.
    """
    batch_size = input_offsets.size(0) - 1
    assert (
        jagged_input_hidden_states.dim() == 2
    ), "jagged input hidden states should be 2D"

    padded_hidden_states = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=jagged_input_hidden_states,
            offsets=[input_offsets],
            max_lengths=[input_max_seqlen],
            padding_value=0.0,
        )
        .view(batch_size, input_max_seqlen, -1)
        .transpose(1, 0)
    )  # [S, B, D]
    return padded_hidden_states


def _transpose_dense_to_jagged(
    dense_hidden_states: torch.Tensor,
    input_offsets: torch.Tensor,
    input_max_seqlen: int,
) -> torch.Tensor:
    """
    Convert the dense hidden states to jagged.
    input is Sequence major, output is Batch major.
    """

    assert dense_hidden_states.dim() == 3, "dense hidden states should be 3D"
    jagged_hidden_states = torch.ops.fbgemm.dense_to_jagged(
        dense_hidden_states.transpose(1, 0),  # [S, B, D] -> [B, S, D]
        [input_offsets],
    )[0]
    return jagged_hidden_states


class SIDGRDecoder(MegatronModule):
    """
    Don't support PP currently. Does not include embedding.

    Supports two backend modes controlled by *use_jagged_flash_attn*:

    * **True** (default) — ``JaggedTransformerBlock``: flattens all batch
      sequences into one (B=1) and uses the arbitrary-mask FlashAttention
      path (``flash_attn.cute``) with ``arbitrary_func`` mask encoding.
      Zero padding.
    * **False** — Megatron-Core ``TransformerBlock``: pads jagged to dense,
      uses ``DotProductAttention`` with a dense arbitrary attention mask.
    """

    def __init__(
        self,
        decoder_config: TransformerConfig,  # decoder config
        transformer_decoder_layer_spec: ModuleSpec,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "relative"
        ] = "learned_absolute",
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        use_jagged_flash_attn: bool = False,
    ):
        super().__init__(config=decoder_config)

        self.config: TransformerConfig = decoder_config

        self.transformer_decoder_layer_spec: ModuleSpec = transformer_decoder_layer_spec
        self.model_type = ModelType.encoder_or_decoder
        self.position_embedding_type = position_embedding_type
        self.use_jagged_flash_attn = use_jagged_flash_attn
        self.decoder_relative_pos_emb = RelativePositionEmbedding(
            bidirectional=False,
            init_method=self.config.init_method,
            num_attention_heads=self.config.num_attention_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
        )

        if use_jagged_flash_attn:
            self.decoder = JaggedTransformerBlock(
                num_layers=self.config.num_layers,
                hidden_size=self.config.hidden_size,
                num_attention_heads=self.config.num_attention_heads,
                ffn_hidden_size=self.config.ffn_hidden_size,
                layernorm_epsilon=getattr(self.config, "layernorm_epsilon", 1e-5),
            )
        else:
            self.decoder = TransformerBlock(
                config=self.config,
                spec=self.transformer_decoder_layer_spec,
            )

    def get_jagged_flash_attn_block(self):
        """Return the inner JaggedFlashAttnBlock used by generate_beam_decode.

        Raises if this decoder was constructed without use_jagged_flash_attn.
        """
        if not self.use_jagged_flash_attn:
            raise RuntimeError(
                "get_jagged_flash_attn_block() requires use_jagged_flash_attn=True"
            )
        # self.decoder is a JaggedTransformerBlock wrapping a JaggedFlashAttnBlock
        return self.decoder.block

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        offsets: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        arbitrary_func: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.use_jagged_flash_attn:
            return self.decoder(
                hidden_states=hidden_states,
                arbitrary_func=arbitrary_func,
            )
        else:
            # mcore path: expects dense [S, B, D] input
            assert offsets is not None and max_seqlen is not None
            padded = _padding_to_dense_and_transpose(hidden_states, offsets, max_seqlen)
            output = self.decoder(
                hidden_states=padded,
                attention_mask=attention_mask,
            )
            return _transpose_dense_to_jagged(output, offsets, max_seqlen)


class SIDGRModel(MegatronModule):
    """
    Don't support PP currently.
    """

    def __init__(
        self,
        decoder_config: TransformerConfig,  # decoder config
        codebook_embedding_config: ShardedEmbeddingConfig,  # all codebooks share the same embedding
        codebook_sizes: List[int],
        num_hierarchies: int,
        transformer_decoder_layer_spec: ModuleSpec,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "relative"
        ] = "relative",
        user_embedding_config: Optional[ShardedEmbeddingConfig] = None,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        should_add_sep_token: bool = True,
        top_k_for_generation: int = 10,  # this is used for eval
        eval_metrics: Tuple[str, ...] = (),  # this is used for eval
        share_lm_head_across_hierarchies: bool = True,
        use_jagged_flash_attn: bool = False,
    ):
        super(SIDGRModel, self).__init__(config=decoder_config)
        assert (
            position_embedding_type == "relative"
        ), "only relative position embedding is supported"
        # TODO, use different embedding dim???
        self.embedding_dim = decoder_config.hidden_size
        self.codebook_size = codebook_sizes[0]
        self.add_bos_to_history_for_training = (
            decoder_config.bos_token_mode & BOSMode.HISTORY
        ) != 0

        assert all(
            size == self.codebook_size for size in codebook_sizes
        ), "all codebook sizes should be the same"
        self._num_hierarchies = num_hierarchies
        self._codebooks_collection = ShardedEmbedding(
            [codebook_embedding_config]
        )  # codebooks can be fused into single table
        self._user_embedding_collection = (
            ShardedEmbedding([user_embedding_config])
            if user_embedding_config is not None
            else None
        )  # user embedding can be fused into single table
        self.decoder = SIDGRDecoder(
            decoder_config,
            transformer_decoder_layer_spec,
            position_embedding_type="relative",
            use_jagged_flash_attn=use_jagged_flash_attn,
        )
        self.codebook_sizes = codebook_sizes
        assert codebook_embedding_config.vocab_size >= sum(
            codebook_sizes
        ), "codebook size should be greater than the sum of codebook sizes"
        assert (
            len(codebook_sizes) == num_hierarchies
        ), "number of codebook sizes should match the number of hierarchies"
        # bos_token used to prompt the decoder to generate the first token
        # this is duplicated across dp+cp+tp ranks. (DP+CP) be broadcasted, TP same seed.
        self.bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )
        # sep_token used to separate between different items
        self.sep_token = (
            torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
            if should_add_sep_token
            else None
        )

        self.share_lm_head_across_hierarchies = share_lm_head_across_hierarchies
        # output projection for the decoder to project the hidden state to the vocabulary space
        # TODO@junzhang, TEColumnParallelLinear does not support gather_output=True
        if not share_lm_head_across_hierarchies:
            # TODO, combine into single grouped linear layer!
            self._decoder_mlp = torch.nn.ModuleList(
                [
                    TEColumnParallelLinear(
                        input_size=self.embedding_dim,
                        output_size=codebook_size,
                        init_method=self.config.init_method,
                        config=self.config,
                        bias=False,
                        gather_output=False,
                        skip_bias_add=True,
                        is_expert=False,
                    )
                    for codebook_size in self.codebook_sizes
                ]
            )
        else:
            self._decoder_mlp = TEColumnParallelLinear(
                input_size=self.embedding_dim,
                output_size=sum(self.codebook_sizes),
                init_method=self.config.init_method,
                config=self.config,
                bias=False,
                gather_output=False,
                skip_bias_add=True,
                is_expert=False,
            )

        self.loss_module = GPTSIDLossModule(
            reduction="none",
        )

        self._training_dtype = (
            torch.float16
            if decoder_config.fp16
            else (torch.bfloat16 if decoder_config.bf16 else torch.float32)
        )
        for metric_spec in eval_metrics:
            metric_name, top_k = metric_spec.split("@")
            assert metric_name.lower() in [
                "ndcg",
                "recall",
                "hitrate",
            ], "invalid metric name"
            assert (
                int(top_k) <= top_k_for_generation
            ), "top_k for evaluation should be less than top_k for generation"
        # below are used for eval
        self.top_k_for_generation = top_k_for_generation  # beam search width.

        # below comments are reserved for multiple evaluators and debugging purpose
        # _evaluators = {}
        # for i in range(1, num_hierarchies + 1):
        #   _evaluators[f"eval_hierarchy_{i}"] = SIDRetrievalEvaluator(eval_metrics, i)
        # self.evaluator = MultipleEvaluatorWrapper(_evaluators)

        self.evaluator = SIDRetrievalEvaluator(eval_metrics, num_hierarchies)
        self.beam_search = BeamSearch(
            beam_width=top_k_for_generation,
            num_hierarchies=num_hierarchies,
            codebook_sizes=codebook_sizes,
            record_history=True,  # for debugging purpose
        )

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the decoder & mlp module.

        """
        self.decoder.bfloat16()
        self._decoder_mlp.bfloat16()
        self.bos_token.data = self.bos_token.data.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the decoder & mlp module.

        """
        self.decoder.half()
        self._decoder_mlp.half()
        self.bos_token.data = self.bos_token.data.half()
        return self

    # TODO
    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return id_embeddings, attention_mask

    def _concat_jagged(
        self,
        jagged_embeddings: List[torch.Tensor],
        jagged_offsets: List[torch.Tensor],
        jagged_max_seqlens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        assert (
            len(jagged_embeddings) == len(jagged_offsets) == len(jagged_max_seqlens)
        ), "all jagged tensors should have the same length"
        if len(jagged_embeddings) == 1:
            return jagged_embeddings[0], jagged_offsets[0], jagged_max_seqlens[0]
        max_seqlen_concat = sum(jagged_max_seqlens)

        cated_hidden_states, cated_seqlens = jagged_2D_tensor_concat(
            jagged_embeddings,
            jagged_offsets,
            jagged_max_seqlens,
        )
        cated_offsets = length_to_complete_offsets(cated_seqlens)
        return cated_hidden_states, cated_offsets, max_seqlen_concat

    def _prepare_embeddings(
        self,
        batch: GPTSIDBatch,
        add_bos_to_history: bool = False,
        is_generation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        input has 3 possible cases:
          generation: [history,bos]
          loss on candidate:[history, bos, candidate]
          loss on history and candidate:[history_with_bos_interleaved, bos, candidate]; but note that candidate might be empty.
        """
        history_feature_name = batch.history_feature_name
        candidate_feature_name = batch.candidate_feature_name
        history_features = batch.features[history_feature_name]
        max_seqlen_history = batch.feature_to_max_seqlen[history_feature_name]
        max_seqlen_candidate = batch.feature_to_max_seqlen[candidate_feature_name]
        actual_batch_size = batch.actual_batch_size
        history_offsets = history_features.offsets()
        if is_generation:
            assert (
                not add_bos_to_history
            ), "No need to add bos to history for generation"
        # 1. embedding lookup
        embeddings: Dict[str, JaggedTensor] = self._codebooks_collection(batch.features)
        # TODO, remove the assertion
        assert all(
            feature_name in embeddings.keys() for feature_name in batch.features.keys()
        ), "all embedding feature names should be valid"

        history_embeddings = (
            embeddings[history_feature_name].values().to(self._training_dtype)
        )
        assert (
            self._num_hierarchies == batch._num_hierarchies
        ), "number of hierarchies must match"

        jagged_embeddings = []
        jagged_offsets = []
        jagged_max_seqlens = []
        # 2. if add_bos_to_history, we insert bos token after each item (except the last one)
        if add_bos_to_history:
            # each item is a tuple of sid, and we need to insert bos token after each item (except the last one).
            # [[item0, item1, item2, ...], [item3, item4, item5, ...], ...] ->
            # [[{item0| bos, item1| bos, item2|...| bos, itemN}], [{item3| bos, item4| bos, item5|...| bos, itemM}]]
            # we use cat to implement this.
            history_embeddings = history_embeddings.view(
                -1, self._num_hierarchies, self.embedding_dim
            )
            bos_token = (
                self.bos_token.view(1, 1, -1)
                .expand_as(history_embeddings)[:, :1, :]
                .to(self._training_dtype)
            )
            history_embeddings = torch.cat([history_embeddings, bos_token], dim=1).view(
                -1, self.embedding_dim
            )
            history_offsets = history_offsets // self._num_hierarchies + history_offsets
            max_seqlen_history = (
                max_seqlen_history + max_seqlen_history // self._num_hierarchies
            )
            # remove the last bos token of each sequence
            last_bos_offsets = torch.arange(
                history_offsets.size(0),
                device=history_offsets.device,
                dtype=history_offsets.dtype,
            ).clamp(max=batch.actual_batch_size)
            history_embeddings, _ = triton_split_2D_jagged(
                history_embeddings,
                max_seq_len=max_seqlen_history,
                offsets_a=history_offsets - last_bos_offsets,
                offsets_b=last_bos_offsets,
            )
            max_seqlen_history -= 1
            history_offsets -= last_bos_offsets
        jagged_embeddings.append(history_embeddings)
        jagged_offsets.append(history_offsets)
        jagged_max_seqlens.append(max_seqlen_history)
        if is_generation or max_seqlen_candidate > 0:
            # when is_generation, we need to append bos
            # when include_candidate, we need to append [bos, candidate]

            # [[item0, item1, item2, ...], [item3, item4, item5, ...], ...] ->
            # [[{item0, item1, item2, ..., itemN} | {bos}], [{item3, item4, item5, ..., itemM} | {bos}]]
            # the last bos of each sequence is retained for later decoding.
            # we use jagged concat
            # note that we use batch_size here instead of actual_batch_size intentionally
            candidate_bos_offsets = torch.arange(
                0,
                batch.batch_size + 1,
                device=history_offsets.device,
                dtype=history_offsets.dtype,
            ).clamp(max=actual_batch_size)
            bos_token = (
                self.bos_token.repeat(actual_batch_size, 1)
                .contiguous()
                .to(self._training_dtype)
            )  # seqlens * num_hierarchies
            jagged_embeddings.append(bos_token)
            jagged_offsets.append(candidate_bos_offsets)
            jagged_max_seqlens.append(1)

        # For generation, we skip this step
        # 3. append candidate.
        if not is_generation and max_seqlen_candidate > 0:
            # [[{item0| bos, item1| bos, item2|...| bos, itemN}, {bos, candidate0}], [{item3| bos, item4| bos, item5|...| bos, itemM}, {bos, candidate1}]]
            candidate_feature_name = batch.candidate_feature_name
            jagged_embeddings.append(
                embeddings[candidate_feature_name].values().to(self._training_dtype)
            )
            jagged_offsets.append(batch.features[candidate_feature_name].offsets())
            jagged_max_seqlens.append(
                batch.feature_to_max_seqlen[candidate_feature_name]
            )
        (
            input_hidden_states,
            input_offsets,
            input_max_seqlen,
        ) = self._concat_jagged(
            jagged_embeddings,
            jagged_offsets,
            jagged_max_seqlens,
        )

        return input_hidden_states, input_offsets, input_max_seqlen

    def _postprocess_output(
        self,
        jagged_output_hidden_states: torch.Tensor,
        input_max_seqlen: int,
        input_offsets: torch.Tensor,
        actual_batch_size: int,
        history_offsets: torch.Tensor,
        output_hierarchies: int,
        add_bos_to_history: bool = False,
    ) -> torch.Tensor:
        """
        input has 2 possible cases:
          loss on candidate:[history, bos, candidate]
          loss on history and candidate:[history_with_bos_interleaved, bos, candidate]

          but note that candidate might be empty.
        """
        # split history, candidate, note that we append a bos token,
        # history are dropped.
        # [[{item0| bos, item1| bos, item2|...| bos, itemN}, {bos, candidate0}], [{item3| bos, item4| bos, item5|...| bos, itemM}, {bos,candidate1}]] or
        # [[{item0,item1,item2... itemN}, {bos, candidate0}], [{item3, item4, item5... itemM}, {bos,candidate1}]]
        prefix_offsets_to_remove = (
            torch.arange(
                history_offsets.size(0),
                device=history_offsets.device,
                dtype=history_offsets.dtype,
            ).clamp(max=actual_batch_size)
            * self._num_hierarchies
            if add_bos_to_history
            else history_offsets
        )
        # [bos, s0,s1,s2(dropped), bos,s3,s4,s5(dropped), bos,s6,s7,s8(dropped), ... bos,c_n, c_n+1, c_n+2(dropped)]
        _, bos_and_candidate_hidden_states = triton_split_2D_jagged(
            jagged_output_hidden_states,
            max_seq_len=input_max_seqlen,
            offsets_a=prefix_offsets_to_remove,
            offsets_b=input_offsets - prefix_offsets_to_remove,
        )
        candidate_hidden_states = bos_and_candidate_hidden_states.view(
            -1, self._num_hierarchies + 1, self.embedding_dim
        )[:, :output_hierarchies, :]
        return candidate_hidden_states

    def decoder_step(
        self,
        input_hidden_states: torch.Tensor,
        input_offsets: torch.Tensor,
        input_max_seqlen: int,
        attention_mask: Optional[torch.Tensor] = None,
        arbitrary_func: Optional[torch.Tensor] = None,
        *,
        default_mask_add_bos_to_history: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Jagged in/out. Routes Megatron-Core vs jagged Flash Attention.

        If both ``attention_mask`` and ``arbitrary_func`` are omitted, builds
        the usual causal mask for the active backend (same behavior as before
        the explicit-mask refactor). Pass either tensor to override.

        * FA path:    ``arbitrary_func`` (B=1 flattened arbitrary mask).
        * mcore path: ``attention_mask`` (dense ``[B, 1, N, N]``).

        ``default_mask_add_bos_to_history`` controls optional-BOS layout for the
        built mcore mask: ``None`` uses ``self.add_bos_to_history_for_training``
        (training-style); ``generate`` passes ``False``.
        """
        if attention_mask is None and arbitrary_func is None:
            if self.decoder.use_jagged_flash_attn:
                total_tokens = int(input_offsets[-1].item())
                arbitrary_func = build_jagged_causal_arbitrary_func(
                    input_offsets, total_tokens
                )
            else:
                add_bos = (
                    self.add_bos_to_history_for_training
                    if default_mask_add_bos_to_history is None
                    else default_mask_add_bos_to_history
                )
                attention_mask = padded_causal_mask_with_optional_bos(
                    input_offsets,
                    input_max_seqlen,
                    add_bos_to_history=add_bos,
                    bos_interval=self._num_hierarchies,
                )

        if self.decoder.use_jagged_flash_attn:
            # If caller provided a dense mask but we're on the jagged FA
            # path, convert it to a flattened (B=1) arbitrary_func.
            if arbitrary_func is None:
                assert attention_mask is not None, (
                    "decoder_step: at least one of attention_mask / "
                    "arbitrary_func must be set"
                )
                total_tokens = int(input_offsets[-1].item())
                valid_mask = ~attention_mask
                arbitrary_func = dense_mask_to_jagged_arbitrary_func(
                    valid_mask, input_offsets, total_tokens
                )
            return self.decoder(
                hidden_states=input_hidden_states,
                arbitrary_func=arbitrary_func,
            )
        assert attention_mask is not None
        return self.decoder(
            hidden_states=input_hidden_states,
            attention_mask=attention_mask,
            offsets=input_offsets,
            max_seqlen=input_max_seqlen,
        )

    def forward(
        self,
        batch: GPTSIDBatch,
        attention_mask: Optional[torch.Tensor] = None,
        arbitrary_func: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. prepare embeddings: embedding lookup + history, bos and candidate concat
        (
            input_hidden_states,
            input_offsets,
            input_max_seqlen,
        ) = self._prepare_embeddings(
            batch,
            add_bos_to_history=self.add_bos_to_history_for_training,
            is_generation=False,
        )
        history_offsets = batch.features[batch.history_feature_name].offsets()

        jagged_output_hidden_states = self.decoder_step(
            input_hidden_states,
            input_offsets,
            input_max_seqlen,
            attention_mask=attention_mask,
            arbitrary_func=arbitrary_func,
        )
        # 3. postprocess: only keep the candidate hidden states
        candidate_hidden_states = self._postprocess_output(
            jagged_output_hidden_states,
            input_max_seqlen,
            input_offsets,
            batch.actual_batch_size,
            history_offsets,
            batch._num_hierarchies,
            add_bos_to_history=self.add_bos_to_history_for_training,
        )
        losses_per_hierarchy = []
        logits_per_hierarchy = []
        merged_labels = batch.labels.values().view(-1, batch._num_hierarchies)
        # 4. output linear projection & loss
        # TODO, merge into single grouped linear layer
        for hierarchy_idx in range(batch._num_hierarchies):
            # TODO: remove this for debugging purpose
            mlp = (
                self._decoder_mlp[hierarchy_idx]
                if not self.share_lm_head_across_hierarchies
                else self._decoder_mlp
            )
            tuple_or_tensor = mlp(candidate_hidden_states[:, hierarchy_idx, :])
            candidate_hierarchy_logits = (
                tuple_or_tensor[0]
                if isinstance(tuple_or_tensor, tuple)
                else tuple_or_tensor
            )
            losses_per_hierarchy.append(
                self.loss_module(
                    candidate_hierarchy_logits.float(), merged_labels[:, hierarchy_idx]
                )
            )  # loss needs to be float for
            logits_per_hierarchy.append(candidate_hierarchy_logits)
        # (T, num_hierarchies)
        merged_losses = torch.stack(losses_per_hierarchy, dim=1).view(-1)
        merged_logits = torch.stack(logits_per_hierarchy, dim=1).view(
            -1, self.codebook_size
        )
        return merged_losses, merged_logits

    @torch.no_grad
    def generate(
        self,
        batch: GPTSIDBatch,
        attention_mask: Optional[torch.Tensor] = None,
        arbitrary_func: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate the output sids for the given batch. The generation will autogressively generate the output sids with a constrained fixed-width beam search strategy.
        Args:
          batch (GPTSIDBatch): The batch of data.
        Returns:
          torch.Tensor: The generated sids.
        """

        # 0. prepare history and bos embeddings. Note that we do not append bos to history.
        (
            history_embeddings,
            input_offsets,
            input_max_seqlen,
        ) = self._prepare_embeddings(
            batch, add_bos_to_history=False, is_generation=True
        )
        batch_size = batch.actual_batch_size
        input_offsets = input_offsets[: batch_size + 1]
        topk_prev_step = 1
        self.beam_search.reset()
        for i in range(self._num_hierarchies):
            generated_sids = self.beam_search.get_sids()
            # 1. prepare embeddings: [concat history, generated sids]
            if generated_sids is not None:
                # topk might be not always equal to the beam width because we have validation check.
                batch_size, topk_prev_step, candidate_length = generated_sids.shape
                assert (
                    candidate_length == i
                ), "current step should match the hierarchy index"

                # we must append hist. This is the defect of torchrec. Considering using torch.nn.Embedding
                generated_sids_kjt = KeyedJaggedTensor.from_lengths_sync(
                    keys=[
                        batch.candidate_feature_name,
                        batch.history_feature_name,
                    ],
                    values=generated_sids.view(-1),
                    lengths=torch.cat(
                        [
                            torch.full(
                                (batch_size,),
                                topk_prev_step * candidate_length,
                                device=generated_sids.device,
                                dtype=torch.long,
                            ),
                            torch.zeros(
                                (batch_size,),
                                device=generated_sids.device,
                                dtype=torch.long,
                            ),
                        ]
                    ),
                )
                generated_embeddings = (
                    self._codebooks_collection(generated_sids_kjt)[
                        batch.candidate_feature_name
                    ]
                    .values()
                    .to(self._training_dtype)
                )
                candidate_offsets = generated_sids_kjt[
                    batch.candidate_feature_name
                ].offsets()
                # Jagged concat!
                (
                    cated_hidden_states,
                    cated_offsets,
                    cated_max_seqlen,
                ) = self._concat_jagged(
                    [history_embeddings, generated_embeddings],
                    [input_offsets, candidate_offsets],
                    [input_max_seqlen, topk_prev_step * candidate_length],
                )
            else:
                # when we are at the first step, we do not have any generated sids and only bos token appended to the input.
                candidate_length = 0
                cated_hidden_states = history_embeddings
                cated_offsets = input_offsets
                cated_max_seqlen = input_max_seqlen

                # for first step, a single bos token for each sequence
                candidate_offsets = torch.arange(
                    0,
                    batch.actual_batch_size + 1,
                    device=input_offsets.device,
                    dtype=input_offsets.dtype,
                )

            # 2. Build the beam-isolating attention mask for this step.
            # Each beam is its own "target region" of length candidate_length
            # so beams don't see each other's tokens. At step 0 there are no
            # generated codes yet (just history+BOS), so num_target_region=0.
            # If the caller passed an explicit mask/arbitrary_func via the
            # generate() args, honour it; otherwise build the proper one.
            step_attention_mask = attention_mask
            step_arbitrary_func = arbitrary_func
            if step_attention_mask is None and step_arbitrary_func is None:
                if self.decoder.use_jagged_flash_attn:
                    # Direct vectorised build of the flattened arbitrary_func.
                    # Avoids materialising the dense [B, 1, N, N] mask and
                    # then converting it via a Python loop, which is two
                    # orders of magnitude slower for typical seqlens.
                    total_tokens = int(cated_offsets[-1].item())
                    history_seqlen = torch.diff(input_offsets)
                    # The flattened sequence layout is [hist+BOS, beam0, ...]
                    # so the per-sample "history" length here includes BOS,
                    # which is exactly torch.diff(input_offsets).
                    step_arbitrary_func = build_jagged_target_aware_arbitrary_func(
                        history_seqlen=history_seqlen,
                        num_target_region=(0 if i == 0 else topk_prev_step),
                        target_max_seqlen_per_region=candidate_length,
                        offsets=cated_offsets,
                        total_tokens=total_tokens,
                    )
                else:
                    step_attention_mask = padded_target_aware_causal_mask(
                        torch.diff(input_offsets),
                        input_max_seqlen,
                        0 if i == 0 else topk_prev_step,
                        candidate_length,
                    )

            jagged_output_hidden_states = self.decoder_step(
                cated_hidden_states,
                cated_offsets,
                cated_max_seqlen,
                attention_mask=step_attention_mask,
                arbitrary_func=step_arbitrary_func,
                default_mask_add_bos_to_history=False,
            )
            # remove history[batchsize * topk_last_step * max(1,i), embedding_dim]
            _, candidate_hidden_states = triton_split_2D_jagged(
                jagged_output_hidden_states,
                max_seq_len=cated_max_seqlen,
                offsets_a=cated_offsets - candidate_offsets,
                offsets_b=candidate_offsets,
            )
            # 4. calculate the probs for the current step
            candidate_hidden_states = candidate_hidden_states.view(
                batch_size, topk_prev_step, -1, self.embedding_dim
            )[:, :, -1, :]
            mlp = (
                self._decoder_mlp[i]
                if not self.share_lm_head_across_hierarchies
                else self._decoder_mlp
            )
            tuple_or_tensor: Union[
                Tuple[torch.Tensor, torch.Tensor], torch.Tensor
            ] = mlp(candidate_hidden_states)
            # [batch_size, topk_last_step, current_codebook_size]
            candidates_logits = (
                tuple_or_tensor[0]
                if isinstance(tuple_or_tensor, tuple)
                else tuple_or_tensor
            )
            probs_this_step: torch.Tensor = torch.nn.functional.log_softmax(
                candidates_logits.float(), dim=-1
            )
            # 5. filter the topk candidates, update the generated_sids and log_probs for the next step
            self.beam_search.propagate(probs_this_step)
        # only for debugging purpose
        generated_sids = self.beam_search.get_sids()
        log_probs = self.beam_search.get_log_probs()
        return generated_sids, log_probs

    @torch.no_grad
    def generate_beam_decode(
        self,
        batch: GPTSIDBatch,
        backend: str = "3kernel",
        phase_times: Optional[Dict[str, float]] = None,
        use_jagged_kv: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate using beam_decode_attn kernel with KV cache.

        Architecture:
          1. Prefill: run [history + BOS] through JaggedFlashAttnBlock,
             cache per-layer context K/V.
          2. Decode loop: for each hierarchy step, embed the newly selected
             code, run through layers using beam_decode_attn (context KV
             shared, beam KV per-beam via topk_indices), then MLP → logits
             → beam_search.propagate.

        The beam_decode_attn kernel separates context KV (shared across all
        beams, from prefill) and beam KV (per-beam, accumulated from decode
        steps), using topk_indices to encode the tree-structured ancestor
        chain. This avoids re-processing the full sequence at each step.

        Backend choice (passed to ``beam_decode_attn``):
          The kernel exposes two implementations of the same math:

          - ``"3kernel"`` (default): pipelined implementation. Supports
            ``seqused_k``, so it correctly masks padded K positions when
            the batch contains samples with different history lengths.
            Safe for arbitrary batches.
          - ``"dsl"``: fused implementation. ~1.5x faster than 3kernel
            on SM80 / SM90 / SM120 at typical SID-GR shapes (measured
            2026-05 on H100 NVL: ~0.111 ms vs ~0.164 ms per call). The
            fused path silently ignores ``seqused_k``, so it can only
            be used when every sample in the batch has the same history
            length. We check this at runtime and raise if you pick
            ``"dsl"`` with a non-uniform batch.

          On SM100 (B200) the kernel auto-routes to the pipelined
          implementation regardless of the ``backend`` arg, because the
          fused path is slower there.

          Note: end-to-end this only saves ~0.15 ms on a ~6.7 ms decode
          loop (~3%); dense projections dominate. Default stays
          ``"3kernel"`` because it works for all batch shapes.

        Args:
            batch: input batch.
            backend: ``"3kernel"`` (default) or ``"dsl"``. See "Backend
                choice" above. Pass ``"dsl"`` only when all samples in
                the batch share the same history length.
            phase_times: optional dict; if given, populated with measured
                phase latencies (in milliseconds) using cuda events:
                  - "prefill_ms": time spent in prefill (including embedding).
                  - "decode_loop_ms": time spent in the decode loop.
                Useful for benchmarking. Adds tiny overhead.
            use_jagged_kv: default ``True``. Prefill runs as a packed
                ``[total_tokens, D]`` stream through FA's
                ``flash_attn_varlen_func`` (``causal=True`` +
                ``cu_seqlens``) and the resulting jagged
                ``[total_k, H, D]`` K/V caches are fed to the kernel via
                ``cu_seqlens_k``. No padding compute anywhere.
                ``False`` is a legacy fallback that pads history to
                ``[B, max_seqlen, D]`` and uses ``seqused_k`` to mask
                pad positions; measured 17 – 48 % slower at ``B=16``
                across ``hist ∈ {256, 1024, 2048}`` and kept only for
                callers that already produce padded inputs.
                Requires ``backend="3kernel"``.
        """
        # Backend whitelist: the kernel's interface silently treats any
        # non-"3kernel" string as the fused/dsl path, so reject typos
        # here rather than letting them slip through.
        if backend not in ("3kernel", "dsl"):
            raise ValueError(f"backend must be '3kernel' or 'dsl', got {backend!r}")

        # use_jagged_kv requires the pipelined backend. The fused path
        # doesn't consume cu_seqlens_k; silently routing through dense
        # would discard the caller's choice.
        if use_jagged_kv and backend != "3kernel":
            raise ValueError(
                f"use_jagged_kv=True requires backend='3kernel'; got "
                f"backend={backend!r}. The fused/dsl path does not accept "
                f"cu_seqlens_k."
            )

        # generate_beam_decode uses JaggedFlashAttnBlock for prefill and
        # decode; SIDGRDecoder must have been constructed with
        # use_jagged_flash_attn=True. Fail at entry rather than after
        # _prepare_embeddings does work that has to be thrown away.
        if not self.decoder.use_jagged_flash_attn:
            raise RuntimeError(
                "generate_beam_decode() requires use_jagged_flash_attn=True "
                "because it uses JaggedFlashAttnBlock for prefill and beam "
                "decode."
            )

        # Kernel-side preconditions on BeamSearch configuration.
        # beam_decode_attn asserts uniform beam widths via
        # k_beam.shape[1] == decode_nums * beam_width, and we accumulate
        # beam_kv with a fixed stride per step. Reject non-uniform here.
        beam_widths = self.beam_search.beam_widths
        if not all(w == beam_widths[0] for w in beam_widths):
            raise ValueError(
                f"generate_beam_decode requires uniform beam_widths across "
                f"hierarchy steps for the beam_decode_attn kernel; got "
                f"{beam_widths}"
            )
        # propagate() clamps actual topk to
        # min(beam_widths[s], topk_previous_step * codebook_size_this_step).
        # If any codebook_size < beam_width, propagate would shrink topk,
        # making k_beam.shape[1] inconsistent with decode_nums * beam_width.
        bw = beam_widths[0]
        cs_min = min(self.codebook_sizes)
        if bw > cs_min:
            raise ValueError(
                f"generate_beam_decode requires beam_width ({bw}) <= "
                f"min(codebook_sizes) ({cs_min}); otherwise propagate would "
                f"shrink the actual topk and the kernel's "
                f"k_beam.shape[1] == decode_nums * beam_width assertion fails"
            )

        # Capability probe for use_jagged_kv=True: fail at entry rather
        # than deep in decode_beam. Catches two cases:
        #   (a) installed kernel lacks the cu_seqlens_k kwarg (upstream).
        #   (b) resolver fell back to _beam_decode_attn_reference, whose
        #       signature includes cu_seqlens_k but only to raise
        #       NotImplementedError on use.
        if use_jagged_kv:
            import inspect

            from .jagged_flash_attn_block import (
                _beam_decode_attn_reference,
                _get_beam_decode_attn,
            )

            kernel = _get_beam_decode_attn()
            if kernel is _beam_decode_attn_reference:
                raise RuntimeError(
                    "use_jagged_kv=True requires the real CuTe "
                    "beam_decode_attn kernel; the PyTorch reference "
                    "fallback does not implement jagged context K/V "
                    "(it only raises NotImplementedError when actually "
                    "called). Ensure corelib/gr_decode_atten is on "
                    "PYTHONPATH (Docker image sets this automatically), "
                    "or use use_jagged_kv=False."
                )
            try:
                _kernel_sig = inspect.signature(kernel)
            except (TypeError, ValueError) as exc:
                # Fail closed: an uninspectable callable (e.g. a C
                # extension) cannot be proven to accept cu_seqlens_k.
                # Better an explicit entry error than a deep
                # "unexpected keyword argument" surprise mid-decode.
                raise RuntimeError(
                    "use_jagged_kv=True requires a beam_decode_attn "
                    "callable whose signature can be inspected for "
                    "cu_seqlens_k support. The installed kernel's "
                    "signature is not inspectable, so cu_seqlens_k "
                    "support cannot be verified. Use use_jagged_kv=False "
                    "or verify corelib/gr_decode_atten is on PYTHONPATH."
                ) from exc
            if "cu_seqlens_k" not in _kernel_sig.parameters:
                raise RuntimeError(
                    "use_jagged_kv=True requires a beam_decode_attn "
                    "entry point that accepts cu_seqlens_k. The vendored "
                    "corelib/gr_decode_atten kernel includes it; the "
                    "currently-resolved kernel does not. Check "
                    "PYTHONPATH, or use use_jagged_kv=False."
                )

        # Optional phase timing via cuda events
        record_times = phase_times is not None
        if record_times:
            ev_t0 = torch.cuda.Event(enable_timing=True)
            ev_t1 = torch.cuda.Event(enable_timing=True)
            ev_t2 = torch.cuda.Event(enable_timing=True)
            ev_t0.record()

        # 0. Prepare history + BOS embeddings
        (
            history_embeddings,
            input_offsets,
            input_max_seqlen,
        ) = self._prepare_embeddings(
            batch, add_bos_to_history=False, is_generation=True
        )
        batch_size = batch.actual_batch_size
        input_offsets = input_offsets[: batch_size + 1]

        # Access the inner JaggedFlashAttnBlock through the decoder helper.
        fa_block = self.decoder.get_jagged_flash_attn_block()

        # 1. Prefill — produces per-layer context K/V caches in either
        # jagged or dense layout, depending on use_jagged_kv.
        history_seqlens = torch.diff(input_offsets)  # [B]
        seqused_k = None
        cu_seqlens_k = None

        # backend="dsl" silently ignores both seqused_k and cu_seqlens_k —
        # so it can only handle uniform-length history. Validate that
        # before paying any prefill cost (host sync on .all().item() is
        # cheap; the actual prefill is 2-3 ms).
        if backend != "3kernel":
            if not bool((history_seqlens == history_seqlens[0]).all().item()):
                raise ValueError(
                    f"backend={backend!r} (fused/dsl path) ignores "
                    f"seqused_k / cu_seqlens_k, so all samples must share "
                    f"the same history length. Got history_seqlens="
                    f"{history_seqlens.tolist()}. Use backend='3kernel' "
                    f"or pre-pad/crop history."
                )

        if backend == "3kernel" and use_jagged_kv:
            # Jagged-native prefill: feed the concatenated [total_tokens, D]
            # stream through FA's `flash_attn_varlen_func` fast path. The
            # K/V caches that fall out are already jagged [total_tokens, H, D]
            # — exactly the layout the kernel expects when cu_seqlens_k is
            # supplied. We hand cu_seqlens directly to FA (varlen + causal)
            # instead of going through arbitrary_func + block-sparsity, which
            # is ~6× slower at hist=2048 for the same semantics.
            total_tokens = int(input_offsets[-1].item())
            flat_history = history_embeddings.unsqueeze(0).to(
                self._training_dtype
            )  # [1, total_tokens, D]
            prefill_output, context_kv_caches = fa_block.prefill(
                flat_history,
                cu_seqlens=input_offsets.to(torch.int32),
                seqlen=total_tokens,
            )
            prefill_output = prefill_output.squeeze(0)  # [total_tokens, D]
            context_kv_caches = [
                (k.squeeze(0), v.squeeze(0)) for k, v in context_kv_caches
            ]
            cu_seqlens_k = input_offsets.to(torch.int32)
            # BOS hidden = last valid position per sample, in flat layout
            bos_offsets_flat = (input_offsets[1:] - 1).clamp(min=0)
            bos_hidden = prefill_output[bos_offsets_flat]  # [B, D]
        else:
            # Dense prefill: pad history, run [B, max_seqlen, D] through
            # FA's causal=True fast path.
            padded_history = (
                torch.ops.fbgemm.jagged_to_padded_dense(
                    values=history_embeddings,
                    offsets=[input_offsets],
                    max_lengths=[input_max_seqlen],
                    padding_value=0.0,
                )
                .view(batch_size, input_max_seqlen, -1)
                .to(self._training_dtype)
            )
            prefill_output, context_kv_caches = fa_block.prefill(
                padded_history, arbitrary_func=None, seqlen=input_max_seqlen
            )
            bos_positions = (history_seqlens - 1).clamp(min=0)  # [B]
            bos_hidden = prefill_output[
                torch.arange(batch_size, device=prefill_output.device),
                bos_positions,
            ]  # [B, D]
            if backend == "3kernel":
                seqused_k = history_seqlens.to(torch.int32)
            # backend="dsl" already validated above (uniform history).

        # MLP → logits for step 0
        self.beam_search.reset()
        bos_hidden = bos_hidden.unsqueeze(1)  # [B, 1, D]
        mlp = (
            self._decoder_mlp[0]
            if not self.share_lm_head_across_hierarchies
            else self._decoder_mlp
        )
        tuple_or_tensor: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor] = mlp(
            bos_hidden
        )
        candidates_logits = (
            tuple_or_tensor[0]
            if isinstance(tuple_or_tensor, tuple)
            else tuple_or_tensor
        )
        probs_step0: torch.Tensor = torch.nn.functional.log_softmax(
            candidates_logits.float(), dim=-1
        )
        self.beam_search.propagate(probs_step0)

        if record_times:
            ev_t1.record()

        # 2. Decode loop for remaining hierarchy steps
        num_heads = fa_block.layers[0].num_heads
        num_layers = len(fa_block.layers)

        # Per-layer accumulated beam KV: None initially
        beam_kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * num_layers

        for d in range(self._num_hierarchies - 1):
            # d is the decode step (0-indexed); the codes we embed here
            # were chosen by propagate(step=d).
            hierarchy_step = d + 1

            # Use the actual top-k count from beam_search (may be less than
            # the configured beam_width when topk is bounded by
            # topk_previous_step * codebook_size).
            step_codes = self.beam_search.generated_sids[:, :, d]  # [B, current_topk]
            current_topk = step_codes.shape[1]
            step_codes_flat = step_codes.reshape(-1)

            codes_kjt = KeyedJaggedTensor.from_lengths_sync(
                keys=[
                    batch.candidate_feature_name,
                    batch.history_feature_name,
                ],
                values=step_codes_flat,
                lengths=torch.cat(
                    [
                        torch.full(
                            (batch_size,),
                            current_topk,
                            device=step_codes_flat.device,
                            dtype=torch.long,
                        ),
                        torch.zeros(
                            (batch_size,),
                            device=step_codes_flat.device,
                            dtype=torch.long,
                        ),
                    ]
                ),
            )
            code_embeddings = (
                self._codebooks_collection(codes_kjt)[batch.candidate_feature_name]
                .values()
                .to(self._training_dtype)
            )  # [B * current_topk, D]
            code_embeddings = code_embeddings.view(
                batch_size, current_topk, self.embedding_dim
            )

            # Build topk_indices for this decode step
            decode_nums = d + 1  # include self
            topk_indices = self.beam_search.build_beam_topk_indices(
                decode_step=d,
                num_heads=num_heads,
            )

            # Decode through all layers
            hidden_states, new_beam_kvs = fa_block.decode_beam(
                code_embeddings,
                context_kv_caches,
                beam_kv_caches,
                topk_indices,
                decode_nums,
                seqused_k=seqused_k,
                cu_seqlens_k=cu_seqlens_k,
                backend=backend,
            )  # hidden_states: [B, W, D]

            # Accumulate beam KV for next step
            for l in range(num_layers):
                k_new, v_new = new_beam_kvs[l]
                cache = beam_kv_caches[l]
                if cache is not None:
                    k_prev, v_prev = cache
                    beam_kv_caches[l] = (
                        torch.cat([k_prev, k_new], dim=1),
                        torch.cat([v_prev, v_new], dim=1),
                    )
                else:
                    beam_kv_caches[l] = (k_new, v_new)

            # MLP → logits → beam_search.propagate
            mlp = (
                self._decoder_mlp[hierarchy_step]
                if not self.share_lm_head_across_hierarchies
                else self._decoder_mlp
            )
            tuple_or_tensor = mlp(hidden_states)
            candidates_logits = (
                tuple_or_tensor[0]
                if isinstance(tuple_or_tensor, tuple)
                else tuple_or_tensor
            )
            probs_this_step: torch.Tensor = torch.nn.functional.log_softmax(
                candidates_logits.float(), dim=-1
            )
            self.beam_search.propagate(probs_this_step)

        if phase_times is not None:
            ev_t2.record()
            torch.cuda.synchronize()
            phase_times["prefill_ms"] = ev_t0.elapsed_time(ev_t1)
            phase_times["decode_loop_ms"] = ev_t1.elapsed_time(ev_t2)

        generated_sids = self.beam_search.get_sids()
        log_probs = self.beam_search.get_log_probs()
        return generated_sids, log_probs
