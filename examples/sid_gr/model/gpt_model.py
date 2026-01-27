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
from commons.datasets.gpt_sid_batch import GPTSIDBatch, to_packed_seq_params
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
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from modules.eval_metrics import SIDRetrievalEvaluator
from modules.gpt_loss_module import GPTSIDLossModule
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

from .attention_mask import (
    padded_causal_mask_with_optional_bos,
    padded_target_aware_causal_mask,
)


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
    Don't support PP currently. Does not inclu de embedding
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
    ):
        super().__init__(config=decoder_config)

        self.config: TransformerConfig = decoder_config

        self.transformer_decoder_layer_spec: ModuleSpec = transformer_decoder_layer_spec
        # TODO, add position encoder
        self.model_type = ModelType.encoder_or_decoder
        self.position_embedding_type = position_embedding_type
        self.decoder_relative_pos_emb = RelativePositionEmbedding(
            bidirectional=False,
            init_method=self.config.init_method,
            num_attention_heads=self.config.num_attention_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
        )
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
        )

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # decoder attention mask, always causal
        *,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        attention_bias = None
        # if self.position_embedding_type == 'relative':
        #     # attention bias is supported by cudnn, but not fa.
        #     # TODO@junzhang add jagged support once we have attention kernels
        #     query_seq_length = input_max_seqlen
        #     key_seq_length = query_seq_length
        #     attention_bias = self.decoder_relative_pos_emb(query_seq_length, key_seq_length)
        output = self.decoder(
            hidden_states=hidden_states,  # query
            attention_mask=attention_mask,  # attention mask
            packed_seq_params=packed_seq_params,  # query and kv seqlens
            attention_bias=attention_bias,
            **kwargs,
        )
        return output


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
        padding_to_dense: bool = True,
        add_bos_to_history: bool = False,
    ) -> torch.Tensor:
        """
        Input and Output are both jagged.
        attention_mask is used only when padding_to_dense is True.
        When attention mask is None, we will construct a causal attention mask if padding_to_dense is True.

        We now only support dense input.
        """
        if add_bos_to_history:
            assert (
                attention_mask is None
            ), "attention mask should be None when adding bos to history"
        # TODO, remove the padding.
        input_offsets[-1].item()
        if padding_to_dense:
            decoder_input_hidden_states = _padding_to_dense_and_transpose(
                input_hidden_states,
                input_offsets,
                input_max_seqlen,
            )
            packed_seq_params = None
            if attention_mask is None:
                attention_mask = padded_causal_mask_with_optional_bos(
                    input_offsets,
                    input_max_seqlen,
                    add_bos_to_history=add_bos_to_history,
                    bos_interval=self._num_hierarchies,
                )
        else:
            # THD still needs batch dimension
            # we need to unsqueeze the hidden states to [T, 1, hidden_size] and unsqueeze back after decoder
            assert input_hidden_states.dim() == 2, "input_hidden_states should be 2D"
            decoder_input_hidden_states = input_hidden_states.unsqueeze(1)
            attention_mask = None
            packed_seq_params = to_packed_seq_params(
                input_offsets,
                input_max_seqlen,
            )
        decoder_output_hidden_states = self.decoder(
            hidden_states=decoder_input_hidden_states,  # input_hidden_states,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,  # we now enforce arbitrary attention mask + dense padding
        )

        if padding_to_dense:
            output_hidden_states = _transpose_dense_to_jagged(
                decoder_output_hidden_states,
                input_offsets,
                input_max_seqlen,
            )
        else:
            # remove batch dim if THD
            output_hidden_states = decoder_output_hidden_states.squeeze(1)
        return output_hidden_states

    def forward(
        self,
        batch: GPTSIDBatch,
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

        # 2. decoder step
        jagged_output_hidden_states = self.decoder_step(
            input_hidden_states,
            input_offsets,
            input_max_seqlen,
            attention_mask=None,
            add_bos_to_history=self.add_bos_to_history_for_training,
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
    def generate(self, batch: GPTSIDBatch) -> torch.Tensor:
        """
        Generate the output sids for the given batch. The generation will autogressively generate the output sids with a constrained fixed-width beam search strategy.
        Args:
          batch (GPTSIDBatch): The batch of data.
        Returns:
          torch.Tensor: The generated sids.
        """

        attention_mask: Optional[torch.Tensor] = None
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

            # 2. prepare the attention mask
            attention_mask = padded_target_aware_causal_mask(
                torch.diff(input_offsets),
                input_max_seqlen,
                0 if i == 0 else topk_prev_step,
                candidate_length,
            )
            # 3. we need a decoder step with the concatenated hidden states and offsets. Note that we do not add bos to history for generation.
            jagged_output_hidden_states = self.decoder_step(
                cated_hidden_states,
                cated_offsets,
                cated_max_seqlen,
                attention_mask=attention_mask,
                padding_to_dense=True,
                add_bos_to_history=False,
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
