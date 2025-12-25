from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from beam_search.beam_search import BeamSearch
from commons.modules.embedding import ShardedEmbedding, ShardedEmbeddingConfig
from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
from commons.ops.length_to_offsets import length_to_complete_offsets
from commons.ops.triton_ops.triton_jagged import triton_split_2D_jagged
from data.gpt_sid_batch import GPTSIDBatch, to_packed_seq_params
from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from modules.eval_metrics import SIDRetrievalEvaluator
from modules.gpt_loss_module import GPTSIDLossModule
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


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


def _get_padded_dense_attention_mask(
    input_offsets: torch.Tensor,
    input_max_seqlen: int,
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


def _create_multi_region_candidate_causal_mask(
    batchsize,
    history_seqlen: int,
    max_history_seqlen: int,
    num_candidate_region: int,
    candidate_max_seqlen_per_region: int,
    device: torch.device,
) -> torch.Tensor:
    """
    input sequence is : [history, candidate_region_0, candidate_region_1, ... padding_0, padding_1, ...],
                        where history length is history_seqlen, each candidate region length is candidate_max_seqlen_per_region,
                        and padding length is (max_history_seqlen - history_seqlen).
    intra region: causal ; inter region: invisible.
    each candidate needs to attend to the history

    return shape [batchsize, 1 , max_history_seqlen + candidate_max_seqlen_per_region * num_candidate_region, max_history_seqlen + candidate_max_seqlen_per_region * num_candidate_region ]
    """
    total_seqlen = (
        max_history_seqlen + num_candidate_region * candidate_max_seqlen_per_region
    )
    valid_seqlen = (
        history_seqlen + candidate_max_seqlen_per_region * num_candidate_region
    )
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

    # candidate to history (row >= max_history_seqlen AND col < max_history_seqlen)
    is_candidate_row = (row_indices >= history_seqlen) & (row_indices < valid_seqlen)
    candidate_attend_history = is_candidate_row & is_history_col

    # intra region: causal
    candidate_row_idx = row_indices - history_seqlen  # candidate region row index
    candidate_col_idx = col_indices - history_seqlen  # candidate region col index
    row_region_id = candidate_row_idx // candidate_max_seqlen_per_region
    row_offset = candidate_row_idx % candidate_max_seqlen_per_region

    col_region_id = candidate_col_idx // candidate_max_seqlen_per_region
    col_offset = candidate_col_idx % candidate_max_seqlen_per_region

    # intra candidate region: causal
    is_candidate_col = (col_indices >= history_seqlen) & (col_indices < valid_seqlen)
    same_region = row_region_id == col_region_id
    causal_within_region = row_offset >= col_offset
    candidate_internal_mask = (
        is_candidate_row & is_candidate_col & same_region & causal_within_region
    )

    mask = history_causal | candidate_attend_history | candidate_internal_mask

    # expand batch dimension: [batchsize, 1, total_seqlen, total_seqlen]
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batchsize, 1, -1, -1)

    # note that we return the inverse of the mask to match the attention mask format.
    return ~mask


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
    ) -> torch.Tensor:
        #
        output = self.decoder(
            hidden_states=hidden_states,  # query
            attention_mask=attention_mask,  # attention mask
            packed_seq_params=packed_seq_params,  # query and kv seqlens
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
    ):
        super(SIDGRModel, self).__init__(config=decoder_config)
        assert (
            position_embedding_type == "relative"
        ), "only relative position embedding is supported"
        # TODO, use different embedding dim???
        self.embedding_dim = decoder_config.hidden_size
        self.codebook_size = codebook_sizes[0]
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

        # output projection for the decoder to project the hidden state to the vocabulary space
        # TODO, combine into single linear layer!
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
        self.evaluator = SIDRetrievalEvaluator(eval_metrics)
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
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the decoder & mlp module.

        """
        self.decoder.half()
        self._decoder_mlp.half()
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
    ) -> torch.Tensor:
        assert (
            len(jagged_embeddings) == len(jagged_offsets) == len(jagged_max_seqlens)
        ), "all jagged tensors should have the same length"
        # [
        #    [s0_0,s1_0,s2_0], [bos], [c0_0, c1_0, c2_0],
        #    [s0_1,s1_1,s2_1; s0_2,s1_2,s2_2], [bos], [c0_1, c1_1, c2_1],
        #    [s0_4,s1_4,s2_4], [bos], [c0_4, c1_4, c2_4],
        # ]
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
        include_candidate: bool = True,  # False for generation
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        history_feature_name = batch.history_feature_name
        history_features = batch.features[history_feature_name]
        max_seqlen_history = batch.feature_to_max_seqlen[history_feature_name]
        history_offsets = history_features.offsets()

        # 1. embedding lookup
        embeddings: Dict[str, JaggedTensor] = self._codebooks_collection(batch.features)
        # TODO, remove the assertion
        assert all(
            feature_name in embeddings.keys() for feature_name in batch.features.keys()
        ), "all embedding feature names should be valid"
        # assert (
        #     self._num_hierarchies == batch._num_hierarchies
        # ), "number of hierarchies must match"
        bos_offsets = torch.arange(
            0,
            batch.batch_size + 1,
            device=history_offsets.device,
            dtype=history_offsets.dtype,
        )
        # 2. concat history, bos, optionally candidate ( TODO: add position encoder )
        bos_token = (
            self.bos_token.repeat(batch.batch_size, 1)
            .contiguous()
            .to(self._training_dtype)
        )  # seqlens * num_hierarchies
        jagged_embeddings: List[torch.Tensor] = [
            embeddings[history_feature_name].values().to(self._training_dtype),
            bos_token,
        ]
        jagged_offsets: List[torch.Tensor] = [history_offsets, bos_offsets]
        jagged_max_seqlens: List[int] = [max_seqlen_history, 1]
        if include_candidate:
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

        # TODO, considering removing the bos_offsets from the return value for better readability.
        return input_hidden_states, input_offsets, input_max_seqlen, bos_offsets

    def _postprocess_output(
        self,
        jagged_output_hidden_states: torch.Tensor,
        input_max_seqlen: int,
        history_offsets: torch.Tensor,
        candidate_offsets: torch.Tensor,
        bos_offsets: torch.Tensor,
        max_seqlen_candidate: int,
        output_hierarchies: int,
    ) -> torch.Tensor:
        # split history, candidate, note that we append a bos token,
        # history are dropped.
        # TODO, replace with one-shot op
        _, output_hidden_states_bos_candidate = triton_split_2D_jagged(
            jagged_output_hidden_states,
            max_seq_len=input_max_seqlen,
            offsets_a=history_offsets,
            offsets_b=candidate_offsets + bos_offsets,
        )
        # remove the last token.
        output_hidden_states, _ = triton_split_2D_jagged(
            output_hidden_states_bos_candidate,
            max_seq_len=max_seqlen_candidate + 1,
            offsets_a=candidate_offsets,
            offsets_b=bos_offsets,
        )
        # the output shape should be [(sum(candidate_seqlen)) , num_hierarchies, hidden_size]
        candidate_hidden_states = output_hidden_states.reshape(
            -1, output_hierarchies, self.embedding_dim
        )
        return candidate_hidden_states

    def decoder_step(
        self,
        input_hidden_states: torch.Tensor,
        input_offsets: torch.Tensor,
        input_max_seqlen: int,
        attention_mask: Optional[torch.Tensor] = None,
        padding_to_dense: bool = True,
    ) -> torch.Tensor:
        """
        Input and Output are both jagged.
        attention_mask is used only when padding_to_dense is True.
        When attention mask is None, we will construct a causal attention mask if padding_to_dense is True.
        """
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
                attention_mask = _get_padded_dense_attention_mask(
                    input_offsets,
                    input_max_seqlen,
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

        # causal self-attention
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
            bos_offsets,
        ) = self._prepare_embeddings(batch)
        history_offsets = batch.features[batch.history_feature_name].offsets()
        candidate_offsets = batch.features[batch.candidate_feature_name].offsets()
        max_seqlen_candidate = batch.feature_to_max_seqlen[batch.candidate_feature_name]

        # 2. decoder step
        jagged_output_hidden_states = self.decoder_step(
            input_hidden_states,
            input_offsets,
            input_max_seqlen,
            attention_mask=None,
        )
        # 3. postprocess: only keep the candidate hidden states
        candidate_hidden_states = self._postprocess_output(
            jagged_output_hidden_states,
            input_max_seqlen,
            history_offsets,
            candidate_offsets,
            bos_offsets,
            max_seqlen_candidate,
            batch._num_hierarchies,
        )
        losses_per_hierarchy = []
        logits_per_hierarchy = []
        merged_labels = batch.labels.view(-1, batch._num_hierarchies)

        # 4. output linear projection & loss
        # TODO, merge into single linear layer
        for hierarchy_idx, mlp in enumerate[Any](self._decoder_mlp):
            # TODO: remove this for debugging purpose
            if hierarchy_idx >= batch._num_hierarchies:
                break
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
        # 0. prepare history and bos embeddings.
        (
            history_embeddings,
            input_offsets,
            input_max_seqlen,
            _,
        ) = self._prepare_embeddings(batch, include_candidate=False)
        # TODO : fix the incomplete batch
        batch_size = input_offsets.size(0) - 1
        actual_history_seqlen = input_offsets[-1].item()
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
                    batch_size + 1,
                    device=input_offsets.device,
                    dtype=input_offsets.dtype,
                )

            # 2. prepare the attention mask
            attention_mask = _create_multi_region_candidate_causal_mask(
                batch_size,
                actual_history_seqlen,
                input_max_seqlen,
                0 if i == 0 else topk_prev_step,
                candidate_length,
                device=cated_hidden_states.device,
            )

            # 3. we need a decoder step with the concatenated hidden states and offsets
            jagged_output_hidden_states = self.decoder_step(
                cated_hidden_states,
                cated_offsets,
                cated_max_seqlen,
                attention_mask=attention_mask,
                padding_to_dense=True,
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
            tuple_or_tensor: Union[
                Tuple[torch.Tensor, torch.Tensor], torch.Tensor
            ] = self._decoder_mlp[i](candidate_hidden_states)
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
