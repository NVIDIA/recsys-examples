from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
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
from modules.gpt_loss_module import GPTSIDLossModule
from torchrec.sparse.jagged_tensor import JaggedTensor


class SIDGRDecoder(MegatronModule):
    """
    Don't support PP currently. Does not include embedding
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

    def _concat_history_bos_candidate(
        self,
        history_hidden_states: torch.Tensor,  # sid
        candidate_hidden_states: torch.Tensor,  # sid
        cu_seqlens_history: torch.Tensor,  # sid
        cu_seqlens_candidate: torch.Tensor,  # sid
        cu_seqlens_bos: torch.Tensor,  # bos
        max_seqlen_history: int,
        max_seqlen_candidate: int,
        bos_token: torch.Tensor,
        batch_size: int,
        num_hierarchies: int,
    ) -> torch.Tensor:
        assert cu_seqlens_history.size(0) == cu_seqlens_candidate.size(
            0
        ), "history and candidate must have the same batch size"

        # we now only have one bos token.
        bos_token = bos_token.repeat(
            batch_size, 1
        ).contiguous()  # seqlens * num_hierarchies

        # [
        #    [s0_0,s1_0,s2_0], [bos], [c0_0, c1_0, c2_0],
        #    [s0_1,s1_1,s2_1; s0_2,s1_2,s2_2], [bos], [c0_1, c1_1, c2_1, c0_2, c1_2, c2_2],
        #    [s0_4,s1_4,s2_4], [bos], [c0_4, c1_4, c2_4], [c0_5, c1_5, c2_5],
        # ]
        max_seqlen_concat = max_seqlen_history + 1 + max_seqlen_candidate
        cated_hidden_states, cated_seqlens = jagged_2D_tensor_concat(
            [history_hidden_states, bos_token, candidate_hidden_states],
            [cu_seqlens_history, cu_seqlens_bos, cu_seqlens_candidate],
            [max_seqlen_history, 1, max_seqlen_candidate],
        )
        cated_offsets = length_to_complete_offsets(cated_seqlens)
        return cated_hidden_states, cated_offsets, max_seqlen_concat

    def forward(
        self,
        batch: GPTSIDBatch,
    ) -> torch.Tensor:
        history_feature_name = batch.history_feature_name
        candidate_feature_name = batch.candidate_feature_name
        # here assume the input sids are hierarchically properly biased.
        history_features = batch.features[history_feature_name]
        candidate_features = batch.features[candidate_feature_name]

        # 1. embedding lookuo
        embeddings: Dict[str, JaggedTensor] = self._codebooks_collection(batch.features)
        # TODO, remove the assertion
        assert all(
            feature_name in embeddings.keys() for feature_name in batch.features.keys()
        ), "history and candidate feature names must be in the embeddings"
        assert (
            self._num_hierarchies == batch._num_hierarchies
        ), "number of hierarchies must match"
        # total_seqlen = history_features.size(0) // self._num_hierarchies
        # candidate_seqlen = candidate_features.size(0) // self._num_hierarchies
        max_seqlen_history = batch.feature_to_max_seqlen[history_feature_name]
        max_seqlen_candidate = batch.feature_to_max_seqlen[candidate_feature_name]
        bos_offsets = torch.arange(
            0,
            batch.batch_size + 1,
            device=history_features.offsets().device,
            dtype=history_features.offsets().dtype,
        )
        history_offsets = history_features.offsets()
        candidate_offsets = candidate_features.offsets()
        # 2. preprocess:concat history, bos, candidate ( TODO: add position encoder )
        (
            input_hidden_states,
            input_offsets,
            input_max_seqlen,
        ) = self._concat_history_bos_candidate(
            embeddings[history_feature_name]
            .values()
            .to(
                self._training_dtype
            ),  # embedding output type is decoupeld, we need to cvt to training dtype.
            embeddings[candidate_feature_name]
            .values()
            .to(
                self._training_dtype
            ),  # embedding output type is decoupeld, we need to cvt to training dtype.
            history_offsets,
            candidate_offsets,
            bos_offsets,
            max_seqlen_history,  # note that this is already multiplied by num_hierarchies
            max_seqlen_candidate,  # note that this is already multiplied by num_hierarchies
            self.bos_token.to(
                self._training_dtype
            ),  # embedding output type is decoupeld, we need to cvt to training dtype.
            batch.batch_size,
            self._num_hierarchies,
        )
        packed_seq_params = to_packed_seq_params(
            input_offsets,
            input_max_seqlen,
        )
        # we need to unsqueeze the hidden states to [T, 1, hidden_size] and unsqueeze back after decoder
        input_hidden_states = input_hidden_states.unsqueeze(1)
        # 3. decoder (causal self-attention)
        output_hidden_states = self.decoder(
            hidden_states=input_hidden_states,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )
        output_hidden_states = output_hidden_states.squeeze(1)

        # 4. postprocess: split history, candidate, note that we append a bos token, so we need to remove the last token.
        # history are dropped.
        # TODO, replace with one-shottorch op
        _, output_hidden_states_bos_candidate = triton_split_2D_jagged(
            output_hidden_states,
            max_seq_len=input_max_seqlen,
            offsets_a=history_offsets,
            offsets_b=candidate_offsets + bos_offsets,
        )
        _, output_hidden_states = triton_split_2D_jagged(
            output_hidden_states_bos_candidate,
            max_seq_len=max_seqlen_candidate + 1,
            offsets_a=bos_offsets,
            offsets_b=candidate_offsets,
        )
        # the output shape should be [(sum(candidate_seqlen)) , num_hierarchies, hidden_size]
        candidate_hidden_states = output_hidden_states.reshape(
            -1, self._num_hierarchies, self.embedding_dim
        )
        losses_per_hierarchy = []
        logits_per_hierarchy = []
        merged_labels = batch.labels.view(-1, self._num_hierarchies)
        # 5. output linear projection & loss
        # TODO, merge into single linear layer
        # note that the labels
        for hierarchy_idx, mlp in enumerate[Any](self._decoder_mlp):
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
