from typing import Dict, List, Literal, Optional, Tuple

import torch
from commons.modules.embedding import ShardedEmbedding, ShardedEmbeddingConfig
from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
from commons.ops.length_to_complete_offsets import length_to_complete_offsets
from commons.ops.triton_ops.triton_jagged import triton_split_2D_jagged
from data.GPTBatch import to_packed_seq_params
from data.T5Batch import GPTSIDBatch
from megatron.core import MegatronModule, TransformerConfig, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_sequence_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
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
        max_sequence_length: int,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "relative"
        ] = "learned_absolute",
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=decoder_config, pg_collection=pg_collection)

        self.config: TransformerConfig = decoder_config

        self.transformer_decoder_layer_spec: ModuleSpec = transformer_decoder_layer_spec
        self.max_sequence_length = max_sequence_length
        # TODO, add position encoder
        self.model_type = ModelType.endoer_or_decoder
        self.decoder = TransformerBlock(
            config=self.decoder_config,
            spec=self.transformer_decoder_layer_spec,
            pg_collection=pg_collection,
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
        max_sequence_length: int,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "relative"
        ] = "learned_absolute",
        user_embedding_config: Optional[ShardedEmbeddingConfig] = None,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        should_add_sep_token: bool = True,
        pg_collection: ProcessGroupCollection = None,
    ):
        super(SIDGRModel, self).__init__(
            config=decoder_config, pg_collection=pg_collection
        )
        assert (
            position_embedding_type == "relative"
        ), "only relative position embedding is supported"
        # TODO, use different embedding dim???
        self.embedding_dim = decoder_config.hidden_size
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
            max_sequence_length,
            pg_collection=pg_collection,
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
                tensor_parallel.ColumnParallelLinear(
                    self.embedding_dim,
                    codebook_size,
                    bias=False,
                )
                for codebook_size in self.codebook_sizes
            ]
        )

        self.loss_module = GPTSIDLossModule(
            reduction="none",
        )

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
        batch_size: int,
        num_hierarchies: int,
    ) -> torch.Tensor:
        assert history_hidden_states.size(0) == candidate_hidden_states.size(
            0
        ), "history and candidate must have the same batch size"
        # we now only have one bos token.
        bos_token = self.bos_token.expand(
            batch_size
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
        embeddings: Dict[str, JaggedTensor] = self._embedding_collection(batch.features)
        # TODO, remove the assertion
        assert all(
            feature_name in embeddings.keys() for feature_name in batch.features.keys()
        ), "history and candidate feature names must be in the embeddings"
        assert (
            self._num_hierarchies == batch.num_hierarchies
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

        # 2. preprocess: concat history, bos, candidate ( TODO: add position encoder )
        (
            input_hidden_states,
            input_offsets,
            input_max_seqlen,
        ) = self._concat_history_bos_candidate(
            embeddings[history_feature_name].values(),
            embeddings[candidate_feature_name].values(),
            history_features.offsets(),
            candidate_features.offsets(),
            bos_offsets,
            max_seqlen_history,  # note that this is already multiplied by num_hierarchies
            max_seqlen_candidate,  # note that this is already multiplied by num_hierarchies
            batch.batch_size,
            self._num_hierarchies,
        )
        packed_seq_params = to_packed_seq_params(
            input_offsets,
            input_max_seqlen,
        )
        # 3. decoder (causal self-attention)
        output_hidden_states = self.decoder(
            hidden_states=input_hidden_states,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )

        # 4. postprocess: split history, candidate, note that we append a bos token, so we need to remove the last token.
        # history are dropped.
        _, output_hidden_states_bos_candidate = triton_split_2D_jagged(
            output_hidden_states,
            max_seq_len=input_max_seqlen,
            offsets_a=input_offsets - candidate_features - bos_offsets,
            offsets_b=candidate_features + bos_offsets,
            dense_size=self._num_hierarchies,
            n_prefix_to_right=1,
        )
        output_hidden_states_bos_candidate = output_hidden_states_bos_candidate.reshape(
            -1, self._num_hierarchies + 1, self.embedding_dim
        )
        # the output shape should be [sum(candidate_seqlen), num_hierarchies, hidden_size]
        candidate_hidden_states = output_hidden_states_bos_candidate[:, :-1, :]
        losses_per_hierarchy = []
        logits_per_hierarchy = []
        merged_labels = batch.labels.values().view(-1, self._num_hierarchies)

        # 5. output linear projection & loss
        # TODO, merge into single linear layer
        # note that the labels
        for hierarchy_idx, mlp in enumerate(self._decoder_mlp):
            candidate_hierarchy_logits = mlp(
                candidate_hidden_states[:, hierarchy_idx, :]
            )
            losses_per_hierarchy.append(
                self.loss_module(
                    candidate_hierarchy_logits, merged_labels[:, hierarchy_idx]
                )
            )
            logits_per_hierarchy.append(candidate_hierarchy_logits)

        # (T, num_hierarchies)
        merged_losses = torch.cat(losses_per_hierarchy, dim=1)
        merged_logits = torch.cat(logits_per_hierarchy, dim=1)
        return merged_losses, merged_logits


# class (MegatronModule):
# class
