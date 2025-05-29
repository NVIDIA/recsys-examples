# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Optional, Union

import torch
from commons.utils.nvtx_op import output_nvtx_hook
from configs.hstu_config import HSTUConfig, HSTULayerType
from dataset.utils import RankingBatch, RetrievalBatch
from megatron.core.transformer.module import MegatronModule
from modules.fused_hstu_layer import FusedHSTULayer
from modules.jagged_data import JaggedData
from modules.native_hstu_layer import HSTULayer
from modules.paged_hstu_infer_layer import PagedHSTUInferLayer
from modules.position_encoder import HSTUPositionalEncoder
from ops.jagged_tensor_op import concat_2D_jagged_tensors
from ops.length_to_offsets import length_to_complete_offsets
from ops.triton_ops.triton_jagged import (  # type: ignore[attr-defined]
    triton_concat_2D_jagged,
    triton_split_2D_jagged,
)
from torchrec.sparse.jagged_tensor import JaggedTensor


class HSTUBlock(MegatronModule):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        config: HSTUConfig,
        inference_mode: bool = False,
    ):
        super().__init__(config=config)
        self._training_dtype = torch.float32
        if self.config.bf16:
            self._training_dtype = torch.bfloat16
        if self.config.fp16:
            self._training_dtype = torch.float16

        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if config.position_encoding_config is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                num_position_buckets=config.position_encoding_config.num_position_buckets,
                num_time_buckets=config.position_encoding_config.num_time_buckets,
                embedding_dim=config.hidden_size,
                is_inference=False,
                use_time_encoding=config.position_encoding_config.use_time_encoding,
                training_dtype=self._training_dtype,
            )
        
        if not inference_mode:
            HSTULayerImpl = (
                FusedHSTULayer
                if config.hstu_layer_type == HSTULayerType.FUSED
                else HSTULayer
            )
            self._attention_layers = torch.nn.ModuleList(
                [HSTULayerImpl(config) for l in range(self.config.num_layers)]
            )
        else:
            self._attention_layers = torch.nn.ModuleList(
                [PagedHSTUInferLayer(config, layer_idx) for layer_idx in range(self.config.num_layers)]
            )

    @output_nvtx_hook(nvtx_tag="hstu_preprocess")
    def hstu_preprocess(
        self, embeddings: Dict[str, JaggedTensor], batch: RankingBatch
    ) -> JaggedData:
        """
        Preprocesses the embeddings for use in the HSTU architecture.

        This method performs the following steps:
        1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
        2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample, following the order specified in the batch.
        3. **Position Encoding**: Applies position encoding to the concatenated embeddings.

        Args:
            embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
            batch (RankingBatch): The batch of ranking data.

        Returns:
            JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
        """
        item_jt = embeddings[batch.item_feature_name]  # history + candidate
        sequence_embeddings = item_jt.values()
        sequence_embeddings_lengths = item_jt.lengths()
        sequence_embeddings_lengths_offsets = item_jt.offsets()
        sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

        if batch.action_feature_name is not None:
            action_jt = embeddings[batch.action_feature_name]
            jagged_size = sequence_embeddings.size(0)
            embedding_dim = sequence_embeddings.size(1)
            sequence_embeddings = torch.cat(
                [sequence_embeddings, action_jt.values()], dim=1
            ).view(2 * jagged_size, embedding_dim)
            sequence_embeddings_lengths = sequence_embeddings_lengths * 2
            sequence_embeddings_lengths_offsets = (
                sequence_embeddings_lengths_offsets * 2
            )
            sequence_max_seqlen = sequence_max_seqlen * 2

        if batch.num_candidates is not None and batch.action_feature_name is not None:
            num_candidates = batch.num_candidates * 2
            max_num_candidates = batch.max_num_candidates * 2
        else:
            num_candidates = batch.num_candidates
            max_num_candidates = batch.max_num_candidates

        contextual_max_seqlen = 0
        contextual_seqlen = None
        contextual_seqlen_offsets = None
        if len(batch.contextual_feature_names) > 0:
            contextual_max_seqlens = [
                batch.feature_to_max_seqlen[name]
                for name in batch.contextual_feature_names
            ]
            contextual_embedding, contextual_seqlen = concat_2D_jagged_tensors(
                jagged_tensors=[
                    embeddings[name] for name in batch.contextual_feature_names
                ],
                max_seqlens=contextual_max_seqlens,
            )

            contextual_max_seqlen = max(
                len(batch.contextual_feature_names), sum(contextual_max_seqlens)
            )
            contextual_seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                contextual_seqlen
            )

            sequence_embeddings = triton_concat_2D_jagged(
                max_seq_len=contextual_max_seqlen + sequence_max_seqlen,
                values_a=contextual_embedding,
                values_b=sequence_embeddings,
                offsets_a=contextual_seqlen_offsets,
                offsets_b=sequence_embeddings_lengths_offsets,
            )

            sequence_embeddings_lengths = (
                contextual_seqlen + sequence_embeddings_lengths
            )
            sequence_embeddings_lengths_offsets = (
                contextual_seqlen_offsets + sequence_embeddings_lengths_offsets
            )
            sequence_max_seqlen = sequence_max_seqlen + contextual_max_seqlen

        if self._positional_encoder is not None:
            sequence_embeddings = self._positional_encoder(
                max_seq_len=sequence_max_seqlen,
                seq_lengths=sequence_embeddings_lengths,
                seq_offsets=sequence_embeddings_lengths_offsets,
                seq_timestamps=None,
                seq_embeddings=sequence_embeddings,
                num_targets=num_candidates,
            )

        sequence_embeddings = torch.nn.functional.dropout(
            sequence_embeddings,
            p=0.2,
            training=self.training,
        )
        return JaggedData(
            values=sequence_embeddings.to(self._training_dtype),
            seqlen=sequence_embeddings_lengths.to(
                torch.int32
            ),  # contextual + history + candidate
            seqlen_offsets=sequence_embeddings_lengths_offsets.to(torch.int32),
            max_seqlen=sequence_max_seqlen,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates.to(torch.int32)
            if num_candidates is not None
            else None,
            num_candidates_offsets=length_to_complete_offsets(num_candidates).to(
                torch.int32
            )
            if num_candidates is not None
            else None,
            contextual_max_seqlen=contextual_max_seqlen,
            contextual_seqlen=contextual_seqlen.to(torch.int32)
            if contextual_seqlen is not None
            else None,
            contextual_seqlen_offsets=contextual_seqlen_offsets.to(torch.int32)
            if contextual_seqlen_offsets is not None
            else None,
            has_interleaved_action=batch.action_feature_name is not None,
        )

    @output_nvtx_hook(nvtx_tag="hstu_postprocess")
    def hstu_postprocess(self, jd: JaggedData) -> JaggedData:
        """
        Postprocess the output from the HSTU architecture.
        1. If max_num_candidates > 0, split and only keep last ``num_candidates`` embeddings as candidates embedding for further processing.
        2. Remove action embeddings if present. Only use item embedding for further processing.

        Args:
            jd (JaggedData): The jagged data output from the HSTU architecture that needs further processing.

        Returns:
            JaggedData: The postprocessed jagged data.
        """

        sequence_embeddings: torch.Tensor
        seqlen_offsets: torch.Tensor
        max_seqlen: int
        if jd.max_num_candidates > 0:
            seqlen_offsets = jd.num_candidates_offsets
            max_seqlen = jd.max_num_candidates
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
                offsets_b=seqlen_offsets,
            )
        elif jd.contextual_max_seqlen > 0:
            seqlen_offsets = jd.seqlen_offsets - jd.contextual_seqlen_offsets
            max_seqlen = jd.max_seqlen - jd.contextual_max_seqlen
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.contextual_seqlen_offsets,
                offsets_b=seqlen_offsets,
            )
        else:
            sequence_embeddings = jd.values
            seqlen_offsets = jd.seqlen_offsets
            max_seqlen = jd.max_seqlen

        if jd.has_interleaved_action:
            sequence_embeddings = sequence_embeddings[0::2, ...]
            seqlen_offsets = seqlen_offsets // 2
            max_seqlen = max_seqlen // 2

        sequence_embeddings = sequence_embeddings / torch.linalg.norm(
            sequence_embeddings, ord=2, dim=-1, keepdim=True
        ).clamp(min=1e-6)

        return JaggedData(
            values=sequence_embeddings,
            seqlen=(seqlen_offsets[1:] - seqlen_offsets[:-1]).to(jd.seqlen.dtype),
            seqlen_offsets=seqlen_offsets.to(jd.seqlen_offsets.dtype),
            max_seqlen=max_seqlen,
            has_interleaved_action=False,
        )

    @output_nvtx_hook(nvtx_tag="HSTUBlock", hook_tensor_attr_name="values")
    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: Union[RankingBatch, RetrievalBatch],
    ) -> JaggedData:
        """
        Forward pass of the HSTUBlock.

        Args:
            embeddings (Dict[str, JaggedTensor]): The input embeddings.
            batch (RankingBatch): The input batch.

        Returns:
            JaggedData: The output jagged data.
        """
        jd = self.hstu_preprocess(embeddings, batch)
        for hstu_layer in self._attention_layers:
            jd = hstu_layer(jd)
        return self.hstu_postprocess(jd)

    @output_nvtx_hook(nvtx_tag="hstu_predict")
    @torch.inference_mode()
    def predict(self, jd: JaggedData, kv_cache_handler) -> JaggedData:
        for hstu_layer in self._attention_layers:
            jd = hstu_layer(jd, kv_cache_handler)
        return self.hstu_postprocess(jd)
