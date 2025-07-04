# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Optional, Union

import torch
from commons.utils.nvtx_op import output_nvtx_hook
from configs.hstu_config import HSTUConfig, HSTULayerType
from dataset.utils import RankingBatch, RetrievalBatch
from megatron.core.transformer.module import MegatronModule
from modules.debug.debug_hstu_layer import HSTULayer as DebugHSTULayer
from modules.fused_hstu_layer import FusedHSTULayer
from modules.jagged_data import JaggedData
from modules.native_hstu_layer import HSTULayer as NativeHSTULayer
from modules.position_encoder import HSTUPositionalEncoder
from modules.utils import hstu_postprocess_embeddings, hstu_preprocess_embeddings
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
        HSTULayerImpl = (
            FusedHSTULayer
            if config.hstu_layer_type == HSTULayerType.FUSED
            else DebugHSTULayer
            if config.hstu_layer_type == HSTULayerType.DEBUG
            else NativeHSTULayer
        )
        self._attention_layers = torch.nn.ModuleList(
            [HSTULayerImpl(config) for l in range(self.config.num_layers)]
        )
        self._dropout_ratio = config.hidden_dropout

    @output_nvtx_hook(nvtx_tag="HSTUBlock preprocess", hook_key_or_attr_name="values")
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
        # Interleaving & concatenation
        jd = hstu_preprocess_embeddings(embeddings, batch, is_inference=False)

        if self._positional_encoder is not None:
            jd.values = self._positional_encoder(
                max_seq_len=jd.max_seqlen,
                seq_lengths=jd.seqlen,
                seq_offsets=jd.seqlen_offsets,
                seq_timestamps=None,
                seq_embeddings=jd.values,
                num_targets=jd.num_candidates,
            )

        jd.values = torch.nn.functional.dropout(
            jd.values,
            p=self._dropout_ratio,
            training=self.training,
        ).to(self._training_dtype)
        return jd

    @output_nvtx_hook(nvtx_tag="HSTUBlock postprocess", hook_key_or_attr_name="values")
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

        return hstu_postprocess_embeddings(jd, is_inference=False)

    @output_nvtx_hook(nvtx_tag="HSTUBlock", hook_key_or_attr_name="values")
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
