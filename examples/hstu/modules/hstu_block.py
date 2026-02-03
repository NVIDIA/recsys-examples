# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Tuple

import torch
from commons.datasets.hstu_batch import HSTUBatch
from commons.utils.nvtx_op import output_nvtx_hook
from configs.hstu_config import HSTUConfig, HSTULayerType
from megatron.core.transformer.module import MegatronModule
from modules.debug.debug_hstu_layer import HSTULayer as DebugHSTULayer
from modules.fused_hstu_layer import FusedHSTULayer
from modules.hstu_processor import HSTUBlockPostprocessor, HSTUBlockPreprocessor
from modules.jagged_data import JaggedData
from modules.native_hstu_layer import HSTULayer as NativeHSTULayer
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

        self._preprocessor = HSTUBlockPreprocessor(
            config, is_inference=False
        )  # sequence parallel is from config
        self._postprocessor = HSTUBlockPostprocessor(
            is_inference=False, sequence_parallel=config.sequence_parallel
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

    @output_nvtx_hook(nvtx_tag="HSTUBlock", hook_key_or_attr_name="values")
    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: HSTUBatch,
    ) -> Tuple[JaggedData, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the HSTUBlock.

        Args:
            embeddings (Dict[str, JaggedTensor]): The input embeddings.
            batch (HSTUBatch): The input batch.

        Returns:
            JaggedData: The output jagged data.
        """
        jd = self._preprocessor(embeddings, batch)
        seqlen_after_preprocessor = jd.seqlen
        num_contextuals_after_preprocessor = (
            jd.contextual_seqlen
            if jd.contextual_seqlen is not None
            else torch.zeros_like(seqlen_after_preprocessor)
        )
        num_candidates_after_preprocessor = (
            jd.num_candidates
            if jd.num_candidates is not None
            else torch.zeros_like(seqlen_after_preprocessor)
        )
        for hstu_layer in self._attention_layers:
            jd = hstu_layer(jd)
        return self._postprocessor(jd), (
            seqlen_after_preprocessor.detach(),
            num_contextuals_after_preprocessor.detach(),
            num_candidates_after_preprocessor.detach(),
        )
