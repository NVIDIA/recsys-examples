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
from configs.kv_cache_config import KVCacheConfig, KVCacheMetadata
import math


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
        kvcache_config: KVCacheConfig = None,

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
        
        self._inference = inference_mode
        if not self._inference:
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
                [PagedHSTUInferLayer(config, kvcache_config, layer_idx) for layer_idx in range(self.config.num_layers)]
            )
        
        self._hstu_graph = None

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

        if not self._inference:
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
    def predict(self, batch_size: int, num_tokens: int, hidden_states: torch.Tensor, jd: JaggedData, kv_cache_metadata) -> JaggedData:
        if self._hstu_graph is None:
            hidden_data = hidden_states
            for hstu_layer in self._attention_layers:
                hidden_data = hstu_layer(num_tokens, hidden_data, jd, kv_cache_metadata)
            jd.values = hidden_data
            return jd
        else:
            batch_size = 2**math.ceil(math.log2(batch_size))
            num_tokens_pow2 = max(32, 2**math.ceil(math.log2(num_tokens)))
            self._hstu_graph[batch_size][num_tokens_pow2].replay()
            hidden_states[:num_tokens, ...].copy_(
                self._attention_layers[-1].output_buffer_[:num_tokens, ...], non_blocking = True)
            jd.values = hidden_states
            return jd

    def predict_naive(self, batch_size: int, num_tokens: int, hidden_states: torch.Tensor, jd: JaggedData, kv_cache_metadata) -> JaggedData:
        with torch.inference_mode():
            jagged_metadata = JaggedData(
                values=None,
                max_seqlen=jd.max_seqlen,
                seqlen=jd.seqlen[:batch_size],
                seqlen_offsets=jd.seqlen_offsets[:batch_size+1],
                max_num_candidates=jd.max_num_candidates,
                num_candidates=jd.num_candidates[:batch_size],
                num_candidates_offsets=jd.num_candidates_offsets[:batch_size+1],
                contextual_max_seqlen=jd.contextual_max_seqlen,
                contextual_seqlen=jd.contextual_seqlen,
                contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
                has_interleaved_action=jd.has_interleaved_action,
            )
            kv_cache_metadata.delta_history_token_nnz = num_tokens
            hidden_data = hidden_states
            for hstu_layer in self._attention_layers:
                hidden_data = hstu_layer(batch_size, num_tokens, hidden_data, jagged_metadata, kv_cache_metadata)
            jd.values = hidden_data
            return jd

    def set_cudagraph(self, max_batch_size, max_seq_len, static_hidden_states, static_jagged_metadata, static_kvcache_metadata):
        if self._hstu_graph is None:

            self._hstu_graph = dict()
            self._hstu_graph[max_batch_size] = dict()

            start_free_memory = torch.cuda.mem_get_info()[0]

            max_num_tokens = max_batch_size * max_seq_len
            graph_max = self.capture_graph(max_batch_size, max_num_tokens, static_hidden_states, static_jagged_metadata, static_kvcache_metadata)
            self._hstu_graph[max_batch_size][max_num_tokens] = graph_max

            bs_list = [ 2 ** i for i in range(math.ceil(math.log2(max_batch_size)) + 1) ]
            num_tokens_list =  [ 2 ** i for i in range(5, math.ceil(math.log2(max_num_tokens)) + 1) ]

            for batch_size in bs_list:
                if batch_size not in self._hstu_graph:
                    self._hstu_graph[batch_size] = dict()
                for num_tokens in num_tokens_list:
                    if num_tokens // batch_size > max_seq_len:
                        break
                    if num_tokens in self._hstu_graph[batch_size]:
                        continue
                    self._hstu_graph[batch_size][num_tokens] = self.capture_graph(batch_size, num_tokens, 
                        static_hidden_states, static_jagged_metadata, static_kvcache_metadata,
                        graph_max.pool())
            
            end_free_memory = torch.cuda.mem_get_info()[0]
            print('total cuda graph memory: %fGB'%((start_free_memory - end_free_memory) / 1024 / 1024 / 1024))
        
    
    def capture_graph(self, batch_size, num_tokens, static_hidden_states, static_jagged_metadata, static_kvcache_metadata, memory_pool=None):
        start_free_memory = torch.cuda.mem_get_info()[0]

        # Create CUDA stream
        graph_capture_warmup_stream = torch.cuda.Stream()
        graph_capture_warmup_stream.wait_stream(torch.cuda.current_stream())

        seqlen = num_tokens // batch_size
        static_jagged_metadata.seqlen_offsets[:batch_size+1].copy_(
            torch.arange(end=batch_size+1,
                         dtype = static_jagged_metadata.num_candidates.dtype,
                         device = static_jagged_metadata.num_candidates.device
            ) * seqlen)
        
        default_num_candidates = seqlen // 2
        torch.full((batch_size, ), default_num_candidates, out=static_jagged_metadata.num_candidates[:batch_size])
        static_jagged_metadata.num_candidates_offsets[:batch_size+1].copy_(
            torch.arange(end=batch_size+1,
                         dtype = static_jagged_metadata.num_candidates.dtype,
                         device = static_jagged_metadata.num_candidates.device
            ) * default_num_candidates)

        # Warmup
        with torch.cuda.stream(graph_capture_warmup_stream):
            for _ in range(3):
                static_output = self.predict_naive(batch_size, num_tokens, static_hidden_states, static_jagged_metadata, static_kvcache_metadata)
                torch.cuda.synchronize()

        # Create and capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=memory_pool):
            static_output = self.predict_naive(batch_size, num_tokens, static_hidden_states, static_jagged_metadata, static_kvcache_metadata)
        end_free_memory = torch.cuda.mem_get_info()[0]
        print("Capture graph for", (batch_size, num_tokens), 'cuda graph memory: %fGB'%((start_free_memory - end_free_memory) / 1024 / 1024 / 1024))
        return graph
