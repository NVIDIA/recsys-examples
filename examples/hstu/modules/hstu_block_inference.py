# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Any, Dict, Optional

import torch
from configs import InferenceHSTUConfig, KVCacheConfig
from dataset.utils import Batch
from modules.jagged_data import JaggedData
from modules.paged_hstu_infer_layer import PagedHSTUInferLayer
from modules.position_encoder import HSTUPositionalEncoder
from modules.utils import hstu_postprocess_embeddings, hstu_preprocess_embeddings
from ops.triton_ops.triton_jagged import (  # type: ignore[attr-defined]
    triton_concat_2D_jagged,
    triton_split_2D_jagged,
)
from torchrec.sparse.jagged_tensor import JaggedTensor


class HSTUBlockInference(torch.nn.Module):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (InferenceHSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        config: InferenceHSTUConfig,
        kvcache_config: KVCacheConfig,
    ):
        super().__init__()
        self.config = config
        if self.config.bf16:
            self._dtype = torch.bfloat16
        if self.config.fp16:
            self._dtype = torch.float16

        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if config.position_encoding_config is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                num_position_buckets=config.position_encoding_config.num_position_buckets,
                num_time_buckets=config.position_encoding_config.num_time_buckets,
                embedding_dim=config.hidden_size,
                is_inference=True,
                use_time_encoding=config.position_encoding_config.use_time_encoding,
                training_dtype=self._dtype,
            )
        self._attention_layers = torch.nn.ModuleList(
            [
                PagedHSTUInferLayer(config, kvcache_config, layer_idx)
                for layer_idx in range(self.config.num_layers)
            ]
        )
        self._hstu_graph: Optional[Dict[int, Any]] = None  # type: ignore

    def hstu_preprocess(
        self, embeddings: Dict[str, JaggedTensor], batch: Batch
    ) -> JaggedData:
        """
        Preprocesses the embeddings for use in the HSTU architecture.

        This method performs the following steps:
        1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings (candidates excluded).
        2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample, following the order specified in the batch.
        3. **Position Encoding**: Applies position encoding to the concatenated embeddings.

        Args:
            embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
            batch (Batch): The batch of ranking data.

        Returns:
            JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
        """

        # Interleaving & Concatenation
        jd = hstu_preprocess_embeddings(
            embeddings, batch, is_inference=True, dtype=self._dtype
        )
        device = jd.seqlen_offsets.device
        jd.num_candidates = jd.num_candidates.to(device=device)
        jd.num_candidates_offsets = jd.num_candidates_offsets.to(device=device)

        # Position Encoding
        if self._positional_encoder is not None:
            jd.values = self._positional_encoder(
                max_seq_len=jd.max_seqlen,
                seq_lengths=jd.seqlen,
                seq_offsets=jd.seqlen_offsets,
                seq_timestamps=None,
                seq_embeddings=jd.values,
                num_targets=jd.num_candidates,
            )

        return jd

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

        return hstu_postprocess_embeddings(jd, is_inference=True)

    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: Batch,
    ) -> JaggedData:
        """
        Forward pass of the HSTUBlock.

        Args:
            embeddings (Dict[str, JaggedTensor]): The input embeddings.
            batch (Batch): The input batch.

        Returns:
            JaggedData: The output jagged data.
        """
        jd = self.hstu_preprocess(embeddings, batch)
        for hstu_layer in self._attention_layers:
            jd = hstu_layer(jd)
        return self.hstu_postprocess(jd)

    def predict(
        self,
        batch_size: int,
        num_tokens: int,
        hidden_states: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
        use_cudagraph: bool = True,
    ) -> torch.Tensor:
        if self._hstu_graph is None or not use_cudagraph:
            hidden_data = hidden_states
            for hstu_layer in self._attention_layers:
                hidden_data = hstu_layer.forward_naive(
                    batch_size, num_tokens, hidden_data, jd, kv_cache_metadata
                )
            return hidden_data
        else:
            return self.predict_cudagraph(
                batch_size, num_tokens, hidden_states, kv_cache_metadata
            )

    def predict_naive(
        self,
        batch_size: int,
        num_tokens: int,
        hidden_states: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> torch.Tensor:
        with torch.inference_mode():
            jagged_metadata = JaggedData(
                values=None,
                max_seqlen=jd.max_seqlen,
                seqlen=jd.seqlen[:batch_size],
                seqlen_offsets=jd.seqlen_offsets[: batch_size + 1],
                max_num_candidates=jd.max_num_candidates,
                num_candidates=jd.num_candidates[:batch_size],
                num_candidates_offsets=jd.num_candidates_offsets[: batch_size + 1],
                contextual_max_seqlen=jd.contextual_max_seqlen,
                contextual_seqlen=jd.contextual_seqlen,
                contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
                has_interleaved_action=jd.has_interleaved_action,
            )
            kv_cache_metadata.new_history_nnz = num_tokens
            hidden_data = hidden_states
            for hstu_layer in self._attention_layers:
                hstu_layer.forward_input(
                    batch_size,
                    num_tokens,
                    hidden_data,
                    jagged_metadata,
                    kv_cache_metadata,
                )
                hidden_data = hstu_layer.forward_output(
                    batch_size,
                    num_tokens,
                    hidden_data,
                    jagged_metadata,
                    kv_cache_metadata,
                )
            return hidden_data

    def predict_cudagraph(
        self,
        batch_size: int,
        num_tokens: int,
        hidden_states: torch.Tensor,
        kv_cache_metadata,
    ) -> torch.Tensor:
        with torch.inference_mode():
            batch_size = 2 ** math.ceil(math.log2(batch_size))
            if num_tokens not in self._hstu_graph[batch_size]:  # type: ignore
                num_tokens_pow2 = max(32, 2 ** math.ceil(math.log2(num_tokens)))
            else:
                num_tokens_pow2 = num_tokens

            self._hstu_graph[batch_size][num_tokens_pow2][0].replay()  # type: ignore
            for idx in range(1, self.config.num_layers + 1):
                kv_cache_metadata.onload_history_kv_events[idx - 1].wait(
                    torch.cuda.current_stream()
                )
                self._hstu_graph[batch_size][num_tokens_pow2][idx].replay()  # type: ignore

            hstu_output = torch.zeros_like(hidden_states[:num_tokens, ...])
            hstu_output.copy_(
                self._attention_layers[-1].output_buffer_[:num_tokens, ...],
                non_blocking=True,
            )
            return hstu_output

    def set_cudagraph(
        self,
        max_batch_size,
        max_seq_len,
        static_hidden_states,
        static_jagged_metadata,
        static_kvcache_metadata,
    ):
        print("Setting up cuda graphs ...")
        if self._hstu_graph is None:
            self._hstu_graph = dict()
            self._hstu_graph[max_batch_size] = dict()

            torch.cuda.mem_get_info()[0]

            max_num_tokens = max_batch_size * max_seq_len
            graph_max = self.capture_graph(
                max_batch_size,
                max_num_tokens,
                static_hidden_states,
                static_jagged_metadata,
                static_kvcache_metadata,
                None,
            )
            self._hstu_graph[max_batch_size][max_num_tokens] = graph_max

            bs_list = [2**i for i in range(math.ceil(math.log2(max_batch_size)) + 1)]
            num_tokens_list = [
                2**i for i in range(5, math.ceil(math.log2(max_num_tokens)) + 1)
            ]

            for batch_size in bs_list:
                if batch_size not in self._hstu_graph:
                    self._hstu_graph[batch_size] = dict()
                for num_tokens in num_tokens_list:
                    if num_tokens // batch_size > max_seq_len:
                        break
                    if num_tokens in self._hstu_graph[batch_size]:
                        continue
                    self._hstu_graph[batch_size][num_tokens] = self.capture_graph(
                        batch_size,
                        num_tokens,
                        static_hidden_states,
                        static_jagged_metadata,
                        static_kvcache_metadata,
                        graph_max[0].pool(),
                    )

            torch.cuda.mem_get_info()[0]

    def capture_graph(
        self,
        batch_size,
        num_tokens,
        static_hidden_states,
        static_jagged_metadata,
        static_kvcache_metadata,
        memory_pool=None,
    ):
        torch.cuda.mem_get_info()[0]

        # Create CUDA stream
        graph_capture_warmup_stream = torch.cuda.Stream()
        graph_capture_warmup_stream.wait_stream(torch.cuda.current_stream())

        seqlen = num_tokens // batch_size
        static_jagged_metadata.seqlen_offsets[: batch_size + 1].copy_(
            torch.arange(
                end=batch_size + 1,
                dtype=static_jagged_metadata.num_candidates.dtype,
                device=static_jagged_metadata.num_candidates.device,
            )
            * seqlen
        )

        default_num_candidates = seqlen // 2
        torch.full(
            (batch_size,),
            default_num_candidates,
            out=static_jagged_metadata.num_candidates[:batch_size],
        )
        static_jagged_metadata.num_candidates_offsets[: batch_size + 1].copy_(
            torch.arange(
                end=batch_size + 1,
                dtype=static_jagged_metadata.num_candidates.dtype,
                device=static_jagged_metadata.num_candidates.device,
            )
            * default_num_candidates
        )

        static_kvcache_metadata.total_history_offsets += (
            static_jagged_metadata.num_candidates_offsets
        )
        static_kvcache_metadata.new_history_nnz = num_tokens

        # Warmup
        with torch.cuda.stream(graph_capture_warmup_stream):
            for _ in range(3):
                static_output = self.predict_naive(
                    batch_size,
                    num_tokens,
                    static_hidden_states,
                    static_jagged_metadata,
                    static_kvcache_metadata,
                )
                torch.cuda.synchronize()

        # Create and capture the graph
        num_layers = self.config.num_layers
        graph = [torch.cuda.CUDAGraph() for _ in range(num_layers + 1)]
        input_buffer = [static_hidden_states] + [
            self._attention_layers[layer_idx].output_buffer_
            for layer_idx in range(num_layers)
        ]

        with torch.cuda.graph(graph[0], pool=memory_pool):
            static_uvqk = self._attention_layers[0].forward_input(
                batch_size,
                num_tokens,
                static_hidden_states,
                static_jagged_metadata,
                static_kvcache_metadata,
            )

        if memory_pool is None:
            memory_pool = graph[0].pool()

        for layer_idx in range(0, num_layers - 1):
            with torch.cuda.graph(graph[layer_idx + 1], pool=memory_pool):
                static_output = self._attention_layers[layer_idx].forward_output(
                    batch_size,
                    num_tokens,
                    input_buffer[layer_idx],
                    static_jagged_metadata,
                    static_kvcache_metadata,
                )
                static_uvqk = self._attention_layers[layer_idx + 1].forward_input(
                    batch_size,
                    num_tokens,
                    static_output,
                    static_jagged_metadata,
                    static_kvcache_metadata,
                )

        with torch.cuda.graph(graph[num_layers], pool=memory_pool):
            static_output = self._attention_layers[-1].forward_output(
                batch_size,
                num_tokens,
                input_buffer[num_layers - 1],
                static_jagged_metadata,
                static_kvcache_metadata,
            )

        torch.cuda.mem_get_info()[0]

        static_kvcache_metadata.total_history_offsets -= (
            static_jagged_metadata.num_candidates_offsets
        )
        print(
            "Capture cuda graphs for batch_size = {0} and num_tokens = {1}".format(
                batch_size, num_tokens
            )
        )
        return graph
