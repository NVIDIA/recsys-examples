# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Any, Dict, Optional

import torch
from configs import InferenceHSTUConfig, KVCacheConfig
from dataset.utils import Batch
from modules.hstu_processor import HSTUBlockPostprocessor, HSTUBlockPreprocessor
from modules.jagged_data import JaggedData
from modules.paged_hstu_infer_layer import PagedHSTUInferLayer
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

        self._preprocessor = HSTUBlockPreprocessor(config, is_inference=True)
        self._postprocessor = HSTUBlockPostprocessor(is_inference=True)

        self._attention_layers = torch.nn.ModuleList(
            [
                PagedHSTUInferLayer(config, kvcache_config, layer_idx)
                for layer_idx in range(self.config.num_layers)
            ]
        )
        self._hstu_graph: Optional[Dict[int, Any]] = None  # type: ignore

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
        with torch.inference_mode():
            jd = self._preprocessor(embeddings, batch)
            for hstu_layer in self._attention_layers:
                jd = hstu_layer(jd)
            return self._postprocessor(jd)

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
        cudagraph_configs=None,
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
            seqlen_list = [
                2**i for i in range(5, math.ceil(math.log2(max_seq_len)) + 1)
            ]
            if cudagraph_configs is not None:
                bs_list = cudagraph_configs["batch_size"]
                seqlen_list = cudagraph_configs["length_per_sequence"]
            print("Cudagraph setup configs:")
            print("  Batch size:", bs_list)
            print("  Length per sequence", seqlen_list)

            for batch_size in bs_list:
                if batch_size not in self._hstu_graph:
                    self._hstu_graph[batch_size] = dict()
                for seq_len in seqlen_list:
                    if seq_len > max_seq_len:
                        break
                    num_tokens = seq_len * batch_size
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
