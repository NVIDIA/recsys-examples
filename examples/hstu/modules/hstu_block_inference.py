# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Any, Dict, Optional

import torch
from commons.datasets.hstu_batch import HSTUBatch
from configs import InferenceHSTUConfig
from modules.hstu_processor import HSTUBlockPostprocessor, HSTUBlockPreprocessor
from modules.jagged_data import JaggedData
from modules.paged_hstu_infer_layer import PagedHSTUInferLayer
from torchrec.sparse.jagged_tensor import JaggedTensor


def _initialize_kvcache_metadata_for_cudagraph(
    kvcache_metadata,
    batch_size: int,
    seqlen: int,
    num_candidates: int,
) -> None:
    """Populate static paged-KV metadata with a valid capture-time workload.

    The metadata buffers are allocated with ``torch.empty`` because normal
    inference fills them through the KV-cache manager. CUDA graph capture runs
    before that first allocation, so it needs a synthetic, internally
    consistent batch for warmup and capture.
    """
    if not kvcache_metadata.kv_cache_table:
        raise ValueError("CUDA graph KV metadata must reference a KV cache table")
    if batch_size <= 0 or seqlen < 0:
        raise ValueError(
            f"Invalid CUDA graph shape: batch_size={batch_size}, seqlen={seqlen}"
        )
    if num_candidates < 0 or num_candidates > seqlen:
        raise ValueError(
            f"Invalid candidate count {num_candidates} for sequence length {seqlen}"
        )

    history_len = seqlen - num_candidates
    new_history_nnz = batch_size * history_len
    max_new_history_nnz = batch_size * seqlen

    first_cache_table = kvcache_metadata.kv_cache_table[0]
    if first_cache_table.ndim != 5:
        raise ValueError(
            "KV cache table must have shape "
            "[num_pages, 2, page_size, num_heads, head_dim]"
        )
    page_size = first_cache_table.shape[2]
    if page_size <= 0:
        raise ValueError(f"Invalid KV cache page size {page_size}")

    pages_per_sequence = math.ceil(history_len / page_size)
    num_pages = batch_size * pages_per_sequence
    for layer_idx, cache_table in enumerate(kvcache_metadata.kv_cache_table):
        if cache_table.ndim != 5 or cache_table.shape[2] != page_size:
            raise ValueError(
                f"KV cache table {layer_idx} has an inconsistent shape or page size"
            )
        if cache_table.shape[0] < num_pages:
            raise ValueError(
                f"KV cache table {layer_idx} has {cache_table.shape[0]} pages, "
                f"but CUDA graph capture needs {num_pages}"
            )

    required_capacities = {
        "kv_indices": num_pages,
        "kv_indptr": batch_size + 1,
        "kv_last_page_len": batch_size,
        "total_history_lengths": batch_size,
        "total_history_offsets": batch_size + 1,
        "new_history_offsets": batch_size + 1,
        "batch_indices": new_history_nnz,
        "position": new_history_nnz,
        "new_history_nnz_cuda": 1,
        "kv_seqlens": batch_size,
        "kv_seqlen_offsets": batch_size + 1,
    }
    for name, required in required_capacities.items():
        tensor = getattr(kvcache_metadata, name)
        if tensor is None or tensor.numel() < required:
            actual = 0 if tensor is None else tensor.numel()
            raise ValueError(
                f"KV metadata buffer {name} has capacity {actual}, "
                f"but CUDA graph capture needs {required}"
            )

    # Initialize complete backing views, including the inactive tail. Reusing
    # these buffers for a smaller graph must never expose stale values from a
    # previously captured shape.
    kvcache_metadata.kv_indices.zero_()
    kvcache_metadata.kv_indptr.zero_()
    kvcache_metadata.kv_last_page_len.zero_()
    kvcache_metadata.total_history_lengths.zero_()
    kvcache_metadata.total_history_offsets.zero_()
    kvcache_metadata.new_history_offsets.zero_()
    kvcache_metadata.batch_indices.zero_()
    kvcache_metadata.position.zero_()
    kvcache_metadata.new_history_nnz_cuda.zero_()
    kvcache_metadata.kv_seqlens.zero_()
    kvcache_metadata.kv_seqlen_offsets.zero_()

    device = kvcache_metadata.kv_indptr.device
    dtype = kvcache_metadata.kv_indptr.dtype
    if num_pages > 0:
        kvcache_metadata.kv_indices[:num_pages].copy_(
            torch.arange(
                num_pages,
                dtype=kvcache_metadata.kv_indices.dtype,
                device=device,
            )
        )
    kvcache_metadata.kv_indptr[: batch_size + 1].copy_(
        torch.arange(batch_size + 1, dtype=dtype, device=device) * pages_per_sequence
    )
    if kvcache_metadata.kv_indptr.numel() > batch_size + 1:
        kvcache_metadata.kv_indptr[batch_size + 1 :].fill_(num_pages)

    last_page_len = history_len % page_size
    if history_len > 0 and last_page_len == 0:
        last_page_len = page_size
    kvcache_metadata.kv_last_page_len[:batch_size].fill_(last_page_len)
    kvcache_metadata.total_history_lengths[:batch_size].fill_(history_len)

    history_offsets = (
        torch.arange(batch_size + 1, dtype=dtype, device=device) * history_len
    )
    kvcache_metadata.total_history_offsets[: batch_size + 1].copy_(history_offsets)
    kvcache_metadata.new_history_offsets[: batch_size + 1].copy_(history_offsets)
    if kvcache_metadata.total_history_offsets.numel() > batch_size + 1:
        kvcache_metadata.total_history_offsets[batch_size + 1 :].fill_(new_history_nnz)
        kvcache_metadata.new_history_offsets[batch_size + 1 :].fill_(new_history_nnz)

    if new_history_nnz > 0:
        kvcache_metadata.batch_indices[:new_history_nnz].copy_(
            torch.arange(batch_size, dtype=dtype, device=device).repeat_interleave(
                history_len
            )
        )
        kvcache_metadata.position[:new_history_nnz].copy_(
            torch.arange(history_len, dtype=dtype, device=device).repeat(batch_size)
        )
    kvcache_metadata.new_history_nnz_cuda.fill_(new_history_nnz)

    kv_seqlen = history_len + num_candidates
    kvcache_metadata.kv_seqlens[:batch_size].fill_(kv_seqlen)
    kv_seqlen_offsets = (
        torch.arange(batch_size + 1, dtype=dtype, device=device) * kv_seqlen
    )
    kvcache_metadata.kv_seqlen_offsets[: batch_size + 1].copy_(kv_seqlen_offsets)
    if kvcache_metadata.kv_seqlen_offsets.numel() > batch_size + 1:
        kvcache_metadata.kv_seqlen_offsets[batch_size + 1 :].fill_(
            batch_size * kv_seqlen
        )

    # This Python integer is baked into the captured launch dimensions. Keep
    # the full token count as the upper bound because replayed requests can
    # contain fewer candidates (and therefore more new-history tokens).
    kvcache_metadata.new_history_nnz = max_new_history_nnz


class HSTUBlockInference(torch.nn.Module):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (InferenceHSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        config: InferenceHSTUConfig,
    ):
        super().__init__()
        self.config = config

        self._preprocessor = HSTUBlockPreprocessor(config, is_inference=True)
        self._postprocessor = HSTUBlockPostprocessor(is_inference=True)

        self._attention_layers = torch.nn.ModuleList(
            [
                PagedHSTUInferLayer(config, layer_idx)
                for layer_idx in range(self.config.num_layers)
            ]
        )
        self._hstu_graph: Optional[Dict[int, Any]] = None  # type: ignore
        self._contextual_hstu_graph: Optional[Dict[int, Any]] = None  # type: ignore

    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: HSTUBatch,
    ) -> JaggedData:
        """
        Forward pass of the HSTUBlock.

        Args:
            embeddings (Dict[str, JaggedTensor]): The input embeddings.
            batch (HSTUBatch): The input batch.

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
        if self._hstu_graph is None or not use_cudagraph or jd.scaling_seqlen != -1:
            hidden_data = hidden_states
            for hstu_layer in self._attention_layers:
                hidden_data = hstu_layer.forward_naive(
                    batch_size, num_tokens, hidden_data, jd, kv_cache_metadata
                )
            return hidden_data
        else:
            return self.predict_cudagraph(
                batch_size,
                num_tokens,
                hidden_states,
                kv_cache_metadata,
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
                scaling_seqlen=jd.scaling_seqlen,
            )
            if kv_cache_metadata is not None:
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
            if batch_size not in self._hstu_graph:  # type: ignore
                batch_size_padded = min(
                    [k for k in self._hstu_graph.keys() if k > batch_size], default=-1  # type: ignore
                )
                if batch_size_padded == -1:
                    raise ValueError(
                        f"No CUDA graph captured for batch size {batch_size}"
                    )
                batch_size = batch_size_padded
            if num_tokens not in self._hstu_graph[batch_size]:  # type: ignore
                num_tokens_padded = min(
                    [k for k in self._hstu_graph[batch_size].keys() if k > num_tokens],  # type: ignore
                    default=-1,
                )
                if num_tokens_padded == -1:
                    raise ValueError(f"No CUDA graph captured for #tokens {num_tokens}")
            else:
                num_tokens_padded = num_tokens

            self._hstu_graph[batch_size][num_tokens_padded][0].replay()  # type: ignore
            for idx in range(1, self.config.num_layers + 1):
                kv_onload_handle = (
                    kv_cache_metadata.kv_onload_handle
                    if kv_cache_metadata is not None
                    else None
                )
                if (
                    kv_onload_handle is not None
                    and kv_onload_handle.handle is not None
                    and kv_onload_handle.is_layerwise
                ):
                    kv_onload_handle.wait_layer(idx - 1)
                self._hstu_graph[batch_size][num_tokens_padded][idx].replay()  # type: ignore

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
        max_num_tokens = max_batch_size * max_seq_len
        bs_list = [2**i for i in range(math.ceil(math.log2(max_batch_size)) + 1)]
        seqlen_list = [2**i for i in range(5, math.ceil(math.log2(max_seq_len)) + 1)]
        if cudagraph_configs is not None:
            bs_list = cudagraph_configs["batch_size"]
            seqlen_list = cudagraph_configs["length_per_sequence"]

        shared_graph_mempool = None
        print("Setting up cuda graphs ...")
        print("Cudagraph setup configs:")
        print("  Batch size:", bs_list)
        print("  Length per sequence", seqlen_list)

        if self._hstu_graph is None:
            self._hstu_graph = dict()
            self._hstu_graph[max_batch_size] = dict()

            graph_max = self.capture_graph(
                max_batch_size,
                max_num_tokens,
                static_hidden_states,
                static_jagged_metadata,
                static_kvcache_metadata,
                shared_graph_mempool,
            )
            if shared_graph_mempool is None:
                shared_graph_mempool = graph_max[0].pool()
            self._hstu_graph[max_batch_size][max_num_tokens] = graph_max

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
                        shared_graph_mempool,
                    )

    def capture_graph(
        self,
        batch_size,
        num_tokens,
        static_hidden_states,
        static_jagged_metadata,
        static_kvcache_metadata,
        memory_pool=None,
    ):
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

        if static_kvcache_metadata is not None:
            _initialize_kvcache_metadata_for_cudagraph(
                static_kvcache_metadata,
                batch_size=batch_size,
                seqlen=seqlen,
                num_candidates=default_num_candidates,
            )

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

        # if static_kvcache_metadata is not None:
        #     static_kvcache_metadata.total_history_offsets -= (
        #         static_jagged_metadata.num_candidates_offsets
        #     )
        print(
            "Capture cuda graphs for batch_size = {0} and num_tokens = {1}".format(
                batch_size, num_tokens
            )
        )
        # print((before_gmem - after_gmem), (before_gmem - after_gmem) / 1024. / 1024. / 1024.)
        return graph
