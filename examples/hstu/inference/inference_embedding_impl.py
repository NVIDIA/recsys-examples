# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation module for exportable inference embedding demo.

This module owns:
1. Loading `inference_emb_ops.so` before any `dynamicemb` imports
2. Importing the `dynamicemb` symbols required by the inference demo
3. The `InferenceLinearBucketTable` and `InferenceEmbeddingTable` implementations
"""

import os
from typing import List, Optional

import torch

_SEARCH_PATHS = [
    os.path.join(
        os.path.dirname(__file__),
        "../../../corelib/dynamicemb/torch_binding_build/inference_emb_ops.so",
    ),
    os.path.join(
        os.path.dirname(__file__),
        "../../../corelib/dynamicemb/build_inference_ops_check/inference_emb_ops.so",
    ),
    os.path.join(
        os.path.dirname(__file__),
        "../../../corelib/dynamicemb/build/inference_emb_ops.so",
    ),
    "inference_emb_ops.so",
]

AOTI_DEMO_DIR = os.path.join(os.path.dirname(__file__), "aoti_demo")
AOTI_MODEL_PATH = os.path.join(AOTI_DEMO_DIR, "model.pt2")

_ops_loaded: bool = False
_ops_load_attempted: bool = False


def _load_inference_emb_ops() -> bool:
    """Load `inference_emb_ops.so` once and return whether it succeeded."""
    global _ops_loaded, _ops_load_attempted

    if _ops_loaded:
        return True
    if _ops_load_attempted:
        return False

    _ops_load_attempted = True
    for _path in _SEARCH_PATHS:
        if os.path.exists(_path):
            try:
                torch.ops.load_library(_path)
                print(f"[INFO] Loaded inference_emb_ops.so from {_path}")
                _ops_loaded = True
            except Exception as _e:
                print(f"[WARN] Failed to load {_path}: {_e}")
            break

    if not _ops_loaded:
        print("[WARN] Could not find inference_emb_ops.so. Custom ops may not be available.")

    return _ops_loaded


# Load operators before importing dynamicemb.
_load_inference_emb_ops()

try:
    import dynamicemb
    import dynamicemb.index_range_meta as _index_range_meta
    import dynamicemb.lookup_meta as _lookup_meta
    from dynamicemb import (
        DynamicEmbInitializerArgs,
        DynamicEmbInitializerMode,
        DynamicEmbPoolingMode,
        DynamicEmbScoreStrategy,
        DynamicEmbTableOptions,
    )
    from dynamicemb.batched_dynamicemb_tables import (
        BatchedDynamicEmbeddingTablesV2,
        encode_checkpoint_file_path,
        encode_meta_json_file_path,
        get_loading_files,
    )
    from dynamicemb.key_value_table import _iter_batches_from_files, load_from_json
    from dynamicemb.scored_hashtable import ScorePolicy
    from dynamicemb_extensions import table_insert, expand_table_ids_cuda

    if _ops_loaded:
        if not _index_range_meta.REGISTERED:
            _index_range_meta.register_index_range_fake()
        if not _lookup_meta.REGISTERED:
            _lookup_meta.register_lookup_fake()
except ImportError:
    pass


__all__ = [
    "_load_inference_emb_ops",
    "AOTI_DEMO_DIR",
    "AOTI_MODEL_PATH",
    "InferenceLinearBucketTable",
    "InferenceEmbeddingTable",
    "DynamicEmbInitializerArgs",
    "DynamicEmbInitializerMode",
    "DynamicEmbPoolingMode",
    "DynamicEmbScoreStrategy",
    "DynamicEmbTableOptions",
    "BatchedDynamicEmbeddingTablesV2",
    "encode_checkpoint_file_path",
    "encode_meta_json_file_path",
    "ScorePolicy",
]


# ---------------------------------------------------------------------------
# Helpers for InferenceEmbeddingTable construction
# ---------------------------------------------------------------------------


def _resolve_capacity(opt: "DynamicEmbTableOptions") -> int:
    """Return the capacity for a single table option.

    Prefers ``init_capacity``; falls back to ``max_capacity``.
    Raises ``ValueError`` if neither is set or positive.
    """
    cap = opt.init_capacity if opt.init_capacity is not None else opt.max_capacity
    if cap is None or cap <= 0:
        raise ValueError(
            "Each table option must provide init_capacity or max_capacity > 0"
        )
    return int(cap)



def _derive_grouped_offsets(feature_table_map: List[int]) -> List[int]:
    """Derive boundary-style offsets from a per-feature table-id list.

    For example, ``[0, 0, 1, 2]`` → ``[0, 2, 3, 4]``.
    The result is analogous to ``table_bucket_offsets_`` in ``LinearBucketTable``.
    """
    offsets = [0]
    prev = feature_table_map[0]
    for i, tid in enumerate(feature_table_map[1:], start=1):
        if tid != prev:
            offsets.append(i)
            prev = tid
    offsets.append(len(feature_table_map))
    return offsets


class InferenceLinearBucketTable(torch.nn.Module):
    """Simple exportable hash table wrapper for inference lookup using custom op.

    This is a minimal demo version that focuses on lookup-only, non-pooled inference.
    For the full production version, see LinearBucketTable in scored_hashtable.py.
    """

    def __init__(
        self,
        capacity: List[int],
        key_type: torch.dtype = torch.int64,
        bucket_capacity: int = 128,
        device: Optional[torch.device] = None,
    ):
        """Initialize demo hash table.

        Args:
            capacity: List of per-table capacities
            key_type: torch.int64 or torch.uint64
            bucket_capacity: slots per bucket
            device: CUDA device (defaults to current)
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.device = device
        self.key_type_ = key_type
        self.bucket_capacity_ = bucket_capacity
        self.num_tables_ = len(capacity)

        per_table_num_buckets = []
        bucket_offset_list = [0]
        for cap in capacity:
            nb = (cap + bucket_capacity - 1) // bucket_capacity
            per_table_num_buckets.append(nb)
            bucket_offset_list.append(bucket_offset_list[-1] + nb)

        total_buckets = bucket_offset_list[-1]
        self.capacity_ = total_buckets * self.bucket_capacity_

        bytes_per_slot = 8 + 1 + 8
        total_storage_bytes = bytes_per_slot * bucket_capacity * total_buckets

        self.register_buffer(
            "table_storage_",
            torch.zeros(total_storage_bytes, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "table_bucket_offsets_",
            torch.tensor(bucket_offset_list, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "bucket_sizes",
            torch.zeros(total_buckets, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "_ref_counter",
            torch.zeros(self.capacity_, dtype=torch.int32, device=self.device),
        )

    def lookup(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score_value: Optional[torch.Tensor] = None,
        score_policy: int = int(ScorePolicy.CONST),
    ) -> tuple:
        """Lookup keys in the hash table using the custom operator."""
        score_out, founds, indices = torch.ops.INFERENCE_EMB.table_lookup(
            self.table_storage_,
            self.table_bucket_offsets_,
            self.bucket_capacity_,
            keys,
            table_ids,
            score_value,
            score_policy,
            None,
            0,
            None,
        )

        return score_out, founds, indices


class InferenceEmbeddingTable(torch.nn.Module):
    """Simplified, export-compatible embedding table using custom ops."""

    def __init__(
        self,
        table_options: List["DynamicEmbTableOptions"],
        table_names: Optional[List[str]] = None,
        feature_table_map: Optional[List[int]] = None,
        output_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        key_type: torch.dtype = torch.int64,
    ):
        super().__init__()

        if not table_options:
            raise ValueError("table_options must be non-empty")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if key_type not in (torch.int64, torch.uint64):
            raise ValueError(f"unsupported key_type: {key_type}")

        capacities = [_resolve_capacity(opt) for opt in table_options]
        num_tables = len(table_options)

        if table_names is None:
            table_names = [f"table_{i}" for i in range(num_tables)]
        if len(table_names) != num_tables:
            raise ValueError("table_names size must match table_options")

        if feature_table_map is None:
            feature_table_map = list(range(num_tables))
        if not isinstance(feature_table_map, list) or len(feature_table_map) == 0:
            raise ValueError("feature_table_map must be a non-empty list")
        if any(t < 0 or t >= num_tables for t in feature_table_map):
            raise ValueError(
                f"feature_table_map contains out-of-range table id (must be in [0, {num_tables}))"
            )

        feature_offsets = _derive_grouped_offsets(feature_table_map)

        self.device = device
        self.output_dtype_ = output_dtype
        self.key_type_ = key_type
        self.num_tables_ = num_tables
        self.table_names_ = table_names

        self.register_buffer(
            "feature_table_map_",
            torch.tensor(feature_table_map, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "feature_offsets_",
            torch.tensor(feature_offsets, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "capacity_list_",
            torch.tensor(capacities, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "table_offsets_",
            torch.zeros(num_tables + 1, dtype=torch.int64, device=device),
        )
        torch.cumsum(self.capacity_list_, dim=0, out=self.table_offsets_[1:])

        self.hash_table = InferenceLinearBucketTable(
            capacity=capacities,
            key_type=key_type,
            bucket_capacity=128,
            device=device,
        )

        emb_dim = 128
        total_rows = int(self.capacity_list_.sum().item())
        self.register_buffer(
            "linear_mem_table_",
            torch.zeros(total_rows, emb_dim, dtype=torch.float32, device=device),
        )

    def load(
        self,
        save_dir: str,
        table_names: Optional[List[str]] = None,
    ) -> None:
        if not os.path.exists(save_dir):
            raise RuntimeError(f"Save directory does not exist: {save_dir}")

        if "get_loading_files" not in globals() or "_iter_batches_from_files" not in globals():
            raise RuntimeError(
                "dynamicemb load helpers are unavailable. Ensure dynamicemb and inference operators are importable."
            )

        if table_names is None:
            table_names = self.table_names_

        requested_table_names = set(table_names)
        dim = self.linear_mem_table_.size(1)
        device = self.device

        self.hash_table.table_storage_.zero_()
        self.hash_table.bucket_sizes.zero_()
        self.hash_table._ref_counter.zero_()
        self.linear_mem_table_.zero_()

        for table_id, table_name in enumerate(self.table_names_):
            if table_name not in requested_table_names:
                continue

            meta_json_file = encode_meta_json_file_path(save_dir, table_name)
            if os.path.exists(meta_json_file):
                try:
                    _ = load_from_json(meta_json_file)
                except Exception as e:
                    print(
                        f"[WARN] Failed to read meta json for {table_name} at {meta_json_file}: {e}"
                    )

            (
                emb_key_files,
                emb_value_files,
                emb_score_files,
                _opt_value_files,
                _counter_key_files,
                _counter_frequency_files,
            ) = get_loading_files(
                save_dir,
                table_name,
                rank=0,
                world_size=1,
            )

            if len(emb_key_files) == 0:
                print(f"[INFO] No checkpoint files found for table: {table_name}")
                continue

            num_key_files = len(emb_key_files)
            for i in range(num_key_files):
                score_file = emb_score_files[i] if i < len(emb_score_files) else None
                for keys, embeddings, scores, _opt_states in _iter_batches_from_files(
                    emb_key_files[i],
                    emb_value_files[i],
                    score_file,
                    None,
                    dim,
                    0,
                    device,
                ):
                    if keys.numel() == 0:
                        continue

                    table_ids = torch.full(
                        (keys.numel(),),
                        table_id,
                        dtype=torch.int64,
                        device=device,
                    )
                    policy = (
                        ScorePolicy.ASSIGN if scores is not None else ScorePolicy.CONST
                    )
                    indices = table_insert(
                        self.hash_table.table_storage_,
                        self.hash_table.table_bucket_offsets_,
                        self.hash_table.bucket_capacity_,
                        self.hash_table.bucket_sizes,
                        keys,
                        table_ids,
                        scores,
                        policy,
                        self.hash_table._ref_counter,
                        None,
                        None,
                    )

                    valid_mask = indices >= 0
                    if not torch.all(valid_mask):
                        num_failed = (~valid_mask).sum().item()
                        print(
                            f"[WARN] table_insert failed for {num_failed} keys in table {table_name}."
                        )

                    valid_indices = indices[valid_mask].to(torch.int64)
                    if valid_indices.numel() == 0:
                        continue

                    max_index = valid_indices.max().item()
                    table_cap = int(self.capacity_list_[table_id].item())
                    if max_index >= table_cap:
                        raise RuntimeError(
                            f"linear_mem_table_ has insufficient rows ({table_cap}) for loaded index {max_index}."
                        )

                    self.linear_mem_table_[
                        self.table_offsets_[table_id] : self.table_offsets_[table_id + 1]
                    ].index_copy_(
                        0,
                        valid_indices,
                        embeddings[valid_mask].to(self.linear_mem_table_.dtype),
                    )

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        num_elements = indices.size(0)

        table_ids = torch.ops.INFERENCE_EMB.expand_table_ids(
            offsets, None, self.num_tables_, 1, num_elements
        )

        _scores, _founds, table_indices = self.hash_table.lookup(
            keys=indices,
            table_ids=table_ids,
            score_value=None,
            score_policy=int(ScorePolicy.CONST),
        )

        global_table_offsets = torch.index_select(self.table_offsets_, 0, table_ids)
        table_indices = table_indices + global_table_offsets

        safe_table_indices = torch.where(
            _founds,
            table_indices,
            torch.zeros_like(table_indices),
        )
        embeddings = torch.index_select(self.linear_mem_table_, 0, safe_table_indices)
        embeddings = torch.where(
            _founds.unsqueeze(-1),
            embeddings,
            torch.zeros_like(embeddings),
        )
        return embeddings
