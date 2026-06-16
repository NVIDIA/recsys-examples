# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation module for exportable inference embedding demo.

This module owns:
1. Loading `inference_emb_ops.so` before any `dynamicemb` imports
2. Importing the `dynamicemb` symbols required by the inference demo
3. The `InferenceLinearBucketTable` and `InferenceEmbeddingTable` implementations
"""

import itertools
import os
from typing import Dict, List, Optional

import pynve.torch.nve_layers as nve_layers
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import (
    encode_meta_json_file_path,
    get_loading_files,
)
from dynamicemb.key_value_table import _iter_batches_from_files, load_from_json
from dynamicemb.scored_hashtable import ScorePolicy
from dynamicemb_extensions import table_insert
from torch.nn import ModuleDict
from torchrec.modules.embedding_configs import EmbeddingConfig

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


def _resolve_embedding_dim(table_options: List["DynamicEmbTableOptions"]) -> int:
    dims = {int(opt.dim) for opt in table_options if opt.dim is not None}
    if len(dims) != 1:
        raise ValueError(
            "InferenceEmbeddingTable requires exactly one shared embedding dim across all table_options"
        )
    emb_dim = dims.pop()
    if emb_dim <= 0:
        raise ValueError("Embedding dim must be > 0")
    return emb_dim


def _resolve_gpu_cache_size(
    table_options: List["DynamicEmbTableOptions"],
    total_size_bytes: int,
) -> int:
    values = {int(opt.global_hbm_for_values or 0) for opt in table_options}
    if len(values) != 1:
        raise ValueError(
            "All table_options must have the same global_hbm_for_values for NVE inference table"
        )
    gpu_cache_size = values.pop()
    if gpu_cache_size <= 0:
        gpu_cache_size = int(total_size_bytes)
        print(
            "[INFO] global_hbm_for_values is 0 for all tables; "
            f"using fallback gpu_cache_size={gpu_cache_size}"
        )
    return gpu_cache_size


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


# ---------------------------------------------------------------------------
# Modules for InferenceEmbeddingTable and its hash table
# ---------------------------------------------------------------------------


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

        self.score_policy = int(ScorePolicy.CONST)

    def lookup(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score_value: Optional[torch.Tensor] = None,
        score_policy: int = 0,
    ) -> tuple:
        """Lookup keys in the hash table using the custom operator."""
        score_out, founds, indices = torch.ops.INFERENCE_EMB.table_lookup(
            self.table_storage_,
            self.table_bucket_offsets_,
            self.bucket_capacity_,
            keys,
            table_ids,
            score_value,
            self.score_policy,
            None,
            0,
            None,
        )

        return score_out, founds, indices


class InferenceEmbeddingCollection(torch.nn.Module):
    """Export-compatible embedding table using custom ops.

    The pooling mode is fixed at construction time so that each exported
    artifact corresponds to exactly one pooling behaviour:

    - ``pooling_mode=-1``: no pooling; ``forward()`` returns ``(N, D)``.
    - ``pooling_mode=1``:  sum pooling; ``forward()`` returns ``(B, D)``.
    - ``pooling_mode=2``:  mean pooling; ``forward()`` returns ``(B, D)``.
    """

    def __init__(
        self,
        table_options: List["EmbeddingConfig"],
        use_dynamic_hash: bool,
        pooling_mode: int,
        table_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        feature_table_map: Optional[List[int]] = None,
        output_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        key_type: torch.dtype = torch.int64,
    ):
        super().__init__()
        self.embedding_configs: List[EmbeddingConfig] = table_options

        if pooling_mode not in (-1, 1, 2):
            raise ValueError(
                f"pooling_mode must be -1 (no pooling), 1 (sum), or 2 (mean), "
                f"got {pooling_mode}"
            )
        if not table_options:
            raise ValueError("table_options must be non-empty")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if key_type not in (torch.int64, torch.uint64):
            raise ValueError(f"unsupported key_type: {key_type}")
        if output_dtype not in (torch.float32, torch.float16):
            raise ValueError(f"unsupported output_dtype: {output_dtype}")

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
        for i in range(1, len(feature_table_map)):
            if feature_table_map[i] < feature_table_map[i - 1]:
                raise ValueError(
                    "feature_table_map must be non-decreasing (features for the same table must be contiguous)"
                )

        feature_offsets = _derive_grouped_offsets(feature_table_map)

        self.device = device
        self.output_dtype_ = output_dtype
        self.key_type_ = key_type
        self.num_tables_ = num_tables
        self.table_names_ = table_names
        self.feature_names_ = feature_names
        self.pooling_mode_ = (
            pooling_mode  # plain Python int – compile-time constant for torch.export
        )
        self.score_policy = int(
            ScorePolicy.CONST
        )  # plain Python int – compile-time constant for torch.export

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
        self.capacity_list_ += (
            1  # reserve the first row of each section for "not found" entries
        )
        torch.cumsum(
            torch.cat([torch.ones((1,), device=device), self.capacity_list_]),
            dim=0,
            out=self.table_offsets_,
        )

        self.use_dynamic_hash = use_dynamic_hash
        if use_dynamic_hash:
            self.hash_table = InferenceLinearBucketTable(
                capacity=capacities,
                key_type=key_type,
                bucket_capacity=128,
                device=device,
            )

        self.emb_dim_ = _resolve_embedding_dim(table_options)
        total_rows = int(self.capacity_list_.sum().item())
        dtype_size = torch.finfo(output_dtype).bits // 8
        total_size_bytes = total_rows * self.emb_dim_ * dtype_size
        self.gpu_cache_size_ = _resolve_gpu_cache_size(table_options, total_size_bytes)

        if self.pooling_mode_ == -1:
            self.nve_embedding_ = nve_layers.NVEmbedding(
                num_embeddings=total_rows,
                embedding_size=self.emb_dim_,
                data_type=output_dtype,
                cache_type=nve_layers.CacheType.LinearUVM,
                gpu_cache_size=int(self.gpu_cache_size_),
                optimize_for_training=False,
                device=device,
            )
        else:
            if self.pooling_mode_ == 1:
                mode = "sum"
            elif self.pooling_mode_ == 2:
                mode = "mean"
            self.nve_embedding_ = nve_layers.NVEmbeddingBag(
                num_embeddings=total_rows,
                embedding_size=self.emb_dim_,
                data_type=output_dtype,
                cache_type=nve_layers.CacheType.LinearUVM,
                mode=mode,
                gpu_cache_size=int(self.gpu_cache_size_),
                optimize_for_training=False,
                device=device,
            )

    def load_from_embedding_table(self, table_weights: torch.Tensor) -> None:
        """Load embedding weights from a pre-extracted table tensor.

        This is used when the training checkpoint contains a pre-extracted embedding table
        that can be directly copied into the NVE weight, bypassing the hash table insertion.

        Args:
            table_weights: (sum of all table capacities, emb_dim) float tensor containing the embedding weights for all tables, ordered by table and then by row within each table.
        """
        assert (
            table_weights.size(0)
            <= self.nve_embedding_.weight.size(0) - self.num_tables_
        ), f"Provided table_weights has more rows ({table_weights.size(0)}) than the NVE embedding capacity ({self.nve_embedding_.weight.size(0) - self.num_tables_} excluding for reserved 'not found' rows)"
        assert (
            table_weights.size(1) == self.emb_dim_
        ), f"Provided table_weights has embedding dim {table_weights.size(1)}, expected {self.emb_dim_}"

        self.nve_embedding_.weight.data.zero_()  # zero out the "not found" row in each table
        for table_id in range(self.num_tables_):
            self.nve_embedding_.weight.data[
                self.table_offsets_[table_id].item(), :
            ].zero_()

            current_offsets = self.table_offsets_[
                table_id
            ].item()  # table_offsets_ already skipped the reserved row
            original_offsets = self.table_offsets_[table_id].item() - table_id - 1
            num_rows = (
                int(self.capacity_list_[table_id].item()) - 1
            )  # exclude reserved row
            if table_id == self.num_tables_ - 1:
                num_rows = (
                    table_weights.size(0) - original_offsets
                )  # use remaining rows for the last table

            self.nve_embedding_.weight.data[
                current_offsets : current_offsets + num_rows, :
            ].copy_(
                table_weights[original_offsets : original_offsets + num_rows, :].to(
                    self.nve_embedding_.weight.dtype
                )
            )

    def load_from_dynamicemb_file(
        self,
        save_dir: str,
        table_names: Optional[List[str]] = None,
    ) -> None:
        if not os.path.exists(save_dir):
            raise RuntimeError(f"Save directory does not exist: {save_dir}")

        if (
            "get_loading_files" not in globals()
            or "_iter_batches_from_files" not in globals()
        ):
            raise RuntimeError(
                "dynamicemb load helpers are unavailable. Ensure dynamicemb and inference operators are importable."
            )

        if table_names is None:
            table_names = self.table_names_

        requested_table_names = set(table_names)
        dim = self.emb_dim_
        device = self.device
        weight = self.nve_embedding_.weight.data

        self.hash_table.table_storage_.zero_()
        self.hash_table.bucket_sizes.zero_()
        self.hash_table._ref_counter.zero_()
        weight.zero_()

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
                            f"nve_embedding has insufficient rows ({table_cap}) for loaded index {max_index}."
                        )

                    abs_indices = valid_indices + self.table_offsets_[table_id]
                    weight.index_copy_(
                        0,
                        abs_indices.to(torch.int64),
                        embeddings[valid_mask].to(weight.dtype),
                    )

    def forward(
        self,
        keys: torch.Tensor,
        offsets: torch.Tensor,
        pooling_offsets: Optional[torch.Tensor] = None,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run embedding lookup with optional pooling.

        Args:
            keys:            (N,) int64 – flat lookup keys; may span multiple
                             tables and pooling bags.
            offsets:         (T*B+1,) int64 – CSR boundaries that map segments
                             of ``keys`` to per-table feature slots; used to
                             derive a per-key table id via
                             ``INFERENCE_EMB::expand_table_ids``.
            pooling_offsets: (B+1,) int64 – CSR boundaries that map segments of
                             ``keys`` to pooling bags; required when
                             ``self.pooling_mode_ >= 0``, otherwise unused.

        Returns:
            - ``pooling_mode_ == -1``: ``(N, D)`` float tensor of per-key
              embeddings.
                        - ``pooling_mode_ == 1 or 2``: ``(B, D)`` float tensor of pooled
              embeddings, where ``B = pooling_offsets.size(0) - 1``.
        """
        # Derive per-key table id from the table-segment offsets.
        table_ids = torch.ops.INFERENCE_EMB.expand_table_ids(
            offsets,
            keys,
            self.feature_offsets_,
            self.num_tables_,
            1,
        )

        if self.use_dynamic_hash:
            # Hash-table lookup: keys → per-table row indices.
            _scores, _founds, table_indices = self.hash_table.lookup(
                keys=keys,
                table_ids=table_ids,
                score_value=None,
                score_policy=self.score_policy,
            )
        else:
            # Use identical-mapping per-table indices
            table_indices = keys

        # Convert per-table indices to absolute embedding row ids.
        global_table_offsets = torch.index_select(self.table_offsets_, 0, table_ids)
        global_indices = table_indices + global_table_offsets

        # self.pooling_mode_ is a plain Python int attribute – a compile-time
        # constant for torch.export.  The branch below is statically resolved
        # during tracing: only one code path is included in the exported graph.
        if self.pooling_mode_ < 0:
            # Non-pooled path: return one embedding vector per key.
            return self.nve_embedding_(global_indices)
        else:
            # Pooled path: reduce each bag of keys to one embedding vector.
            return self.nve_embedding_(
                global_indices,  # (N,) – absolute row ids
                pooling_offsets,  # (B+1,) – CSR bag boundaries
                per_sample_weights,  # optional per-key weights
            )


def create_inference_embedding_collection(
    embedding_configs: List[EmbeddingConfig],
    pooling_mode: int = -1,  # non pooling
    use_dynamic: bool = True,
) -> InferenceEmbeddingCollection:
    table_names = [config.name for config in embedding_configs]
    feature_names = list(
        itertools.chain(*[config.feature_names for config in embedding_configs])
    )
    feature_table_map = list(
        itertools.chain(
            *[
                [idx] * len(config.feature_names)
                for idx, config in enumerate(embedding_configs)
            ]
        )
    )
    table_options = [
        DynamicEmbTableOptions(
            embedding_dtype=torch.float32,
            dim=config.embedding_dim,
            max_capacity=config.num_embeddings,
            local_hbm_for_values=0,
            bucket_capacity=128,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL,
            ),
            training=False,
        )
        for config in embedding_configs
    ]
    return InferenceEmbeddingCollection(
        table_options,
        use_dynamic,
        pooling_mode,
        table_names,
        feature_names,
        feature_table_map,
        device=torch.device("cuda"),
    )


def apply_inference_embedding_collection(
    model: torch.nn.Module,
    dynamic_table_configs: Dict[str, bool],
    trained_emb_table_sizes: Dict[str, int],
):
    """
    Replace torchrec.EmbeddingCollection in the model with dynamicemb.InferenceEmbeddingCollection.

    Args:
        model (torch.nn.Module): The input training model.
        dynamic_table_configs (Dict[str, bool]): A dictionary mapping table names to their use_dynamic flag.
        trained_emb_table_sizes (Dict[str, int]): A dictionary mapping table names to their corresponding vocabulary sizes in the trained model.
    Returns:
        torch.nn.Module: The training model with the exportable embedding.
    """
    from torch.nn.modules.sparse import Embedding

    check_modules = set()
    while True:
        name = None
        for name, module in model.named_modules():
            if isinstance(module, ModuleDict):
                submodules = [
                    submodule
                    for name, submodule in module.named_modules()
                    if name != ""
                ]
                # skip if not torchrec.EmbeddingCollection
                if len({type(m) for m in submodules}) != 1 or not isinstance(
                    submodules[0], Embedding
                ):
                    continue
                # skip if already converted
                if (
                    isinstance(module, nve_layers.NVEmbedding)
                    or isinstance(module, nve_layers.NVEmbeddingBag)
                    or isinstance(module, InferenceEmbeddingCollection)
                ):
                    continue
                # skip if already checked
                if name in check_modules:
                    continue
                break
        else:
            break

        embedding_configs = None
        for parent_name, parent_module in model.named_modules():
            if parent_name == name.removesuffix(".embeddings"):
                embedding_configs = parent_module.embedding_configs()
                break
        assert (
            embedding_configs is not None
        ), f"Cannot find embedding configs from parent module {name.removesuffix('.embeddings')}"

        parent_name = name.removesuffix(".embeddings")
        check_modules.add(name)
        check_modules.add(parent_name)

        # Adjust training vocab sizes to inference vocab sizes based on trained_emb_table_sizes
        # TODO(junyiq): Try exact the freezed vocab size from embedding files/model states.
        for config in embedding_configs:
            if config.name not in trained_emb_table_sizes:
                print(
                    f"[WARNING] Table {config.name} in module {name} is missing the trained vocab size for inference.\n"
                    + f"          Using {config.vocab_size} rows from the training config."
                )
            config.num_embeddings = trained_emb_table_sizes.get(
                config.name, config.num_embeddings
            )

        use_dynamic = {
            config.name: dynamic_table_configs[config.name]
            for config in embedding_configs
            if config.name in dynamic_table_configs
        }
        assert (
            len(use_dynamic) > 0
        ), "At least one table in the embedding collection module should have a config in dynamic_table_configs."
        assert (
            len(set(use_dynamic.values())) == 1
        ), f"All tables in the same embedding collection module should have the same config in dynamic_table_configs. Got:\n{use_dynamic}"
        use_dynamic = list(use_dynamic.values())[0]

        pooling_config = getattr(embedding_configs[0], "pooling", "NONE")
        if pooling_config == "NONE":
            pooling_mode = -1
        elif pooling_config == "SUM":
            pooling_mode = 1
        elif pooling_config == "MEAN":
            pooling_mode = 2
        else:
            raise ValueError(f"Unsupported pooling config: {pooling_config}")

        embedding_collection = create_inference_embedding_collection(
            embedding_configs,
            pooling_mode,
            use_dynamic,
        )
        assert isinstance(embedding_collection, InferenceEmbeddingCollection)
        embedding_collection.embedding_configs = embedding_configs

        exec("model." + parent_name + "=embedding_collection")
        print(
            f"[INFO] converting {parent_name} to InferenceEmbeddingCollection with use_dynamic={use_dynamic} and tables={embedding_collection.table_names_}"
        )

    return model
