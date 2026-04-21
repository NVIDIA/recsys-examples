#!/usr/bin/env python3
# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
"""
HSTU Training Memory Estimator

Estimates GPU memory required for training one batch, divided into:
1. Parameter memory (Weights/Parameters)
2. Activation memory (Activations)
3. Optimizer states

Supports recompute options to reduce activation memory.

Usage:
    # Using gin-config file
    python estimate_memory.py --gin_config ../configs/h100_16gpu_exp0_baseline.gin
    
    # Using command-line arguments
    python estimate_memory.py --batch_size 32 --max_seq_len 4096 --hidden_size 1024 --num_layers 8
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Add hstu directory to sys.path for importing gin_config_args
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HSTU_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if HSTU_ROOT not in sys.path:
    sys.path.insert(0, HSTU_ROOT)


def _parse_gin_file_manually(
    gin_config_path: str,
) -> Tuple["HSTUModelConfig", "TrainingConfig", str]:
    """
    Manually parse gin config file (used when standard gin.parse_config_file fails).

    Extracts key parameters using regular expressions.
    """
    import re

    with open(gin_config_path, "r") as f:
        content = f.read()

    def extract_value(pattern, default=None, type_fn=str):
        match = re.search(pattern, content)
        if match:
            value = match.group(1).strip()
            # Handle string quotes
            if value.startswith("'") or value.startswith('"'):
                value = value[1:-1]
            try:
                return type_fn(value)
            except:
                return default
        return default

    def extract_list(pattern, default=None):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            list_str = match.group(1)
            # Simple list parsing
            items = re.findall(r"[\d]+", list_str)
            return [int(x) for x in items] if items else default
        return default

    # Extract NetworkArgs parameters
    hidden_size = extract_value(r"NetworkArgs\.hidden_size\s*=\s*(\d+)", 1024, int)
    num_layers = extract_value(r"NetworkArgs\.num_layers\s*=\s*(\d+)", 8, int)
    num_attention_heads = extract_value(
        r"NetworkArgs\.num_attention_heads\s*=\s*(\d+)", 4, int
    )
    kv_channels = extract_value(r"NetworkArgs\.kv_channels\s*=\s*(\d+)", 256, int)
    item_embedding_dim = extract_value(
        r"NetworkArgs\.item_embedding_dim\s*=\s*(\d+)", 128, int
    )
    contextual_embedding_dim = extract_value(
        r"NetworkArgs\.contextual_embedding_dim\s*=\s*(\d+)", 256, int
    )
    recompute_layernorm = (
        extract_value(
            r"NetworkArgs\.recompute_input_layernorm\s*=\s*(True|False)", "False"
        )
        == "True"
    )
    recompute_silu = (
        extract_value(r"NetworkArgs\.recompute_input_silu\s*=\s*(True|False)", "False")
        == "True"
    )
    dtype_str = extract_value(
        r"NetworkArgs\.dtype_str\s*=\s*['\"]([^'\"]+)['\"]", "bfloat16"
    )

    # Extract TrainerArgs parameters
    batch_size = extract_value(r"TrainerArgs\.train_batch_size\s*=\s*(\d+)", 32, int)

    # Extract max_sequence_length
    max_seq_len = extract_value(
        r"FeatureArgs\.max_sequence_length\s*=\s*(\d+)", 4096, int
    )

    # Extract RankingArgs parameters
    ranking_arch = extract_list(
        r"RankingArgs\.prediction_head_arch\s*=\s*\[([^\]]+)\]", [512, 8]
    )
    num_tasks = extract_value(r"RankingArgs\.num_tasks\s*=\s*(\d+)", 8, int)

    # Extract embedding info
    # Need to distinguish item/action embeddings (dim=item_embedding_dim)
    # from other contextual embeddings (dim=contextual_embedding_dim)
    #
    # Item embedding: DynamicEmbeddingArgs or EmbeddingArgs with table_name "item" or "action"
    # Contextual embedding: all other embedding tables

    # Parse all embedding config blocks
    # Format: xxx_emb/DynamicEmbeddingArgs.table_name = 'xxx'
    #         xxx_emb/DynamicEmbeddingArgs.item_vocab_size_or_capacity = N

    # Find all embedding blocks' table_name and vocab_size
    item_embedding_rows = 0
    contextual_embedding_rows = 0
    num_embedding_tables = 0

    # Find all DynamicEmbeddingArgs configs
    # Pattern: xxx/DynamicEmbeddingArgs.table_name = 'yyy'
    table_patterns = re.findall(
        r"(\w+)/DynamicEmbeddingArgs\.table_name\s*=\s*['\"](\w+)['\"]", content
    )

    for prefix, table_name in table_patterns:
        # Find corresponding vocab_size
        vocab_pattern = (
            rf"{prefix}/DynamicEmbeddingArgs\.item_vocab_size_or_capacity\s*=\s*(\d+)"
        )
        vocab_match = re.search(vocab_pattern, content)
        if vocab_match:
            vocab_size = int(vocab_match.group(1))
            num_embedding_tables += 1
            # 'item' and 'action' use item_embedding_dim
            if table_name in ["item", "action"]:
                item_embedding_rows += vocab_size
            else:
                contextual_embedding_rows += vocab_size

    # Find all EmbeddingArgs configs (non-Dynamic, e.g., action)
    static_table_patterns = re.findall(
        r"(\w+)/EmbeddingArgs\.table_name\s*=\s*['\"](\w+)['\"]", content
    )

    for prefix, table_name in static_table_patterns:
        # Find corresponding vocab_size
        vocab_pattern = (
            rf"{prefix}/EmbeddingArgs\.item_vocab_size_or_capacity\s*=\s*(\d+)"
        )
        vocab_match = re.search(vocab_pattern, content)
        if vocab_match:
            vocab_size = int(vocab_match.group(1))
            num_embedding_tables += 1
            # 'item' and 'action' use item_embedding_dim
            if table_name in ["item", "action"]:
                item_embedding_rows += vocab_size
            else:
                contextual_embedding_rows += vocab_size

    total_embedding_rows = item_embedding_rows + contextual_embedding_rows

    # If no embeddings were parsed, use default values
    if total_embedding_rows == 0:
        item_embedding_rows = 50_000_000
        contextual_embedding_rows = 200_000_000
        total_embedding_rows = 250_000_000
        num_embedding_tables = 50

    # Extract GPU cache ratio
    gpu_cache_ratio = extract_value(
        r"DynamicEmbeddingArgs\.item_vocab_gpu_capacity_ratio\s*=\s*([\d.]+)",
        1.0,
        float,
    )

    # Extract optimizer type
    optimizer = extract_value(
        r"OptimizerArgs\.optimizer_str\s*=\s*['\"]([^'\"]+)['\"]", "adam"
    )

    # Extract number of contextual features
    contextual_matches = re.findall(
        r"contextual_feature_names\s*=\s*\[([^\]]+)\]", content, re.DOTALL
    )
    num_contextual = 0
    if contextual_matches:
        # Estimate feature count by counting commas
        for match in contextual_matches:
            num_contextual = max(
                num_contextual, match.count(",") + 1 if match.strip() else 0
            )

    model_config = HSTUModelConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        item_embedding_dim=item_embedding_dim if item_embedding_dim > 0 else 128,
        contextual_embedding_dim=contextual_embedding_dim
        if contextual_embedding_dim > 0
        else 256,
        num_embedding_tables=num_embedding_tables,
        total_embedding_rows=total_embedding_rows,
        item_embedding_rows=item_embedding_rows,
        contextual_embedding_rows=contextual_embedding_rows,
        embedding_gpu_cache_ratio=gpu_cache_ratio,
        ranking_head_arch=ranking_arch,
        num_tasks=num_tasks,
        recompute_input_layernorm=recompute_layernorm,
        recompute_input_silu=recompute_silu,
        dtype="bf16" if dtype_str == "bfloat16" else "fp16",
    )

    train_config = TrainingConfig(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        avg_seq_len=max_seq_len // 2,
        num_contextual_features=num_contextual,
    )

    return model_config, train_config, optimizer


def load_config_from_gin(
    gin_config_path: str,
) -> Tuple["HSTUModelConfig", "TrainingConfig", str]:
    """
    Load configuration from gin-config file.

    Args:
        gin_config_path: Path to gin config file.

    Returns:
        Tuple of (HSTUModelConfig, TrainingConfig, optimizer).
    """
    # Try importing gin
    try:
        import gin
    except ImportError:
        print("❌ Error: 'gin-config' package is not installed.")
        print("   Please install it with: pip install gin-config")
        sys.exit(1)

    # Try importing gin_config_args
    try:
        from utils.gin_config_args import (
            BenchmarkDatasetArgs,
            DynamicEmbeddingArgs,
            NetworkArgs,
            OptimizerArgs,
            RankingArgs,
            TrainerArgs,
        )
    except ImportError as e:
        print(f"❌ Error importing gin_config_args: {e}")
        print("   Please ensure you are running from the correct directory:")
        print("   cd examples/hstu/training/benchmark && python estimate_memory.py ...")
        print("   Or set PYTHONPATH to include the hstu directory")
        sys.exit(1)

    # Clear previous gin config
    gin.clear_config()

    # Parse gin config file
    # Note: Some config params may not be defined in gin_config_args, need to skip them
    try:
        # Try using skip_unknown parameter
        gin.parse_config_file(gin_config_path, skip_unknown=True)
    except TypeError:
        # Older versions of gin-config may not support skip_unknown
        try:
            gin.parse_config_file(gin_config_path)
        except ValueError:
            # If there are unknown params, fall back to manual parsing
            print(
                f"⚠️  Warning: Some gin config parameters are not recognized, parsing manually..."
            )
            model_config, train_config, optimizer = _parse_gin_file_manually(
                gin_config_path
            )
            return model_config, train_config, optimizer
    except ValueError as e:
        # If there are unknown params, try manual parsing
        print(f"⚠️  Warning: {e}")
        print("   Falling back to manual parsing...")
        model_config, train_config, optimizer = _parse_gin_file_manually(
            gin_config_path
        )
        return model_config, train_config, optimizer
    except Exception as e:
        print(f"❌ Error parsing gin config file: {e}")
        sys.exit(1)

    # Get each config object
    try:
        trainer_args = TrainerArgs()
    except Exception:
        trainer_args = None

    try:
        network_args = NetworkArgs()
    except Exception:
        network_args = None

    try:
        ranking_args = RankingArgs()
    except Exception:
        ranking_args = None

    try:
        benchmark_dataset_args = BenchmarkDatasetArgs()
    except Exception:
        benchmark_dataset_args = None

    try:
        optimizer_args = OptimizerArgs()
    except Exception:
        optimizer_args = None

    # Compute embedding-related parameters
    # Distinguish item/action embedding (uses item_embedding_dim) from contextual embedding (uses contextual_embedding_dim)
    total_embedding_rows = 0
    item_embedding_rows = 0
    contextual_embedding_rows = 0
    num_embedding_tables = 0
    embedding_gpu_cache_ratio = 1.0

    if benchmark_dataset_args is not None:
        for emb_arg in benchmark_dataset_args.embedding_args:
            vocab_size = emb_arg.item_vocab_size_or_capacity
            table_name = emb_arg.table_name
            total_embedding_rows += vocab_size
            num_embedding_tables += 1

            # 'item' and 'action' use item_embedding_dim
            if table_name in ["item", "action"]:
                item_embedding_rows += vocab_size
            else:
                contextual_embedding_rows += vocab_size

            # Get GPU cache ratio (use config from the first DynamicEmbeddingArgs)
            if isinstance(emb_arg, DynamicEmbeddingArgs):
                if emb_arg.item_vocab_gpu_capacity_ratio is not None:
                    embedding_gpu_cache_ratio = emb_arg.item_vocab_gpu_capacity_ratio

    # Get max_sequence_length
    max_seq_len = 4096  # Default value
    if benchmark_dataset_args is not None and benchmark_dataset_args.feature_args:
        for feat_arg in benchmark_dataset_args.feature_args:
            if feat_arg.is_jagged and feat_arg.max_sequence_length > max_seq_len:
                max_seq_len = feat_arg.max_sequence_length
            elif feat_arg.is_jagged:
                max_seq_len = max(max_seq_len, feat_arg.max_sequence_length)

    # Build HSTUModelConfig
    model_config = HSTUModelConfig(
        hidden_size=network_args.hidden_size if network_args else 1024,
        num_layers=network_args.num_layers if network_args else 8,
        num_attention_heads=network_args.num_attention_heads if network_args else 4,
        kv_channels=network_args.kv_channels if network_args else 256,
        item_embedding_dim=network_args.item_embedding_dim
        if network_args and network_args.item_embedding_dim > 0
        else 128,
        contextual_embedding_dim=network_args.contextual_embedding_dim
        if network_args and network_args.contextual_embedding_dim > 0
        else 256,
        num_embedding_tables=num_embedding_tables,
        total_embedding_rows=total_embedding_rows,
        item_embedding_rows=item_embedding_rows,
        contextual_embedding_rows=contextual_embedding_rows,
        embedding_gpu_cache_ratio=embedding_gpu_cache_ratio,
        ranking_head_arch=ranking_args.prediction_head_arch
        if ranking_args
        else [512, 8],
        num_tasks=ranking_args.num_tasks if ranking_args else 8,
        recompute_input_layernorm=network_args.recompute_input_layernorm
        if network_args
        else False,
        recompute_input_silu=network_args.recompute_input_silu
        if network_args
        else False,
        dtype="bf16"
        if (network_args and network_args.dtype_str == "bfloat16")
        else "fp16",
    )

    # Get number of contextual features
    num_contextual_features = 0
    if benchmark_dataset_args is not None:
        num_contextual_features = len(benchmark_dataset_args.contextual_feature_names)

    # Build TrainingConfig
    train_config = TrainingConfig(
        batch_size=trainer_args.train_batch_size if trainer_args else 32,
        max_seq_len=max_seq_len,
        avg_seq_len=max_seq_len // 2,  # Estimated average
        num_contextual_features=num_contextual_features,
    )

    # Get optimizer type
    optimizer = "adam"
    if optimizer_args is not None:
        optimizer = optimizer_args.optimizer_str

    return model_config, train_config, optimizer


@dataclass
class HSTUModelConfig:
    """HSTU model configuration."""

    # Basic parameters
    hidden_size: int = 1024
    num_layers: int = 8
    num_attention_heads: int = 4
    kv_channels: int = 256  # Per-head dimension

    # Embedding parameters
    item_embedding_dim: int = 128
    contextual_embedding_dim: int = 256
    num_embedding_tables: int = 50
    total_embedding_rows: int = (
        250_000_000  # Total embedding rows (deprecated, use the next two fields)
    )
    item_embedding_rows: int = (
        50_000_000  # Total rows for item + action (uses item_embedding_dim)
    )
    contextual_embedding_rows: int = 200_000_000  # Total rows for other contextual features (uses contextual_embedding_dim)
    embedding_gpu_cache_ratio: float = 1.0  # GPU cache ratio (1.0 = all on GPU)

    # Ranking head
    ranking_head_arch: List[int] = field(default_factory=lambda: [512, 8])
    num_tasks: int = 8

    # Recompute options
    recompute_input_layernorm: bool = False
    recompute_input_silu: bool = False

    # Data type
    dtype: str = "bf16"  # bf16, fp16, fp32

    def bytes_per_element(self) -> int:
        """Returns bytes per element based on dtype."""
        if self.dtype in ["bf16", "fp16"]:
            return 2
        return 4


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 32
    max_seq_len: int = 4096
    avg_seq_len: int = 2048  # Average sequence length (for variable-length sequences)
    num_contextual_features: int = 48

    # Parallelism configuration
    tensor_parallel_size: int = 1
    data_parallel_size: int = 16
    sequence_parallel: bool = False


def estimate_hstu_layer_weights(config: HSTUModelConfig) -> Dict[str, int]:
    """
    Estimate parameter memory for one HSTU Layer.

    An HSTU Layer includes:
    1. Input LayerNorm: weight + bias
    2. Linear UVQK: weight + bias (hidden -> 4 * num_heads * kv_channels)
    3. Output LayerNorm: weight + bias
    4. Linear Proj: weight + bias (num_heads * kv_channels -> hidden)
    """
    H = config.hidden_size
    N = config.num_attention_heads
    D = config.kv_channels
    bytes_per_elem = config.bytes_per_element()

    # UVQK projection: hidden_size -> 4 * num_heads * kv_channels
    # U, V, Q, K each have one head dimension
    uvqk_out_dim = 4 * N * D
    linear_uvqk_weight = H * uvqk_out_dim * bytes_per_elem
    linear_uvqk_bias = uvqk_out_dim * bytes_per_elem

    # Output projection: num_heads * kv_channels -> hidden_size
    linear_proj_weight = N * D * H * bytes_per_elem
    linear_proj_bias = H * bytes_per_elem

    # LayerNorm: 2 * hidden_size (weight + bias) x 2 (input + output)
    layernorm_params = 2 * H * bytes_per_elem * 2  # Input + output LayerNorm

    return {
        "linear_uvqk_weight": linear_uvqk_weight,
        "linear_uvqk_bias": linear_uvqk_bias,
        "linear_proj_weight": linear_proj_weight,
        "linear_proj_bias": linear_proj_bias,
        "layernorm": layernorm_params,
        "total": linear_uvqk_weight
        + linear_uvqk_bias
        + linear_proj_weight
        + linear_proj_bias
        + layernorm_params,
    }


def estimate_embedding_weights(config: HSTUModelConfig) -> Dict[str, Any]:
    """
    Estimate embedding tables parameter memory.

    ⚠️ Note: Embeddings are always stored as fp32 (regardless of model dtype).

    Calculation:
    - Item embedding (item + action): item_embedding_rows x item_embedding_dim x 4 bytes
    - Contextual embedding (all others): contextual_embedding_rows x contextual_embedding_dim x 4 bytes

    DynamicEmb supports CPU-GPU hybrid storage:
    - Only the cache portion is stored on GPU
    - If gpu_cache_ratio = 1.0, all embeddings are on GPU
    """
    # Embeddings are always fp32 (4 bytes per element)
    bytes_per_elem = 4  # fp32

    # Item embedding (item + action): uses item_embedding_dim
    item_embedding = (
        config.item_embedding_rows * config.item_embedding_dim * bytes_per_elem
    )

    # Contextual embeddings (all other features): uses contextual_embedding_dim
    contextual_embedding = (
        config.contextual_embedding_rows
        * config.contextual_embedding_dim
        * bytes_per_elem
    )

    total_embedding = item_embedding + contextual_embedding

    # Apply GPU cache ratio
    gpu_embedding = int(total_embedding * config.embedding_gpu_cache_ratio)
    cpu_embedding = total_embedding - gpu_embedding

    return {
        "item_embedding_rows": config.item_embedding_rows,
        "contextual_embedding_rows": config.contextual_embedding_rows,
        "item_embedding_total": item_embedding,
        "contextual_embedding_total": contextual_embedding,
        "total_embedding": total_embedding,
        "gpu_embedding": gpu_embedding,
        "cpu_embedding": cpu_embedding,
        "dtype": "fp32",  # Embeddings are always fp32
    }


def estimate_ranking_head_weights(config: HSTUModelConfig) -> Dict[str, Any]:
    """Estimate memory of ranking head parameters."""
    bytes_per_elem = config.bytes_per_element()

    # MLP layers
    layers = [config.hidden_size] + config.ranking_head_arch
    total = 0
    details = {}

    for i in range(len(layers) - 1):
        in_dim = layers[i]
        out_dim = layers[i + 1]
        weight = in_dim * out_dim * bytes_per_elem
        bias = out_dim * bytes_per_elem
        details[f"mlp_layer_{i}"] = weight + bias
        total += weight + bias

    # Final layer: arch[-1] -> num_tasks
    final_weight = config.ranking_head_arch[-1] * config.num_tasks * bytes_per_elem
    final_bias = config.num_tasks * bytes_per_elem
    details["final_layer"] = final_weight + final_bias
    total += final_weight + final_bias

    return {
        "details": details,
        "total": total,
    }


def estimate_optimizer_states(
    dense_params_bytes: int,
    embedding_params_bytes: int,
    optimizer: str = "adam",
    dtype: str = "bf16",
) -> Dict[str, Any]:
    """
    Estimate memory for optimizer states.

    ⚠️ Key distinction:
    - Dense part (HSTU layers + ranking head): bf16 + fp32 master weights
    - Embedding part: always fp32, no master weights needed

    Memory layout during training:
    1. Dense weights (bf16): dense_params_bytes
    2. Dense Master Weights (fp32): separate fp32 copy for accurate gradient updates
    3. Embedding weights (fp32): embedding_params_bytes (no master weights needed)
    4. Optimizer States (fp32): Adam m + v for all parameters

    Adam: master_weights (dense only) + 2x all_params (momentum + variance)
    """
    # Dense part: if bf16/fp16, need fp32 master weights
    if dtype in ["bf16", "fp16"]:
        # bf16/fp16 to fp32: double the size
        dense_fp32_bytes = dense_params_bytes * 2
    else:
        dense_fp32_bytes = dense_params_bytes

    # Master weights: only dense part needs them (embedding is already fp32)
    master_weights = dense_fp32_bytes

    # Optimizer states: needed for all parameters (dense + embedding)
    # Embeddings are already fp32, dense optimizer states also use fp32
    all_params_fp32 = (
        dense_fp32_bytes + embedding_params_bytes
    )  # Embedding is already fp32

    if optimizer.lower() in ["adam", "adamw"]:
        # Adam/AdamW: m (momentum) + v (variance), both fp32
        optimizer_states = all_params_fp32 * 2  # m + v
    elif optimizer.lower() == "sgd":
        # SGD with momentum: 1x fp32 for momentum
        optimizer_states = all_params_fp32
    else:
        optimizer_states = 0

    return {
        "optimizer": optimizer,
        "master_weights": master_weights,  # fp32 master weights for dense part only
        "optimizer_states": optimizer_states,  # Adam m + v for all params
        "total": master_weights + optimizer_states,
    }


def estimate_forward_activations(
    model_config: HSTUModelConfig,
    train_config: TrainingConfig,
) -> Dict[str, int]:
    """
    Estimate memory for forward-pass activations.

    ⚠️ Note: Uses max_seq_len because memory must be allocated for the worst case.

    HSTU Layer Forward:
    1. Input: (total_tokens, hidden_size)
    2. After LayerNorm: (total_tokens, hidden_size) - can be recomputed
    3. After UVQK projection: (total_tokens, 4 * num_heads * kv_channels)
    4. After SiLU: (total_tokens, 4 * num_heads * kv_channels) - can be recomputed
    5. Attention: (num_heads, total_tokens, max_seq_len) - O(n^2)
    6. After Attention: (total_tokens, num_heads * kv_channels)
    7. Output: (total_tokens, hidden_size)
    """
    B = train_config.batch_size
    S_max = train_config.max_seq_len  # ⚠️ Use max_seq_len
    H = model_config.hidden_size
    N = model_config.num_attention_heads
    D = model_config.kv_channels
    L = model_config.num_layers
    bytes_per_elem = model_config.bytes_per_element()

    # Compute total token count with max_seq_len (worst case)
    total_tokens = B * S_max

    activations = {}

    # Activations per layer
    # 1. Input (must be stored for residual connection)
    input_activation = total_tokens * H * bytes_per_elem
    activations["input_per_layer"] = input_activation

    # 2. LayerNorm output (can be recomputed)
    if not model_config.recompute_input_layernorm:
        layernorm_output = total_tokens * H * bytes_per_elem
    else:
        layernorm_output = 0  # Not stored when recompute is enabled
    activations["layernorm_output_per_layer"] = layernorm_output

    # 3. UVQK projection output
    uvqk_output = total_tokens * 4 * N * D * bytes_per_elem
    activations["uvqk_output_per_layer"] = uvqk_output

    # 4. SiLU output (can be recomputed)
    if not model_config.recompute_input_silu:
        silu_output = total_tokens * 4 * N * D * bytes_per_elem
    else:
        silu_output = 0  # Not stored when recompute is enabled
    activations["silu_output_per_layer"] = silu_output

    # 5. Attention scores (largest part!)
    # HSTU attention: softmax(Q @ K^T / sqrt(d)) @ V
    # Attention scores shape: (batch, num_heads, seq_len, seq_len) -> O(n^2)
    # ⚠️ Use max_seq_len^2 since memory must be allocated for the longest sequence
    attention_scores = B * N * S_max * S_max * bytes_per_elem
    activations["attention_scores_per_layer"] = attention_scores

    # 6. Attention output
    attention_output = total_tokens * N * D * bytes_per_elem
    activations["attention_output_per_layer"] = attention_output

    # 7. Projection output
    proj_output = total_tokens * H * bytes_per_elem
    activations["proj_output_per_layer"] = proj_output

    # Total activations per layer
    per_layer_total = (
        input_activation
        + layernorm_output
        + uvqk_output
        + silu_output
        + attention_scores
        + attention_output
        + proj_output
    )
    activations["per_layer_total"] = per_layer_total

    # Total activations for all layers
    # Note: Due to backward pass requirements, activations from all layers are stored
    # (unless using gradient checkpointing)
    activations["all_layers_total"] = per_layer_total * L

    # Embedding activations
    # Item embedding lookup: (total_tokens, item_embedding_dim)
    item_emb_activation = (
        total_tokens * model_config.item_embedding_dim * bytes_per_elem
    )
    # Contextual embeddings: (batch, num_contextual, contextual_embedding_dim)
    contextual_emb_activation = (
        B
        * train_config.num_contextual_features
        * model_config.contextual_embedding_dim
        * bytes_per_elem
    )
    activations["embedding_activations"] = (
        item_emb_activation + contextual_emb_activation
    )

    # Ranking head activations
    ranking_activation = 0
    for dim in model_config.ranking_head_arch:
        ranking_activation += total_tokens * dim * bytes_per_elem
    activations["ranking_head_activations"] = ranking_activation

    # Total activations
    activations["total"] = (
        activations["all_layers_total"]
        + activations["embedding_activations"]
        + activations["ranking_head_activations"]
    )

    return activations


def estimate_backward_gradients(
    model_config: HSTUModelConfig,
    train_config: TrainingConfig,
) -> Dict[str, Any]:
    """
    Estimate memory for backward gradients.

    ⚠️ Note:
    - Dense parts (HSTU layers + ranking head) use fp32 gradient buffer
    - Embedding part does not need a separate gradient buffer (DynamicEmb handles it internally)

    Gradient buffer size = total dense params x 4 bytes (fp32)
    """
    # Gradients for dense parts (HSTU layers + ranking head)
    # Parameters are bf16, but gradients are stored in fp32
    layer_weights = estimate_hstu_layer_weights(model_config)
    layer_params_bytes_bf16 = layer_weights["total"] * model_config.num_layers
    # bf16 -> fp32: double the size
    layer_weight_grads_fp32 = (
        layer_params_bytes_bf16 * 2
        if model_config.dtype in ["bf16", "fp16"]
        else layer_params_bytes_bf16
    )

    ranking_weights = estimate_ranking_head_weights(model_config)
    ranking_params_bytes_bf16 = ranking_weights["total"]
    ranking_weight_grads_fp32 = (
        ranking_params_bytes_bf16 * 2
        if model_config.dtype in ["bf16", "fp16"]
        else ranking_params_bytes_bf16
    )

    total_dense_grads = layer_weight_grads_fp32 + ranking_weight_grads_fp32

    return {
        "layer_weight_grads": layer_weight_grads_fp32,
        "ranking_weight_grads": ranking_weight_grads_fp32,
        "total": total_dense_grads,  # Dense part only
        "dtype": "fp32",
    }


def format_bytes(num_bytes: int) -> str:
    """Format bytes to readable string"""
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def estimate_total_memory(
    model_config: HSTUModelConfig,
    train_config: TrainingConfig,
    optimizer: str = "adam",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Estimate total memory requirement

    Returns:
        Dict with breakdown of memory usage
    """
    results = {
        "model_config": {
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_layers,
            "num_attention_heads": model_config.num_attention_heads,
            "kv_channels": model_config.kv_channels,
            "dtype": model_config.dtype,
            "recompute_input_layernorm": model_config.recompute_input_layernorm,
            "recompute_input_silu": model_config.recompute_input_silu,
        },
        "training_config": {
            "batch_size": train_config.batch_size,
            "max_seq_len": train_config.max_seq_len,
            "avg_seq_len": train_config.avg_seq_len,
            "tensor_parallel_size": train_config.tensor_parallel_size,
        },
    }

    # 1. Parameter memory
    layer_weights = estimate_hstu_layer_weights(model_config)
    total_layer_weights = layer_weights["total"] * model_config.num_layers

    embedding_weights = estimate_embedding_weights(model_config)
    gpu_embedding_weights = embedding_weights["gpu_embedding"]

    ranking_weights = estimate_ranking_head_weights(model_config)
    total_ranking_weights = ranking_weights["total"]

    # Dense part (HSTU layers + ranking head): bf16
    # Embedding part: fp32
    total_dense_weights = total_layer_weights + total_ranking_weights
    total_weights = total_dense_weights + gpu_embedding_weights

    results["weights"] = {
        "hstu_layers": total_layer_weights,
        "hstu_layers_dtype": model_config.dtype,
        "embeddings_gpu": gpu_embedding_weights,
        "embeddings_gpu_dtype": "fp32",  # Embedding always fp32
        "embeddings_cpu": embedding_weights["cpu_embedding"],
        "ranking_head": total_ranking_weights,
        "ranking_head_dtype": model_config.dtype,
        "total_dense": total_dense_weights,
        "total_gpu": total_weights,
    }

    # Embedding details (for report)
    results["embedding_details"] = {
        "item_rows": embedding_weights.get(
            "item_embedding_rows", model_config.item_embedding_rows
        ),
        "contextual_rows": embedding_weights.get(
            "contextual_embedding_rows", model_config.contextual_embedding_rows
        ),
        "item_total": embedding_weights["item_embedding_total"],
        "contextual_total": embedding_weights["contextual_embedding_total"],
        "total": embedding_weights["total_embedding"],
    }

    # 2. Optimizer states (includes master weights + optimizer states)
    # Master weights: needed by dense only (embedding is always fp32)
    opt_states = estimate_optimizer_states(
        dense_params_bytes=total_dense_weights,
        embedding_params_bytes=gpu_embedding_weights,
        optimizer=optimizer,
        dtype=model_config.dtype,
    )
    results["optimizer_states"] = opt_states

    # 3. Activation memory
    activations = estimate_forward_activations(model_config, train_config)
    results["activations"] = {
        "per_layer": activations["per_layer_total"],
        "all_layers": activations["all_layers_total"],
        "embedding": activations["embedding_activations"],
        "ranking_head": activations["ranking_head_activations"],
        "attention_scores_per_layer": activations["attention_scores_per_layer"],
        "total": activations["total"],
    }

    # 4. Gradients
    gradients = estimate_backward_gradients(model_config, train_config)
    results["gradients"] = gradients

    # 5. Total
    # Memory composition:
    #   1. Model parameters (bf16/fp16): total_weights
    #   2. Master Weights (fp32): opt_states["master_weights"]
    #   3. Optimizer States (fp32): opt_states["optimizer_states"]
    #   4. Activations (bf16/fp16): activations["total"]
    #   5. Gradients (bf16/fp16): gradients["total"]
    total_gpu_memory = (
        total_weights
        + opt_states["master_weights"]  # model params (bf16)
        + opt_states["optimizer_states"]  # master weights (fp32)
        + activations["total"]  # optimizer states (fp32)
        + gradients["total"]  # activations  # gradients
    )

    results["total"] = {
        "weights": total_weights,
        "master_weights": opt_states["master_weights"],
        "optimizer_states": opt_states["optimizer_states"],
        "optimizer_total": opt_states["total"],
        "activations": activations["total"],
        "gradients": gradients["total"],
        "total_gpu_memory": total_gpu_memory,
    }

    if verbose:
        print_memory_report(results)

    return results


def print_memory_report(results: Dict):
    """Print memory report"""
    print("\n" + "=" * 80)
    print("HSTU Training Memory Estimation Report")
    print("=" * 80)

    # Config info
    print("\n📋 Model Configuration:")
    mc = results["model_config"]
    print(f"   Hidden Size: {mc['hidden_size']}")
    print(f"   Num Layers: {mc['num_layers']}")
    print(f"   Num Attention Heads: {mc['num_attention_heads']}")
    print(f"   KV Channels (per head): {mc['kv_channels']}")
    print(f"   Dtype: {mc['dtype']}")
    print(f"   Recompute LayerNorm: {mc['recompute_input_layernorm']}")
    print(f"   Recompute SiLU: {mc['recompute_input_silu']}")

    print("\n📋 Training Configuration:")
    tc = results["training_config"]
    print(f"   Batch Size: {tc['batch_size']}")
    print(f"   Max Sequence Length: {tc['max_seq_len']}")
    print(f"   Avg Sequence Length: {tc['avg_seq_len']}")
    print(f"   Tensor Parallel Size: {tc['tensor_parallel_size']}")

    # Parameters
    print("\n" + "-" * 80)
    print("💾 Weights Memory (Model Parameters)")
    print("-" * 80)
    w = results["weights"]
    print(
        f"   HSTU Layers ({w['hstu_layers_dtype']}):     {format_bytes(w['hstu_layers']):>12}"
    )
    print(
        f"   Ranking Head ({w['ranking_head_dtype']}):    {format_bytes(w['ranking_head']):>12}"
    )
    print(f"   ─────────────────────────────────────")
    print(f"   Dense Total ({mc['dtype']}):       {format_bytes(w['total_dense']):>12}")
    print(f"")
    emb = results.get("embedding_details", {})
    if emb:
        print(f"   📦 Embedding Breakdown (fp32):")
        print(
            f"      Item+Action ({emb.get('item_rows', 'N/A'):,} rows × dim):  {format_bytes(emb.get('item_total', 0)):>12}"
        )
        print(
            f"      Contextual  ({emb.get('contextual_rows', 'N/A'):,} rows × dim):  {format_bytes(emb.get('contextual_total', 0)):>12}"
        )
    print(
        f"   Embeddings GPU (fp32):   {format_bytes(w['embeddings_gpu']):>12}  ← Always fp32"
    )
    print(
        f"   Embeddings CPU (fp32):   {format_bytes(w['embeddings_cpu']):>12}  (not counted in GPU)"
    )
    print(f"   ─────────────────────────────────────")
    print(f"   Total Weights (GPU):     {format_bytes(w['total_gpu']):>12}")

    # Optimizer states
    print("\n" + "-" * 80)
    print("⚙️  Optimizer States (Optimizer States + Master Weights)")
    print("-" * 80)
    opt = results["optimizer_states"]
    w = results["weights"]
    print(f"   Optimizer: {opt['optimizer']}")
    print(
        f"   Master Weights (FP32):    {format_bytes(opt['master_weights']):>12}  ← Dense part only ({mc['dtype']}→fp32)"
    )
    print(
        f"                             (Embeddings always fp32, no master weights needed)"
    )
    print(
        f"   Optimizer States (FP32):  {format_bytes(opt['optimizer_states']):>12}  ← Adam m+v (all params)"
    )
    print(f"   ─────────────────────────────────────")
    print(f"   Total Optimizer:          {format_bytes(opt['total']):>12}")

    # Activations
    print("\n" + "-" * 80)
    print("🔥 Activations Memory (Forward Pass Activations)")
    print("-" * 80)
    act = results["activations"]
    print(f"   Per Layer Activation:       {format_bytes(act['per_layer']):>12}")
    print(
        f"     - Attention Scores:       {format_bytes(act['attention_scores_per_layer']):>12} (O(n²) ⚠️)"
    )
    print(
        f"   All Layers ({mc['num_layers']} layers):      {format_bytes(act['all_layers']):>12}"
    )
    print(f"   Embedding Activations:      {format_bytes(act['embedding']):>12}")
    print(f"   Ranking Head Activations:   {format_bytes(act['ranking_head']):>12}")
    print(f"   ─────────────────────────────────────")
    print(f"   Total Activations:          {format_bytes(act['total']):>12}")

    # Gradients
    print("\n" + "-" * 80)
    print("📐 Gradients Memory (Dense Gradient Buffer - fp32)")
    print("-" * 80)
    grad = results["gradients"]
    print(f"   ⚠️  Dense part uses fp32 gradient buffer (for precise gradient update)")
    print(f"   ⚠️  Embedding part does not need a separate gradient buffer")
    print(f"")
    print(f"   HSTU Layers (fp32):     {format_bytes(grad['layer_weight_grads']):>12}")
    print(
        f"   Ranking Head (fp32):    {format_bytes(grad['ranking_weight_grads']):>12}"
    )
    print(f"   ─────────────────────────────────────")
    print(f"   Total Gradients (fp32): {format_bytes(grad['total']):>12}")

    # Total
    print("\n" + "=" * 80)
    print("📊 TOTAL GPU MEMORY SUMMARY")
    print("=" * 80)
    total = results["total"]
    print(f"\n   ┌───────────────────────────────────────────────────────┐")
    print(
        f"   │  Weights (Model Params bf16):      {format_bytes(total['weights']):>12}        │"
    )
    print(
        f"   │  Master Weights (fp32 copy):    {format_bytes(total['master_weights']):>12}        │"
    )
    print(
        f"   │  Optimizer States (Adam m+v):  {format_bytes(total['optimizer_states']):>12}        │"
    )
    print(
        f"   │  Activations (Forward):         {format_bytes(total['activations']):>12}        │"
    )
    print(
        f"   │  Gradients (Backprop):             {format_bytes(total['gradients']):>12}        │"
    )
    print(f"   ├───────────────────────────────────────────────────────┤")
    print(
        f"   │  TOTAL GPU MEMORY:             {format_bytes(total['total_gpu_memory']):>12}        │"
    )
    print(f"   └───────────────────────────────────────────────────────┘")

    # Recommendations
    print("\n" + "-" * 80)
    print("💡 Recommendations")
    print("-" * 80)

    total_gb = total["total_gpu_memory"] / (1024**3)
    act_gb = total["activations"] / (1024**3)
    total["optimizer_total"] / (1024**3)

    if total_gb > 80:
        print(f"   ⚠️  Total memory ({total_gb:.1f} GB) exceeds H100 80GB!")
        print("   Suggestions:")
        print("   - Reduce batch_size or avg_seq_len")
        print("   - Enable recompute_input_layernorm and recompute_input_silu")
        print("   - Reduce embedding_gpu_cache_ratio (use more CPU storage)")
        print("   - Enable Tensor Parallelism (TP > 1)")
    elif total_gb > 70:
        print(f"   ⚠️  Memory usage ({total_gb:.1f} GB) is close to H100 limit!")
        print("   Consider enabling recompute options for safety margin.")
    else:
        print(f"   ✅ Memory usage ({total_gb:.1f} GB) fits within H100 80GB")

    if act_gb > total_gb * 0.5:
        print(
            f"\n   📝 Activations ({act_gb:.1f} GB) dominate memory usage ({act_gb/total_gb*100:.1f}%)"
        )
        print("      This is typical for training with large batch/sequence.")
        if not mc["recompute_input_layernorm"]:
            print("      → Enable recompute_input_layernorm to save ~10% activations")
        if not mc["recompute_input_silu"]:
            print("      → Enable recompute_input_silu to save ~15% activations")

    print("\n" + "=" * 80)


def run_parameter_sweep(
    model_config: HSTUModelConfig,
    batch_sizes: List[int],
    seq_lens: List[int],
    target_gpu_memory_gb: float = 80.0,
) -> List[Dict]:
    """
    Parameter sweep to find configuration sets that meet memory requirement
    """
    valid_configs = []

    print("\n" + "=" * 80)
    print(f"Parameter Sweep (Target GPU Memory: {target_gpu_memory_gb} GB)")
    print("=" * 80)
    print(
        f"\n{'Batch':>8} {'AvgSeq':>8} {'Weights':>10} {'Optim':>10} {'Activat':>10} {'Gradients':>10} {'Total':>10} {'Status':>8}"
    )
    print("-" * 86)

    for batch_size in batch_sizes:
        for avg_seq_len in seq_lens:
            train_config = TrainingConfig(
                batch_size=batch_size,
                avg_seq_len=avg_seq_len,
                max_seq_len=max(4096, avg_seq_len),
            )

            results = estimate_total_memory(model_config, train_config, verbose=False)
            total = results["total"]
            total_gb = total["total_gpu_memory"] / (1024**3)

            status = "✅" if total_gb <= target_gpu_memory_gb else "❌"

            print(
                f"{batch_size:>8} {avg_seq_len:>8} "
                f"{format_bytes(total['weights']):>10} "
                f"{format_bytes(total['optimizer_total']):>10} "
                f"{format_bytes(total['activations']):>10} "
                f"{format_bytes(total['gradients']):>10} "
                f"{format_bytes(total['total_gpu_memory']):>10} "
                f"{status:>8}"
            )

            if total_gb <= target_gpu_memory_gb:
                valid_configs.append(
                    {
                        "batch_size": batch_size,
                        "avg_seq_len": avg_seq_len,
                        "total_memory_gb": total_gb,
                        "details": results["total"],
                    }
                )

    print("\n" + "-" * 86)
    print(f"Found {len(valid_configs)} valid configurations")

    return valid_configs


def main():
    parser = argparse.ArgumentParser(
        description="HSTU Training Memory Estimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use gin-config file (recommended)
  python estimate_memory.py --gin_config ../configs/h100_16gpu_exp0_baseline.gin
  
  # Use command-line arguments
  python estimate_memory.py --batch_size 32 --max_seq_len 4096 --hidden_size 1024 --num_layers 8
  
  # Parameter sweep
  python estimate_memory.py --gin_config ../configs/h100_16gpu_exp0_baseline.gin --sweep
        """,
    )

    # Gin config (highest priority)
    parser.add_argument(
        "--gin_config",
        type=str,
        help="Path to gin config file (e.g., ../configs/h100_16gpu_exp0_baseline.gin)",
    )

    # Model config (used when gin_config is not provided)
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of HSTU layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--kv_channels", type=int, default=256, help="KV channels per head"
    )
    parser.add_argument(
        "--item_embedding_dim", type=int, default=128, help="Item embedding dimension"
    )
    parser.add_argument(
        "--contextual_embedding_dim",
        type=int,
        default=256,
        help="Contextual embedding dimension",
    )
    parser.add_argument(
        "--num_embedding_tables",
        type=int,
        default=50,
        help="Number of embedding tables",
    )
    parser.add_argument(
        "--total_embedding_rows",
        type=int,
        default=250_000_000,
        help="Total embedding rows (deprecated)",
    )
    parser.add_argument(
        "--item_embedding_rows",
        type=int,
        default=50_000_000,
        help="Item+Action embedding rows (uses item_embedding_dim)",
    )
    parser.add_argument(
        "--contextual_embedding_rows",
        type=int,
        default=200_000_000,
        help="Contextual embedding rows (uses contextual_embedding_dim)",
    )
    parser.add_argument(
        "--embedding_gpu_cache_ratio",
        type=float,
        default=1.0,
        help="GPU cache ratio (0-1)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]
    )

    # Recompute options
    parser.add_argument(
        "--recompute_layernorm",
        action="store_true",
        help="Enable recompute input layernorm",
    )
    parser.add_argument(
        "--recompute_silu", action="store_true", help="Enable recompute input silu"
    )

    # Training config
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument(
        "--max_seq_len", type=int, default=4096, help="Max sequence length"
    )
    parser.add_argument(
        "--avg_seq_len", type=int, default=2048, help="Average sequence length"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size"
    )

    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"]
    )

    # Sweep mode
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument(
        "--target_memory_gb", type=float, default=80.0, help="Target GPU memory in GB"
    )

    # Output
    parser.add_argument("--output_json", type=str, help="Output results to JSON file")

    args = parser.parse_args()

    # Choose config source depending on gin_config
    if args.gin_config:
        # Use gin-config file
        print(f"📂 Loading config from: {args.gin_config}")
        model_config, train_config, optimizer = load_config_from_gin(args.gin_config)
        print(f"   ✅ Config loaded successfully")
        print(f"   - Hidden Size: {model_config.hidden_size}")
        print(f"   - Num Layers: {model_config.num_layers}")
        print(f"   - Batch Size: {train_config.batch_size}")
        print(f"   - Max Seq Len: {train_config.max_seq_len}")
        print(f"   - Embedding Tables: {model_config.num_embedding_tables}")
        print(
            f"   - Item Embedding Rows: {model_config.item_embedding_rows:,} (dim={model_config.item_embedding_dim})"
        )
        print(
            f"   - Contextual Embedding Rows: {model_config.contextual_embedding_rows:,} (dim={model_config.contextual_embedding_dim})"
        )
        print(f"   - Total Embedding Rows: {model_config.total_embedding_rows:,}")
        print(f"   - GPU Cache Ratio: {model_config.embedding_gpu_cache_ratio}")
        print(f"   - Optimizer: {optimizer}")
    else:
        # Use command line arguments
        model_config = HSTUModelConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            kv_channels=args.kv_channels,
            item_embedding_dim=args.item_embedding_dim,
            contextual_embedding_dim=args.contextual_embedding_dim,
            num_embedding_tables=args.num_embedding_tables,
            total_embedding_rows=args.total_embedding_rows,
            item_embedding_rows=args.item_embedding_rows,
            contextual_embedding_rows=args.contextual_embedding_rows,
            embedding_gpu_cache_ratio=args.embedding_gpu_cache_ratio,
            dtype=args.dtype,
            recompute_input_layernorm=args.recompute_layernorm,
            recompute_input_silu=args.recompute_silu,
        )
        train_config = TrainingConfig(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            avg_seq_len=args.avg_seq_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        optimizer = args.optimizer

    if args.sweep:
        # Parameter sweep
        batch_sizes = [8, 16, 32, 64, 128]
        seq_lens = [512, 1024, 2048, 4096]
        valid_configs = run_parameter_sweep(
            model_config,
            batch_sizes,
            seq_lens,
            args.target_memory_gb,
        )

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(valid_configs, f, indent=2)
            print(f"\nResults saved to {args.output_json}")
    else:
        # Single estimation
        results = estimate_total_memory(
            model_config,
            train_config,
            optimizer=optimizer,
            verbose=True,
        )

        if args.output_json:
            # Convert to serializable format
            output = {
                "gin_config": args.gin_config if args.gin_config else None,
                "model_config": results["model_config"],
                "training_config": results["training_config"],
                "memory_breakdown": {
                    k: {
                        kk: vv
                        for kk, vv in v.items()
                        if isinstance(vv, (int, float, str))
                    }
                    for k, v in results.items()
                    if isinstance(v, dict)
                    and k not in ["model_config", "training_config"]
                },
            }
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
