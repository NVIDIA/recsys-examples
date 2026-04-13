#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate a gin config file based on exp0_baseline with custom optimization switches.

Usage:
    # Output to terminal
    python generate_gin_config.py --kernel_backend cutlass --recompute_layernorm
    
    # Output to file
    python generate_gin_config.py --kernel_backend cutlass --caching --ratio 0.1 -o my_config.gin
    
    # Generate exp8 full optimization config
    python generate_gin_config.py \\
        --kernel_backend cutlass \\
        --recompute_layernorm \\
        --balanced_shuffler \\
        --caching \\
        --ratio 0.1 \\
        --evict lfu \\
        --pipeline_type prefetch \\
        --tp_size 2 \\
        -o exp8_full.gin

    # Zipf key distribution (hot/cold access pattern)
    python generate_gin_config.py --kernel_backend cutlass --value_dist zipf --value_dist_alpha 1.2
"""

import argparse
import sys
from pathlib import Path


def get_baseline_template():
    """Return the baseline gin config template."""
    return """# ============================================================================
# Generated Gin Config
#
# Optimizations:
#   - Attention Backend: {kernel_backend}
#   - Recompute LayerNorm: {recompute_layernorm}
#   - Workload Balancer: {balanced_shuffler}
#   - DynamicEmb Caching: {caching}
#   - Cache Ratio: {ratio}
#   - Eviction Strategy: {evict}
#   - Pipeline: {pipeline_type}
#   - Tensor Parallel: {tp_size}
# ============================================================================

# ===== Trainer Configuration =====
TrainerArgs.train_batch_size = 32
TrainerArgs.eval_batch_size = 32
TrainerArgs.log_interval = 100
TrainerArgs.eval_interval = 400
TrainerArgs.max_train_iters = 1000
TrainerArgs.max_eval_iters = 50
TrainerArgs.seed = 1234
TrainerArgs.pipeline_type = '{pipeline_type}'
{balanced_shuffler_line}

# Profiling, we need iteration later than num_generated_batches to make sure jit-compiled kernels are cached
TrainerArgs.profile = True
TrainerArgs.profile_step_start = 150
TrainerArgs.profile_step_end = 200

# Checkpoint
TrainerArgs.ckpt_save_dir = './checkpoints/generated_exp'
TrainerArgs.ckpt_save_interval = 999999999

# ===== Dataset Configuration =====
# Main sequence features (item + action)
item_and_action_feature/FeatureArgs.feature_names = ['item', 'action']
item_and_action_feature/FeatureArgs.max_sequence_length = 4096
item_and_action_feature/FeatureArgs.is_jagged = True
item_seqlen_dist/RandomDistribution.dist_type = 'zipf'
item_seqlen_dist/RandomDistribution.alpha = 1.2
item_seqlen_dist/RandomDistribution.low = 1 # 256 is the minimum sequence length
item_and_action_feature/FeatureArgs.seqlen_dist = @item_seqlen_dist/RandomDistribution()

{value_dist_section}
# Contextual Features (only 3)
user_contextual_features/FeatureArgs.feature_names = ['user_id', 'user_age']
user_contextual_features/FeatureArgs.max_sequence_length = 1
user_contextual_features/FeatureArgs.is_jagged = False
{user_id_value_dist_section}

item_contextual_features/FeatureArgs.feature_names = ['item_category_l1']
item_contextual_features/FeatureArgs.max_sequence_length = 1
item_contextual_features/FeatureArgs.is_jagged = False

BenchmarkDatasetArgs.feature_args = [
    @item_and_action_feature/FeatureArgs(),
    @user_contextual_features/FeatureArgs(),
    @item_contextual_features/FeatureArgs(),
]

BenchmarkDatasetArgs.item_feature_name = 'item'
BenchmarkDatasetArgs.contextual_feature_names = [
    'user_id',
    'user_age',
    'item_category_l1',
]  # Total 3 contextual features
BenchmarkDatasetArgs.action_feature_name = 'action'
BenchmarkDatasetArgs.max_num_candidates = 0
BenchmarkDatasetArgs.num_generated_batches = 100

# ===== Embedding Configuration =====
# Item embedding (main ID)
item_embedding/DynamicEmbeddingArgs.feature_names = ['item']
item_embedding/DynamicEmbeddingArgs.table_name = 'item'
item_embedding/DynamicEmbeddingArgs.item_vocab_size_or_capacity = 50000000  # 50M
item_embedding/DynamicEmbeddingArgs.item_vocab_gpu_capacity_ratio = {ratio}
item_embedding/DynamicEmbeddingArgs.evict_strategy = '{evict}'
item_embedding/DynamicEmbeddingArgs.caching = {caching}

# Action embedding
action_embedding/EmbeddingArgs.feature_names = ['action']
action_embedding/EmbeddingArgs.table_name = 'action'
action_embedding/EmbeddingArgs.item_vocab_size_or_capacity = 100
action_embedding/EmbeddingArgs.sharding_type = 'data_parallel'

# Contextual Feature Embeddings (3 total)
user_id_emb/DynamicEmbeddingArgs.feature_names = ['user_id']
user_id_emb/DynamicEmbeddingArgs.table_name = 'user_id'
user_id_emb/DynamicEmbeddingArgs.item_vocab_size_or_capacity = 50000000  # 50M
user_id_emb/DynamicEmbeddingArgs.item_vocab_gpu_capacity_ratio = {ratio}
user_id_emb/DynamicEmbeddingArgs.evict_strategy = '{evict}'
user_id_emb/DynamicEmbeddingArgs.caching = {caching}

user_age_emb/EmbeddingArgs.feature_names = ['user_age']
user_age_emb/EmbeddingArgs.table_name = 'user_age'
user_age_emb/EmbeddingArgs.item_vocab_size_or_capacity = 100
user_age_emb/EmbeddingArgs.sharding_type = 'data_parallel'

item_cat_l1_emb/EmbeddingArgs.feature_names = ['item_category_l1']
item_cat_l1_emb/EmbeddingArgs.table_name = 'item_category_l1'
item_cat_l1_emb/EmbeddingArgs.item_vocab_size_or_capacity = 50
item_cat_l1_emb/EmbeddingArgs.sharding_type = 'data_parallel'

# Aggregate all embedding configs (5 embedding tables total)
BenchmarkDatasetArgs.embedding_args = [
    @item_embedding/DynamicEmbeddingArgs(),
    @action_embedding/EmbeddingArgs(),
    @user_id_emb/DynamicEmbeddingArgs(),
    @user_age_emb/EmbeddingArgs(),
    @item_cat_l1_emb/EmbeddingArgs(),
]

# ===== Network Configuration =====
NetworkArgs.item_embedding_dim = 128
NetworkArgs.contextual_embedding_dim = 128  # Same as item_embedding_dim
NetworkArgs.num_layers = 8
NetworkArgs.num_attention_heads = 4
NetworkArgs.hidden_size = 1024
NetworkArgs.kv_channels = 256

# Kernel config
NetworkArgs.kernel_backend = '{kernel_backend}'
NetworkArgs.recompute_input_layernorm = {recompute_layernorm}
NetworkArgs.recompute_input_silu = False

# ===== Ranking Head =====
RankingArgs.prediction_head_arch = [512, 8]
RankingArgs.prediction_head_bias = True
RankingArgs.num_tasks = 8
RankingArgs.eval_metrics = ['auc']

# ===== Optimizer =====
OptimizerArgs.optimizer_str = 'adam'
OptimizerArgs.learning_rate = 1e-3
OptimizerArgs.adam_beta1 = 0.9
OptimizerArgs.adam_beta2 = 0.999
OptimizerArgs.adam_eps = 1e-8

# ===== Parallelism =====
TensorModelParallelArgs.tensor_model_parallel_size = {tp_size}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a gin config file based on exp0_baseline with custom optimization switches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Output to terminal
  %(prog)s --kernel_backend cutlass --recompute_layernorm
  
  # Output to file
  %(prog)s --kernel_backend cutlass --caching --ratio 0.1 -o my_config.gin
  
  # Generate exp8 full optimization config
  %(prog)s --kernel_backend cutlass --recompute_layernorm --balanced_shuffler \\
           --caching --ratio 0.1 --evict lfu --pipeline_type prefetch --tp_size 2 \\
           -o exp8_full.gin
""",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path. If not specified, output to stdout.",
    )

    # Optimization switches
    parser.add_argument(
        "--kernel_backend",
        type=str,
        choices=["triton", "cutlass"],
        default="triton",
        help="Attention kernel backend (default: triton)",
    )

    parser.add_argument(
        "--recompute_layernorm",
        action="store_true",
        default=False,
        help="Enable recompute for LayerNorm (default: False)",
    )

    parser.add_argument(
        "--balanced_shuffler",
        action="store_true",
        default=False,
        help="Enable workload balancer (default: False)",
    )

    parser.add_argument(
        "--caching",
        action="store_true",
        default=False,
        help="Enable DynamicEmb caching (default: False)",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.0,
        help="GPU cache ratio for large tables (0.0-1.0, default: 0.0)",
    )

    parser.add_argument(
        "--evict",
        type=str,
        choices=["lru", "lfu"],
        default="lru",
        help="Eviction strategy (default: lru)",
    )

    parser.add_argument(
        "--pipeline_type",
        type=str,
        choices=["none", "prefetch", "sw_serial"],
        default="none",
        help="Pipeline type (default: none)",
    )

    parser.add_argument(
        "--tp_size", type=int, default=1, help="Tensor Parallel size (default: 1)"
    )

    parser.add_argument(
        "--value_dist",
        type=str,
        choices=["uniform", "zipf"],
        default="uniform",
        help="Key value distribution for item/user_id (default: uniform)",
    )

    parser.add_argument(
        "--value_dist_alpha",
        type=float,
        default=1.2,
        help="Zipf alpha parameter when --value_dist=zipf (default: 1.2)",
    )

    return parser.parse_args()


def generate_config(args):
    """Generate the gin config content based on args."""

    # Auto-set ratio to 0.1 (10%) when caching is enabled but ratio is 0
    ratio = args.ratio
    if args.caching and ratio == 0:
        ratio = 0.1
        print(
            f"Warning: caching enabled but ratio=0, auto-setting ratio to 0.1 (10%)",
            file=sys.stderr,
        )

    # Generate value distribution sections
    if args.value_dist == "zipf":
        alpha = args.value_dist_alpha
        value_dist_section = (
            f"# Item-ID value distribution: Zipf (long-tail)\n"
            f"item_value_dist/RandomDistribution.dist_type = 'zipf'\n"
            f"item_value_dist/RandomDistribution.alpha = {alpha}\n"
            f"item_and_action_feature/FeatureArgs.value_dists = {{\n"
            f"    'item': @item_value_dist/RandomDistribution(),\n"
            f"}}"
        )
        user_id_value_dist_section = (
            f"user_id_value_dist/RandomDistribution.dist_type = 'zipf'\n"
            f"user_id_value_dist/RandomDistribution.alpha = {alpha}\n"
            f"user_contextual_features/FeatureArgs.value_dists = {{\n"
            f"    'user_id': @user_id_value_dist/RandomDistribution(),\n"
            f"}}"
        )
    else:
        value_dist_section = "# Item-ID value distribution: uniform (default)"
        user_id_value_dist_section = ""

    # Generate balanced_shuffler line
    if args.balanced_shuffler:
        balanced_shuffler_line = (
            "TrainerArgs.enable_balanced_shuffler = True  # Enable workload balancer"
        )
    else:
        balanced_shuffler_line = ""

    # Format the template
    config = get_baseline_template().format(
        kernel_backend=args.kernel_backend,
        recompute_layernorm=str(args.recompute_layernorm),
        balanced_shuffler="Enabled" if args.balanced_shuffler else "Disabled",
        balanced_shuffler_line=balanced_shuffler_line,
        caching=str(args.caching),
        ratio=ratio,  # Use auto-corrected ratio
        evict=args.evict,
        pipeline_type=args.pipeline_type,
        tp_size=args.tp_size,
        value_dist_section=value_dist_section,
        user_id_value_dist_section=user_id_value_dist_section,
    )

    return config


def main():
    args = parse_args()
    config = generate_config(args)

    if args.output:
        # Write to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(config)
        print(f"Generated config written to: {output_path}", file=sys.stderr)
    else:
        # Output to stdout
        print(config)


if __name__ == "__main__":
    main()
