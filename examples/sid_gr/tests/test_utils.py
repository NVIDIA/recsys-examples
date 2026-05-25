from typing import TYPE_CHECKING, List, Tuple

import torch

# Heavy deps (commons / megatron / configs / model) are only needed by
# `create_sid_gr_model_and_optimizer` below. The pure-tensor validators
# `validate_compare_outputs` / `validate_pair_outputs` should be
# callable without them, so callers like
# `test_compare_kv_modes_default_raises_on_validation_failure` can
# exercise the validation contract outside the full Docker stack.
if TYPE_CHECKING:
    from commons.modules.embedding import ShardedEmbeddingConfig


def _check_sids_shapes(*sids_tensors):
    """Validate that all sids tensors share a consistent ``[B, K, H]`` shape."""
    if not sids_tensors:
        return None
    ref_shape = sids_tensors[0].shape
    if len(ref_shape) != 3:
        return (
            f"unexpected rank: sids should be 3-D [B, K, H], " f"got {tuple(ref_shape)}"
        )
    for i, s in enumerate(sids_tensors):
        if s.shape != ref_shape:
            return (
                f"sids shape mismatch at path {i}: "
                f"{tuple(s.shape)} != {tuple(ref_shape)}"
            )
    return None


# Top-K beam set overlap is the only metric used for benchmark-level
# validation. It is bounded in [0, 1] regardless of seqlen / num_layers /
# num_hierarchies, so the same threshold stays meaningful as customers
# scale the workload up. Stricter checks (top-1 exact match, absolute
# log-prob deltas) live in the unit tests where the config is fixed and
# the noise floor is calibrated.
_OVERLAP_THRESHOLD = 0.7


def _min_overlap(sets_x, sets_y, label, issues):
    worst = 1.0
    worst_sample = -1
    top_k = len(next(iter(sets_x))) if sets_x else 0
    for b, (sx, sy) in enumerate(zip(sets_x, sets_y)):
        o = len(sx & sy) / len(sx)
        if o < worst:
            worst, worst_sample = o, b
    if worst < _OVERLAP_THRESHOLD:
        issues.append(
            f"{label} top-{top_k} overlap {worst*100:.0f}% < "
            f"{int(_OVERLAP_THRESHOLD*100)}% on sample {worst_sample}"
        )
    return worst


def _sids_to_sets(sids: torch.Tensor):
    top_k = sids.shape[1]
    return [
        {tuple(sids[b, k].tolist()) for k in range(top_k)} for b in range(sids.shape[0])
    ]


def validate_compare_outputs(
    sids_a: torch.Tensor,
    sids_b: torch.Tensor,
    sids_c: torch.Tensor,
) -> Tuple[bool, str, float]:
    """Validate that three beam-search outputs explore approximately the
    same top-K beam set, using per-sample set overlap >= 70%.

    Returns ``(passed, summary, worst_overlap)`` where ``worst_overlap``
    is the minimum overlap across all three pairs and all samples, in
    ``[0, 1]``. Callers display the value so the metric being checked is
    visible alongside the PASS/FAIL flag.

    Scale-invariant: top-K overlap is bounded ``[0, 1]`` regardless of
    seqlen, num_layers, or hierarchy depth — so the threshold stays
    meaningful as workloads scale up. Strict top-1 match and absolute
    log-prob deltas live in the unit tests, not here.
    """
    shape_issue = _check_sids_shapes(sids_a, sids_b, sids_c)
    if shape_issue is not None:
        return False, shape_issue, 0.0

    issues: List[str] = []
    sets_a = _sids_to_sets(sids_a)
    sets_b = _sids_to_sets(sids_b)
    sets_c = _sids_to_sets(sids_c)

    ov_ab = _min_overlap(sets_a, sets_b, "A vs B", issues)
    ov_bc = _min_overlap(sets_b, sets_c, "B vs C", issues)
    ov_ac = _min_overlap(sets_a, sets_c, "A vs C", issues)
    worst = min(ov_ab, ov_bc, ov_ac)

    summary = (
        f"ov_ab={ov_ab*100:.0f}% ov_bc={ov_bc*100:.0f}% " f"ov_ac={ov_ac*100:.0f}%"
    )
    if issues:
        return False, "; ".join(issues) + " | " + summary, worst
    return True, summary, worst


def validate_pair_outputs(
    sids_a: torch.Tensor,
    sids_b: torch.Tensor,
) -> Tuple[bool, str, float]:
    """Two-path version of ``validate_compare_outputs`` for A vs B.

    Returns ``(passed, summary, overlap)``; ``overlap`` is the worst
    sample's set overlap, in ``[0, 1]``.
    """
    shape_issue = _check_sids_shapes(sids_a, sids_b)
    if shape_issue is not None:
        return False, shape_issue, 0.0

    issues: List[str] = []
    sets_a = _sids_to_sets(sids_a)
    sets_b = _sids_to_sets(sids_b)
    ov_ab = _min_overlap(sets_a, sets_b, "A vs B", issues)

    summary = f"ov_ab={ov_ab*100:.0f}%"
    if issues:
        return False, "; ".join(issues) + " | " + summary, ov_ab
    return True, summary, ov_ab


def create_sid_gr_model_and_optimizer(
    dtype: torch.dtype,
    hidden_size: int,
    num_attention_heads: int,
    kv_channels: int,
    num_layers: int,
    num_hierarchies: int,
    codebook_embedding_config: "ShardedEmbeddingConfig",
    codebook_sizes: List[int],
    should_add_sep_token: bool = False,
    optimizer_type_str: str = "adam",
    pipeline_type: str = "none",
    device: torch.device = None,
    use_jagged_flash_attn: bool = False,
):
    # Lazy imports of the heavy stack — see module-level comment.
    from commons.distributed.sharding import make_optimizer_and_shard
    from commons.optimizer import OptimizerParam
    from configs.gpt_config import get_gpt_config
    from model.gpt_model import SIDGRModel
    from model.mcore_model_specs import get_gpt_decoder_block_spec

    decoder_config = get_gpt_config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        num_layers=num_layers,
        dtype=dtype,
        normalization="LayerNorm",
        norm_epsilon=1e-5,
        hidden_dropout=0.0,
        tensor_model_parallel_size=1,
        loss_on_history=False,
    )
    # thd + causal + TE
    transformer_decoder_layer_spec = get_gpt_decoder_block_spec(
        decoder_config,
        use_transformer_engine=False,
        arbitrary_attention_mask=True,
    )

    sid_gr_model = SIDGRModel(
        decoder_config=decoder_config,
        codebook_embedding_config=codebook_embedding_config,
        codebook_sizes=codebook_sizes,
        num_hierarchies=num_hierarchies,
        transformer_decoder_layer_spec=transformer_decoder_layer_spec,
        should_add_sep_token=should_add_sep_token,
        top_k_for_generation=10,
        eval_metrics=("HitRate@2", "NDCG@10"),
        share_lm_head_across_hierarchies=False,
        use_jagged_flash_attn=use_jagged_flash_attn,
    )

    optimizer_param = OptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3 if optimizer_type_str == "adam" else 1e-1,
        adam_beta1=0.5,  # larger beta1 for better debugging!
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,  # decay is off for better debugging
    )

    model_train, dense_optimizer = make_optimizer_and_shard(
        sid_gr_model,
        config=decoder_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        pipeline_type=pipeline_type,
        device=device,
    )

    return model_train, dense_optimizer
