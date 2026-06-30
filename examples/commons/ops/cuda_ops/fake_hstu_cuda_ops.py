import torch


def _hstu_inference_preprocess_fake(
    item_values,
    item_lengths,
    action_values,
    action_lengths,
    num_candidates,
):
    torch._check(item_values.dim() == 2)
    torch._check(action_values.dim() == 2)
    torch._check(action_lengths.dim() == 1)
    torch._check(item_lengths.dim() == 1)
    torch._check(num_candidates.dim() == 1)
    torch._check(item_values.size(1) == action_values.size(1))
    torch._check(item_lengths.size(0) == action_lengths.size(0))
    torch._check(item_lengths.size(0) == num_candidates.size(0))

    ctx = torch.library.get_ctx()
    output_len = ctx.new_dynamic_size()
    batch_size = item_lengths.size(0)
    return (
        item_values.new_empty((output_len, item_values.size(1))),
        item_lengths.new_empty((batch_size,), dtype=item_lengths.dtype),
        item_lengths.new_empty((batch_size + 1,), dtype=item_lengths.dtype),
        item_lengths.new_empty((batch_size + 1,), dtype=num_candidates.dtype),
    )


if hasattr(torch.ops.hstu_cuda_ops, "hstu_inference_preprocess"):
    torch.library.register_fake("hstu_cuda_ops::hstu_inference_preprocess")(
        _hstu_inference_preprocess_fake
    )


@torch.library.register_fake("hstu_cuda_ops::split_by_lengths")
def _split_by_lengths_fake(values, lengths_1d, num_splits: int):
    torch._check(values.dim() == 1 or values.dim() == 2)
    torch._check(lengths_1d.dim() == 1)

    # Tensor[] return length is static (num_splits), which is export-friendly.
    # Each output tensor length along dim 0 is data-dependent; represent with symbolic sizes.
    # Keep trailing dimensions the same as `values`.
    ctx = torch.library.get_ctx()
    out = []
    for _ in range(num_splits):
        dyn_len = ctx.new_dynamic_size()
        if values.dim() == 1:
            out.append(values.new_empty((dyn_len,)))
        else:
            out.append(values.new_empty((dyn_len, values.size(1))))
    return out


@torch.library.register_fake("hstu_cuda_ops::lengths_splits")
def _lengths_splits_fake(lengths_1d, num_splits: int):
    torch._check(lengths_1d.dim() == 1)
    torch._check(num_splits > 0)

    # num_splits is static at export time.
    out = []
    for _ in range(num_splits):
        out.append(lengths_1d.new_empty((lengths_1d.size(0) // num_splits,)))
    return out


@torch.library.register_fake("hstu_cuda_ops::lengths_reduce_dim1")
def _lengths_reduce_dim1_fake(lengths_1d, num_splits: int):
    torch._check(lengths_1d.dim() == 1)
    torch._check(num_splits > 0)

    # num_splits is static at export time.
    return lengths_1d.new_empty((num_splits,))


@torch.library.register_fake("hstu_cuda_ops::permute_and_split")
def _permute_and_split_fake(
    jagged_features,
    jagged_lengths,
    jagged_offsets,
    num_static_features: int,
    num_dynamic_features: int,
    features_order: list,
):
    # num_static_features and num_dynamic_features are static at export time.

    torch._check(jagged_features.dim() == 1)
    torch._check(jagged_lengths.dim() == 1)
    torch._check(num_static_features > 0)
    torch._check(num_dynamic_features > 0)

    num_features = num_static_features + num_dynamic_features
    ctx = torch.library.get_ctx()
    static_output_len = ctx.new_dynamic_size()
    dynamic_output_len = ctx.new_dynamic_size()
    out = [
        jagged_features.new_empty((static_output_len,)),
        jagged_features.new_empty((dynamic_output_len,)),
        jagged_lengths.new_empty(
            ((jagged_lengths.size(0) // num_features) * num_static_features,)
        ),
        jagged_lengths.new_empty(
            ((jagged_lengths.size(0) // num_features) * num_dynamic_features,)
        ),
    ]
    return out


def _strip_cached_tokens_fake(
    values,
    lengths,
    length_offsets,
    num_cached,
    feature_order: list,
):
    torch._check(values.dim() == 1)
    torch._check(lengths.dim() == 1)
    torch._check(length_offsets.dim() == 1)
    torch._check(num_cached.dim() == 1)
    torch._check(len(feature_order) > 0)
    torch._check(length_offsets.size(0) == lengths.size(0) + 1)
    torch._check(lengths.size(0) == num_cached.size(0) * len(feature_order))

    ctx = torch.library.get_ctx()
    output_len = ctx.new_dynamic_size()
    return values.new_empty((output_len,)), lengths.new_empty(lengths.shape)


if hasattr(torch.ops.hstu_cuda_ops, "strip_cached_tokens"):
    torch.library.register_fake("hstu_cuda_ops::strip_cached_tokens")(
        _strip_cached_tokens_fake
    )
