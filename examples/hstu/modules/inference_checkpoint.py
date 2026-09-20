# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dense checkpoint loading shared by inference backbones."""

import torch


def load_dense_state_dict(module, state_dict, *args, **kwargs):
    """Load dense weights using the selected backbone's checkpoint layout.

    Filter sparse embedding keys and apply legacy HSTU transpositions only to
    HSTU layers. Forward load options to PyTorch and return its incompatible-key
    result. Missing or unexpected dense keys raise RuntimeError even when the
    caller passes strict=False to allow filtering the sparse weights.
    """
    hstu_layout = not module._use_exportable and module._backbone == "hstu"
    converted = {}
    for key, value in state_dict.items():
        if (
            key.startswith(
                "_embedding_collection._data_parallel_embedding_collection.embeddings."
            )
            or "_model_parallel_embedding_collection" in key
        ):
            continue
        new_key = key
        if hstu_layout:
            for old, new, transpose in (
                ("_linear_uvqk_weight", "_linear_uvqk.weight", True),
                ("_linear_uvqk_bias", "_linear_uvqk.bias", False),
                ("_linear_proj_weight", "_linear_proj.weight", True),
            ):
                if key.endswith(old):
                    new_key = key.removesuffix(old) + new
                    value = value.T if transpose else value
                    break
        converted[new_key] = value
    result = torch.nn.Module.load_state_dict(module, converted, *args, **kwargs)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint does not match {module._backbone} backbone: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if hstu_layout:
        with torch.no_grad():
            for layer in module._hstu_block._attention_layers:
                layer._linear_uvqk_weight.copy_(layer._linear_uvqk.weight.T)
                layer._linear_proj_weight.copy_(layer._linear_proj.weight.T)
    return result
