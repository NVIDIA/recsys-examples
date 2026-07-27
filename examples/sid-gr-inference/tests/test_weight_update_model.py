# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-layer primitives: in-place update by name, atomic validate-then-copy,
and the logical-weight dry-run (no mutation). Plus pure unit tests of the
weight-ipc building blocks (bucket round-trip, HF->logical name mapping).
"""

from __future__ import annotations

import pytest

from gr_inference.gr_models.qwen3.weights import materialize_qwen3_checkpoint

import _weight_update_helpers as h

torch = pytest.importorskip("torch")


def test_update_weights_from_tensor_copies_by_module_name() -> None:
    model, _ = h.model()
    before = h.forward_logits(model)

    named = {name: tensor.clone() for name, tensor in model.named_parameters()}
    named["layers.0.ops.qkv_proj.weight"].add_(1.0)
    named["norm.weight"].add_(0.5)
    sentinel = model.get_parameter_by_name("layers.1.ops.down_proj.weight").clone()

    count = model.update_weights_from_tensor(
        {
            "layers.0.ops.qkv_proj.weight": named["layers.0.ops.qkv_proj.weight"],
            "norm.weight": named["norm.weight"],
        }
    )
    assert count == 2
    assert torch.allclose(
        model.get_parameter_by_name("layers.0.ops.qkv_proj.weight"),
        named["layers.0.ops.qkv_proj.weight"],
    )
    # Untouched parameter is unchanged.
    assert torch.allclose(
        model.get_parameter_by_name("layers.1.ops.down_proj.weight"), sentinel
    )
    assert not torch.allclose(before, h.forward_logits(model))


def test_update_weights_from_tensor_is_atomic_on_failure() -> None:
    model, _ = h.model()
    sentinel = model.get_parameter_by_name("embed_tokens.weight").clone()

    # Wrong shape must raise before any copy and leave the model untouched.
    with pytest.raises(ValueError):
        model.update_weights_from_tensor({"embed_tokens.weight": torch.zeros(1, 1)})
    assert torch.allclose(model.get_parameter_by_name("embed_tokens.weight"), sentinel)

    # Unknown name with strict=True raises.
    with pytest.raises(KeyError):
        model.update_weights_from_tensor({"does.not.exist": torch.zeros(1)})

    # Unknown names are skipped when strict=False; known ones still apply.
    good = model.get_parameter_by_name("norm.weight").clone() + 0.25
    count = model.update_weights_from_tensor(
        {"does.not.exist": torch.zeros(1), "norm.weight": good}, strict=False
    )
    assert count == 1


def test_validate_logical_weights_does_not_mutate(tmp_path) -> None:
    model, cfg = h.model()
    checkpoint = h.write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    logical = materialize_qwen3_checkpoint(checkpoint)

    snapshot = {name: tensor.clone() for name, tensor in model.named_parameters()}
    model.validate_logical_weights(logical)  # dry run
    for name, tensor in model.named_parameters():
        assert torch.allclose(tensor, snapshot[name])

    # A real load actually applies the checkpoint (params diverge from snapshot).
    model.load_logical_weights(logical)
    changed = [
        name
        for name, tensor in model.named_parameters()
        if not torch.allclose(tensor, snapshot[name])
    ]
    assert changed, "load_logical_weights did not apply any weights"


def test_weight_ipc_flattened_bucket_roundtrip() -> None:
    from gr_inference.gr_serving.weight_ipc import reconstruct_named_tensors

    named = [
        ("model.norm.weight", torch.randn(8, dtype=torch.float32)),
        ("lm_head.weight", torch.randn(4, 5, dtype=torch.float32)),
    ]
    serialized = h.serialize_bucket(named)
    out = reconstruct_named_tensors([serialized], load_format="flattened_bucket")
    assert [name for name, _ in out] == [name for name, _ in named]
    for (n_in, t_in), (n_out, t_out) in zip(named, out, strict=True):
        assert n_in == n_out
        assert torch.equal(t_in, t_out)


def test_hf_to_logical_name_mapping() -> None:
    from gr_inference.gr_serving.continuous import _hf_to_logical_name

    assert _hf_to_logical_name("model.embed_tokens.weight") == "embed_tokens.weight"
    assert _hf_to_logical_name("model.norm.weight") == "final_norm.weight"
    assert _hf_to_logical_name("lm_head.weight") == "lm_head.weight"
    assert (
        _hf_to_logical_name("model.layers.0.self_attn.q_proj.weight")
        == "layers.0.self_attn.q_proj.weight"
    )
