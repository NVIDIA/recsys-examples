import importlib.util

import pytest


def test_torch_fused_mlp_backend_matches_explicit_ops() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kernels import FusedMLP, TorchFusedMLPBackend
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, TorchQwen3LayerOps

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-gr",
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=16,
        max_seq_len=20,
        max_decode_steps=3,
        max_beam_width=8,
        intermediate_size=64,
    )
    ops = TorchQwen3LayerOps(config)
    hidden_states = torch.randn(2, 3, 32)

    gate, up = ops.gate_up(hidden_states)
    expected = ops.down_proj_only(ops.silu_mul(gate, up))
    actual = FusedMLP(TorchFusedMLPBackend())(hidden_states, ops)

    assert torch.allclose(actual, expected)
