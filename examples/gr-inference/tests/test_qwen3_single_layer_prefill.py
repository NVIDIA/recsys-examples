import importlib.util

import pytest


def test_qwen3_single_decoder_layer_prefill_writes_context_kv() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_kernels.prefill import PrefillAttention, TorchSDPAPrefillBackend
    from gr_inference.gr_kv import ContextKV
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3SingleLayerPrefill

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
        rms_norm_eps=1e-6,
    )
    batch = 2
    seq_len = 16
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    context_k = torch.empty(
        config.num_layers,
        batch,
        seq_len,
        config.num_kv_heads,
        config.head_dim,
    )
    context_v = torch.empty_like(context_k)
    context_kv = ContextKV(context_k, context_v)

    layer = Qwen3SingleLayerPrefill(
        config,
        layer_idx=0,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    output = layer.forward_prefill(hidden_states, context_kv)

    assert tuple(output.shape) == (batch, seq_len, config.hidden_size)
    assert layer.ops.qkv_proj.out_features == (
        config.num_attention_heads + 2 * config.num_kv_heads
    ) * config.head_dim
    assert layer.ops.gate_up_proj.out_features == 2 * config.intermediate_size
    assert layer.ops.down_proj.in_features == config.intermediate_size
    assert tuple(context_kv.key.shape) == (
        config.num_layers,
        batch,
        seq_len,
        config.num_kv_heads,
        config.head_dim,
    )
    assert torch.isfinite(context_kv.key).all()
    assert torch.isfinite(context_kv.value).all()
    assert torch.isfinite(output).all()


def test_qwen3_rope_preserves_shape_and_dtype() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_models.qwen3 import apply_qwen3_rope

    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 1, 8)

    q_out, k_out = apply_qwen3_rope(q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype


def test_qwen3_rms_norm_matches_manual_reference() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_models.qwen3 import Qwen3RMSNorm

    norm = Qwen3RMSNorm(8, eps=1e-6)
    hidden_states = torch.randn(2, 3, 8)
    output = norm(hidden_states)

    reference = hidden_states.float()
    variance = reference.pow(2).mean(dim=-1, keepdim=True)
    reference = reference * torch.rsqrt(variance + norm.eps)
    reference = norm.weight * reference

    assert torch.allclose(output, reference.to(output.dtype), atol=1e-6, rtol=1e-6)


def test_qwen3_sgl_kernel_mlp_path_uses_packed_activation(monkeypatch) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    import gr_inference.gr_models.qwen3.layers as qwen3_layers
    from gr_inference.gr_kernels import CAP_FUSED_MLP
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig
    from gr_inference.gr_models.qwen3.layers import TorchQwen3LayerOps

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-gr",
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_context_len=8,
        max_seq_len=10,
        max_decode_steps=2,
        max_beam_width=4,
        intermediate_size=16,
    )
    ops = TorchQwen3LayerOps(config)
    hidden_states = torch.randn(2, 3, config.hidden_size)
    fused_inputs = []

    def selected_backend(capability: str) -> str:
        return "sgl_kernel" if capability == CAP_FUSED_MLP else "torch"

    def fake_silu_and_mul(packed_gate_up):
        fused_inputs.append(packed_gate_up)
        dim = packed_gate_up.shape[-1] // 2
        return torch.nn.functional.silu(packed_gate_up[..., :dim]) * packed_gate_up[
            ...,
            dim:,
        ]

    monkeypatch.setattr(qwen3_layers, "_selected_kernel_backend", selected_backend)
    monkeypatch.setattr(qwen3_layers, "_is_cuda_tensor", lambda _tensor: True)
    monkeypatch.setattr(qwen3_layers, "_sgl_kernel_silu_and_mul", lambda: fake_silu_and_mul)

    actual = ops.mlp(hidden_states)
    gate, up = ops.gate_up(hidden_states)
    expected = ops.down_proj_only(torch.nn.functional.silu(gate) * up)

    assert len(fused_inputs) == 1
    assert fused_inputs[0].shape[-1] == 2 * config.intermediate_size
    assert torch.allclose(actual, expected)


def test_qwen3_flashinfer_rmsnorm_flattens_leading_dims() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_models.qwen3.layers import _reshape_for_flashinfer_rmsnorm

    hidden_states = torch.randn(2, 3, 4, 8)
    flattened = _reshape_for_flashinfer_rmsnorm(hidden_states)

    assert tuple(flattened.shape) == (24, 8)
    assert flattened.reshape_as(hidden_states).shape == hidden_states.shape


def test_qwen3_flashinfer_fused_add_rmsnorm_helper_matches_reference() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_models.qwen3.layers import (
        _apply_flashinfer_fused_add_rmsnorm,
    )

    residual = torch.randn(2, 3, 8)
    projected = torch.randn(2, 3, 8)
    weight = torch.randn(8)
    eps = 1e-6

    def fake_fused_add_rmsnorm(input_tensor, residual_tensor, weight, eps):
        residual_tensor.add_(input_tensor)
        reference = residual_tensor.float()
        variance = reference.pow(2).mean(dim=-1, keepdim=True)
        reference = reference * torch.rsqrt(variance + eps)
        input_tensor.copy_((weight * reference).to(input_tensor.dtype))
        return None

    residual_output, norm_output = _apply_flashinfer_fused_add_rmsnorm(
        fake_fused_add_rmsnorm,
        projected.clone(),
        residual.clone(),
        weight,
        eps,
    )

    expected_residual = residual + projected
    expected_norm = expected_residual.float()
    variance = expected_norm.pow(2).mean(dim=-1, keepdim=True)
    expected_norm = weight * expected_norm * torch.rsqrt(variance + eps)

    assert torch.allclose(residual_output, expected_residual)
    assert torch.allclose(norm_output, expected_norm.to(norm_output.dtype))


def test_qwen3_trtllm_fused_qk_norm_rope_helper_splits_packed_qkv() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_models.qwen3.layers import (
        _apply_trtllm_fused_qk_norm_rope,
    )

    batch = 2
    seq_len = 3
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = torch.zeros(batch, seq_len, q_size + 2 * kv_size)
    q = qkv[..., :q_size].reshape(batch, seq_len, num_q_heads, head_dim)
    k = qkv[..., q_size : q_size + kv_size].reshape(
        batch,
        seq_len,
        num_kv_heads,
        head_dim,
    )

    captured_args = {}

    def fake_fused_qk_norm_rope(qkv_flat, *_args):
        captured_args["is_neox"] = _args[9]
        qkv_flat[:, :q_size].fill_(1.0)
        qkv_flat[:, q_size : q_size + kv_size].fill_(2.0)

    q_out, k_out = _apply_trtllm_fused_qk_norm_rope(
        fake_fused_qk_norm_rope,
        qkv,
        q,
        k,
        num_attention_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_size=q_size,
        kv_size=kv_size,
        q_norm_weight=torch.ones(head_dim),
        k_norm_weight=torch.ones(head_dim),
        eps=1e-6,
        rope_theta=1_000_000.0,
        position_ids=16,
    )

    assert tuple(q_out.shape) == (batch, seq_len, num_q_heads, head_dim)
    assert tuple(k_out.shape) == (batch, seq_len, num_kv_heads, head_dim)
    assert captured_args["is_neox"] is True
    assert torch.all(q_out == 1.0)
    assert torch.all(k_out == 2.0)


def test_gr_trtllm_fused_qk_norm_rope_reference_matches_composed_path() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    import gr_inference_trtllm_kernels  # noqa: F401
    from gr_inference.gr_models.qwen3 import apply_qwen3_rope

    batch = 1
    seq_len = 3
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 8
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    eps = 1e-6
    rope_theta = 1_000_000.0
    q_weight = torch.randn(head_dim)
    k_weight = torch.randn(head_dim)
    qkv = torch.randn(batch * seq_len, q_size + 2 * kv_size)
    position_ids = torch.arange(seq_len, dtype=torch.int32)

    q_ref = qkv[:, :q_size].reshape(batch, seq_len, num_q_heads, head_dim)
    k_ref = qkv[:, q_size : q_size + kv_size].reshape(
        batch,
        seq_len,
        num_kv_heads,
        head_dim,
    )
    q_ref = _manual_rmsnorm(q_ref, q_weight, eps)
    k_ref = _manual_rmsnorm(k_ref, k_weight, eps)
    q_ref, k_ref = apply_qwen3_rope(
        q_ref,
        k_ref,
        rope_theta=rope_theta,
        position_ids=position_ids.reshape(batch, seq_len),
    )

    qkv_out = qkv.clone()
    torch.ops.gr_trtllm.fused_qk_norm_rope(
        qkv_out,
        num_q_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        eps,
        q_weight,
        k_weight,
        rope_theta,
        True,
        position_ids,
        1.0,
        0,
        0,
        1.0,
        True,
    )

    q_out = qkv_out[:, :q_size].reshape(batch, seq_len, num_q_heads, head_dim)
    k_out = qkv_out[:, q_size : q_size + kv_size].reshape(
        batch,
        seq_len,
        num_kv_heads,
        head_dim,
    )
    assert torch.allclose(q_out, q_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_out, k_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(qkv_out[:, q_size + kv_size :], qkv[:, q_size + kv_size :])


def test_gr_trtllm_gated_mlp_reference_matches_composed_path() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    import gr_inference_trtllm_kernels

    gr_inference_trtllm_kernels.reset_call_counts()
    hidden_states = torch.randn(2, 3, 8)
    gate_up_weight = torch.randn(32, 8)
    down_weight = torch.randn(8, 16)

    gate_up = torch.matmul(hidden_states, gate_up_weight.transpose(0, 1))
    gate, up = gate_up.split([16, 16], dim=-1)
    expected = torch.matmul(
        torch.nn.functional.silu(gate) * up,
        down_weight.transpose(0, 1),
    )

    actual = torch.ops.gr_trtllm.gated_mlp(
        hidden_states,
        gate_up_weight,
        down_weight,
    )

    assert torch.allclose(actual, expected)
    assert gr_inference_trtllm_kernels.call_counts()["gated_mlp_reference"] == 1


def _manual_rmsnorm(hidden_states, weight, eps: float):
    import torch

    reference = hidden_states.float()
    variance = reference.pow(2).mean(dim=-1, keepdim=True)
    reference = reference * torch.rsqrt(variance + eps)
    return (weight * reference).to(hidden_states.dtype)


def test_qwen3_rope_accepts_decode_position_ids() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_models.qwen3 import apply_qwen3_rope

    q = torch.randn(1, 3, 2, 8)
    k = torch.randn(1, 3, 1, 8)
    position_ids = torch.full((1, 3), 16, dtype=torch.long)

    q_out, k_out = apply_qwen3_rope(q, k, position_ids=position_ids)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_qwen3_layer_accepts_custom_ops_backend() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_kernels.prefill import PrefillAttention, TorchSDPAPrefillBackend
    from gr_inference.gr_models.qwen3 import (
        Qwen3GRConfig,
        Qwen3SingleLayerPrefill,
        TorchQwen3LayerOps,
    )

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
    layer = Qwen3SingleLayerPrefill(
        config,
        layer_idx=0,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
        ops=ops,
    )

    assert layer.ops is ops
    assert tuple(layer.ops.mlp(torch.randn(1, 2, 32)).shape) == (1, 2, 32)


def test_torch_qwen3_layer_ops_loads_logical_weights() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

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
    prefix = "layers.0"
    qkv = torch.randn(64, 32)
    gate_up = torch.randn(128, 32)
    weights = {
        f"{prefix}.input_layernorm.weight": torch.randn(32),
        f"{prefix}.self_attn.qkv_proj.weight": qkv,
        f"{prefix}.self_attn.o_proj.weight": torch.randn(32, 32),
        f"{prefix}.self_attn.q_norm.weight": torch.randn(8),
        f"{prefix}.self_attn.k_norm.weight": torch.randn(8),
        f"{prefix}.post_attention_layernorm.weight": torch.randn(32),
        f"{prefix}.mlp.gate_up_proj.weight": gate_up,
        f"{prefix}.mlp.down_proj.weight": torch.randn(32, 64),
    }

    ops.load_logical_weights(weights, layer_idx=0)

    assert torch.equal(ops.qkv_proj.weight, qkv)
    assert torch.equal(ops.gate_up_proj.weight, gate_up)
