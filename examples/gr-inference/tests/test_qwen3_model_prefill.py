import importlib.util

import pytest


def test_qwen3_model_prefill_writes_all_context_kv_layers() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-gr",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=16,
        max_seq_len=20,
        max_decode_steps=3,
        max_beam_width=8,
        intermediate_size=64,
        vocab_size=128,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    input_ids = torch.randint(0, config.vocab_size, (2, 16))

    logits, context_kv = model.forward_prefill(input_ids)

    assert tuple(logits.shape) == (2, 16, config.vocab_size)
    assert tuple(context_kv.key.shape) == (
        config.num_layers,
        2,
        16,
        config.num_kv_heads,
        config.head_dim,
    )
    assert torch.isfinite(context_kv.key).all()
    assert torch.isfinite(context_kv.value).all()
    assert torch.isfinite(logits).all()


def test_qwen3_rope_matches_hf_half_split_layout() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_models.qwen3.layers import apply_qwen3_rope

    q = torch.arange(1 * 3 * 1 * 8, dtype=torch.float32).reshape(1, 3, 1, 8)
    k = q + 1000
    theta = 1_000_000.0

    q_rope, k_rope = apply_qwen3_rope(q, k, rope_theta=theta)

    def expected_rope(x):
        inv_freq = 1.0 / (theta ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
        positions = torch.arange(3, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        x_first = x[..., :4]
        x_second = x[..., 4:]
        return torch.cat(
            (x_first * cos - x_second * sin, x_first * sin + x_second * cos),
            dim=-1,
        )

    assert torch.allclose(q_rope, expected_rope(q))
    assert torch.allclose(k_rope, expected_rope(k))


def test_qwen3_model_allocate_context_kv_validates_context_len() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel

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
        vocab_size=128,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )

    with pytest.raises(ValueError, match="max_context_len"):
        model.allocate_context_kv(batch_size=1, context_len=17)


def test_qwen3_model_decode_step_writes_beam_kv_and_returns_logits() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kernels.attention import GRDecodeAttention
    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel
    from gr_inference.gr_runtime import GRDecodeEngine, GRGenerationState

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-gr",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=16,
        max_seq_len=20,
        max_decode_steps=3,
        max_beam_width=8,
        intermediate_size=64,
        vocab_size=128,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )

    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    prefill = model.forward_prefill(input_ids, return_result=True)
    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=config.max_decode_steps,
        max_beam_width=config.max_beam_width,
        fixed_beam_width=3,
    )
    selection = generation.initialize_beams()
    beam_token_ids = torch.tensor([selection.token_ids], dtype=torch.long)

    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=generation.fixed_beam_width,
    )
    logits = model.forward_decode_step(
        beam_token_ids,
        generation,
        decode_engine,
        step=0,
    )

    assert tuple(logits.shape) == (1, 3, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(generation.beam_kv.key[:, :, 0, :3]).all()
    assert torch.isfinite(generation.beam_kv.value[:, :, 0, :3]).all()


def test_qwen3_model_generate_fixed_beam_two_steps() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kernels.attention import GRDecodeAttention
    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel
    from gr_inference.gr_runtime import GRDecodeEngine, GRGenerationState

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-gr",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=16,
        max_seq_len=20,
        max_decode_steps=3,
        max_beam_width=8,
        intermediate_size=64,
        vocab_size=128,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    prefill = model.forward_prefill(input_ids, return_result=True)
    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=config.max_decode_steps,
        max_beam_width=config.max_beam_width,
        fixed_beam_width=3,
    )
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=generation.fixed_beam_width,
    )

    result = model.generate_fixed_beam(
        generation,
        decode_engine,
        max_steps=2,
    )

    assert len(result.steps) == 2
    assert len(result.final_token_ids) == 3
    assert generation.beam_path.steps_done == 3
    assert tuple(result.steps[0].logits.shape) == (1, 3, config.vocab_size)
    assert torch.isfinite(generation.beam_kv.key[:, :, :2, :3]).all()


def test_qwen3_model_generate_fixed_beam_with_trie_constraints() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kernels.attention import GRDecodeAttention
    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel
    from gr_inference.gr_runtime import (
        GRDecodeEngine,
        GRGenerationState,
        TokenTrie,
        TrieItemMaskProvider,
    )

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-gr",
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=8,
        max_seq_len=10,
        max_decode_steps=1,
        max_beam_width=2,
        intermediate_size=64,
        vocab_size=32,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    prefill = model.forward_prefill(input_ids, return_result=True)
    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=config.max_decode_steps,
        max_beam_width=config.max_beam_width,
    )
    provider = TrieItemMaskProvider(
        TokenTrie.from_sequences([[1, 10], [2, 20]]),
        vocab_size=config.vocab_size,
    )
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=generation.fixed_beam_width,
    )

    result = model.generate_fixed_beam(
        generation,
        decode_engine,
        max_steps=1,
        item_mask_provider=provider,
    )

    assert set(generation.beam_path.entries[0].token_ids).issubset({1, 2})
    assert len(result.final_token_ids) == 2


def test_qwen3_model_loads_logical_weights() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kernels.prefill import (
        PrefillAttention,
        TorchSDPAPrefillBackend,
    )
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel

    config = Qwen3GRConfig(
        model_name="tiny-qwen3-gr",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=8,
        max_seq_len=10,
        max_decode_steps=1,
        max_beam_width=2,
        intermediate_size=64,
        vocab_size=32,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    weights = {
        "embed_tokens.weight": torch.randn(32, 32),
        "final_norm.weight": torch.randn(32),
        "lm_head.weight": torch.randn(32, 32),
    }
    for layer_idx in range(config.num_layers):
        prefix = f"layers.{layer_idx}"
        weights.update(
            {
                f"{prefix}.input_layernorm.weight": torch.randn(32),
                f"{prefix}.self_attn.qkv_proj.weight": torch.randn(64, 32),
                f"{prefix}.self_attn.o_proj.weight": torch.randn(32, 32),
                f"{prefix}.self_attn.q_norm.weight": torch.randn(8),
                f"{prefix}.self_attn.k_norm.weight": torch.randn(8),
                f"{prefix}.post_attention_layernorm.weight": torch.randn(32),
                f"{prefix}.mlp.gate_up_proj.weight": torch.randn(128, 32),
                f"{prefix}.mlp.down_proj.weight": torch.randn(32, 64),
            }
        )

    model.load_logical_weights(weights)

    assert torch.equal(model.embed_tokens.weight, weights["embed_tokens.weight"])
    assert torch.equal(model.norm.weight, weights["final_norm.weight"])
    assert torch.equal(model.lm_head.weight, weights["lm_head.weight"])
    assert torch.equal(
        model.layers[1].ops.qkv_proj.weight,
        weights["layers.1.self_attn.qkv_proj.weight"],
    )
