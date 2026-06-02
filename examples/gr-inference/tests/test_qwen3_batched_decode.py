import importlib.util

import pytest


def test_qwen3_model_decode_step_accepts_batched_beam_tokens() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_kernels.attention import GRDecodeAttention
    from gr_inference.gr_kernels.prefill import PrefillAttention, TorchSDPAPrefillBackend
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig, Qwen3GRModel
    from gr_inference.gr_runtime import GRDecodeEngine, GRGenerationState

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
        max_beam_width=4,
        intermediate_size=64,
        vocab_size=64,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    prefill = model.forward_prefill(input_ids, return_result=True)
    generation = GRGenerationState.from_prefill(
        request_id="batched-req",
        prefill=prefill,
        max_decode_steps=1,
        max_beam_width=4,
        fixed_beam_width=3,
    )
    beam_token_ids = torch.randint(0, config.vocab_size, (2, 3))
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=3,
    )

    logits = model.forward_decode_step(
        beam_token_ids,
        generation,
        decode_engine,
        step=0,
        active_beam_width=3,
    )

    assert tuple(logits.shape) == (2, 3, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(generation.beam_kv.key[:, :, 0, :3]).all()
    assert torch.isfinite(generation.beam_kv.value[:, :, 0, :3]).all()

