import importlib.util

import pytest
from gr_inference.gr_kv import ContextKV, TensorSpec
from gr_inference.gr_runtime import (
    BeamSelection,
    GRGenerationState,
    PrefillResult,
    allocate_beam_kv_like_context,
)


def test_allocate_beam_kv_from_context_tensor_spec() -> None:
    context_kv = ContextKV(
        TensorSpec("context_k", (2, 1, 16, 2, 8)),
        TensorSpec("context_v", (2, 1, 16, 2, 8)),
    )

    beam_kv = allocate_beam_kv_like_context(
        context_kv,
        max_decode_steps=3,
        max_beam_width=8,
    )

    assert beam_kv.key_shape == (2, 1, 3, 8, 2, 8)
    assert beam_kv.flattened_beam_shape() == (1, 24, 2, 8)


def test_generation_state_builds_request_state() -> None:
    prefill = PrefillResult(
        logits=TensorSpec("logits", (1, 16, 128)),
        context_kv=ContextKV(
            TensorSpec("context_k", (2, 1, 16, 2, 8)),
            TensorSpec("context_v", (2, 1, 16, 2, 8)),
        ),
    )

    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=3,
        max_beam_width=8,
        fixed_beam_width=4,
    )
    request = generation.request_state()

    assert request.request_id == "req-1"
    assert request.context_kv is prefill.context_kv
    assert request.beam_kv.key_shape == (2, 1, 3, 8, 2, 8)
    assert request.beam_path.max_decode_steps == 4
    assert request.beam_path.active_beam_width == 1
    assert generation.fixed_beam_width == 4


def test_generation_state_initializes_beams_from_prefill_logits() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    prefill = PrefillResult(
        logits=torch.tensor([[[0.0, 5.0, 1.0, 3.0, 2.0]]]),
        context_kv=ContextKV(
            TensorSpec("context_k", (2, 1, 16, 2, 8)),
            TensorSpec("context_v", (2, 1, 16, 2, 8)),
        ),
    )
    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=3,
        max_beam_width=4,
        fixed_beam_width=3,
    )

    selection = generation.initialize_beams()

    assert selection.token_ids == (1, 3, 4)
    assert generation.beam_path.steps_done == 1
    assert generation.beam_path.active_beam_width == 3
    assert generation.beam_path.token_trace(beam=1) == (3,)


def test_generation_state_initializes_beams_from_precomputed_selection() -> None:
    prefill = PrefillResult(
        logits=TensorSpec("logits", (1, 16, 128)),
        context_kv=ContextKV(
            TensorSpec("context_k", (2, 1, 16, 2, 8)),
            TensorSpec("context_v", (2, 1, 16, 2, 8)),
        ),
    )
    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=3,
        max_beam_width=4,
        fixed_beam_width=3,
    )
    selection = BeamSelection(
        token_ids=(7, 5, 3),
        scores=(9.0, 8.0, 7.0),
        parent_beams=(0, 0, 0),
    )

    returned = generation.initialize_beams_from_selection(selection)

    assert returned is selection
    assert generation.beam_path.steps_done == 1
    assert generation.beam_path.active_beam_width == 3
    assert generation.beam_path.token_trace(beam=1) == (5,)


def test_generation_state_updates_beams_from_decode_logits() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    prefill = PrefillResult(
        logits=torch.tensor([[[0.0, 5.0, 1.0, 3.0, 2.0]]]),
        context_kv=ContextKV(
            TensorSpec("context_k", (2, 1, 16, 2, 8)),
            TensorSpec("context_v", (2, 1, 16, 2, 8)),
        ),
    )
    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=3,
        max_beam_width=4,
        fixed_beam_width=2,
    )
    generation.initialize_beams()
    logits = torch.tensor([[[0.0, 10.0, 0.0], [9.0, 0.0, 0.0]]])

    selection = generation.update_beams_from_logits(logits)

    assert selection.parent_beams == (0, 1)
    assert generation.beam_path.steps_done == 2
    assert generation.beam_path.token_trace(beam=0) == (1, 1)


def test_qwen3_model_prefill_can_return_generation_state() -> None:
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
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    prefill = model.forward_prefill(input_ids, return_result=True)

    generation = GRGenerationState.from_prefill(
        request_id="req-1",
        prefill=prefill,
        max_decode_steps=config.max_decode_steps,
        max_beam_width=config.max_beam_width,
    )

    assert tuple(prefill.logits.shape) == (1, 16, config.vocab_size)
    assert generation.request_state().beam_kv.key_shape == (1, 1, 3, 8, 2, 8)
