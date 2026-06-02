import importlib.util

import pytest

from gr_inference.gr_kv import BeamKV, ContextKV, TensorSpec


def test_context_kv_shape_properties() -> None:
    key = TensorSpec("context_key", (28, 2, 4700, 8, 128))
    value = TensorSpec("context_value", (28, 2, 4700, 8, 128))

    kv = ContextKV(key, value)

    assert kv.num_layers == 28
    assert kv.batch_size == 2
    assert kv.context_len == 4700
    assert kv.num_kv_heads == 8
    assert kv.head_dim == 128
    assert kv.expected_layer_shape() == (2, 4700, 8, 128)


def test_context_kv_rejects_mismatched_shapes() -> None:
    key = TensorSpec("context_key", (28, 2, 4700, 8, 128))
    value = TensorSpec("context_value", (28, 2, 4700, 8, 64))

    with pytest.raises(ValueError, match="key/value shapes differ"):
        ContextKV(key, value)


def test_context_kv_slice_batch() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    key = torch.randn(2, 3, 4, 1, 8)
    value = torch.randn(2, 3, 4, 1, 8)
    kv = ContextKV(key, value)

    sliced = kv.slice_batch(1)

    assert sliced.key_shape == (2, 1, 4, 1, 8)
    assert torch.equal(sliced.key[:, 0], key[:, 1])


def test_beam_kv_step_major_layout() -> None:
    key = TensorSpec("beam_key", (28, 2, 3, 128, 8, 128))
    value = TensorSpec("beam_value", (28, 2, 3, 128, 8, 128))

    kv = BeamKV(key, value)
    step = kv.validate_step(step=2, active_beam_width=64)

    assert kv.flattened_beam_shape() == (2, 384, 8, 128)
    assert step.expected_step_shape == (28, 2, 64, 8, 128)
    assert step.flat_offset == 256


def test_beam_kv_rejects_width_overflow() -> None:
    kv = BeamKV(
        TensorSpec("beam_key", (28, 1, 3, 128, 8, 128)),
        TensorSpec("beam_value", (28, 1, 3, 128, 8, 128)),
    )

    with pytest.raises(ValueError, match="exceeds max_beam_width"):
        kv.validate_step(step=0, active_beam_width=129)
