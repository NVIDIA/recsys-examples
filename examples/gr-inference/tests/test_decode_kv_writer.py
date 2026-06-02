import importlib.util

import pytest
from gr_inference.gr_kv import BeamKV, TensorSpec
from gr_inference.gr_runtime import BeamKVWriter


def test_beam_kv_writer_validates_shape_only_specs() -> None:
    beam_kv = BeamKV(
        TensorSpec("beam_k", (2, 1, 3, 8, 2, 8)),
        TensorSpec("beam_v", (2, 1, 3, 8, 2, 8)),
    )
    writer = BeamKVWriter(beam_kv)

    write = writer.validate_layer_step(
        layer_idx=1,
        step=2,
        active_beam_width=4,
        k=TensorSpec("k_step", (1, 4, 2, 8)),
        v=TensorSpec("v_step", (1, 4, 2, 8)),
    )

    assert write.expected_layer_step_shape == (1, 4, 2, 8)
    assert write.flat_offset == 16


def test_beam_kv_writer_rejects_bad_shape() -> None:
    beam_kv = BeamKV(
        TensorSpec("beam_k", (2, 1, 3, 8, 2, 8)),
        TensorSpec("beam_v", (2, 1, 3, 8, 2, 8)),
    )
    writer = BeamKVWriter(beam_kv)

    with pytest.raises(ValueError, match="k step shape"):
        writer.validate_layer_step(
            layer_idx=0,
            step=0,
            active_beam_width=4,
            k=TensorSpec("k_step", (1, 5, 2, 8)),
            v=TensorSpec("v_step", (1, 4, 2, 8)),
        )


def test_beam_kv_writer_writes_torch_tensors() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    beam_k = torch.zeros(2, 1, 3, 8, 2, 8)
    beam_v = torch.zeros_like(beam_k)
    beam_kv = BeamKV(beam_k, beam_v)
    writer = BeamKVWriter(beam_kv)
    k_step = torch.ones(1, 4, 2, 8)
    v_step = torch.full((1, 4, 2, 8), 2.0)

    writer.write_layer_step(
        layer_idx=1,
        step=2,
        active_beam_width=4,
        k=k_step,
        v=v_step,
    )

    assert torch.equal(beam_k[1, :, 2, :4], k_step)
    assert torch.equal(beam_v[1, :, 2, :4], v_step)
    assert torch.equal(beam_k[0], torch.zeros_like(beam_k[0]))
    assert torch.equal(beam_k[1, :, 2, 4:], torch.zeros_like(beam_k[1, :, 2, 4:]))


def test_cuda_beam_kv_writer_handles_distinct_k_v_strides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    monkeypatch.setenv("GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT", "1")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "torch_ext"))

    from gr_inference_trtllm_kernels import qwen3

    qwen3._CUDA_EXTENSION = None
    qwen3._CUDA_EXTENSION_FAILED = False
    qwen3.reset_call_counts()
    if qwen3._cuda_extension() is None:
        pytest.skip("gr_trtllm CUDA extension is not available")

    batch, beam_width, q_heads, kv_heads, head_dim = 2, 4, 4, 2, 8
    qkv = torch.randn(
        batch,
        beam_width,
        (q_heads + 2 * kv_heads) * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    k_step = qkv[..., q_size : q_size + kv_size].view(
        batch,
        beam_width,
        kv_heads,
        head_dim,
    )
    # FlashInfer RoPE returns a contiguous K tensor, while V remains the QKV
    # split view. The fused writer must honor the independent V stride.
    k_step = k_step.contiguous()
    v_step = qkv[..., q_size + kv_size :].view(
        batch,
        beam_width,
        kv_heads,
        head_dim,
    )
    assert k_step.stride() != v_step.stride()

    beam_k = torch.zeros(
        3,
        batch,
        2,
        beam_width,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    beam_v = torch.zeros_like(beam_k)
    expected_k = beam_k.clone()
    expected_v = beam_v.clone()
    expected_k[1, :, 1, :beam_width] = k_step
    expected_v[1, :, 1, :beam_width] = v_step

    writer = BeamKVWriter(BeamKV(beam_k, beam_v))
    writer.write_layer_step(
        layer_idx=1,
        step=1,
        active_beam_width=beam_width,
        k=k_step,
        v=v_step,
    )
    torch.cuda.synchronize()

    assert qwen3.call_counts().get("beam_kv_write_cuda") == 1
    assert torch.equal(beam_k, expected_k)
    assert torch.equal(beam_v, expected_v)
