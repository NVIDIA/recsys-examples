# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import math
import re
import struct

import pytest
from gr_inference.gr_kernels.attention import (
    ExistingGRDecodeAttentionBackend,
    GRDecodeAttentionInputs,
    MissingKernelBackend,
    existing_kernel_backend,
)
from gr_inference.gr_kv import BeamKV, BeamPath, ContextKV

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


@pytest.fixture
def interface():
    try:
        ExistingGRDecodeAttentionBackend().ensure_available()
    except MissingKernelBackend as exc:
        pytest.skip(str(exc))
    return importlib.import_module("interface")


@pytest.fixture
def fused(interface):
    return importlib.import_module(
        "gr_inference.gr_kernels.attention.fused_beam_decode"
    )


@pytest.fixture
def fused_launches(fused, monkeypatch):
    launches = []
    launch = fused._launch_fused_beam_combine
    monkeypatch.setattr(
        fused,
        "_launch_fused_beam_combine",
        lambda *args: launches.append(1) or launch(*args),
    )
    return launches


def _inputs(
    decode_nums,
    q_heads=16,
    head_dim=128,
    dtype=torch.bfloat16,
    width=64,
    max_decode=4,
    batch=3,
    context_len=257,
):
    gen = torch.Generator(device="cuda").manual_seed(7)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=dtype)

    kv_heads = 4
    q = randn(batch, 1, width, q_heads, head_dim)
    context_k = randn(batch, context_len, kv_heads, head_dim)
    context_v = randn(batch, context_len, kv_heads, head_dim)
    beam_k = randn(batch, decode_nums * width, kv_heads, head_dim)
    beam_v = randn(batch, decode_nums * width, kv_heads, head_dim)
    topk_kv = torch.randint(
        0,
        max(decode_nums * width, 1),
        (batch, 1, kv_heads, max_decode, width),
        generator=gen,
        device="cuda",
    ).to(torch.int32)
    topk = topk_kv.repeat_interleave(q_heads // kv_heads, dim=2)
    return q, context_k, context_v, beam_k, beam_v, topk, decode_nums


def _seqused_k(batch=3):
    lens = (17, 129, 257, 64)[:batch]
    return torch.tensor(lens, device="cuda", dtype=torch.int32)


def _reference(
    interface,
    q,
    k_context,
    v_context,
    k_beam,
    v_beam,
    topk,
    decode_nums,
    seqused_k=None,
    cu_seqlens_k=None,
):
    """The public 3-kernel pipeline with one context split."""
    B, _, W, Hq, D = q.shape
    scale = 1.0 / math.sqrt(D)
    o_partial = torch.empty(2, B, W, Hq, D, device="cuda", dtype=torch.float32)
    lse_partial = torch.empty(2, B, Hq, W, device="cuda", dtype=torch.float32)
    out = torch.empty(B, W, Hq, D, device="cuda", dtype=q.dtype)
    lse = torch.empty(B, Hq, W, device="cuda", dtype=torch.float32).transpose(-1, -2)
    interface._context_attention(
        q,
        k_context,
        v_context,
        scale,
        out=o_partial[0],
        lse=lse_partial[0],
        seqused_k=seqused_k,
        cu_seqlens_k=cu_seqlens_k,
    )
    interface._beam_sparse_attention(
        q,
        k_beam,
        v_beam,
        topk,
        decode_nums,
        scale,
        out=o_partial[1],
        lse=lse_partial[1],
    )
    interface._combine(o_partial, lse_partial.transpose(-1, -2), out, lse)
    return out.unsqueeze(1), lse.unsqueeze(1)


def _assert_bit_identical(actual, expected):
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("q_heads", [4, 8, 16])  # 1, 2 and 4 query heads per KV head
@pytest.mark.parametrize("decode_nums", [2, 4])
@pytest.mark.parametrize("captured", [False, True])
def test_fused_beam_decode_is_bit_identical_to_the_public_three_kernel_path(
    interface, fused, fused_launches, decode_nums, captured, q_heads, dtype
) -> None:
    args = _inputs(decode_nums, q_heads, dtype=dtype)
    seqused_k = _seqused_k()

    def launch():
        return fused.beam_decode_fused(
            *args, return_lse=True, backend="3kernel", seqused_k=seqused_k
        )

    launch()  # compiles outside the capture
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = launch()
        args[0].copy_(torch.randn_like(args[0]))  # the replay reads the live inputs
        graph.replay()
        torch.cuda.synchronize()
    else:
        out = launch()

    assert fused_launches
    _assert_bit_identical(out, _reference(interface, *args, seqused_k=seqused_k))


@pytest.mark.parametrize("width", [1, 5, 33, 300])  # not multiples of the block shape
def test_fused_beam_decode_is_bit_identical_at_any_beam_width(
    interface, fused, fused_launches, width
) -> None:
    args = _inputs(2, 8, dtype=torch.float16, width=width)

    out = fused.beam_decode_fused(*args, return_lse=True, backend="3kernel")

    assert fused_launches
    _assert_bit_identical(out, _reference(interface, *args))


@pytest.mark.parametrize("context", ["dense", "seqused_k", "cu_seqlens_k"])
def test_fused_beam_decode_is_bit_identical_for_every_context_layout(
    interface, fused, fused_launches, context
) -> None:
    q, context_k, context_v, *beam = _inputs(4, 8)
    kwargs = {}
    if context == "seqused_k":
        kwargs["seqused_k"] = _seqused_k()
    if context == "cu_seqlens_k":
        lens = _seqused_k().tolist()
        context_k = torch.cat([context_k[b, :n] for b, n in enumerate(lens)])
        context_v = torch.cat([context_v[b, :n] for b, n in enumerate(lens)])
        kwargs["cu_seqlens_k"] = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
    args = (q, context_k, context_v, *beam)

    out = fused.beam_decode_fused(*args, return_lse=True, backend="3kernel", **kwargs)

    assert fused_launches
    _assert_bit_identical(out, _reference(interface, *args, **kwargs))


@pytest.mark.parametrize("q_heads", [4, 8, 16])
# many register stages, some ending mid-stage
@pytest.mark.parametrize("decode_nums", [9, 40])
def test_fused_beam_decode_is_bit_identical_at_long_decode_depths(
    interface, fused, fused_launches, decode_nums, q_heads
) -> None:
    args = _inputs(decode_nums, q_heads, max_decode=40)

    out = fused.beam_decode_fused(*args, return_lse=True, backend="3kernel")

    assert fused_launches
    _assert_bit_identical(out, _reference(interface, *args))


def test_fused_beam_decode_never_follows_top_k_entries_past_decode_nums(
    interface, fused, fused_launches
) -> None:
    *args, topk, decode_nums = _inputs(2, 8, dtype=torch.float16)
    bad_topk = topk.clone()
    bad_topk[:, :, :, 2:, :] = torch.tensor(
        [2**30, -(2**30)], dtype=torch.int32, device="cuda"
    )[torch.arange(topk.shape[-1], device="cuda") % 2]

    out = fused.beam_decode_fused(
        *args, bad_topk, decode_nums, return_lse=True, backend="3kernel"
    )
    torch.cuda.synchronize()

    assert fused_launches
    _assert_bit_identical(out, _reference(interface, *args, topk, decode_nums))


def test_fused_beam_decode_handles_zero_decode_nums(
    interface, fused, fused_launches
) -> None:
    args = _inputs(0, 8)

    out = fused.beam_decode_fused(*args, return_lse=True, backend="3kernel")

    assert fused_launches
    _assert_bit_identical(
        out, interface.beam_decode_attn(*args, return_lse=True, backend="3kernel")
    )


def test_fused_beam_decode_writes_into_out(interface, fused, fused_launches) -> None:
    args = _inputs(2, 8)
    out = torch.empty_like(args[0])

    result, lse = fused.beam_decode_fused(*args, out=out, backend="3kernel")

    assert fused_launches
    assert result is out and lse is None
    assert torch.equal(out, _reference(interface, *args)[0])


def _assert_runs_the_public_path(interface, fused, fused_launches, args, **kwargs):
    out = fused.beam_decode_fused(*args, return_lse=True, **kwargs)

    assert not fused_launches
    _assert_bit_identical(
        out, interface.beam_decode_attn(*args, return_lse=True, **kwargs)
    )


def test_fused_beam_decode_runs_the_public_path_for_head_dims_it_is_not_validated_for(
    interface, fused, fused_launches
) -> None:
    args = _inputs(4, 8, head_dim=64)

    _assert_runs_the_public_path(
        interface, fused, fused_launches, args, backend="3kernel"
    )


def test_fused_beam_decode_runs_the_public_path_for_int64_top_k(
    interface, fused, fused_launches, monkeypatch
) -> None:
    # The public sparse-attention compile key leaves out the top-k dtype.
    monkeypatch.setattr(interface, "_compile_cache", {})
    *args, topk, decode_nums = _inputs(2, 8)

    _assert_runs_the_public_path(
        interface,
        fused,
        fused_launches,
        (*args, topk.to(torch.int64), decode_nums),
        backend="3kernel",
    )


def test_fused_beam_decode_runs_the_public_path_where_context_attention_splits(
    interface, fused, fused_launches
) -> None:
    # One request with a long context leaves most SMs idle, so the public heuristic splits it.
    args = _inputs(2, 4, batch=1, context_len=2048)
    assert fused._context_num_splits(args[0], args[1], None, None) > 1

    _assert_runs_the_public_path(
        interface, fused, fused_launches, args, backend="3kernel"
    )


def test_fused_beam_decode_follows_the_public_backend_choice(
    interface, fused, fused_launches
) -> None:
    # The public "dsl" backend runs the 3-kernel pipeline only on SM10x/SM11x.
    args = _inputs(2, 8)

    out = fused.beam_decode_fused(*args, return_lse=True, backend="dsl")

    assert bool(fused_launches) == (torch.cuda.get_device_capability()[0] in (10, 11))
    _assert_bit_identical(
        out, interface.beam_decode_attn(*args, return_lse=True, backend="dsl")
    )


def test_fused_beam_decode_checks_its_inputs(fused) -> None:
    *args, decode_nums = _inputs(2, 8)

    with pytest.raises(ValueError, match="host int"):
        fused.beam_decode_fused(
            *args,
            torch.tensor(decode_nums, dtype=torch.int32, device="cuda"),
            backend="3kernel",
        )

    with pytest.raises(AssertionError, match="k_beam shape"):
        fused.beam_decode_fused(*args, decode_nums - 1, backend="3kernel")


def _backend_inputs():
    gen = torch.Generator(device="cuda").manual_seed(11)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.bfloat16)

    layers, batch, width, active_width, q_heads, kv_heads = 2, 3, 64, 48, 16, 4
    head_dim, context_len, max_decode, decode_nums = 128, 129, 4, 2
    context_kv = ContextKV(
        randn(layers, batch, context_len, kv_heads, head_dim),
        randn(layers, batch, context_len, kv_heads, head_dim),
    )
    beam_shape = (layers, batch, max_decode, width, kv_heads, head_dim)
    beam_kv = BeamKV(randn(*beam_shape), randn(*beam_shape))
    topk = torch.randint(
        0,
        decode_nums * active_width,
        (batch, 1, q_heads, max_decode, active_width),
        generator=gen,
        device="cuda",
    ).to(torch.int32)
    return GRDecodeAttentionInputs(
        q=randn(batch, active_width, q_heads, head_dim),
        context_kv=context_kv,
        beam_kv=beam_kv,
        beam_path=BeamPath(max_decode_steps=max_decode, max_beam_width=width),
        layer_idx=1,
        step=decode_nums,
        active_beam_width=active_width,
        topk_indices=topk,
        decode_nums=decode_nums,
        return_lse=True,
        backend_name="3kernel",
    )


def test_backend_runs_the_fused_kernel_bit_identically(
    interface, fused_launches, monkeypatch
) -> None:
    # The kernel runs on any GPU; open the SM10x gate so the routing runs wherever this test does.
    monkeypatch.setattr(
        existing_kernel_backend, "_cuda_capability_major", lambda _device: 10
    )
    inputs = _backend_inputs()

    out = ExistingGRDecodeAttentionBackend(fused_beam_combine=True)(inputs)
    assert fused_launches
    _assert_bit_identical(out, ExistingGRDecodeAttentionBackend()(inputs))


@pytest.mark.parametrize("fused_beam_combine", [True, False])
def test_backend_runs_on_the_tensors_device(
    interface, fused_launches, monkeypatch, fused_beam_combine
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    monkeypatch.setattr(
        existing_kernel_backend, "_cuda_capability_major", lambda _device: 10
    )
    backend = ExistingGRDecodeAttentionBackend(fused_beam_combine=fused_beam_combine)
    with torch.cuda.device(1):
        inputs = _backend_inputs()
        expected = backend(inputs)

    _assert_bit_identical(backend(inputs), expected)
    assert bool(fused_launches) is fused_beam_combine


def _compile_for_sm100(interface, fused, tmp_path, max_decode):
    """The fused kernel's PTX and cubin for SM100 (B200), compiled without running,
    so any GPU can check them."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from src.common.cute_dsl_utils import to_cute_tensor

    tmp_path.mkdir(parents=True, exist_ok=True)
    batch, width, q_heads, kv_heads, head_dim = 1, 64, 8, 4, 128

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(*shape, device="cuda", dtype=dtype)

    kernel = fused.BeamSparseCombineVec(
        interface.TORCH_TO_CUTLASS[torch.bfloat16],
        head_dim,
        q_heads // kv_heads,
        max_decode,
    )
    rows = zeros(batch, max_decode * width, kv_heads, head_dim)
    out = zeros(batch * width, 1, q_heads, head_dim)
    lse = to_cute_tensor(
        zeros(batch, q_heads, width, dtype=torch.float32), assumed_align=4
    )
    cute.compile(
        kernel,
        to_cute_tensor(out),
        to_cute_tensor(rows),
        to_cute_tensor(rows),
        to_cute_tensor(zeros(batch * width, 1, q_heads, head_dim, dtype=torch.float32)),
        lse,
        to_cute_tensor(out),
        lse,
        0.1,
        to_cute_tensor(zeros(batch, kv_heads, max_decode, width, dtype=torch.int32)),
        width,
        to_cute_tensor(zeros(1, dtype=torch.int32)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--keep-ptx --keep-cubin --gpu-arch sm_100a --dump-dir {tmp_path}",
    )
    ptx = next(tmp_path.glob("*.ptx")).read_text()
    return kernel, ptx, next(tmp_path.glob("*.cubin")).read_bytes()


def _registers_and_frame_bytes(cubin):
    """EIATTR_REGCOUNT and the largest EIATTR_FRAME_SIZE (local memory, where spills go)
    from the cubin's .nv.info."""
    shoff = struct.unpack_from("<Q", cubin, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", cubin, 0x3A)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + i * shentsize)
        for i in range(shnum)
    ]
    names = headers[shstrndx][4]
    info = next(
        h
        for h in headers
        if cubin[names + h[0] : cubin.index(b"\0", names + h[0])] == b".nv.info"
    )
    data = cubin[info[4] : info[4] + info[5]]
    registers, frame, at = None, 0, 0
    while at < len(data):
        fmt, attr = data[at], data[at + 1]
        size = struct.unpack_from("<H", data, at + 2)[0] if fmt == 0x04 else 0
        if attr == 0x2F:
            registers = struct.unpack_from("<I", data, at + 8)[0]
        elif attr == 0x11:
            frame = max(frame, struct.unpack_from("<I", data, at + 8)[0])
        # every record without a size field is 4 bytes, value or padding
        at += 4 + size if fmt == 0x04 else 4
    return registers, frame


@pytest.mark.parametrize("max_decode", [4, 40])
def test_fused_kernel_issues_its_first_top_k_indices_before_any_other_load(
    interface, fused, tmp_path, max_decode
) -> None:
    # The gathered rows wait on these indices, so they must not queue behind the Q load.
    kernel, ptx, _ = _compile_for_sm100(interface, fused, tmp_path, max_decode)
    lines = [line.strip() for line in ptx.splitlines()]
    index_loads = [
        i for i, line in enumerate(lines) if line.startswith("ld.volatile.global.s32")
    ]
    first_vector_load = next(
        i for i, line in enumerate(lines) if re.match(r"ld\.global(\.\w+)*\.v4", line)
    )

    assert sum(i < first_vector_load for i in index_loads) == min(
        max_decode, kernel.stage_rows
    )


def test_fused_kernel_never_spills_and_stops_adding_registers_with_decode_depth(
    interface, fused, tmp_path
) -> None:
    resources = [
        _registers_and_frame_bytes(
            _compile_for_sm100(interface, fused, tmp_path / str(depth), depth)[2]
        )
        for depth in (4, 40, 128)
    ]
    registers, frames = zip(*resources)

    assert frames == (0, 0, 0)
    assert registers[1] == registers[2]
