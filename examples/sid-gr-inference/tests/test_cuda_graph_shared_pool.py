# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared CUDA-graph private pool (decode + prefill runners).

Covers the deduplicated pool-sharing helper that now lives on
``CudaGraphCacheMixin``: env opt-out, lazy handle creation + reuse, and the
contract that prefill/decode runners inherit (do not redefine) the helper with
distinct ``_separate_pools_env`` class attributes.

CPU-only (no GPU required): the real ``torch`` is faked at the call site.
"""

from gr_inference.gr_serving.cuda_graph_utils import CudaGraphCacheMixin


class _FakeHandle:
    def __init__(self, tag: int) -> None:
        self.tag = tag


class _FakeTorchCuda:
    def __init__(self) -> None:
        self.calls = 0

    def graph_pool_handle(self):
        self.calls += 1
        return _FakeHandle(self.calls)


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeTorchCuda()


class _Runner(CudaGraphCacheMixin):
    _separate_pools_env = "TEST_SEPARATE_POOLS_ENV"

    def __init__(self) -> None:
        self._graph_pool = None


def test_capture_kwargs_shares_pool_and_reuses_handle():
    runner = _Runner()
    torch = _FakeTorch()
    kw1 = runner._graph_capture_kwargs(torch)
    kw2 = runner._graph_capture_kwargs(torch)

    assert "pool" in kw1
    assert kw1["pool"] is kw2["pool"]      # handle reused, not re-created
    assert torch.cuda.calls == 1           # handle created exactly once


def test_capture_kwargs_opts_out_via_env(monkeypatch):
    monkeypatch.setenv("TEST_SEPARATE_POOLS_ENV", "1")
    runner = _Runner()
    torch = _FakeTorch()

    assert runner._graph_capture_kwargs(torch) == {}   # no pool kwarg
    assert torch.cuda.calls == 0                       # handle never requested


def test_capture_kwargs_ignores_env_when_attr_unset():
    # Default _separate_pools_env == "" -> no opt-out possible, always shares.
    class _BareRunner(CudaGraphCacheMixin):
        pass

    runner = _BareRunner()
    torch = _FakeTorch()
    assert "pool" in runner._graph_capture_kwargs(torch)


def test_prefill_and_decode_inherit_helper_with_distinct_env_names():
    from gr_inference.gr_serving.prefill_cuda_graph import GRPrefillCudaGraphRunner
    from gr_inference.gr_serving.decode_cuda_graph import GRDecodeCudaGraphRunner

    assert (
        GRPrefillCudaGraphRunner._separate_pools_env
        == "GR_INFERENCE_PREFILL_CUDA_GRAPH_SEPARATE_POOLS"
    )
    assert (
        GRDecodeCudaGraphRunner._separate_pools_env
        == "GR_INFERENCE_DECODE_CUDA_GRAPH_SEPARATE_POOLS"
    )
    # Dedup contract: neither runner redefines the helper; both inherit it.
    assert "_graph_capture_kwargs" not in GRPrefillCudaGraphRunner.__dict__
    assert "_graph_capture_kwargs" not in GRDecodeCudaGraphRunner.__dict__
