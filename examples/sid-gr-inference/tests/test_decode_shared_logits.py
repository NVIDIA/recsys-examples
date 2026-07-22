# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared decode logits output buffer.

Covers:
  * ``_linear_project(..., out=buffer)`` writes the lm_head projection into the
    caller-supplied buffer (and returns it) instead of allocating fresh.
  * ``GRDecodeCudaGraphRunner._shared_logits_for`` returns one buffer per
    ``(active_beam_width, bucket)`` shape, reused across calls.

CPU-only (no GPU required).
"""

import types

import torch

from gr_inference.gr_models.qwen3.model import _linear_project
from gr_inference.gr_serving.decode_cuda_graph import GRDecodeCudaGraphRunner


def _linear(hidden: int, vocab: int) -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(hidden, vocab, bias=False)


# _linear_project is an inference helper; in production it always runs under
# forward_decode_step's @torch.no_grad(). Mirror that here: matmul's out= form
# rejects autograd-tracked inputs (the lm_head weight requires grad).
@torch.no_grad()
def test_linear_project_out_writes_into_buffer_and_returns_it():
    linear = _linear(hidden=8, vocab=16)
    hidden_states = torch.randn(2, 3, 8)            # [B, beam, hidden]
    out = torch.empty(2, 3, 16)

    ret = _linear_project(linear, hidden_states, out=out)

    expected = linear(hidden_states.reshape(-1, 8)).reshape(2, 3, 16)
    assert ret is out
    assert torch.allclose(out, expected)


@torch.no_grad()
def test_linear_project_default_path_unchanged():
    linear = _linear(hidden=8, vocab=16)
    hidden_states = torch.randn(2, 3, 8)

    result = _linear_project(linear, hidden_states)

    expected = linear(hidden_states.reshape(-1, 8)).reshape(2, 3, 16)
    assert torch.allclose(result, expected)


class _FakeModel:
    def __init__(self, vocab: int) -> None:
        self.lm_head = torch.nn.Linear(8, vocab, bias=False)


class _FakeDecodeEngine:
    pass


def _runner(vocab: int = 16) -> GRDecodeCudaGraphRunner:
    return GRDecodeCudaGraphRunner(_FakeModel(vocab), _FakeDecodeEngine())


def test_shared_logits_for_caches_per_beam_bucket():
    runner = _runner()
    bt1 = torch.empty((1, 1024), dtype=torch.long)   # bucket=1
    bt2 = torch.empty((2, 1024), dtype=torch.long)   # bucket=2

    b1 = runner._shared_logits_for(bt1, active_beam_width=1024)
    b1_again = runner._shared_logits_for(bt1, active_beam_width=1024)
    b2 = runner._shared_logits_for(bt2, active_beam_width=1024)

    assert tuple(b1.shape) == (1, 1024, 16)
    assert tuple(b2.shape) == (2, 1024, 16)
    assert b1 is b1_again        # same (beam, bucket) -> one shared buffer
    assert b1 is not b2          # different bucket -> distinct buffer
    assert len(runner._shared_logits) == 2


def test_shared_logits_for_separates_by_beam_width():
    runner = _runner()
    bt = torch.empty((1, 1024), dtype=torch.long)

    wide = runner._shared_logits_for(bt, active_beam_width=1024)
    narrow = runner._shared_logits_for(bt, active_beam_width=256)

    assert wide is not narrow                     # different beam -> distinct
    assert tuple(narrow.shape) == (1, 256, 16)


def test_shared_logits_for_matches_lm_head_dtype_and_vocab():
    runner = _runner(vocab=32)
    bt = torch.empty((4, 1024), dtype=torch.long)

    buf = runner._shared_logits_for(bt, active_beam_width=1024)

    assert tuple(buf.shape) == (4, 1024, 32)
    assert buf.dtype == runner.model.lm_head.weight.dtype


def _beam_token_ids(bucket: int):
    return torch.empty((bucket, 1024), dtype=torch.long)


def _decode_entry(beam_width: int, bucket: int):
    """Minimal stand-in for ``_DecodeGraphEntry`` for store/eviction tests."""
    return types.SimpleNamespace(
        active_beam_width=beam_width,
        beam_token_ids=torch.empty((bucket, beam_width), dtype=torch.long),
    )


def test_shared_logits_buffer_freed_when_its_shape_is_evicted():
    runner = _runner()
    runner.max_entries = 1  # only one live graph: storing a 2nd evicts the 1st

    runner._shared_logits_for(_beam_token_ids(1), active_beam_width=1024)
    runner._store_graph(("a",), _decode_entry(1024, 1))
    assert (1024, 1) in runner._shared_logits

    runner._shared_logits_for(_beam_token_ids(2), active_beam_width=1024)
    runner._store_graph(("b",), _decode_entry(1024, 2))  # evicts ("a",)

    assert (1024, 1) not in runner._shared_logits  # evicted shape -> buffer freed
    assert (1024, 2) in runner._shared_logits       # live shape -> buffer kept
    assert len(runner._shared_logits) == 1


def test_shared_logits_buffer_kept_until_last_live_graph_evicted():
    runner = _runner()
    runner.max_entries = 2

    runner._shared_logits_for(_beam_token_ids(1), active_beam_width=1024)
    runner._store_graph(("a",), _decode_entry(1024, 1))
    runner._store_graph(("b",), _decode_entry(1024, 1))  # two live, same shape

    runner._shared_logits_for(_beam_token_ids(2), active_beam_width=1024)
    runner._store_graph(("c",), _decode_entry(1024, 2))  # evicts ("a",); ("b",) keeps (1024,1)
    assert (1024, 1) in runner._shared_logits

    runner._shared_logits_for(_beam_token_ids(3), active_beam_width=1024)
    runner._store_graph(("d",), _decode_entry(1024, 3))  # evicts ("b",); no live (1024,1)
    assert (1024, 1) not in runner._shared_logits


def test_shared_logits_buffer_kept_for_just_captured_entry_when_caching_disabled():
    runner = _runner()
    runner.max_entries = 0  # base _store_graph retains no graph entries

    buf = runner._shared_logits_for(_beam_token_ids(1), active_beam_width=1024)
    runner._store_graph(("a",), _decode_entry(1024, 1))

    # max_entries=0 leaves _graphs empty, but the caller replays the just-captured
    # graph immediately after _capture, so its output buffer must stay alive (and
    # is reused by the next same-shape capture instead of being re-allocated).
    assert (1024, 1) in runner._shared_logits
    assert runner._shared_logits[(1024, 1)] is buf
    assert runner._shared_logits_for(_beam_token_ids(1), active_beam_width=1024) is buf
