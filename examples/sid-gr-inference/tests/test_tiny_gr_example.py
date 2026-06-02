# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from argparse import Namespace

import pytest


def test_tiny_gr_example_run_demo() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    from run_tiny_gr import run_demo

    summary = run_demo(
        Namespace(
            layers=2,
            context_len=16,
            decode_steps=2,
            beam_width=3,
            vocab_size=128,
            seed=0,
            constraint_demo=False,
            beam_schedule=None,
        )
    )

    assert summary["input_shape"] == (1, 16)
    assert summary["context_kv_shape"] == (2, 1, 16, 2, 8)
    assert summary["beam_kv_shape"] == (2, 1, 2, 3, 2, 8)
    assert summary["steps"] == 2
    assert summary["beam_path_steps"] == 3
    assert len(summary["final_token_ids"]) == 3


def test_tiny_gr_example_constraint_demo() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    from run_tiny_gr import run_demo

    summary = run_demo(
        Namespace(
            layers=1,
            context_len=8,
            decode_steps=2,
            beam_width=2,
            vocab_size=64,
            seed=0,
            constraint_demo=True,
            beam_schedule=None,
        )
    )

    assert summary["constraint_demo"] is True
    assert summary["beam_path_steps"] == 3
    assert len(summary["final_token_ids"]) == 2


def test_tiny_gr_example_dynamic_beam_schedule() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    from run_tiny_gr import run_demo

    summary = run_demo(
        Namespace(
            layers=1,
            context_len=8,
            decode_steps=2,
            beam_width=3,
            vocab_size=64,
            seed=0,
            constraint_demo=False,
            beam_schedule="0:3,1:2,2:1",
        )
    )

    assert summary["beam_schedule"] == "0:3,1:2,2:1"
    assert summary["beam_path_steps"] == 3
    assert len(summary["final_token_ids"]) == 1
