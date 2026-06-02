# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime output metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecodeStepOutput:
    """Output from one fixed-beam decode attention step."""

    request_id: str
    layer_idx: int
    step: int
    active_beam_width: int
    attention_output: Any
