# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving metrics helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ServingMetrics:
    """Simple per-request timing metrics."""

    values_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values_ms[name] = (time.perf_counter() - start) * 1000

    def to_metadata(self) -> dict[str, float]:
        return dict(self.values_ms)
