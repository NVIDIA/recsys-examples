# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight timing helpers for profiling runtime stages."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingRecorder:
    """Accumulate wall-clock milliseconds by section name."""

    sync_module: Any | None = None
    detail: str = "coarse"
    emit_nvtx: bool | None = None
    sync_timing: bool = True
    totals_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self) -> None:
        if self.emit_nvtx is None:
            self.emit_nvtx = os.environ.get("GR_INFERENCE_NVTX", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

    @contextmanager
    def section(self, name: str):
        nvtx_pushed = self._nvtx_push(name)
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals_ms[name] += (time.perf_counter() - start) * 1000
            self.counts[name] += 1
            if nvtx_pushed:
                self._nvtx_pop()

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "total_ms": total,
                "count": self.counts[name],
                "avg_ms": total / self.counts[name],
            }
            for name, total in sorted(self.totals_ms.items())
        }

    def _sync(self) -> None:
        if not self.sync_timing:
            return
        module = self.sync_module
        if (
            module is not None
            and hasattr(module, "cuda")
            and module.cuda.is_available()
        ):
            module.cuda.synchronize()

    def _nvtx_push(self, name: str) -> bool:
        if not self.emit_nvtx:
            return False
        module = self.sync_module
        nvtx = getattr(getattr(module, "cuda", None), "nvtx", None)
        push = getattr(nvtx, "range_push", None)
        if push is None:
            return False
        push(name)
        return True

    def _nvtx_pop(self) -> None:
        module = self.sync_module
        nvtx = getattr(getattr(module, "cuda", None), "nvtx", None)
        pop = getattr(nvtx, "range_pop", None)
        if pop is not None:
            pop()
