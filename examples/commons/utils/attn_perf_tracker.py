# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HSTU Attention performance tracker — measures per-iteration forward / backward
TFLOPS and MFU for the attention kernels.

Controlled by environment variables:
  - ``PRINT_HSTU_PERF``        : "1" to enable (default "0")
  - ``PRINT_HSTU_PERF_START``  : first iteration to print (default 0)
  - ``PRINT_HSTU_PERF_STOP``   : stop printing at this iteration, -1 = unlimited (default -1)

Usage: the perf hooks are automatically registered when ``PRINT_HSTU_PERF=1``
by :func:`create_hstu_attention` in ``hstu_attention.py``.  Each rank prints
its own statistics independently via ``debug_rank_all``.
"""

import os
from typing import List, Optional, Tuple

import torch
from commons.utils.logger import debug_rank_all
from commons.utils.perf import get_current_device_spec

# ---------------------------------------------------------------------------
# Environment variable knobs
# ---------------------------------------------------------------------------
PRINT_HSTU_PERF: bool = os.environ.get("PRINT_HSTU_PERF", "0") == "1"
_PRINT_HSTU_PERF_START: int = int(os.environ.get("PRINT_HSTU_PERF_START", "0"))
_PRINT_HSTU_PERF_STOP: int = int(os.environ.get("PRINT_HSTU_PERF_STOP", "-1"))

# ---------------------------------------------------------------------------
# Global accumulator
# ---------------------------------------------------------------------------


class _AttnPerfAccumulator:
    """Collects CUDA-event timing across all attention layers within one
    iteration, and lazily prints the *previous* iteration's stats at the start
    of the next iteration (to avoid synchronisation bubbles).

    Supports both **training** (fwd + bwd per iteration) and **inference /
    eval** (fwd-only) modes.  The mode is auto-detected:

    * *Training*: ``_num_layers`` is set when the first backward pass
      completes (``add_bwd`` count == ``add_fwd`` count).
    * *Inference after training*: ``_num_layers`` is already known; when
      ``add_fwd`` is called ``_num_layers`` times without any ``add_bwd``,
      the iteration is finalised automatically.
    * *Pure inference* (``_num_layers`` never set by bwd): call
      :meth:`step` at the end of each iteration, or the accumulator will
      auto-detect ``_num_layers`` from the *first* ``step()`` call and
      handle subsequent iterations automatically.
    """

    def __init__(self) -> None:
        self._num_layers: int = 0  # auto-detected from first iteration
        self._iter_idx: int = 0

        # Current iteration bookkeeping
        self._cur_fwd_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cur_fwd_flops: float = 0.0
        self._cur_bwd_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cur_bwd_flops: float = 0.0
        self._cur_fwd_count: int = 0
        self._cur_bwd_count: int = 0

        # Previous iteration (for lazy print)
        self._prev: Optional[dict] = None

        # Lazy GPU peak
        self._peak_tflops: Optional[float] = None

    # -- public API called by hooks / instrumentation -------------------------

    def add_fwd(
        self,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        flops: float,
    ) -> None:
        # Detect fwd-only (inference) iteration boundary:
        # If _num_layers is known and we already collected a full set of fwd
        # calls with zero bwd calls, the previous fwd-only iteration is done.
        if (
            self._num_layers > 0
            and self._cur_fwd_count == self._num_layers
            and self._cur_bwd_count == 0
        ):
            self._finalize_iteration()

        if self._cur_fwd_count == 0:
            # First layer's forward → safe to print previous iter
            self._maybe_print_prev()

        self._cur_fwd_events.append((start, end))
        self._cur_fwd_flops += flops
        self._cur_fwd_count += 1

    def add_bwd(
        self,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        flops: float,
    ) -> None:
        self._cur_bwd_events.append((start, end))
        self._cur_bwd_flops += flops
        self._cur_bwd_count += 1

        # Auto-detect num_layers on the first training iteration
        if self._num_layers == 0 and self._cur_bwd_count == self._cur_fwd_count:
            self._num_layers = self._cur_fwd_count

        # Check if the iteration is complete (training mode)
        if self._num_layers > 0 and self._cur_bwd_count == self._num_layers:
            self._finalize_iteration()

    def step(self) -> None:
        """Explicitly mark the end of the current iteration.

        This is useful for **pure inference** where ``_num_layers`` has not
        been auto-detected from a prior training backward pass.  If
        ``_num_layers`` is unknown, it will be inferred from the number of
        ``add_fwd`` calls accumulated so far.  Subsequent fwd-only iterations
        will then be finalised automatically.

        Calling ``step()`` during training (after bwd already finalised the
        iteration) is a safe no-op.
        """
        if self._cur_fwd_count == 0:
            # Nothing accumulated (e.g. bwd already finalised) — no-op.
            return
        # Auto-detect num_layers from fwd count if still unknown.
        if self._num_layers == 0 and self._cur_fwd_count > 0:
            self._num_layers = self._cur_fwd_count
        self._finalize_iteration()

    # -- internals ------------------------------------------------------------

    def _finalize_iteration(self) -> None:
        self._prev = {
            "fwd_events": self._cur_fwd_events,
            "bwd_events": self._cur_bwd_events,
            "fwd_flops": self._cur_fwd_flops,
            "bwd_flops": self._cur_bwd_flops,
            "iter_idx": self._iter_idx,
        }
        self._cur_fwd_events = []
        self._cur_bwd_events = []
        self._cur_fwd_flops = 0.0
        self._cur_bwd_flops = 0.0
        self._cur_fwd_count = 0
        self._cur_bwd_count = 0
        self._iter_idx += 1

    @staticmethod
    def _should_print(iter_idx: int) -> bool:
        if iter_idx < _PRINT_HSTU_PERF_START:
            return False
        if _PRINT_HSTU_PERF_STOP >= 0 and iter_idx >= _PRINT_HSTU_PERF_STOP:
            return False
        return True

    def _maybe_print_prev(self) -> None:
        if self._prev is None:
            return
        prev = self._prev
        self._prev = None
        if not self._should_print(prev["iter_idx"]):
            return

        # Synchronise once — previous iter's events should already be complete
        torch.cuda.synchronize()

        fwd_ms = sum(s.elapsed_time(e) for s, e in prev["fwd_events"])
        fwd_tflops = prev["fwd_flops"] / (fwd_ms * 1e-3) / 1e12 if fwd_ms > 0 else 0.0

        if self._peak_tflops is None:
            spec = get_current_device_spec()
            self._peak_tflops = spec.peak_tflops.get(
                "bf16", spec.peak_tflops.get("fp16", 312.0)
            )
        fwd_mfu = fwd_tflops / self._peak_tflops * 100.0

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if prev["bwd_events"]:
            # Training mode — print both fwd and bwd stats
            bwd_ms = sum(s.elapsed_time(e) for s, e in prev["bwd_events"])
            bwd_tflops = (
                prev["bwd_flops"] / (bwd_ms * 1e-3) / 1e12 if bwd_ms > 0 else 0.0
            )
            bwd_mfu = bwd_tflops / self._peak_tflops * 100.0
            debug_rank_all(
                f"[HSTU Attn Perf] iter={prev['iter_idx']} rank={rank}  "
                f"fwd: {fwd_ms:.2f}ms  {fwd_tflops:.2f} TFLOPS  (MFU={fwd_mfu:.1f}%)  "
                f"bwd: {bwd_ms:.2f}ms  {bwd_tflops:.2f} TFLOPS  (MFU={bwd_mfu:.1f}%)"
            )
        else:
            # Inference / eval mode — fwd only
            debug_rank_all(
                f"[HSTU Attn Perf] iter={prev['iter_idx']} rank={rank}  "
                f"fwd: {fwd_ms:.2f}ms  {fwd_tflops:.2f} TFLOPS  (MFU={fwd_mfu:.1f}%)"
            )


# Module-level singleton
_global_accum: Optional[_AttnPerfAccumulator] = None


def _get_attn_perf_accum() -> _AttnPerfAccumulator:
    global _global_accum
    if _global_accum is None:
        _global_accum = _AttnPerfAccumulator()
    return _global_accum
