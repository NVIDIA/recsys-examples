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
import logging

# Set up logger with RichHandler if not already configured
# Control via environment variable:
#   DISABLE_RICH=1  - Disable Rich, use plain print instead
import os

import torch
from rich.console import Console
from rich.logging import RichHandler

# Set up logger with RichHandler if not already configured

console = Console(soft_wrap=True, width=240)
_logger = logging.getLogger("rich_rank0")

handler = RichHandler(
    console=console, show_time=True, show_path=False, rich_tracebacks=True
)
_logger.addHandler(handler)
_logger.propagate = False
_logger.setLevel(
    getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
)


def print_rank_0(message, level=logging.INFO):
    """If distributed is initialized, print on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            _logger.log(level, message)
    else:
        print(message, flush=True)


def info_rank_0(message):
    """If distributed is initialized, print on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            _logger.info(message)
    else:
        print(message, flush=True)


def debug_rank_0(message):
    """If distributed is initialized, print on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            _logger.debug(message)
    else:
        print(message, flush=True)


def print_rank_all(message, level=logging.INFO):
    """If distributed is initialized, print on all ranks."""
    if torch.distributed.is_initialized():
        _logger.log(level, message)
    else:
        print(message, flush=True)


def info_rank_all(message):
    """If distributed is initialized, print on all ranks."""
    if torch.distributed.is_initialized():
        _logger.info(message)
    else:
        print(message, flush=True)


def debug_rank_all(message):
    """If distributed is initialized, print on all ranks."""
    if torch.distributed.is_initialized():
        _logger.debug(message)
    else:
        print(message, flush=True)


# ============================================================================
# GPU Memory Debug Logging
# ============================================================================
_MEM_DEBUG = os.environ.get("MEM_DEBUG", "0") == "1"
_mem_debug_iter = 0


def log_mem(tag: str) -> None:
    """Log GPU physical free/total memory (includes NCCL buffers).

    Enable with MEM_DEBUG=1 environment variable.
    Reports physical free memory (via torch.cuda.mem_get_info) which includes
    memory held by NCCL and other non-PyTorch CUDA allocations, unlike
    torch.cuda.memory_allocated() which only tracks the caching allocator.

    The iteration counter auto-increments on each call.
    """
    if not _MEM_DEBUG:
        return
    global _mem_debug_iter
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(
        f"[rank{rank}] [MEM iter={_mem_debug_iter}] {tag}: "
        f"physical_free={free // 1024 // 1024}MB "
        f"pt_alloc={alloc // 1024 // 1024}MB "
        f"pt_reserved={reserved // 1024 // 1024}MB "
        f"total={total // 1024 // 1024}MB",
        flush=True,
    )
    _mem_debug_iter += 1
