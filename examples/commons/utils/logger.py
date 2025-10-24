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

import torch
from rich.console import Console
from rich.logging import RichHandler

# Set up logger with RichHandler if not already configured

console = Console()
_LOGGER = logging.getLogger("rich_rank0")

if not _LOGGER.hasHandlers():
    handler = RichHandler(
        console=console, show_time=True, show_path=False, rich_tracebacks=True
    )
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False
    _LOGGER.setLevel(logging.INFO)


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            _LOGGER.info(message)
    else:
        print(message, flush=True)
