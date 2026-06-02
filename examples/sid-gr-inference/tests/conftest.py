# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test import path bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPO_ROOT / "tools",
    REPO_ROOT / "benchmarks",
    REPO_ROOT / "examples",
)
for root in reversed(IMPORT_ROOTS):
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
