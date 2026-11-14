#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

python -m pytest -q test/unit_tests/test_input_dist_indices.py \
    test/unit_tests/test_storage_row_copy.py
python -m torch.distributed.run --standalone --nproc_per_node=2 -m pytest -q \
    test/unit_tests/test_input_dist_indices.py
