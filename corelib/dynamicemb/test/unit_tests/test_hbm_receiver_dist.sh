# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
torchrun --standalone --nproc_per_node=2 -m pytest -q \
    test/unit_tests/test_hbm_receiver.py \
    -k 'split_fusion_keeps_each_collections_receiver_state or native_receiver_type_stays_owned_until_awaitable_is_dropped'
