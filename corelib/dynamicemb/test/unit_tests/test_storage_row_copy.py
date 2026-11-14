# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live snapshot publication copies full rows, without a host-sized join."""

import pytest
import torch
from dynamicemb_extensions import copy_matching_storage_rows


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [7, 64, 1024, 3072])
@pytest.mark.parametrize("masked", [False, True])
def test_storage_row_copy_preserves_table_key_admission_and_moments(
    dtype, width, masked
):
    # The high-bit keys test the full uint64 domain, represented bijectively
    # as signed int64 on both sides of the sorted index.
    keys = [(1 << 64) - 3, 0, 1, (1 << 63) + 5, 0, 3]
    targets = [3, (1 << 63) + 5, 0, 1, (1 << 64) - 3, 0, 3]
    tids = [0, 0, 0, 0, 1, 1, 1]
    signed = torch.tensor(keys, device="cuda", dtype=torch.uint64).to(torch.int64)
    signed_targets = torch.tensor(targets, device="cuda", dtype=torch.uint64).to(
        torch.int64
    )
    sorted_parts = [signed[:4].sort(), signed[4:].sort()]
    sorted_keys = torch.cat([part.values for part in sorted_parts])
    sorted_rows = torch.cat([sorted_parts[0].indices, sorted_parts[1].indices + 4])
    source = (
        torch.arange(len(keys), device="cuda", dtype=torch.float32)[:, None]
        + torch.arange(width, device="cuda", dtype=torch.float32)[None, :] / 4096
    ).to(dtype)
    target = torch.full((len(targets), width), 99, device="cuda", dtype=dtype)
    source_mask = [True, False, True, True, True, True]
    target_mask = [True, True, True, True, True, False, True]
    expected = target.clone()
    source_tids = [0, 0, 0, 0, 1, 1]
    for row, (key, tid) in enumerate(zip(targets, tids)):
        for src, (other, source_tid) in enumerate(zip(keys, source_tids)):
            if key == other and tid == source_tid:
                if not masked or (source_mask[src] and target_mask[row]):
                    expected[row] = source[src]
    copy_matching_storage_rows(
        sorted_keys,
        sorted_rows,
        torch.tensor([0, 4, 6], device="cuda"),
        source,
        signed_targets,
        torch.tensor(tids, device="cuda"),
        target,
        torch.tensor(source_mask, device="cuda") if masked else None,
        torch.tensor(target_mask, device="cuda") if masked else None,
    )
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "source_sizes,target_sizes", [([0, 2], [2, 0]), ([0, 0], [2, 1]), ([2, 0], [0, 0])]
)
def test_storage_row_copy_handles_empty_tables_and_batches(source_sizes, target_sizes):
    device = "cuda"
    source_count, target_count = sum(source_sizes), sum(target_sizes)
    source = torch.ones((source_count, 7), device=device)
    target = torch.zeros((target_count, 7), device=device)
    copy_matching_storage_rows(
        torch.arange(source_count, device=device),
        torch.arange(source_count, device=device),
        torch.tensor([0, source_sizes[0], source_count], device=device),
        source,
        torch.zeros(target_count, dtype=torch.int64, device=device),
        torch.tensor(
            [0] * target_sizes[0] + [1] * target_sizes[1],
            dtype=torch.int64,
            device=device,
        ),
        target,
    )
    # All target rows address a source table with zero rows.
    torch.testing.assert_close(target, torch.zeros_like(target), rtol=0, atol=0)
