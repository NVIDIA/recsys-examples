# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fusing row projection must not turn unique-key pins into occurrence pins."""

import pytest
import torch
from dynamicemb_extensions import table_update_counter_with_layout


@pytest.mark.parametrize("num_tables", [1, 3])
@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize("delta", [-1, 1])
@pytest.mark.parametrize(
    "num_unique,num_occurrences",
    [(0, 0), (1, 0), (1, 19), (17, 3), (17, 67), (513, 2049)],
)
def test_counter_row_projection_matches_separate_operations(
    num_tables, overflow, delta, num_unique, num_occurrences
):
    device = torch.device("cuda", torch.cuda.current_device())
    capacities = [1024] if num_tables == 1 else [512, 768, 1024]
    overflow_capacity = 128 if overflow else 0
    offsets = [0]
    for capacity in capacities:
        offsets.append(offsets[-1] + capacity // 128)
    # Include overflow and main slots, and the same slot number in different
    # tables. Every non-negative (table, slot) pair is unique within this call.
    candidates = [
        list(range(capacity, capacity + overflow_capacity)) + list(range(capacity))
        for capacity in capacities
    ]
    pairs = [
        (table, slots[position])
        for position in range(max(map(len, candidates)))
        for table, slots in enumerate(candidates)
        if position < len(slots)
    ][:num_unique]
    tids = torch.tensor([table for table, _ in pairs], device=device, dtype=torch.int64)
    slots = torch.tensor([slot for _, slot in pairs], device=device, dtype=torch.int64)
    if num_unique > 1:
        slots[-1] = -1
    rows = torch.arange(num_unique, device=device) * 13 + 7
    if num_unique > 1:
        rows[-1] = -1
    reverse = (
        (torch.arange(num_occurrences, device=device) * 37 + 11) % num_unique
        if num_unique
        else torch.empty(0, device=device, dtype=torch.int64)
    )
    main_capacity = sum(capacities)
    before = torch.full(
        (main_capacity + num_tables * overflow_capacity,),
        7,
        device=device,
        dtype=torch.int32,
    )
    expected, actual = before.clone(), before.clone()
    layout = dict(
        table_bucket_offsets=torch.tensor(offsets, device=device),
        bucket_capacity=128,
        main_capacity=main_capacity,
        num_tables=num_tables,
        table_ids=tids,
        overflow_output_offsets=torch.tensor(capacities, device=device)
        if overflow
        else None,
        overflow_bucket_capacity=overflow_capacity,
    )
    # The original call form still returns None and updates exactly these pins.
    assert table_update_counter_with_layout(expected, slots, delta, **layout) is None
    occurrence_rows = torch.empty_like(reverse)
    assert (
        table_update_counter_with_layout(
            actual,
            slots,
            delta,
            **layout,
            value_rows=rows,
            reverse_indices=reverse,
            occurrence_rows=occurrence_rows
        )
        is None
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(occurrence_rows, rows[reverse], rtol=0, atol=0)
    assert (actual != before).sum().item() == (slots >= 0).sum().item()


@pytest.mark.parametrize(
    "invalid",
    ["partial", "rows_size", "reverse_dtype", "strided", "output_size", "cpu_rows"],
)
def test_counter_projection_validates_metadata_before_pinning(invalid):
    device = torch.device("cuda", torch.cuda.current_device())
    counter = torch.zeros(128, device=device, dtype=torch.int32)
    slots = torch.arange(4, device=device)
    args = dict(
        value_rows=torch.arange(4, device=device),
        reverse_indices=torch.tensor([3, 0], device=device),
        occurrence_rows=torch.empty(2, device=device, dtype=torch.int64),
    )
    if invalid == "partial":
        del args["occurrence_rows"]
    elif invalid == "rows_size":
        args["value_rows"] = args["value_rows"][:3]
    elif invalid == "reverse_dtype":
        args["reverse_indices"] = args["reverse_indices"].int()
    elif invalid == "strided":
        args["reverse_indices"] = torch.arange(4, device=device)[::2]
    elif invalid == "output_size":
        args["occurrence_rows"] = args["occurrence_rows"][:1]
    else:
        args["value_rows"] = args["value_rows"].cpu()
    with pytest.raises(RuntimeError):
        table_update_counter_with_layout(
            counter, slots, 1, torch.tensor([0, 1], device=device), 128, 128, 1, **args
        )
    assert torch.count_nonzero(counter).item() == 0
