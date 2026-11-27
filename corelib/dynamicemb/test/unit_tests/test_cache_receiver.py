# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist
from dynamicemb.key_value_table import load_from_flat
from dynamicemb.types import CopyMode
from test_hbm_receiver import (  # noqa: F401
    batch,
    check_debug,
    distribute,
    distributed,
    make_model,
    output,
    tables,
)


def test_cache_eviction_reinitializes_rows_and_adam_state():
    model = make_model(init_capacity=256, max_capacity=256, layout="cache")
    seen_rows = {}
    reused = False
    per_table = tables(model)[0].cache._state.capacity // len(
        tables(model)[0]._dynamicemb_options
    )
    count = max(32, per_table * dist.get_world_size() // 2)
    for iteration in range(6):
        features = batch(start=iteration * count, count=count)
        ctx, arrived = distribute(model, features)
        groups = arrived[0].split(model._lookups[0]._feature_splits)
        for table, group in zip(tables(model), groups):
            state = group.prefetch_state
            assert (state.value_row_indices >= 0).all()
            values = load_from_flat(
                table.cache._state,
                state.value_row_indices,
                state.unique_table_ids,
                copy_mode=CopyMode.VALUE,
            )
            # Every key is new. A reused row must have neither the old embedding
            # nor Adam's moments, regardless of which hash slot owns it now.
            assert (values[:, 8:] == 0).all()
            for tid, row in zip(
                state.unique_table_ids.tolist(), state.value_row_indices.tolist()
            ):
                key = (id(table), tid)
                rows = seen_rows.setdefault(key, set())
                reused |= row in rows
                rows.add(row)
        result = output(model, ctx, arrived)
        check_debug(result, features)
        sum(jt.values().sum() for jt in result.values()).backward()
    assert reused, "the workload did not actually reuse a value row"
