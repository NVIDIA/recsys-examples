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

"""What the tests of an erase on a module have in common: a module whose keys carry
values that show whose they are, a training step that checks what it reads, and a way to
ask the table what it holds."""

from typing import Dict, List, Optional, Sequence, Union

import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.scored_hashtable import ScoreArg
from dynamicemb_extensions import ScorePolicy

DIM = 8
# The DEBUG initializer writes ``key % DEBUG_MOD`` into every element of a new row, so a
# row can be checked against the key that should own it.
DEBUG_MOD = 100_000
BUCKET = 128


def table_names(num_tables: int) -> List[str]:
    return [f"t_{i}" for i in range(num_tables)]


def build_de_table(
    device_id: int,
    score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
    *,
    num_tables: int = 1,
    caching: bool = False,
    local_hbm_for_values: int = 1024**3,
    max_capacity: int = 1 << 20,
    init_capacity: int = 1 << 10,
    names: Optional[List[str]] = None,
    **options,
) -> BatchedDynamicEmbeddingTablesV2:
    """``num_tables`` tables of one feature each, values that show whose they are, and an
    update that changes nothing (learning rate 0), so that what training reads is always
    the key's own initial value."""

    def one() -> DynamicEmbTableOptions:
        return DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device_id,
            dim=DIM,
            max_capacity=max_capacity,
            init_capacity=init_capacity,
            bucket_capacity=BUCKET,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=local_hbm_for_values,
            score_strategy=score_strategy,
            caching=caching,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG
            ),
            **options,
        )

    return BatchedDynamicEmbeddingTablesV2(
        table_options=[one() for _ in range(num_tables)],
        output_dtype=torch.float32,
        table_names=table_names(num_tables) if names is None else names,
        feature_table_map=list(range(num_tables)),
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.0,
        stochastic_rounding=False,
    )


def train_step(
    module: BatchedDynamicEmbeddingTablesV2,
    keys: Union[Sequence[int], Sequence[Sequence[int]]],
    device,
    scores: Optional[Sequence[int]] = None,
    score: Union[None, int, Dict[str, int]] = None,
) -> None:
    """Forward and backward over ``keys``, one key per bag, then check the output.

    ``keys`` is a list of keys for a module of one table, or a list of such lists, one
    per feature (table), of the same length. A key's score is its own in ``scores`` (for a
    table that takes a score per key), or ``score`` for the table -- one number for every
    table, or a dict by table name. Whatever an erase is doing, each key must read its own
    initial value."""
    per_feature = [list(k) for k in keys] if isinstance(keys[0], (list, tuple)) else [list(keys)]
    if isinstance(score, dict):
        module.set_score(score)
    elif score is not None:
        module.set_score({name: score for name in module.table_names})
    flat = [k for feature in per_feature for k in feature]
    indices = torch.tensor(flat, dtype=torch.int64, device=device)
    offsets = torch.arange(0, len(flat) + 1, dtype=torch.int64, device=device)
    if scores is None:
        out = module(indices, offsets)
    else:
        out = module(
            indices,
            offsets,
            scores=torch.tensor(scores, dtype=torch.int64, device=device),
        )
    out.sum().backward()
    torch.cuda.synchronize()
    expected = torch.cat(
        [
            torch.tensor([k % DEBUG_MOD for k in feature], device=out.device)
            .to(out.dtype)
            .unsqueeze(1)
            .expand(-1, DIM)
            for feature in per_feature
        ],
        dim=1,
    )
    torch.testing.assert_close(out, expected)


def new_counts(device) -> torch.Tensor:
    """The ``[erased, spared]`` a scan adds to."""
    return torch.zeros(2, dtype=torch.int64, device=device)


def erased_and_spared(counts: torch.Tensor):
    torch.cuda.synchronize()
    return int(counts[0]), int(counts[1])


def found(
    module: BatchedDynamicEmbeddingTablesV2,
    keys,
    device,
    table_id: int = 0,
    *,
    cache: bool = False,
):
    """Which of ``keys`` table ``table_id`` holds: in the storage, which is where an erase
    works and behind any cache, or with ``cache=True`` in the cache."""
    state = (module.cache if cache else module._storage)._state
    keys = torch.as_tensor(keys, dtype=torch.int64, device=device)
    arg = ScoreArg(name=state.score_policy.name, value=None, policy=ScorePolicy.CONST)
    _, founds, _ = state.key_index_map.lookup(keys, torch.full_like(keys, table_id), arg)
    return founds


def size(module: BatchedDynamicEmbeddingTablesV2, table_id: int = 0) -> int:
    """How many keys table ``table_id`` holds, in the storage behind any cache."""
    return int(module._storage.key_index_map.size(table_id))
