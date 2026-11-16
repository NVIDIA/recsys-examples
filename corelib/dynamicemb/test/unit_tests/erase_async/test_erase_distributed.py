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

"""``erase_by_score_async`` on a model TorchRec has sharded.

Run with ``torchrun --nproc_per_node 2``. The model is a ``DistributedModelParallel``
around an ``EmbeddingCollection`` of dynamicemb tables sharded row-wise: every key lives
on the rank its hash picks, so no rank holds a table, and the call has to find the fused
tables inside the wrapper, erase on each rank what that rank holds, and add it up.

Each case trains a group of "old" keys and then a group of "new" ones, erases by score from
every rank, and looks at what the tables hold, gathered over the ranks:

* the call finds the tables in the wrapped model; between the ranks they erase every old
  key once (the counts add up to the number of old keys), and none is spared;
* the new keys are all still there, and the old ones come back when trained again;
* the same when training goes on while the erase runs;
* with the keys scored per table (``CUSTOMIZED``), per key (``PRECISE_CUSTOMIZED``), and
  per key in tables that never evict -- the use the feature is for;
* with and without ``use_index_dedup``;
* a layout that cannot be scanned is refused, and nothing is erased.
"""

import os
import sys
from typing import List, Set

import torch
import torch.distributed as dist
from dynamicemb import (
    DynamicEmbScoreStrategy,
    ScoredKeyedJaggedTensor,
    erase_by_score_async,
)
from dynamicemb.dump_load import get_dynamic_emb_module
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# test_embedding_dump_load is the fixture module the tests in unit_tests share, so it
# stays a directory up and has to be put on the path from here.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hier_switch  # noqa: E402
from test_embedding_dump_load import (  # noqa: E402
    create_model,
    get_optimizer_kwargs,
    idx_to_name,
)

S = DynamicEmbScoreStrategy

NUM_TABLES = 2
NUM_EMBEDDINGS = 40_000
KEYS_PER_STEP = 200  # of a feature, on a rank, in one step
OLD_KEYS = list(range(0, 2000))
NEW_KEYS = list(range(10_000, 12_000))
LATER_KEYS = list(range(20_000, 22_000))
OLD_SCORE, NEW_SCORE, THRESHOLD = 10, 200, 100

# What a case is: how the tables are scored, what is asked for to erase them by, and
# whether the input carries a score for every key.
CASES = {
    "per table": (S.CUSTOMIZED, S.CUSTOMIZED, False),
    "per key": (S.PRECISE_CUSTOMIZED, S.PRECISE_CUSTOMIZED, True),
    "per key, never evicting": (
        (S.NO_EVICTION, S.PRECISE_CUSTOMIZED),
        S.PRECISE_CUSTOMIZED,
        True,
    ),
}


def _device() -> torch.device:
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def _model(strategy, use_index_dedup=False, hbm_scale=1.0):
    names = [idx_to_name(0, i) for i in range(NUM_TABLES)]
    model = create_model(
        num_embedding_collections=1,
        num_embeddings=[NUM_EMBEDDINGS] * NUM_TABLES,
        embedding_dim=16,
        optimizer_kwargs=get_optimizer_kwargs("sgd"),
        score_strategy=strategy,
        use_index_dedup=use_index_dedup,
        global_hbm_budget_scale=hbm_scale,
    )
    return model, [f for f, _ in names], [t for _, t in names]


def _features(feature_names: List[str], keys: List[int], per_key_score=None):
    """One key per bag, the same keys in every feature; ``per_key_score`` is the score
    every key carries, for tables that take one per key."""
    values = torch.tensor(
        keys * len(feature_names), dtype=torch.int64, device=_device()
    )
    fields = dict(
        keys=feature_names,
        values=values,
        lengths=torch.ones(len(values), dtype=torch.int64, device=_device()),
    )
    if per_key_score is None:
        return KeyedJaggedTensor(**fields)
    return ScoredKeyedJaggedTensor(
        scores=torch.full(
            (len(values),), per_key_score, dtype=torch.int64, device=_device()
        ),
        **fields,
    )


def _train(model, feature_names, keys, per_key: bool, score: int) -> None:
    """Every rank feeds its share of ``keys``, a step at a time, forward and backward.
    The score is the input's for each key, or the table's, set on the tables."""
    mine = keys[dist.get_rank() :: dist.get_world_size()]
    for start in range(0, len(mine), KEYS_PER_STEP):
        if not per_key:
            for module in get_dynamic_emb_module(model):
                module.set_score({name: score for name in module.table_names})
        features = _features(
            feature_names,
            mine[start : start + KEYS_PER_STEP],
            per_key_score=score if per_key else None,
        )
        model(features).sum().backward()
    torch.cuda.synchronize()


def _held(model, table_names: List[str]) -> List[Set[int]]:
    """The keys each table holds, over every rank."""
    torch.cuda.synchronize()
    mine: List[Set[int]] = [set() for _ in table_names]
    for module in get_dynamic_emb_module(model):
        for table_id, name in enumerate(module.table_names):
            for keys, _, _, _ in module.tables.export_keys_values(
                _device(), 65536, table_id=table_id
            ):
                mine[table_names.index(name)].update(keys.tolist())
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    merged: List[Set[int]] = [set() for _ in table_names]
    for per_rank in gathered:
        for table, held in zip(merged, per_rank):
            assert not (table & held), "a key is held by two ranks"
            table |= held
    return merged


def _everywhere(count) -> int:
    """The sum of ``count`` over the ranks."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, count)
    return sum(gathered)


def _expect(model, table_names, keys, what: str) -> None:
    held = _held(model, table_names)
    assert held == [set(keys)] * len(table_names), (
        f"{what}: the tables hold {[len(h) for h in held]} keys, "
        f"expected {len(set(keys))} in each"
    )


def check_erase(case: str, use_index_dedup: bool) -> None:
    strategy, asked, per_key = CASES[case]
    model, feature_names, table_names = _model(strategy, use_index_dedup)
    label = f"{case}, dedup={use_index_dedup}"
    assert get_dynamic_emb_module(model), "the call would find no table in the model"

    _train(model, feature_names, OLD_KEYS, per_key, OLD_SCORE)
    _train(model, feature_names, NEW_KEYS, per_key, NEW_SCORE)
    _expect(model, table_names, OLD_KEYS + NEW_KEYS, f"{label}, before the erase")

    erased, spared = erase_by_score_async(model, asked, THRESHOLD).result()
    # Every table's old keys, once, between the ranks.
    assert (_everywhere(erased), _everywhere(spared)) == (
        NUM_TABLES * len(OLD_KEYS),
        0,
    ), f"{label}: counts {(erased, spared)} on rank {dist.get_rank()}"
    _expect(model, table_names, NEW_KEYS, f"{label}, after the erase")

    # The model goes on: the old keys are new keys again.
    _train(model, feature_names, OLD_KEYS, per_key, NEW_SCORE + 100)
    _expect(
        model, table_names, OLD_KEYS + NEW_KEYS, f"{label}, after training them again"
    )
    hier_switch.release()


def check_erase_beside_training(case: str) -> None:
    strategy, asked, per_key = CASES[case]
    model, feature_names, table_names = _model(strategy)
    label = f"{case}, beside training"

    _train(model, feature_names, OLD_KEYS, per_key, OLD_SCORE)
    _train(model, feature_names, NEW_KEYS, per_key, NEW_SCORE)
    handle = erase_by_score_async(model, asked, THRESHOLD)
    _train(
        model, feature_names, LATER_KEYS, per_key, NEW_SCORE
    )  # the call has returned

    erased, spared = handle.result()
    assert (_everywhere(erased), _everywhere(spared)) == (
        NUM_TABLES * len(OLD_KEYS),
        0,
    ), f"{label}: counts {(erased, spared)} on rank {dist.get_rank()}"
    _expect(model, table_names, NEW_KEYS + LATER_KEYS, label)
    hier_switch.release()


def check_host_only_is_refused() -> None:
    """No HBM for values: the tables are on the host, with no key map to scan on the GPU.
    The call says so on every rank, and nothing was erased."""
    model, feature_names, table_names = _model(S.CUSTOMIZED, hbm_scale=0.0)
    _train(model, feature_names, OLD_KEYS, False, OLD_SCORE)
    try:
        erase_by_score_async(model, S.CUSTOMIZED, THRESHOLD)
    except NotImplementedError as error:
        assert "HOST_ONLY" in str(error), str(error)
    else:
        raise AssertionError("an erase of a host-only table was not refused")
    _expect(model, table_names, OLD_KEYS, "host only")
    hier_switch.release()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    for case in CASES:
        check_erase(case, use_index_dedup=False)
        check_erase(case, use_index_dedup=True)
        check_erase_beside_training(case)
    check_host_only_is_refused()
    dist.barrier()
    if dist.get_rank() == 0:
        print("test_erase_distributed: all cases passed", flush=True)
    dist.destroy_process_group()
