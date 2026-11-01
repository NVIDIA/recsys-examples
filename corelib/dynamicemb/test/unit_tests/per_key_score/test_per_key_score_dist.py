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

"""What a row-wise collection that takes a score per key does with the scores.

Run with ``torchrun --nproc_per_node 2``. Three parts:

1. ``check_input_dist`` -- an ``EmbeddingCollection`` as far as its input
   distribution (and, with ``use_index_dedup``, its dedup):
   - a plain ``KeyedJaggedTensor`` is refused, not read as "no scores";
   - a ``ScoredKeyedJaggedTensor`` reaches the row-wise distributor, and what the
     distribution hands on, with every score still beside its key, nothing lost,
     and the context the output distribution reads its splits from filled in;
   - the lookup hands the scores of each group to the tables' ``forward`` and
     ``prefetch``, still beside their keys.
   Every score there is derived from its key (``score = key * 10 + 1``), so
   ``scores == values * 10 + 1`` says a score is still beside its key.

2. ``check_ec_scores`` -- the scores the tables of an ``EmbeddingCollection``
   hold after ``model(...)`` and ``backward()``, gathered over the ranks, against
   what each key should hold, without and with ``use_index_dedup``, and with a
   cache above the storage:
   - the largest of the scores a batch gave a key, wherever in the batch, and on
     whichever rank, the key was;
   - from a later batch, the later batch's -- a score is assigned on every hit,
     not raised -- and a key the later batch did not have is left as it was;
   - backward does not change them.

   Also: a host-only tier, ``(NO_EVICTION, CUSTOMIZED)``, an input whose features
   are not in the tables' order, a dump and a load, evaluation without scores,
   and a negative score under WARNING and ERROR.

3. ``check_pooled_scores`` -- the same through ``EmbeddingBagCollection``, whose
   lookup is not prefetched, so the scores reach the tables through ``forward``
   alone.
"""

import os
import shutil
import sys
import warnings
from typing import Dict, List

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    ScoredKeyedJaggedTensor,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dump_load import (
    DynamicEmbDump,
    DynamicEmbLoad,
    find_sharded_modules,
    get_dynamic_emb_module,
)
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingBagCollectionSharder
from dynamicemb.shard.input_dist import RwSparseFeaturesDist
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# test_embedding_dump_load is the fixture module the tests in unit_tests share,
# so it stays a directory up and has to be put on the path from here.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_embedding_dump_load import (  # noqa: E402
    create_model,
    get_optimizer_kwargs,
    idx_to_name,
)

NUM_TABLES = 3
NUM_EMBEDDINGS = 1000
BATCH_PER_RANK = 32
INPUT_BATCH_PER_RANK = 4

# --- 1. input distribution ---------------------------------------------------

# The tables' entry points the lookup hands the scores to.
seen_names = ("forward", "prefetch")


def _input_model(use_index_dedup: bool):
    names = [idx_to_name(0, i) for i in range(NUM_TABLES)]
    return create_model(
        num_embedding_collections=1,
        num_embeddings=[NUM_EMBEDDINGS] * NUM_TABLES,
        embedding_dim=16,
        optimizer_kwargs=get_optimizer_kwargs("sgd"),
        score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
        use_index_dedup=use_index_dedup,
        table_option_overrides={table: {"per_key_score": True} for _, table in names},
    ), [feature for feature, _ in names]


def _input_features(feature_names, device):
    generator = torch.Generator().manual_seed(1234 + dist.get_rank())
    lengths = torch.randint(
        0, 4, (len(feature_names) * INPUT_BATCH_PER_RANK,), generator=generator
    )
    values = torch.randint(
        0, NUM_EMBEDDINGS, (int(lengths.sum()),), generator=generator
    )
    values, lengths = values.to(device), lengths.to(device)
    return dict(keys=feature_names, values=values, lengths=lengths), values * 10 + 1


def _spy_row_wise_dist():
    """Record what ``RwSparseFeaturesDist.forward`` is handed."""
    seen = []
    real = RwSparseFeaturesDist.forward

    def spy(self, sparse_features):
        seen.append(sparse_features)
        return real(self, sparse_features)

    RwSparseFeaturesDist.forward = spy
    return seen, lambda: setattr(RwSparseFeaturesDist, "forward", real)


def check_input_dist(use_index_dedup: bool) -> None:
    model, feature_names = _input_model(use_index_dedup=use_index_dedup)
    ((_, _, sharded),) = find_sharded_modules(model, "")
    assert sharded._per_key_score

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    fields, scores = _input_features(feature_names, device)

    # A plain KeyedJaggedTensor is refused.
    try:
        sharded.input_dist(sharded.create_context(), KeyedJaggedTensor(**fields))
    except ValueError as error:
        assert "ScoredKeyedJaggedTensor" in str(error), error
    else:
        raise AssertionError("a plain KeyedJaggedTensor was accepted")

    # A scored one reaches the distributor with its scores beside its keys.
    seen, restore = _spy_row_wise_dist()
    try:
        scored = ScoredKeyedJaggedTensor(scores=scores, **fields)
        context = sharded.create_context()
        distributed = sharded.input_dist(context, scored).wait().wait()
    finally:
        restore()
    assert len(seen) == 1, len(seen)
    (received,) = seen
    assert isinstance(received, ScoredKeyedJaggedTensor), type(received)
    if use_index_dedup:
        assert received.values().numel() <= scores.numel()
    else:
        assert received.values().numel() == scores.numel()
    assert torch.equal(received.scores(), received.values() * 10 + 1)

    # What comes out of the all2all: scores still beside their keys (the default
    # dist_type keeps a key global), and as many of them as were sent.
    (arrived,) = distributed
    assert isinstance(arrived, ScoredKeyedJaggedTensor), type(arrived)
    assert arrived.scores().numel() == arrived.values().numel()
    assert torch.equal(arrived.scores(), arrived.values() * 10 + 1)
    # TorchRec read the splits the output distribution needs off the awaitable.
    (sharding_context,) = context.sharding_contexts
    assert sharding_context.input_splits and sharding_context.output_splits
    counts = torch.tensor(
        [received.values().numel(), arrived.values().numel()],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(counts)
    assert counts[0] == counts[1], counts

    check_lookup_gets_scores(sharded, arrived)


class _Stop(Exception):
    """Raised by a spy once it has seen what it was called with."""


def check_lookup_gets_scores(sharded, arrived) -> None:
    """What the lookup hands to the tables' ``forward`` and ``prefetch``."""
    (lookup,) = sharded._lookups
    seen = {}

    def spy(name):
        def record(self, indices, offsets, scores=None, **_):
            seen[name] = (indices, scores)
            raise _Stop

        return record

    real = {name: getattr(BatchedDynamicEmbeddingTablesV2, name) for name in seen_names}
    try:
        for name in seen_names:
            setattr(BatchedDynamicEmbeddingTablesV2, name, spy(name))
            try:
                getattr(lookup, name)(arrived)
            except _Stop:
                pass
    finally:
        for name, function in real.items():
            setattr(BatchedDynamicEmbeddingTablesV2, name, function)

    for name in seen_names:
        assert name in seen, f"the lookup never called the tables' {name}"
        indices, scores = seen[name]
        assert scores is not None, name
        # Every score is still beside its key (the default dist_type keeps a key
        # global).
        assert torch.equal(scores, indices * 10 + 1), name


# --- 2. scores in an EmbeddingCollection's tables ------------------------------


def _batch(rank: int, seed: int, feature_names: List[str]):
    """The keys, lengths and scores one rank feeds in one batch."""
    generator = torch.Generator().manual_seed(1000 * seed + rank)
    lengths = torch.randint(
        1, 5, (len(feature_names) * BATCH_PER_RANK,), generator=generator
    )
    # Few distinct keys for the number drawn, so keys repeat, in a row and
    # across the ranks.
    values = torch.randint(0, 200, (int(lengths.sum()),), generator=generator)
    scores = torch.randint(1, 10**12, (values.numel(),), generator=generator)
    return values, lengths, scores


def _per_feature(values, lengths, scores, num_features):
    """Split what a batch carries by feature: ``[(keys, scores)]``."""
    per_feature, start = [], 0
    for lengths_of_feature in lengths.view(num_features, -1):
        count = int(lengths_of_feature.sum())
        per_feature.append(
            (values[start : start + count], scores[start : start + count])
        )
        start += count
    return per_feature


def _expected_after(batch_seed, feature_names, held, world_size):
    """``held`` with every key of the batch given the largest score it got."""
    held = [dict(table) for table in held]
    for rank in range(world_size):
        values, lengths, scores = _batch(rank, batch_seed, feature_names)
        for table, (keys, key_scores) in zip(
            held, _per_feature(values, lengths, scores, len(feature_names))
        ):
            largest = {}
            for key, score in zip(keys.tolist(), key_scores.tolist()):
                largest[key] = max(score, largest.get(key, 0))
            # Pooled over the ranks first, then assigned: the batch of a rank is
            # not applied on its own, the keys of all are one lookup per owner.
            table.setdefault("__batch__", {})
            for key, score in largest.items():
                table["__batch__"][key] = max(score, table["__batch__"].get(key, 0))
    for table in held:
        table.update(table.pop("__batch__"))
    return held


def _table_scores(model, table_names: List[str]) -> List[Dict[int, int]]:
    """What the tables hold, over every rank: ``[{key: score}]`` per table."""
    torch.cuda.synchronize()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    mine: List[Dict[int, int]] = [{} for _ in table_names]
    for _, _, sharded in find_sharded_modules(model, ""):
        for module in get_dynamic_emb_module(sharded):
            module.flush()
            for table_id, name in enumerate(module.table_names):
                for keys, _, _, scores in module.tables.export_keys_values(
                    device, 65536, table_id=table_id
                ):
                    index = table_names.index(name)
                    for key, score in zip(keys.tolist(), scores.view(-1).tolist()):
                        mine[index][key] = score
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    merged: List[Dict[int, int]] = [{} for _ in table_names]
    for per_rank in gathered:
        for table, held in zip(merged, per_rank):
            assert not (table.keys() & held.keys()), "a key is held by two ranks"
            table.update(held)
    return merged


def _features_of(rank, batch_seed, feature_names, device, reverse=False, scored=True):
    """The input of one rank; ``reverse`` lists its features in the opposite order
    to the tables', which is what makes the collection permute them."""
    values, lengths, scores = _batch(rank, batch_seed, feature_names)
    counts = [int(c.sum()) for c in lengths.view(len(feature_names), -1)]
    parts = list(
        zip(
            feature_names,
            values.split(counts),
            lengths.view(len(feature_names), -1),
            scores.split(counts),
        )
    )
    if reverse:
        parts.reverse()
    fields = dict(
        keys=[name for name, *_ in parts],
        values=torch.cat([v for _, v, _, _ in parts]).to(device),
        lengths=torch.cat([l for _, _, l, _ in parts]).to(device),
    )
    if not scored:
        return KeyedJaggedTensor(**fields)
    return ScoredKeyedJaggedTensor(
        scores=torch.cat([c for _, _, _, c in parts]).to(device), **fields
    )


def _train_on(model, batch_seed, feature_names, reverse=False):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    features = _features_of(
        dist.get_rank(), batch_seed, feature_names, device, reverse=reverse
    )
    model(features).sum().backward()
    torch.cuda.synchronize()


def _ec_model(
    use_index_dedup=False,
    caching=False,
    score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
    host_only=False,
    **options,
):
    # With caching the budget is the cache's (cache_capacity_ratio of the table,
    # a half), and global_hbm_budget_scale is not read.
    assert not (host_only and caching)
    names = [idx_to_name(0, i) for i in range(NUM_TABLES)]
    model = create_model(
        num_embedding_collections=1,
        num_embeddings=[NUM_EMBEDDINGS] * NUM_TABLES,
        embedding_dim=16,
        optimizer_kwargs=get_optimizer_kwargs("sgd"),
        score_strategy=score_strategy,
        use_index_dedup=use_index_dedup,
        caching=caching,
        # Without caching, 1.0 holds every value in HBM; 0.0 leaves none there: one
        # host tier, the storage mode that is neither a cache nor HBM-only.
        global_hbm_budget_scale=0.0 if host_only else 1.0,
        table_option_overrides={
            table: {"per_key_score": True, **options} for _, table in names
        },
    )
    return model, [f for f, _ in names], [t for _, t in names]


def check_ec_scores(
    use_index_dedup: bool = False,
    caching: bool = False,
    score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
    host_only: bool = False,
    reverse: bool = False,
) -> None:
    model, feature_names, table_names = _ec_model(
        use_index_dedup=use_index_dedup,
        caching=caching,
        score_strategy=score_strategy,
        host_only=host_only,
    )
    world_size = dist.get_world_size()
    empty = [{} for _ in table_names]

    _train_on(model, 1, feature_names, reverse=reverse)
    expected = _expected_after(1, feature_names, empty, world_size)
    assert _table_scores(model, table_names) == expected

    # A later batch assigns: its score replaces the old one, lower or higher.
    _train_on(model, 2, feature_names, reverse=reverse)
    expected = _expected_after(2, feature_names, expected, world_size)
    assert _table_scores(model, table_names) == expected


def check_dump_load() -> None:
    """What a dump holds of the scores is what a load puts back."""
    model, feature_names, table_names = _ec_model()
    world_size = dist.get_world_size()
    _train_on(model, 1, feature_names)
    _train_on(model, 2, feature_names)
    expected = _expected_after(
        2,
        feature_names,
        _expected_after(1, feature_names, [{}] * NUM_TABLES, world_size),
        world_size,
    )
    assert _table_scores(model, table_names) == expected

    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "per_key_score_dump")
    if dist.get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()
    DynamicEmbDump(path, model)

    restored, _, _ = _ec_model()
    assert _table_scores(restored, table_names) != expected
    DynamicEmbLoad(path, restored)
    assert _table_scores(restored, table_names) == expected
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)


def check_eval_needs_no_scores(use_index_dedup: bool = False) -> None:
    """Evaluation reads no score: an input without, or with, scores both go
    through, and neither changes what the tables hold."""
    model, feature_names, table_names = _ec_model(use_index_dedup=use_index_dedup)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    _train_on(model, 1, feature_names)
    held = _table_scores(model, table_names)

    model.eval()
    with torch.no_grad():
        # Ranks that are given different kinds of input must still agree on the
        # collectives, so the last pass gives one kind to the even ranks and the
        # other to the odd ones.
        for scored in (False, True, dist.get_rank() % 2 == 0):
            out = model(
                _features_of(dist.get_rank(), 2, feature_names, device, scored=scored)
            )
            assert out.numel() > 0
    model.train()
    assert _table_scores(model, table_names) == held


def check_negative_score() -> None:
    """A negative score warns under WARNING and raises under ERROR."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for mode in (DynamicEmbCheckMode.WARNING, DynamicEmbCheckMode.ERROR):
        model, feature_names, _ = _ec_model(safe_check_mode=mode)
        values, lengths, scores = _batch(dist.get_rank(), 1, feature_names)
        # Every score, so every rank's tables receive a negative one.
        negative = ScoredKeyedJaggedTensor(
            keys=feature_names,
            values=values.to(device),
            lengths=lengths.to(device),
            scores=(-scores).to(device),
        )
        if mode == DynamicEmbCheckMode.WARNING:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model(negative).sum().backward()
            assert any("non-negative" in str(w.message) for w in caught), caught
        else:
            try:
                model(negative)
            except ValueError as error:
                assert "non-negative" in str(error), error
            else:
                raise AssertionError("a negative score was accepted under ERROR")
        torch.cuda.synchronize()


# --- 3. scores in an EmbeddingBagCollection's tables ---------------------------

EMBEDDING_DIM = 16
TABLE_NAMES = [f"t{i}" for i in range(NUM_TABLES)]
FEATURE_NAMES = [f"f{i}" for i in range(NUM_TABLES)]


def _build_ebc(device: torch.device):
    ebc = torchrec.EmbeddingBagCollection(
        device=torch.device("meta"),
        tables=[
            torchrec.EmbeddingBagConfig(
                name=table,
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=NUM_EMBEDDINGS,
                feature_names=[feature],
                pooling=torchrec.PoolingType.SUM,
            )
            for table, feature in zip(TABLE_NAMES, FEATURE_NAMES)
        ],
    )
    constraints = {
        table: DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            bounds_check_mode=BoundsCheckMode.NONE,
            enforce_hbm=True,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024**3,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.CONSTANT, value=1.0
                ),
                score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
                per_key_score=True,
            ),
        )
        for table in TABLE_NAMES
    }
    planner = DynamicEmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=dist.get_world_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
            hbm_cap=80 * 1024**3,
            ddr_cap=1024**4 // dist.get_world_size(),
        ),
        constraints=constraints,
        batch_size=64,
        debug=False,
    )
    sharder = DynamicEmbeddingBagCollectionSharder(
        fused_params={
            "output_dtype": SparseType.FP32,
            "optimizer": EmbOptimType.EXACT_SGD,
            "learning_rate": 0.1,
        }
    )
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    return DistributedModelParallel(
        module=ebc, device=device, sharders=[sharder], plan=plan
    )


def _train_pooled(model, batch_seed: int, device: torch.device) -> None:
    values, lengths, scores = _batch(dist.get_rank(), batch_seed, FEATURE_NAMES)
    features = ScoredKeyedJaggedTensor(
        keys=FEATURE_NAMES,
        values=values.to(device),
        lengths=lengths.to(device),
        scores=scores.to(device),
    )
    model(features).values().sum().backward()
    torch.cuda.synchronize()


def check_pooled_scores() -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = _build_ebc(device)
    world_size = dist.get_world_size()

    # A plain KeyedJaggedTensor is refused, not read as "no scores".
    values, lengths, _ = _batch(dist.get_rank(), 1, FEATURE_NAMES)
    plain = torchrec.KeyedJaggedTensor(
        keys=FEATURE_NAMES, values=values.to(device), lengths=lengths.to(device)
    )
    try:
        model(plain)
    except ValueError as error:
        assert "ScoredKeyedJaggedTensor" in str(error), error
    else:
        raise AssertionError("a plain KeyedJaggedTensor was accepted")

    _train_pooled(model, 1, device)
    expected = _expected_after(1, FEATURE_NAMES, [{} for _ in TABLE_NAMES], world_size)
    assert _table_scores(model, TABLE_NAMES) == expected

    # A later batch assigns: its score replaces the old one, lower or higher.
    _train_pooled(model, 2, device)
    expected = _expected_after(2, FEATURE_NAMES, expected, world_size)
    assert _table_scores(model, TABLE_NAMES) == expected

    # Evaluation reads no score, so it takes an input without them.
    model.eval()
    with torch.no_grad():
        model(plain)
    model.train()
    assert _table_scores(model, TABLE_NAMES) == expected


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    check_input_dist(use_index_dedup=False)
    check_input_dist(use_index_dedup=True)
    check_ec_scores(use_index_dedup=False)
    # Merging the repeats of a key before the all2all, as use_index_dedup does,
    # must leave the tables holding what they hold without it.
    check_ec_scores(use_index_dedup=True)
    # With a cache above the storage: the scores live in the storage, and every
    # key's reaches it, whether the cache held the key or not.
    check_ec_scores(caching=True)
    # One host tier: neither the cache path nor the HBM one.
    check_ec_scores(host_only=True)
    # (NO_EVICTION, CUSTOMIZED): the table grows instead of evicting, and still
    # takes the scores of the input.
    check_ec_scores(
        score_strategy=(
            DynamicEmbScoreStrategy.NO_EVICTION,
            DynamicEmbScoreStrategy.CUSTOMIZED,
        )
    )
    # An input whose features are not in the order of the tables is permuted.
    check_ec_scores(reverse=True)
    check_ec_scores(reverse=True, use_index_dedup=True)
    check_dump_load()
    check_eval_needs_no_scores()
    # The dedup before the all2all has no scores to merge in an input without.
    check_eval_needs_no_scores(use_index_dedup=True)
    check_negative_score()
    check_pooled_scores()
    dist.barrier()
    if dist.get_rank() == 0:
        print("per-key score distributed ok")
    dist.destroy_process_group()
