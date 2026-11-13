# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Runs an existing test again on the hierarchical output dist, with its
configuration unchanged (docs/hier_output_dist.md, section 18).

``DYNAMICEMB_TEST_OUTPUT_DIST`` unset, empty or ``torchrec`` changes nothing:
:func:`options` gives no field, and the other functions do nothing. With
``hierarchical``, the tables of a sharded EmbeddingCollection whose row-wise
sharding the dist serves (:func:`serves`) get ``output_dist="hierarchical"``,
and the tables of any other collection ``"auto"``, which must take TorchRec's
dists (section 6). Both get ``output_dist_capacity``:
``DYNAMICEMB_TEST_OUTPUT_DIST_CAPACITY``, else ``1 << 17``.

A test gives the DynamicEmb tables of each sharded EmbeddingCollection the
fields of :func:`options`, calls :func:`track` on the model, and
:func:`check_path` after the model's first training forward. At its end it
calls :func:`release`; a pytest file calls it at the teardown of each test,
which every rank reaches, passed or not. test/unit_test.sh runs the files
that do this again on the dist, after their usual runs. To run one by hand,
from ``corelib/dynamicemb``::

    DYNAMICEMB_TEST_OUTPUT_DIST=hierarchical PYHKV_DEBUG_STRICT=1 bash test/unit_tests/test_lfu_scores.sh
"""

import gc
import os
from typing import Any, Collection, Dict, List, Optional

import torch.nn as nn
from dynamicemb.dynamicemb_config import (
    AUTO_OUTPUT_DIST,
    HIERARCHICAL_OUTPUT_DIST,
    TORCHREC_OUTPUT_DIST,
    takes_score_per_key,
)
from dynamicemb.shard import node_size, quiesce
from dynamicemb.shard.rw_sharding import RwSequenceDynamicEmbeddingSharding
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# An empty variable reads as unset.
OUTPUT_DIST = os.environ.get("DYNAMICEMB_TEST_OUTPUT_DIST") or TORCHREC_OUTPUT_DIST
if OUTPUT_DIST not in (TORCHREC_OUTPUT_DIST, HIERARCHICAL_OUTPUT_DIST):
    raise ValueError(
        f"DYNAMICEMB_TEST_OUTPUT_DIST is {OUTPUT_DIST!r}: set it to "
        f"{HIERARCHICAL_OUTPUT_DIST!r}, or to {TORCHREC_OUTPUT_DIST!r} (as unset)"
    )
# Twice the most keys a call of these tests has: 8 ranks, a local batch of
# 2048 and 4 tables of one key a sample (test_twin_module.py).
OUTPUT_DIST_CAPACITY = int(
    os.environ.get("DYNAMICEMB_TEST_OUTPUT_DIST_CAPACITY") or 1 << 17
)

# The models given the hierarchical dist, in the order they were built (the
# same on every rank), until release() quiesces their ops.
_models: List[nn.Module] = []


def serves(
    *,
    world_size: int,
    row_bytes: int,
    all_dynamicemb: bool = True,
    dist_types: Collection[str] = (),
    score_strategies: Collection[Any] = (),
) -> bool:
    """Whether the hierarchical dist serves the row-wise sharding of a
    collection (docs/hier_output_dist.md, section 6): more than one rank,
    DynamicEmb tables alone, rows (``row_bytes``, in the lookup's output
    dtype) of a multiple of 16 bytes, no ``"continuous"`` table and no table
    that takes a score per key. The sharder's index dedup matters only with
    gradient clipping, which none of these tests uses: a call on the dist
    deduplicates the sharding's keys itself."""
    return (
        world_size > 1
        and all_dynamicemb
        and row_bytes % 16 == 0
        and "continuous" not in dist_types
        and not any(takes_score_per_key(s) for s in score_strategies)
    )


def options(**facts) -> Dict[str, Any]:
    """The output_dist fields of every DynamicEmb table of one collection
    (the tables of a sharding must agree on them), from the keyword
    arguments of :func:`serves`. Empty (no field) while the switch is unset."""
    if OUTPUT_DIST == TORCHREC_OUTPUT_DIST:
        return {}
    return {
        "output_dist": (
            HIERARCHICAL_OUTPUT_DIST if serves(**facts) else AUTO_OUTPUT_DIST
        ),
        "output_dist_capacity": OUTPUT_DIST_CAPACITY,
    }


def check_path(
    model: nn.Module,
    options: Dict[str, Any],
    tables: Optional[Collection[str]] = None,
) -> None:
    """After the model's first training forward: each row-wise sharding of
    DynamicEmb tables (of ``tables`` alone, if given) took the hierarchical
    dist exactly when ``options`` give it ``"hierarchical"``. A sharding
    that falls back under ``"hierarchical"`` only warns, so the test would
    pass without the dist. A sharding given ``"auto"`` must fall back, and
    so must every sharding on GPUs of a node that NVLink does not connect
    (a host of PCIe GPUs: :func:`unlinked`)."""
    if not options:
        return
    asked = options["output_dist"] == HIERARCHICAL_OUTPUT_DIST
    num_checked = 0
    for module in model.modules():
        shardings = getattr(module, "_sharding_type_to_sharding", None) or {}
        for sharding in shardings.values():
            if not isinstance(sharding, RwSequenceDynamicEmbeddingSharding):
                continue
            names = sorted(table.name for table in sharding.embedding_tables())
            if tables is not None and not set(names) <= set(tables):
                continue
            if all(
                config.compute_kernel is not EmbeddingComputeKernel.CUSTOMIZED_KERNEL
                for config in sharding._grouped_embedding_configs
            ):
                continue  # TorchRec's tables alone: output_dist does not apply
            hier = sharding._hier
            took = hier is not None and hier.takes_hier_dist
            assert took is not None, f"{names}: the sharding has had no call yet"
            expected = asked and not (hier is not None and unlinked(hier))
            assert took == expected, (
                f"{names}: given output_dist={options['output_dist']!r}, the "
                f"sharding {'took' if took else 'did not take'} the hierarchical dist"
            )
            num_checked += 1
    assert num_checked or not asked, "no row-wise sharding of DynamicEmb tables"


def unlinked(hier) -> bool:
    """Whether the hierarchical dist refused the nodes of a sharding's ranks
    because NVLink does not connect the GPUs of one (node_size's error,
    which the sharding's first input dist read and kept: no collective
    here)."""
    try:
        node_size(hier._pg)
    except RuntimeError as err:
        return "NVLink does not connect" in str(err)
    return False


def check_totals(kjt: KeyedJaggedTensor, options: Dict[str, Any]) -> None:
    """After a model given ``options`` read ``kjt``: the totals per feature
    of ``kjt`` are the sums of its lengths. The hierarchical dist writes
    these totals into the caller's KJT (docs/hier_output_dist.md, section
    7.4), and a model that reads the same KJT next splits it by them."""
    if not options:
        return
    totals = kjt.lengths().view(len(kjt.keys()), -1).sum(1).tolist()
    assert (
        kjt.length_per_key() == totals
    ), f"length_per_key {kjt.length_per_key()}, but the lengths add up to {totals}"


def track(model: nn.Module, *collection_options: Dict[str, Any]) -> None:
    """Keeps ``model`` for :func:`release` when the options of one of its
    collections give it ``"hierarchical"``. A model given ``"auto"`` falls
    back, builds no op, and so has nothing to release."""
    if any(
        each.get("output_dist") == HIERARCHICAL_OUTPUT_DIST
        for each in collection_options
    ):
        _models.append(model)


def release() -> None:
    """Collective, at the end of a test, on every rank: quiesces the ops of
    the models that :func:`track` kept, in the order it kept them, the same
    on every rank, then lets the models go (docs/hier_output_dist.md, section
    4.6). Without it, an op's teardown waits for its peers' teardowns, and
    the cyclic GC, which drops these models, runs at a different time on
    each rank."""
    if not _models:
        return
    for model in _models:
        quiesce(model)
    _models.clear()
    gc.collect()
