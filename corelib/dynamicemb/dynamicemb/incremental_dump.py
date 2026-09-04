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

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module
from dynamicemb.dynamicemb_config import ReplayContent
from dynamicemb.types import ReplayStats
from torch import nn


@dataclass
class DeltaDumpResult:
    """Incremental-dump result for one embedding collection.

    All lists are column-aligned by index: element ``i`` of every list refers to
    the same table, ``table_names[i]``. A collection with ``k`` dumped tables has
    length-``k`` lists.

    Attributes
    ----------
    table_names : List[str]
        Names of the dynamic embedding tables dumped from this collection.
    keys : List[torch.Tensor]
        Per-table matched keys on host (created/modified keys to upsert).
    values : List[torch.Tensor]
        Per-table ``[N, dim]`` embeddings on host, aligned with ``keys``.
        **Embeddings only**: the rest of each stored row is in
        ``optimizer_states``, and concatenating the two along dim 1 reproduces
        the row as the table holds it.
    optimizer_states : List[Optional[torch.Tensor]]
        Per-table optimizer state on host, aligned with ``keys`` -- the trailing
        part of the same value row as ``values``, at the **same width the file
        checkpoint uses**, so a delta and a checkpoint describe a row
        identically. That is narrower than the runtime row for rowwise Adagrad,
        which reserves a fixed 16 bytes per row in the fused layout but fills
        only one accumulator scalar; ``replay_increment`` pads it back out.
        ``None`` for a table whose optimizer keeps no per-row state, e.g. plain
        SGD.
    scores : List[torch.Tensor]
        Per-table ``[N, num_scores]`` score words on host, aligned with ``keys``,
        in the table's configured (logical) column order. Timestamp columns hold
        an **age** (``meta[i]["current_score"] - score``) rather than a raw
        timestamp, because ``%globaltimer`` is per device and resets across
        boots; a consumer rebases them onto its own clock. Every other column
        (LFU frequency, STEP, CUSTOMIZED, NO_EVICTION row) is carried verbatim.
    evicted_keys : List[Optional[torch.Tensor]]
        Per-table keys on host (keys only, no value/score) that the table
        **evicted** to make room since the last ``incremental_dump``. ``None``
        unless the table was configured with ``evicted_item_mode=RETAIN_KEY`` --
        retaining evictions has to be decided up front, since it swaps in a
        collecting insert kernel.

        ``replay_increment`` never applies this list: the key that took an
        evicted key's slot is in the same delta and overwrites it, so the
        eviction is reproduced by the write. It is here for other consumers of
        the dump.
    erased_keys : List[Optional[torch.Tensor]]
        Per-table keys on host that an **explicit erase** removed since the last
        ``incremental_dump`` *and asked to have recorded* -- that is each
        ``erase`` call's decision (its own ``EvictedItemMode``), not a table
        setting, so this is always populated, with an empty tensor when nothing
        was recorded. ``replay_increment`` reads ``None`` as empty too, for
        deltas assembled by hand.

        These are the removals a replica has to perform for itself -- nothing
        takes over the slot, so no write reproduces them. Whether
        ``replay_increment`` applies them is ``meta[i]["replay_mode"]``'s call.

    Both lists drain and release their buffer, so each key is reported exactly
    once across successive ``incremental_dump`` calls, and both hold only keys
    that were really in the table.
    meta : List[Dict[str, Any]]
        Per-table dump metadata, aligned with ``table_names``. A flat dict with:
            meta[i]["current_score"]:    int  -- table's score after this dump;
                also the next forward pass's score / next ``incremental_dump``
                threshold.
            meta[i]["slot_index"]:       torch.Tensor -- int64 host tensor aligned
                with ``keys[i]``; the storage slot each dumped key occupies, used
                by ``replay_increment``. For NO_EVICTION tables it packs the key
                slot (high 32 bits) and value row (low 32 bits) into one int64.
            meta[i]["current_capacity"]: int  -- table's current capacity (slots).
            meta[i]["bucket_capacity"]:  int  -- slots per hash bucket; replay
                compares it to decide whether the source slots are usable.
            meta[i]["num_scores"]:       int  -- score words per key; part of the
                slot layout replay compares against.
            meta[i]["world_size"]:       int  -- ranks the source table was
                sharded across at creation (global WORLD), used by replay to
                reconstruct key->rank. NOT the gather ``pg`` (a comm scope only).
            meta[i]["table_options"]:    DynamicEmbTableOptions -- the table's
                config object (a ``DynamicEmbTableOptions`` instance).
    """

    table_names: List[str] = field(default_factory=list)
    keys: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    optimizer_states: List[Optional[torch.Tensor]] = field(default_factory=list)
    scores: List[torch.Tensor] = field(default_factory=list)
    evicted_keys: List[Optional[torch.Tensor]] = field(default_factory=list)
    erased_keys: List[Optional[torch.Tensor]] = field(default_factory=list)
    meta: List[Dict[str, Any]] = field(default_factory=list)


def is_valid_score_threshold(score_threshold: Any) -> bool:
    """
    Check if score_threshold is instance of `Dict[str, Dict[str, int]]`.
    """
    if not isinstance(score_threshold, dict):
        return False

    for key, value in score_threshold.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, dict):
            return False

        for inner_key, inner_value in value.items():
            if not isinstance(inner_key, str):
                return False
            if not isinstance(inner_value, int):
                return False

    return True


def set_score(
    model: torch.nn.Module, table_score: Union[int, Dict[str, Dict[str, int]]]
) -> None:
    """Set the score for each dynamic embedding table. It will not reset the scores of each embedding table, but register a score for the

    Args:
        model(torch.nn.Module): The model containing dynamic embedding tables.
        table_score(Union[int, Dict[str, Dict[str, int]]):
            int: all embedding table's scores will be set to this integer.
            Dict[str, Dict[str, int]]: the first `str` is the name of embedding collection in the model. 'str' in Dict[str, int] is the name of dynamic embedding table, and `int` in Dict[str, int] is the table's score which will broadcast to all scores in the same batch for the table.

    Returns:
        None.
    """
    # TODO:do we need a cuda sync?
    # if torch.cuda.is_available():
    #    torch.cuda.synchronize()
    if isinstance(table_score, int):
        set_all_table = True
    elif is_valid_score_threshold(table_score):
        set_all_table = False
    else:
        raise ValueError(f"DynamicEmb Error:table_score should be int or Dict")

    # find embedding collections
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or ShardedEmbeddingBagCollection module, can't get score!",
            UserWarning,
        )
        return
    # check if the model have dynamic embedding
    check_dynamic_emb_modules_lists: List[List[nn.Module]] = []

    for i, tmp_collection in enumerate(collections_list):
        _, _, tmp_collection_module = tmp_collection
        check_dynamic_emb_modules_lists.append(
            get_dynamic_emb_module(tmp_collection_module)
        )

    has_dynamic_emb = False
    for check_dynamic_emb_module_list in check_dynamic_emb_modules_lists:
        if len(check_dynamic_emb_module_list) > 0:
            has_dynamic_emb = True
            break

    if not has_dynamic_emb:
        warnings.warn(
            "Input model don't have any Dynamic embedding tables, can't set score!",
            UserWarning,
        )
        return
    if not set_all_table:
        # filter the embedding collection
        collection_names_in_module = set()
        filtered_collections_list = []

        for tmp_module_path, tmp_module_name, module in collections_list:
            collection_names_in_module.add(tmp_module_name)
            if tmp_module_name in table_score.keys():
                filtered_collections_list.append(
                    (tmp_module_path, tmp_module_name, module)
                )

        collections_list = filtered_collections_list

        # maybe user input shared module name wrong ,here raise a warning tell user that model don't have the module name
        for tmp_input_collection_name in table_score.keys():
            if tmp_input_collection_name not in collection_names_in_module:
                warnings.warn(
                    f"sharded module '{tmp_input_collection_name}' specified in table_score not found in the model",
                    UserWarning,
                )

    for i, tmp_collection in enumerate(collections_list):
        collection_path, tmp_collection_name, tmp_collection_module = tmp_collection
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)

        for j, dynamic_emb_module in enumerate(tmp_dynamic_emb_module_list):
            tmp_table_names = dynamic_emb_module.table_names

            filtered_table_names: List[str] = []
            filtered_table_scores: List[int] = []
            # TODO:need a warning
            if not set_all_table:
                tmp_collection_scores = table_score[tmp_collection_name]
                tmp_input_names = tmp_collection_scores.keys()
                for name in tmp_input_names:
                    if name in tmp_table_names:
                        index = tmp_table_names.index(name)
                        filtered_table_names.append(tmp_table_names[index])
                        filtered_table_scores.append(tmp_collection_scores[name])
            else:
                filtered_table_names = tmp_table_names
                filtered_table_scores.extend([table_score] * len(tmp_table_names))
            if len(filtered_table_names) == 0:
                continue
            # do set score
            dynamic_emb_module.set_score(
                dict(zip(filtered_table_names, filtered_table_scores))
            )

    return


def get_score(model: torch.nn.Module) -> Union[Dict[str, Dict[str, int]], None]:
    """Get score for each dynamic embediing table.

    Args:
        model(torch.nn.Module): The model containing dynamic embedding tables.

    Returns:
        Dict[str, Dict[str,int]]:
            - The first `str` is the name of embedding collection in the model.
            - The second `str` is the name of dynamic embedding table.
            - `int` represents:
                * TIMESTAMP mode: global timer of device
                * STEP mode: table's step after last forward pass
                * CUSTOMIZED mode: score set in last forward pass
            - Returns None if no dynamic embedding tables exist or scores unavailable
    """

    # TODO:do we need a cuda sync?
    # if torch.cuda.is_available():
    #    torch.cuda.synchronize()

    # find embedding collections
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or ShardedEmbeddingBagCollection module, can't get score!",
            UserWarning,
        )
        return

    # check if the model have dynamic embedding
    check_dynamic_emb_modules_lists: List[List[nn.Module]] = []

    for i, tmp_collection in enumerate(collections_list):
        _, _, tmp_collection_module = tmp_collection
        check_dynamic_emb_modules_lists.append(
            get_dynamic_emb_module(tmp_collection_module)
        )

    has_dynamic_emb = False
    for check_dynamic_emb_module_list in check_dynamic_emb_modules_lists:
        if len(check_dynamic_emb_module_list) > 0:
            has_dynamic_emb = True
            break

    if not has_dynamic_emb:
        warnings.warn(
            "Input model don't have any Dynamic embedding tables, can't get score!",
            UserWarning,
        )
        return
    ret_score_dict: Dict[str, Dict[str, int]] = {}
    for i, tmp_collection in enumerate(collections_list):
        collection_path, tmp_collection_name, tmp_collection_module = tmp_collection
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)

        # do get score
        table_score_map: Dict[str, int] = {}
        for j, dynamic_emb_module in enumerate(tmp_dynamic_emb_module_list):
            table_score_map.update(dynamic_emb_module.get_score())
        ret_score_dict[collection_path] = table_score_map
    return ret_score_dict


def incremental_dump(
    model: torch.nn.Module,
    score_threshold: Union[int, Dict[str, Dict[str, int]]],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, "DeltaDumpResult"]:
    """Dump the model's embedding tables incrementally based on the score threshold. The index-embedding pair whose score is not less than the threshold will be returned.

    Returns ``{collection_path: DeltaDumpResult}`` -- one DeltaDumpResult per
    embedding collection, packing that collection's per-table keys/values, evicted
    keys (for retain-enabled tables) and meta (see :class:`DeltaDumpResult`). An
    empty dict if the model has no dynamic embedding tables.

    Args:
        model(nn.Module):The model containing dynamic embedding tables.
         score_threshold(Uinon[int, Dict[str, Dict[str, int]]]):
            int: All embedding table's score threshold will be this integer. It will dump matched results for all tables in the model.
            Dict[str, Dict[str, int]]: the first `str` is the name of embedding collection in the model. 'str' in Dict[str, int] is the name of dynamic embedding table, and `int` in Dict[str, int] is the table's score threshold. It will dump for only tables whose names present in this Dict.
        pg(Optional[dist.ProcessGroup]): optional. The process group used to control the communication scope in the dump. Defaults to None.

    Returns
    -------
    Dict[str, DeltaDumpResult]:
        One :class:`DeltaDumpResult` per embedding collection, keyed by the
        collection's module path. See that class for the per-table columns and
        the ``meta`` keys.
    """

    if isinstance(score_threshold, int):
        set_all_table = True
    elif is_valid_score_threshold(score_threshold):
        set_all_table = False
    else:
        raise ValueError(f"DynamicEmb Error:score_threshold should be int or Dict")

    # find embedding collections
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or ShardedEmbeddingBagCollection module, can't incremental dump!",
            UserWarning,
        )
        return {}

    # check if the model have dynamic embedding
    check_dynamic_emb_modules_lists: List[List[nn.Module]] = []

    for i, tmp_collection in enumerate(collections_list):
        _, _, tmp_collection_module = tmp_collection
        check_dynamic_emb_modules_lists.append(
            get_dynamic_emb_module(tmp_collection_module)
        )

    has_dynamic_emb = False
    for check_dynamic_emb_module_list in check_dynamic_emb_modules_lists:
        if len(check_dynamic_emb_module_list) > 0:
            has_dynamic_emb = True
            break

    if not has_dynamic_emb:
        warnings.warn(
            "Input model don't have any Dynamic embedding tables, can't incremental dump!",
            UserWarning,
        )
        return {}
    if not set_all_table:
        # filter the embedding collection
        collection_paths_in_module = set()
        filtered_collections_list = []

        for tmp_module_path, tmp_module_name, module in collections_list:
            collection_paths_in_module.add(tmp_module_path)
            if tmp_module_path in score_threshold.keys():
                filtered_collections_list.append(
                    (tmp_module_path, tmp_module_name, module)
                )

        collections_list = filtered_collections_list

        # maybe user input shared module name wrong ,here raise a warning tell user that model don't have the module name
        for tmp_input_collection_name in score_threshold.keys():
            if tmp_input_collection_name not in collection_paths_in_module:
                warnings.warn(
                    f"sharded module '{tmp_input_collection_name}' specified in score_threshold not found in the model",
                    UserWarning,
                )

    ret: Dict[str, DeltaDumpResult] = {}
    for i, tmp_collection in enumerate(collections_list):
        collection_path, tmp_collection_name, tmp_collection_module = tmp_collection
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)

        collection_result = DeltaDumpResult()

        for j, dynamic_emb_module in enumerate(tmp_dynamic_emb_module_list):
            tmp_table_names = dynamic_emb_module.table_names

            filtered_table_names: List[str] = []
            filtered_thresholds: List[int] = []
            # TODO:need a warning
            if not set_all_table:
                tmp_collection_scores = score_threshold[collection_path]
                tmp_input_names = tmp_collection_scores.keys()
                for name in tmp_input_names:
                    if name in tmp_table_names:
                        index = tmp_table_names.index(name)
                        filtered_table_names.append(tmp_table_names[index])
                        filtered_thresholds.append(tmp_collection_scores[name])
            else:
                filtered_table_names = tmp_table_names
                filtered_thresholds.extend([score_threshold] * len(tmp_table_names))
            if len(filtered_table_names) == 0:
                continue
            # do incremental dump -> one DeltaDumpResult per module; merge into
            # this collection's result (lists stay column-aligned by table).
            module_result = dynamic_emb_module.incremental_dump(
                dict(zip(filtered_table_names, filtered_thresholds)), pg
            )
            collection_result.table_names.extend(module_result.table_names)
            collection_result.keys.extend(module_result.keys)
            collection_result.values.extend(module_result.values)
            collection_result.optimizer_states.extend(module_result.optimizer_states)
            collection_result.scores.extend(module_result.scores)
            collection_result.evicted_keys.extend(module_result.evicted_keys)
            collection_result.erased_keys.extend(module_result.erased_keys)
            collection_result.meta.extend(module_result.meta)

        ret[collection_path] = collection_result

    return ret


def replay_increment(
    model: torch.nn.Module,
    deltas: Dict[str, DeltaDumpResult],
    pg: Optional[dist.ProcessGroup] = None,
    content: ReplayContent = ReplayContent.ALL,
) -> Dict[str, Dict[str, ReplayStats]]:
    """Write ``incremental_dump`` results back into a model's dynamic embedding tables.

    The inverse of :func:`incremental_dump`: it takes that call's
    ``{collection_path: DeltaDumpResult}`` and restores every key's embedding and
    score into *model*. Typical use is delta replication -- train on one job,
    dump periodically, ship the delta, replay it into a serving replica.

    **Write-back is by slot.** Every key is written at the slot and value row it
    occupied in the source table, leaving the target layout-identical to it. That
    is only meaningful when the two tables share a layout, so the target is
    checked against the delta's ``meta`` (capacity, bucket capacity, score
    layout, dim, dist_type, world size) and a mismatch raises
    :class:`ValueError` before anything is written.

    Writing at the source's slot **overwrites whatever occupies it** -- that is
    what makes a replica converge (the source evicted that occupant to make
    room). It also means a replayed table must be built *only* by
    loading/replaying from its source: a table that also takes independent writes
    can lose a key whose slot a delta key claims.

    **What gets written** is *content*'s call -- embedding, optimizer state,
    score, or any combination (see :class:`ReplayContent`). The default is all
    three, so the replica ends up holding what the source held.

    Dropping ``SCORE`` leaves a restored key scored as if it had just been
    inserted here, so the replica orders its own future evictions by when it
    received a key rather than by how the source ranked it -- the two can evict
    in different orders, but never hold a wrong embedding for a key they share.
    (NO_EVICTION is unaffected either way: its score word is a value row, not a
    score, and is restored exactly.)

    **Sharding.** Replay always keeps only the keys this rank owns, recomputing
    ownership from the key with *this* model's world size. What that filter does
    depends on how the delta was produced:

    - ``incremental_dump(..., pg)`` all-gathers, so every rank holds the whole
      group's keys; hand the same delta to every rank and each takes its share.
    - ``incremental_dump(..., pg=None)`` leaves each rank with only its own keys;
      replay it on the rank that produced it, where the filter is a no-op.
      Replaying it on a *different* rank is not an error -- every key simply
      belongs to someone else and is skipped, which ``ReplayStats.skipped``
      makes visible.

    Only ``roundrobin`` and ``hash_roundrobin`` tables can be replayed:
    ``continuous`` has no per-key rank mapping to reconstruct, so it raises
    whenever the fan-out is more than one rank (``incremental_dump`` already
    refuses to dump such a table at all).

    **Optimizer state** is not part of a delta. A key that already occupies its
    target row keeps its optimizer state; a row taken over from another key (or a
    brand-new one) is reset to the table's initial optimizer state. This is what
    happens when *content* omits ``OPTIMIZER_STATE``; including it writes the
    state the source dumped, for every key.

    Args:
        model (nn.Module): the model containing dynamic embedding tables.
        deltas (Dict[str, DeltaDumpResult]): ``incremental_dump``'s return value,
            keyed by embedding-collection path. Collections or tables that the
            model does not have are skipped with a warning.
        pg (Optional[dist.ProcessGroup]): process group defining this model's
            shard fan-out. Defaults to the world the tables were created against.
        content (ReplayContent): which parts of each dumped row to write back --
            embedding, optimizer state, score, or any combination. Defaults to
            all three.

    Returns:
        Dict[str, Dict[str, ReplayStats]]:
            ``{collection_path: {table_name: ReplayStats}}`` -- keys written,
            removed and skipped per table. Empty dict when the model has no
            dynamic embedding tables.

    Raises:
        ValueError: a target table's layout does not match the source's, or a
            delta is missing the per-key data replay needs. Raised before
            anything is written.
        TypeError: a module's storage is neither ``DynamicEmbStorage`` nor
            ``HybridStorage``.
        NotImplementedError: a table is sharded with ``dist_type="continuous"``
            across more than one rank.
        RuntimeError: a key could not be written at its source slot even though
            the metadata matched. Unlike the checks above this fires mid-write,
            so that table may hold a partial replay.
    """
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or "
            "ShardedEmbeddingBagCollection module, can't replay increment!",
            UserWarning,
        )
        return {}

    collections_by_path = {path: module for path, _, module in collections_list}
    for collection_path in deltas.keys():
        if collection_path not in collections_by_path:
            warnings.warn(
                f"sharded module '{collection_path}' present in the delta was not "
                "found in the model; skipping it.",
                UserWarning,
            )

    ret: Dict[str, Dict[str, ReplayStats]] = {}
    for collection_path, delta in deltas.items():
        collection_module = collections_by_path.get(collection_path)
        if collection_module is None:
            continue
        collection_stats: Dict[str, ReplayStats] = {}
        for dynamic_emb_module in get_dynamic_emb_module(collection_module):
            # Each module owns a subset of the collection's tables and skips the
            # rest, so the delta can be handed to all of them unchanged.
            module_delta = _select_tables(delta, dynamic_emb_module.table_names)
            if not module_delta.table_names:
                continue
            collection_stats.update(
                dynamic_emb_module.replay_increment(
                    module_delta,
                    pg=pg,
                    content=content,
                )
            )
        ret[collection_path] = collection_stats

    if not ret:
        warnings.warn(
            "Input model don't have any Dynamic embedding tables, can't replay "
            "increment!",
            UserWarning,
        )
    return ret


def _select_tables(delta: DeltaDumpResult, table_names: List[str]) -> DeltaDumpResult:
    """The sub-delta covering only *table_names*, keeping all lists column-aligned."""
    wanted = set(table_names)
    out = DeltaDumpResult()
    for i, name in enumerate(delta.table_names):
        if name not in wanted:
            continue
        out.table_names.append(name)
        out.keys.append(delta.keys[i])
        out.values.append(delta.values[i])
        out.optimizer_states.append(delta.optimizer_states[i])
        out.scores.append(delta.scores[i])
        out.evicted_keys.append(delta.evicted_keys[i])
        out.erased_keys.append(delta.erased_keys[i])
        out.meta.append(delta.meta[i])
    return out


def _all_gather_evicted_keys(
    keys: torch.Tensor, pg: Optional[dist.ProcessGroup]
) -> torch.Tensor:
    """All-gather variable-length evicted keys within ``pg``, then concat + unique.

    Returns a host (CPU) tensor; the input may be on host or device (the NCCL
    gather itself runs on the CUDA device). Each rank holds a disjoint set
    (row-wise sharding), so the group-wide union is just the concatenation;
    ``unique`` is applied defensively. Implemented as a size all_gather followed by
    a padded all_gather (NCCL has no native variable-length gather).
    """
    world_size = dist.get_world_size(pg)
    if world_size <= 1:
        return keys.cpu()
    device = torch.device("cuda", torch.cuda.current_device())
    keys = keys.to(device)
    local_size = torch.tensor([keys.numel()], device=device, dtype=torch.int64)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=pg)
    sizes = [int(s.item()) for s in size_list]
    max_size = max(max(sizes), 1)  # avoid a 0-length NCCL all_gather
    padded = torch.zeros(max_size, dtype=keys.dtype, device=device)
    if keys.numel() > 0:
        padded[: keys.numel()] = keys
    gathered = [
        torch.zeros(max_size, dtype=keys.dtype, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, padded, group=pg)
    parts = [g[:sz] for g, sz in zip(gathered, sizes) if sz > 0]
    if not parts:
        return keys[:0].cpu()
    return torch.unique(torch.cat(parts)).cpu()


def _pop_retained_keys(
    model: torch.nn.Module,
    method: str,
    what: str,
    table_names: Optional[Dict[str, List[str]]],
    pg: Optional[dist.ProcessGroup],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Shared body of :func:`pop_evicted_keys` / :func:`pop_erased_keys`.

    The two differ only in which buffer they drain; *method* names the module
    method and *what* is the noun used in the "no such tables" warning.
    """
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or "
            f"ShardedEmbeddingBagCollection module, can't pop {what} keys!",
            UserWarning,
        )
        return {}

    ret: Dict[str, Dict[str, torch.Tensor]] = {}
    for collection_path, _collection_name, collection_module in collections_list:
        if table_names is not None and collection_path not in table_names:
            continue
        wanted = table_names.get(collection_path) if table_names is not None else None

        collection_result: Dict[str, torch.Tensor] = {}
        for dynamic_emb_module in get_dynamic_emb_module(collection_module):
            if not hasattr(dynamic_emb_module, method):
                continue
            local = getattr(dynamic_emb_module, method)(wanted)
            for tname, keys in local.items():
                if pg is not None:
                    keys = _all_gather_evicted_keys(keys, pg)
                collection_result[tname] = keys

        if collection_result:
            ret[collection_path] = collection_result

    return ret


def pop_evicted_keys(
    model: torch.nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return + clear the keys last-tier storage **evicted** to make room, per table.

    Only tables configured with ``evicted_item_mode=RETAIN_KEY`` are included;
    all other tables are omitted from the result. This is the
    incremental "pop" of keys evicted since the previous call -- the retained
    buffers are cleared on this rank as they are read.

    Keys removed by an explicit erase are reported separately, by
    :func:`pop_erased_keys`. The two are kept apart because a consumer usually
    wants different things from them: an eviction says the table ran out of room,
    an erase says someone asked for the key to go.

    Args:
        model (nn.Module): the model containing dynamic embedding tables.
        table_names (Optional[Dict[str, List[str]]]): optional filter, keyed by
            embedding-collection path. ``{collection_path: [table_name, ...]}``
            pops only the listed collections/tables. ``None`` pops every
            retain-enabled table in the model.
        pg (Optional[dist.ProcessGroup]): optional process group. ``None`` returns
            each rank's LOCAL evicted keys (row-wise sharded, hence disjoint across
            ranks; zero communication). When given, keys are all_gathered within
            ``pg`` so every rank in the group receives the group-wide union.
            Clearing always affects only this rank's buffer, regardless of ``pg``.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]:
            ``{collection_path: {table_name: keys}}`` where ``keys`` is a 1-D
            tensor of table-unique evicted keys. Empty dict when the model has no
            tables retaining evictions (or none matched).
    """
    return _pop_retained_keys(model, "pop_evicted_keys", "evicted", table_names, pg)


def pop_erased_keys(
    model: torch.nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return + clear the keys an **explicit erase** removed, per table.

    The counterpart of :func:`pop_evicted_keys`, with the same arguments and the
    same drain-on-read semantics. Every table is included, unlike
    ``pop_evicted_keys``: whether an erase was recorded is that ``erase`` call's
    decision, not the table's, so there is no configuration to filter on -- a
    table nobody asked to record simply returns an empty tensor.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]:
            ``{collection_path: {table_name: keys}}`` where ``keys`` is a 1-D
            tensor of table-unique erased keys. Empty dict when the model has no
            tables retaining erases (or none matched).
    """
    return _pop_retained_keys(model, "pop_erased_keys", "erased", table_names, pg)
