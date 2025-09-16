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

import json
import os
import shutil
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTables
from dynamicemb.dynamicemb_config import dtype_to_bytes, dyn_emb_to_torch
from dynamicemb_extensions import (
    DynamicEmbTable,
    EvictStrategy,
    count_matched,
    dyn_emb_capacity,
    dyn_emb_cols,
    dyn_emb_rows,
    export_batch,
    insert_or_assign,
)
from torch import nn
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.model_parallel import get_unwrapped_module

_PAD_TO_LENGTH = 4096  # Pad strings to multiples of 4096 for broadcasting


def _calculate_pad_length(original_str: str, base: int = _PAD_TO_LENGTH) -> int:
    return ((len(original_str) + base - 1) // base) * base


def debug_check_dynamic_table_is_zero(dynamic_table):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    need_dump_score = dynamic_table.evict_strategy() != EvictStrategy.KLru

    key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
    value_dtype = dyn_emb_to_torch(dynamic_table.value_type())
    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()

    search_capacity = dyn_emb_capacity(dynamic_table)

    local_max_rows = dyn_emb_rows(dynamic_table)

    keys = torch.ones(local_max_rows, dtype=key_dtype, device=device)
    values = torch.ones(
        local_max_rows * (dim + optstate_dim), dtype=value_dtype, device=device
    )
    d_counter = torch.zeros(1, dtype=torch.uint64, device=device)

    if need_dump_score:
        score_dtype = torch.uint64
        scores = torch.zeros(local_max_rows, dtype=score_dtype, device=device)

        export_batch(dynamic_table, search_capacity, 0, d_counter, keys, values, scores)
    else:
        export_batch(dynamic_table, search_capacity, 0, d_counter, keys, values)

    values = values.reshape(-1, dim + optstate_dim)[:, :dim].contiguous().reshape(-1)

    value_is_zero = torch.all(values == 0)
    score_is_one = True
    if need_dump_score:
        score_is_one = torch.all(
            scores == 1
        )  # HKV score have a placeholder value , the value is 0 ,so we use scores == 1
    # Check if all values in the values tensor are zero
    if value_is_zero and score_is_one:
        print("DynamicEmb Debug:All values are zero.")
    else:
        non_zero_indices_for_value = torch.nonzero(values)
        print(
            f"DynamicEmb Debug:Not all values are zero. Non-zero values found at indices: {non_zero_indices_for_value.tolist()}"
        )
        if need_dump_score:
            non_one_indices_for_score = torch.nonzero((scores.to(torch.int64) != 1))
            print(
                f"DynamicEmb Debug:Not all scores are one. Non-one scores found at indices = {non_one_indices_for_score} scores = {scores}"
            )
        raise ValueError(
            f"DynamicEmb Debug:Not all values are zero and not all scores are one. Non-zero values found"
        )


def debug_dump(embedding_collections_list, path, table_names, optim, pg):
    def debug_export(dynamic_table, root_path, name, optim):
        rank = dist.get_rank(group=pg)
        world_size = dist.get_world_size(group=pg)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        need_dump_score = dynamic_table.evict_strategy() != EvictStrategy.KLru

        key_name = name + "_keys"
        value_name = name + "_values"

        key_path = os.path.join(root_path, key_name)
        value_path = os.path.join(root_path, value_name)

        fkey = open(key_path, "wb")
        fvalue = open(value_path, "wb")
        local_max_rows = dyn_emb_rows(dynamic_table)
        dim = dyn_emb_cols(dynamic_table)
        search_capacity = dyn_emb_capacity(dynamic_table)

        key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
        value_dtype = dyn_emb_to_torch(dynamic_table.value_type())
        optstate_dim = dynamic_table.optstate_dim()

        keys = torch.empty(local_max_rows, dtype=key_dtype, device=device)
        values = torch.empty(
            local_max_rows * (dim + optstate_dim), dtype=value_dtype, device=device
        )
        d_counter = torch.zeros(1, dtype=torch.uint64, device=device)

        if need_dump_score:
            score_name = name + "_scores"
            score_path = os.path.join(root_path, score_name)
            fscore = open(score_path, "wb")
            score_dtype = torch.uint64
            scores = torch.empty(local_max_rows, dtype=score_dtype, device=device)

            export_batch(
                dynamic_table, search_capacity, 0, d_counter, keys, values, scores
            )
        else:
            export_batch(dynamic_table, search_capacity, 0, d_counter, keys, values)

        cap = dynamic_table.capacity()
        assert cap == search_capacity
        d_num_matched = torch.zeros(1, dtype=torch.uint64, device=device)
        count_matched(dynamic_table, 0, d_num_matched)
        assert d_num_matched.item() == d_counter.item()
        d_counter_cpu = d_counter.item()
        if d_counter_cpu != local_max_rows:
            raise ValueError(
                f"DynamicEmb Debug: d_counter ({d_counter_cpu}) does not match local_max_rows ({local_max_rows})"
            )

        keys_int64 = keys.to(torch.int64)
        values_float = values.reshape(-1, dim + optstate_dim)[
            :, : dim + optstate_dim if optim else dim
        ].to(torch.float)

        fkey.write(keys_int64.cpu().numpy().tobytes())
        fvalue.write(values_float.cpu().numpy().tobytes())
        fkey.close()
        fvalue.close()

        if need_dump_score:
            scores_uint64 = scores.to(torch.uint64)

            fscore.write(scores_uint64.cpu().numpy().tobytes())
            fscore.close()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values[:] = 0
        if need_dump_score:
            scores[
                :
            ] = 1  # HKV score have a placeholder value , the value is 0 ,so we set scores = 1
            insert_or_assign(dynamic_table, d_counter_cpu, keys, values, scores)
        else:
            insert_or_assign(dynamic_table, d_counter_cpu, keys, values)
        d_num_matched = torch.zeros(1, dtype=torch.uint64, device=device)
        count_matched(dynamic_table, 0, d_num_matched)
        assert d_num_matched.item() == d_counter_cpu

        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)

    debug_root_path = path + "_compare"
    if rank == 0:
        if os.path.exists(debug_root_path):
            if os.path.isfile(debug_root_path):
                os.remove(debug_root_path)
            elif os.path.isdir(debug_root_path):
                shutil.rmtree(debug_root_path)

        os.makedirs(debug_root_path, exist_ok=True)

    search_module_table = {}

    for i, embedding_collection in enumerate(embedding_collections_list):
        (
            collection_path,
            tmp_module_name,
            embedding_collection_module,
        ) = embedding_collection
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(
            embedding_collection_module
        )
        search_module_table[tmp_module_name] = []

        for j, dynamic_emb_module in enumerate(tmp_dynamic_emb_module_list):
            tmp_table_names = dynamic_emb_module.table_names
            for tmp_table_name in tmp_table_names:
                search_module_table[tmp_module_name].append(tmp_table_name)

    data_to_write = {
        "rank": rank,
        "world_size": world_size,
        "data": table_names if table_names is not None else search_module_table,
    }

    if rank == 0:
        with open(os.path.join(debug_root_path, "meta.json"), "w") as f:
            json.dump(data_to_write, f, indent=4)

    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    for i, embedding_collection in enumerate(embedding_collections_list):
        (
            collection_path,
            tmp_module_name,
            embedding_collection_module,
        ) = embedding_collection
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(
            embedding_collection_module
        )

        for j, dynamic_emb_module in enumerate(tmp_dynamic_emb_module_list):
            tmp_table_names = dynamic_emb_module.table_names
            tmp_tables = dynamic_emb_module.tables

            filtered_table_names: List[str] = []
            filtered_dynamic_tables: List[DynamicEmbTable] = []

            # TODO:need a warning
            if table_names is not None:
                tmp_input_names = table_names[tmp_module_name]
                for name in tmp_input_names:
                    if name in tmp_table_names:
                        index = tmp_table_names.index(name)
                        filtered_table_names.append(name)
                        filtered_dynamic_tables.append(tmp_tables[index])
            else:
                filtered_table_names = tmp_table_names
                filtered_dynamic_tables = tmp_tables

            if len(filtered_table_names) == 0:
                continue

            if optim:
                optimizer = dynamic_emb_module.optimizer
                optimizer.get_opt_args()

            tmp_tables_dict: Dict[str, DynamicEmbTable] = {
                name: table
                for name, table in zip(filtered_table_names, filtered_dynamic_tables)
            }
            # Get the rank of the current process

            for k, dump_name in enumerate(filtered_table_names):
                dynamic_table = filtered_dynamic_tables[k]
                debug_export(
                    dynamic_table,
                    debug_root_path,
                    dump_name + "_" + str(rank),
                    optim=optim,
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return


def debug_load(embedding_collections_list, path, table_names, optim, pg):
    def validate_dynamic_embedding(
        debug_root_path, dump_name, rank, dynamic_table, optim
    ):
        need_dump_score = dynamic_table.evict_strategy() != EvictStrategy.KLru

        key_name = dump_name + "_" + str(rank) + "_keys"
        value_name = dump_name + "_" + str(rank) + "_values"

        key_path = os.path.join(debug_root_path, key_name)
        value_path = os.path.join(debug_root_path, value_name)

        key_file_size = os.path.getsize(key_path)
        value_file_size = os.path.getsize(value_path)

        key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
        value_dtype = dyn_emb_to_torch(dynamic_table.value_type())

        key_bytes = dtype_to_bytes(key_dtype)
        value_bytes = dtype_to_bytes(value_dtype)

        total_keys = key_file_size // key_bytes
        total_dim = value_file_size // (total_keys * value_bytes)

        dim = dyn_emb_cols(dynamic_table)
        optstate_dim = dynamic_table.optstate_dim()
        exist_optstate: bool = total_dim > dim

        if need_dump_score:
            score_name = dump_name + "_" + str(rank) + "_scores"
            score_path = os.path.join(debug_root_path, score_name)
            os.path.getsize(score_path)

        if key_file_size == 0:
            return

        local_max_rows = dyn_emb_rows(dynamic_table)

        key_dtype_in_table = dyn_emb_to_torch(dynamic_table.key_type())
        value_dtype_in_table = dyn_emb_to_torch(dynamic_table.value_type())

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        keys_in_dynemb = torch.empty(
            local_max_rows, dtype=key_dtype_in_table, device=device
        )
        values_in_dynemb = torch.empty(
            local_max_rows * (dim + optstate_dim),
            dtype=value_dtype_in_table,
            device=device,
        )

        with open(key_path, "rb") as fkey:
            keys = torch.frombuffer(fkey.read(), dtype=torch.int64).clone()

        with open(value_path, "rb") as fvalue:
            values = torch.frombuffer(fvalue.read(), dtype=torch.float).clone()

        assert not torch.all(values == 0), "AssertionError: All values are zero."

        keys_device = keys.to(device)
        values_device = values.to(device)

        if need_dump_score:
            score_dtype_in_table = torch.uint64
            scores_in_dynemb = torch.empty(
                local_max_rows, dtype=score_dtype_in_table, device=device
            )
            with open(score_path, "rb") as fscore:
                scores = torch.frombuffer(fscore.read(), dtype=torch.uint64).clone()
            assert not torch.all(scores == 1), "AssertionError: All scores are one."
            scores_device = scores.to(device)

        d_counter = torch.zeros(1, dtype=torch.uint64, device=device)
        if need_dump_score:
            export_batch(
                dynamic_table,
                dyn_emb_capacity(dynamic_table),
                0,
                d_counter,
                keys_in_dynemb,
                values_in_dynemb,
                scores_in_dynemb,
            )
        else:
            export_batch(
                dynamic_table,
                dyn_emb_capacity(dynamic_table),
                0,
                d_counter,
                keys_in_dynemb,
                values_in_dynemb,
            )
        if not optim:
            values_in_dynemb = (
                values_in_dynemb.reshape(-1, dim + optstate_dim)[:, :dim]
                .contiguous()
                .reshape(-1)
            )

        keys_in_dynemb_int64 = keys_in_dynemb.to(torch.int64)
        values_in_dynemb_float = values_in_dynemb.to(torch.float)

        sorted_keys_device, sorted_indices_device = torch.sort(keys_device)
        sorted_keys_in_dynemb_int64, sorted_indices_in_dynemb_int64 = torch.sort(
            keys_in_dynemb_int64
        )

        sorted_values_device = values_device.view(
            local_max_rows, dim + optstate_dim if exist_optstate else dim
        )[:, : dim + optstate_dim if optim else dim][sorted_indices_device].view(-1)
        sorted_values_in_dynemb_float = values_in_dynemb_float.view(
            local_max_rows, dim + optstate_dim if optim else dim
        )[sorted_indices_in_dynemb_int64].view(-1)

        if sorted_keys_device.shape != sorted_keys_in_dynemb_int64.shape:
            raise ValueError(
                f"DynamicEmb Debug: Shape mismatch for keys in table {dump_name}: {sorted_keys_device.shape} (dumped) != {sorted_keys_in_dynemb_int64.shape} (loaded)"
            )
        if sorted_values_device.shape != sorted_values_in_dynemb_float.shape:
            raise ValueError(
                f"DynamicEmb Debug: Shape mismatch for values in table {dump_name}: {sorted_values_device.shape} (dumped) != {sorted_values_in_dynemb_float.shape} (loaded)"
            )

        if not torch.equal(sorted_keys_device, sorted_keys_in_dynemb_int64):
            diff_keys = torch.nonzero(sorted_keys_device != sorted_keys_in_dynemb_int64)
            raise ValueError(
                f"DynamicEmb Debug: Mismatch in keys for table {dump_name} at positions {diff_keys.tolist()}"
            )
        if not torch.equal(sorted_values_device, sorted_values_in_dynemb_float):
            diff_values = torch.nonzero(
                sorted_values_device != sorted_values_in_dynemb_float
            )
            raise ValueError(
                f"DynamicEmb Debug: Mismatch in values for table {dump_name} at positions {diff_values.tolist()}"
            )

        if need_dump_score:
            scores_in_dynemb_uint64 = scores_in_dynemb.to(torch.uint64)

            scores_in_dynemb_int64 = scores_in_dynemb_uint64.view(torch.int64)
            scores_device_int64 = scores_device.view(torch.int64)

            sorted_scores_device = scores_device_int64[sorted_indices_device]
            sorted_scores_in_dynemb_int64 = scores_in_dynemb_int64[
                sorted_indices_in_dynemb_int64
            ]

            if sorted_scores_device.shape != sorted_scores_in_dynemb_int64.shape:
                raise ValueError(
                    f"DynamicEmb Debug: Shape mismatch for scores in table {dump_name}: {sorted_scores_device.shape} (dumped) != {sorted_scores_in_dynemb_uint64.shape} (loaded)"
                )

            if not torch.equal(sorted_scores_device, sorted_scores_in_dynemb_int64):
                diff_values = torch.nonzero(
                    sorted_scores_device != sorted_scores_in_dynemb_int64
                )
                raise ValueError(
                    f"DynamicEmb Debug: Mismatch in scores for table {dump_name} at positions {diff_values.tolist()}"
                )

    debug_root_path = path + "_compare"
    meta_path = os.path.join(debug_root_path, "meta.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"DynamicEmb Debug: Meta file not found at {meta_path}")

    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    expected_data = table_names if table_names is not None else {}
    if table_names is None:
        for (
            collection_path,
            tmp_module_name,
            embedding_collection_module,
        ) in embedding_collections_list:
            tmp_dynamic_emb_module_list = get_dynamic_emb_module(
                embedding_collection_module
            )
            expected_data[tmp_module_name] = []
            for dynamic_emb_module in tmp_dynamic_emb_module_list:
                tmp_table_names = dynamic_emb_module.table_names
                expected_data[tmp_module_name].extend(tmp_table_names)

    if meta_data["data"] != expected_data:
        raise ValueError("Mismatch between expected data and meta data from debug dump")

    for i, embedding_collection in enumerate(embedding_collections_list):
        (
            collection_path,
            tmp_module_name,
            embedding_collection_module,
        ) = embedding_collection
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(
            embedding_collection_module
        )
        for j, dynamic_emb_module in enumerate(tmp_dynamic_emb_module_list):
            tmp_table_names = dynamic_emb_module.table_names
            tmp_tables = dynamic_emb_module.tables

            filtered_table_names = []
            filtered_dynamic_tables = []

            if table_names is not None:
                tmp_input_names = table_names[collection_path]
                for name in tmp_input_names:
                    if name in tmp_table_names:
                        index = tmp_table_names.index(name)
                        filtered_table_names.append(name)
                        filtered_dynamic_tables.append(tmp_tables[index])
            else:
                filtered_table_names = tmp_table_names
                filtered_dynamic_tables = tmp_tables

            if len(filtered_table_names) == 0:
                continue

            tmp_tables_dict = {
                name: table
                for name, table in zip(filtered_table_names, filtered_dynamic_tables)
            }

            for k, dump_name in enumerate(filtered_table_names):
                dynamic_table = tmp_tables_dict[dump_name]
                validate_dynamic_embedding(
                    debug_root_path, dump_name, rank, dynamic_table, optim=optim
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error saving data to JSON file: {e}")


def load_from_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data from JSON file: {e}")


def find_sharded_modules(
    module: nn.Module, path: str = ""
) -> List[Tuple[str, str, nn.Module]]:
    sharded_modules: List[Tuple[str, str, nn.Module]] = []
    stack: deque[Tuple[nn.Module, str, str]] = deque(
        [(module, path + "model", "model")]
    )

    while stack:
        current_module, current_path, current_name = stack.pop()
        current_module = get_unwrapped_module(current_module)
        if isinstance(
            current_module, (ShardedEmbeddingCollection, ShardedEmbeddingBagCollection)
        ):
            sharded_modules.append((current_path, current_name, current_module))
        else:
            for name, child in current_module.named_children():
                child_path = current_path + ("." if current_path else "") + name

                stack.append((child, child_path, name))

    return sharded_modules


def check_emb_collection_modules(module: nn.Module, ret_list: List[nn.Module]):
    if isinstance(module, BatchedDynamicEmbeddingTables):
        ret_list.append(module)
        return ret_list

    if isinstance(module, nn.Module):
        if hasattr(module, "_emb_module"):
            check_emb_collection_modules(module._emb_module, ret_list)

        if hasattr(module, "_emb_modules"):
            check_emb_collection_modules(module._emb_modules, ret_list)

        if hasattr(module, "_lookups"):
            tmp_module_list = module._lookups
            for tmp_emb_module in tmp_module_list:
                check_emb_collection_modules(tmp_emb_module, ret_list)

    if isinstance(module, nn.ModuleList):
        for i in range(len(module)):
            tmp_emb_module = module[i]

            if isinstance(tmp_emb_module, nn.Module):
                check_emb_collection_modules(tmp_emb_module, ret_list)
            else:
                continue


def get_dynamic_emb_module(model: nn.Module) -> List[nn.Module]:
    dynamic_emb_module_list: List[nn.Module] = []
    check_emb_collection_modules(model, dynamic_emb_module_list)
    return dynamic_emb_module_list


# TODO:will use in TW TWRW , to gather tablename , and do communication
def pad_to_length(original_str: str, length: int, pad_char: str = " ") -> str:
    """
    Pad the input string to a specified length with a given character.

    Args:
        original_str (str): The input string to be padded.
        length (int): The desired length of the padded string.
        pad_char (str): The character to use for padding. Default is space.

    Returns:
        str: The padded string.
    """
    return original_str.ljust(length, pad_char)


def broadcast_string(
    original_str: str,
    length: int,
    rank: int = 0,
    pg: Optional[dist.ProcessGroup] = None,
) -> str:
    """
    Broadcasts a string from rank 0 to all other ranks.

    Args:
        original_str (str): The input string to be broadcasted.
        rank (int): The rank of the current process.

    Returns:
        str: The resulting string after broadcasting.
    """

    padded_str = pad_to_length(original_str, length)
    ascii_values = [ord(char) for char in padded_str]
    tensor = torch.tensor(ascii_values, dtype=torch.int32).cuda()

    broadcasted_tensor = tensor.unsqueeze(0)
    dist.broadcast(broadcasted_tensor, src=0, group=pg)
    torch.cuda.synchronize()
    result_str = "".join(chr(value.item()) for value in broadcasted_tensor[0]).rstrip()
    return result_str


def export_keys_values(dynamic_table, offset, device, batch_size=65536, optim=False):
    key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
    value_dtype = dyn_emb_to_torch(dynamic_table.value_type())
    score_dtype = torch.uint64
    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()
    total_dim = dim + optstate_dim

    keys = torch.empty(batch_size, dtype=key_dtype, device=device)
    values = torch.empty(batch_size * total_dim, dtype=value_dtype, device=device)
    scores = torch.zeros(batch_size, dtype=score_dtype, device=device)
    d_counter = torch.zeros(1, dtype=torch.uint64, device=device)

    export_batch(dynamic_table, batch_size, offset, d_counter, keys, values, scores)

    if not optim:
        values = values.reshape(batch_size, total_dim)[:, :dim].contiguous().reshape(-1)

    return keys, values, scores, d_counter


def gather_and_export(
    dynamic_table,
    root_path,
    name,
    batch_size=65536,
    pg: Optional[dist.ProcessGroup] = None,
    optim: bool = False,
):
    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    need_dump_score = dynamic_table.evict_strategy() != EvictStrategy.KLru

    key_name = name + "_keys"
    value_name = name + "_values"

    key_path = os.path.join(root_path, key_name)
    value_path = os.path.join(root_path, value_name)

    if rank == 0:
        fkey = open(key_path, "wb")
        fvalue = open(value_path, "wb")

    if need_dump_score:
        score_name = name + "_scores"
        score_path = os.path.join(root_path, score_name)
        if rank == 0:
            fscore = open(score_path, "wb")

    local_max_rows = dyn_emb_rows(dynamic_table)
    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()
    if optim:
        dim += optstate_dim

    search_capacity = dyn_emb_capacity(dynamic_table)

    max_rows_tensor = torch.tensor(local_max_rows, dtype=torch.int64, device=device)
    gathered_local_max_rows = (
        [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)]
        if rank == 0
        else None
    )
    dist.gather(max_rows_tensor, gather_list=gathered_local_max_rows, dst=0, group=pg)

    if rank == 0:
        gathered_local_max_rows = [t.item() for t in gathered_local_max_rows]

    accumulated_counts = [0] * world_size if rank == 0 else None

    max_rows_tensor.item()
    offset = 0

    while offset < search_capacity:
        keys, values, scores, d_counter = export_keys_values(
            dynamic_table, offset, device, batch_size, optim=optim
        )
        d_counter = d_counter.to(dtype=torch.int64)

        # Gather keys and values at the root process (rank 0)
        gathered_keys = (
            [torch.empty_like(keys) for _ in range(world_size)] if rank == 0 else None
        )
        gathered_values = (
            [torch.empty_like(values) for _ in range(world_size)] if rank == 0 else None
        )
        gathered_counts = (
            [torch.empty_like(d_counter) for _ in range(world_size)]
            if rank == 0
            else None
        )

        if need_dump_score:
            scores = scores.contiguous()
            scores_bytes = scores.view(torch.uint8)
            gathered_scores_bytes = (
                [torch.empty_like(scores_bytes) for _ in range(world_size)]
                if rank == 0
                else None
            )

        dist.gather(keys, gather_list=gathered_keys, dst=0, group=pg)
        dist.gather(values, gather_list=gathered_values, dst=0, group=pg)
        dist.gather(d_counter, gather_list=gathered_counts, dst=0, group=pg)

        if need_dump_score:
            dist.gather(
                scores_bytes, gather_list=gathered_scores_bytes, dst=0, group=pg
            )

        if rank == 0:
            if need_dump_score:
                gathered_scores = [
                    tensor.view(torch.uint64) for tensor in gathered_scores_bytes
                ]
            for i in range(world_size):
                actual_length = gathered_counts[i].item()
                if actual_length > 0:
                    tmp_gathered_keys = (
                        gathered_keys[i][:actual_length].to(torch.int64).cpu()
                    )
                    fkey.write(tmp_gathered_keys.numpy().tobytes())
                    tmp_gathered_values = (
                        gathered_values[i][: actual_length * dim].to(torch.float).cpu()
                    )
                    fvalue.write(tmp_gathered_values.numpy().tobytes())
                    if need_dump_score:
                        tmp_gathered_scores = (
                            gathered_scores[i][:actual_length].to(torch.uint64).cpu()
                        )
                        fscore.write(tmp_gathered_scores.numpy().tobytes())
                    accumulated_counts[i] += actual_length
        offset += batch_size
    if rank == 0:
        fkey.close()
        fvalue.close()
        if need_dump_score:
            fscore.close()
        for i in range(world_size):
            assert accumulated_counts[i] == gathered_local_max_rows[i], (
                f"Rank {i} has accumulated count {accumulated_counts[i]} which is different from expected {gathered_local_max_rows[i]}, "
                f"difference: {accumulated_counts[i] - gathered_local_max_rows[i]}"
            )

    return


def load_table(
    dynamic_table,
    root_path,
    name,
    batch_size=65536,
    pg: Optional[dist.ProcessGroup] = None,
    debug_mode: Optional[bool] = False,
    optim: bool = False,
):
    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    need_dump_score = dynamic_table.evict_strategy() != EvictStrategy.KLru

    key_name = name + "_keys"
    value_name = name + "_values"

    key_path = os.path.join(root_path, key_name)
    value_path = os.path.join(root_path, value_name)

    if not os.path.exists(key_path):
        raise Exception("can't find path to load, path:", key_path)

    if not os.path.exists(value_path):
        raise Exception("can't find path to load, path:", value_path)

    key_file_size = os.path.getsize(key_path)
    value_file_size = os.path.getsize(value_path)

    key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
    value_dtype = dyn_emb_to_torch(dynamic_table.value_type())

    key_bytes = dtype_to_bytes(key_dtype)
    value_bytes = dtype_to_bytes(value_dtype)

    total_keys = key_file_size // key_bytes
    total_dim = value_file_size // (total_keys * value_bytes)

    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()
    if total_dim < dim or ((total_dim != dim + optstate_dim) and optim):
        raise Exception(
            "Can't load as mismatch of embedding dtype, dim or optimizer type"
        )

    keys_read_bytes = batch_size * 8  # key in file always int64 ,so is 8
    values_read_bytes = (
        batch_size * total_dim * 4
    )  # value in file always float , so is 4

    if debug_mode:
        debug_check_dynamic_table_is_zero(dynamic_table)
    if need_dump_score:
        score_name = name + "_scores"
        score_path = os.path.join(root_path, score_name)
        if not os.path.exists(score_path):
            raise Exception("can't find path to load, path:", score_path)
        score_dtype = torch.uint64
        scores_read_bytes = batch_size * 8  # score in file always uint64, so is 8
        score_file_size = os.path.getsize(score_path)

    with open(key_path, "rb") as fkey, open(value_path, "rb") as fvalue:
        if need_dump_score:
            fscore = open(score_path, "rb")
        while True:
            remaining_key_bytes = key_file_size - fkey.tell()
            remaining_value_bytes = value_file_size - fvalue.tell()

            if remaining_key_bytes <= 0 or remaining_value_bytes <= 0:
                break

            key_bytes_to_read = min(keys_read_bytes, remaining_key_bytes)
            value_bytes_to_read = min(values_read_bytes, remaining_value_bytes)

            key_bytes = fkey.read(key_bytes_to_read)
            value_bytes = fvalue.read(value_bytes_to_read)

            num_keys = len(key_bytes) // 8  # key in file always int64 ,so is 8

            key_array = np.frombuffer(key_bytes, dtype=np.int64)
            value_array = np.frombuffer(value_bytes, dtype=np.float32).reshape(
                -1, total_dim
            )

            if need_dump_score:
                remaining_score_bytes = score_file_size - fscore.tell()
                score_bytes_to_read = min(scores_read_bytes, remaining_score_bytes)
                score_bytes = fscore.read(score_bytes_to_read)
                score_array = np.frombuffer(score_bytes, dtype=np.uint64)

            # Masking keys and values based on rank
            mask = key_array % world_size == rank
            masked_keys = key_array[mask]
            masked_values = value_array[mask, :]
            if need_dump_score:
                masked_scores = score_array[mask]
            if masked_keys.shape[0] > 0:
                keys_tensor = torch.tensor(masked_keys, dtype=key_dtype, device=device)
                values_tensor = torch.tensor(
                    masked_values, dtype=value_dtype, device=device
                )
                if not optim:
                    optstate = (
                        torch.ones(
                            values_tensor.size(0),
                            optstate_dim,
                            dtype=value_dtype,
                            device=device,
                        )
                        * dynamic_table.get_initial_optstate()
                    )
                    values_tensor = torch.cat(
                        (values_tensor[:, :dim], optstate), dim=1
                    ).contiguous()
                if need_dump_score:
                    scores_tensor = torch.tensor(
                        masked_scores, dtype=score_dtype, device=device
                    )
                    insert_or_assign(
                        dynamic_table,
                        masked_keys.shape[0],
                        keys_tensor,
                        values_tensor,
                        scores_tensor,
                    )
                else:
                    insert_or_assign(
                        dynamic_table, masked_keys.shape[0], keys_tensor, values_tensor
                    )

        if need_dump_score:
            fscore.close()
    return


# TODO: Now only support Row-Wise sharding, will support TW/TWRW in future.
# TODO: Currently, the dump and load functions enforce dumping and loading of all parameters of the optimizer.
#      This mechanism prevents users from controlling certain parameters of the optimizer.
#      In order to allow users to set optimizer parameters more flexibly in the future,
#      we need to add functionality for dumping and loading specific args,
#      allowing for more flexible configuration of optimizer arguments
def DynamicEmbDump(
    path: str,
    model: nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    optim: Optional[bool] = False,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Dump the distributed weights and corresponding optimizer states of dynamic embedding tables from the model to the filesystem.
    The weights of the dynamic embedding table will be stored in each EmbeddingCollection or EmbeddingBagCollection folder.
    The name of the collection is the path of the torch module within the model, with the input module defined as str of model.

    Each dynamic embedding table will be stored as a key binary file and a value binary file, where the dtype of the key is int64_t,
    and the dtype of the value is float. Each optimizer state is also treated as a dynamic embedding table.

    Parameters
    ----------
    path : str
        The main folder for weight files.
    model : nn.Module
        The model containing dynamic embedding tables.
    table_names : Optional[Dict[str, List[str]]], optional
        A dictionary specifying which embedding collection and which table to dump. The key is the name of the embedding collection,
        and the value is a list of dynamic embedding table names within that collection. Defaults to None.
    optim : Optional[bool], optional
        Whether to dump the optimizer states. Defaults to False.
    pg : Optional[dist.ProcessGroup], optional
        The process group used to control the communication scope in the dump. Defaults to None.

    Returns
    -------
    None
    """

    # Due to the difficulty of checking the HKV dump and load operations externally,
    # debug_mode has been added to the code. This is controlled by the environment variable
    # DYNAMICEMB_DUMP_LOAD_DEBUG. When DYNAMICEMB_DUMP_LOAD_DEBUG=1, debug mode is enabled.
    # In debug mode, the state of dynamic emb will be saved before dumping to the file system,
    # and a comparison will be performed after the load operation.
    debug_env_var = os.getenv("DYNAMICEMB_DUMP_LOAD_DEBUG")
    if debug_env_var == "1":
        debug_mode = True
        print("DynamicEmb's dump and load is in debug mode")
    else:
        debug_mode = False

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # create path
    created_dirs = []
    if not os.path.exists(path):
        created_dirs.append(path)
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            raise Exception("can't build path:", path) from e
    else:
        if os.path.isdir(path):
            if not os.listdir(path):
                pass
            else:
                raise Exception(
                    f"DynamicEmb Cannot dump to {path} because it already contains files, "
                    "as it may cause overwriting of existing files with the same name."
                )
        else:
            raise Exception(f"The path '{path}' exists and is not a directory.")
    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    # find embedding collections
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or ShardedEmbeddingBagCollection module, don't dump any embedding tables to filesystem!",
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
            "Input model don't have any Dynamic embedding tables, don't dump any embedding tables to filesystem!",
            UserWarning,
        )
        return

    # filter the embedding collection
    collection_names_in_module = set()
    filtered_collections_list = []

    for tmp_module_path, tmp_module_name, module in collections_list:
        collection_names_in_module.add(tmp_module_name)
        if table_names is None or tmp_module_name in table_names.keys():
            filtered_collections_list.append((tmp_module_path, tmp_module_name, module))

    collections_list = filtered_collections_list
    # maybe user input shared module name wrong ,here raise a warning tell user that model don't have the module name
    if table_names is not None:
        for tmp_input_collection_name in table_names.keys():
            if tmp_input_collection_name not in collection_names_in_module:
                warnings.warn(
                    f"sharded module '{tmp_input_collection_name}' specified in table_names not found in the model",
                    UserWarning,
                )

    # create sub folder
    latest_dir = ""
    try:
        for tmp_collection in collections_list:
            collection_path, _, tmp_collection_module = tmp_collection
            full_collection_path = os.path.join(path, collection_path)
            latest_dir = full_collection_path
            if not os.path.exists(full_collection_path):
                os.makedirs(full_collection_path, exist_ok=True)
                created_dirs.append(full_collection_path)
    except Exception as e:
        for dir_path in reversed(created_dirs):
            try:
                os.rmdir(dir_path)
            except OSError:
                pass
        raise Exception(f"can't build path: {latest_dir}") from e

    # start dump from different embedding_collection
    # TODO:In practice, this approach is not ideal because the order of embedding collections
    # might differ across each process, making it impossible to dump them together.
    # We may need an alternative approach.

    # Get the rank of the current process
    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)

    for i, tmp_collection in enumerate(collections_list):
        collection_path, tmp_collection_name, tmp_collection_module = tmp_collection
        full_collection_path = os.path.join(path, collection_path)
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)

        for j, dynamic_emb_module in enumerate(tmp_dynamic_emb_module_list):
            tmp_table_names = dynamic_emb_module.table_names
            tmp_tables = dynamic_emb_module.tables

            filtered_table_names: List[str] = []
            filtered_dynamic_tables: List[DynamicEmbTable] = []
            # TODO:need a warning
            if table_names is not None:
                tmp_input_names = table_names[tmp_collection_name]
                for name in tmp_input_names:
                    if name in tmp_table_names:
                        index = tmp_table_names.index(name)
                        filtered_table_names.append(tmp_table_names[index])
                        filtered_dynamic_tables.append(tmp_tables[index])
            else:
                filtered_table_names = tmp_table_names
                filtered_dynamic_tables = tmp_tables
            if len(filtered_table_names) == 0:
                continue

            if optim:
                optimizer = dynamic_emb_module.optimizer
                opt_args = optimizer.get_opt_args()

            tmp_tables_dict: Dict[str, DynamicEmbTable] = {
                name: table
                for name, table in zip(filtered_table_names, filtered_dynamic_tables)
            }

            ordered_keys = tmp_tables_dict.keys()
            ordered_keys_str = ",".join(ordered_keys)
            length = _calculate_pad_length(ordered_keys_str)

            if rank != 0:
                ordered_keys_str = ""

            ordered_keys_str = broadcast_string(
                ordered_keys_str, length=length, rank=rank, pg=pg
            )
            ordered_keys = ordered_keys_str.split(",")

            for k, dump_name in enumerate(ordered_keys):
                dynamic_table = tmp_tables_dict[dump_name]
                gather_and_export(
                    dynamic_table, full_collection_path, dump_name, pg=pg, optim=optim
                )

                if optim:
                    args_filename = dump_name + "_opt_args.json"
                    args_path = os.path.join(full_collection_path, args_filename)
                    save_to_json(opt_args, args_path)
                if rank == 0:
                    print(
                        f"DynamicEmb dump table {dump_name} from module {tmp_collection_name} success!"
                    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if debug_mode:
        debug_dump(collections_list, path, table_names, optim, pg)

    # add this barrier to guarantee they finish dump at the same time.
    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    return


# TODO: Now only support Row-Wise sharding, will support TW/TWRW in future.
# TODO: Currently, the dump and load functions enforce dumping and loading of all parameters of the optimizer.
#      This mechanism prevents users from controlling certain parameters of the optimizer.
#      In order to allow users to set optimizer parameters more flexibly in the future,
#      we need to add functionality for dumping and loading specific args,
#      allowing for more flexible configuration of optimizer arguments
def DynamicEmbLoad(
    path: str,
    model: nn.Module,
    table_names: Optional[List[str]] = None,
    optim: bool = False,
    pg: Optional[dist.ProcessGroup] = None,
):
    """
    Load the distributed weights and corresponding optimizer states of dynamic embedding tables from the filesystem into the model.

    Each dynamic embedding table will be stored as a key binary file and a value binary file, where the dtype of the key is int64_t,
    and the dtype of the value is float. Each optimizer state is also treated as a dynamic embedding table.

    Parameters
    ----------
    path : str
        The main folder for weight files.
    model : nn.Module
        The model containing dynamic embedding tables.
    table_names : Optional[Dict[str, List[str]]], optional
        A dictionary specifying which embedding collection and which table to load. The key is the name of the embedding collection,
        and the value is a list of dynamic embedding table names within that collection. Defaults to None.
    optim : bool, optional
        Whether to load the optimizer states. Defaults to False.
    pg : Optional[dist.ProcessGroup], optional
        The process group used to control the communication scope in the load. Defaults to None.

    Returns
    -------
    None
    """
    # Due to the difficulty of checking the HKV dump and load operations externally,
    # debug_mode has been added to the code. This is controlled by the environment variable
    # DYNAMICEMB_DUMP_LOAD_DEBUG. When DYNAMICEMB_DUMP_LOAD_DEBUG=1, debug mode is enabled.
    # In debug mode, the state of dynamic emb will be saved before dumping to the file system,
    # and a comparison will be performed after the load operation.

    debug_env_var = os.getenv("DYNAMICEMB_DUMP_LOAD_DEBUG")
    if debug_env_var == "1":
        debug_mode = True
        print("DynamicEmb's dump and load is in debug mode")
    else:
        debug_mode = False

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if not os.path.exists(path):
        raise Exception("can't find path to load, path:", path)

    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        warnings.warn(
            "Input model don't have any TorchREC ShardedEmbeddingCollection or ShardedEmbeddingBagCollection module, can't load any embedding tables from filesystem!",
            UserWarning,
        )
        return

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
            "Input model don't have any Dynamic embedding tables, can't load any embedding tables from filesystem!",
            UserWarning,
        )
        return

    collection_names_in_module = set()
    filtered_collections_list = []

    for tmp_shared_path, tmp_shared_name, module in collections_list:
        collection_names_in_module.add(tmp_shared_name)
        if table_names is None or tmp_shared_name in table_names:
            filtered_collections_list.append((tmp_shared_path, tmp_shared_name, module))

    collections_list = filtered_collections_list

    if table_names is not None:
        for tmp_input_collection_name in table_names.keys():
            if tmp_input_collection_name not in collection_names_in_module:
                warnings.warn(
                    f"sharded module '{tmp_input_collection_name}' specified in table_names not found in the model",
                    UserWarning,
                )

    rank = dist.get_rank(group=pg)

    for i, tmp_collection in enumerate(collections_list):
        # get collection mpdule
        collection_path, tmp_collection_name, tmp_collection_module = tmp_collection
        full_collection_path = os.path.join(path, collection_path)
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)

        for j, dynamic_emb_module in enumerate(tmp_dynamic_emb_module_list):
            tmp_table_names = dynamic_emb_module.table_names
            tmp_tables = dynamic_emb_module.tables

            filtered_table_names: List[str] = []
            filtered_dynamic_tables: List[DynamicEmbTable] = []

            # TODO:need a warning
            if table_names is not None:
                tmp_input_names = table_names[tmp_collection_name]
                for name in tmp_input_names:
                    if name in tmp_table_names:
                        index = tmp_table_names.index(name)
                        filtered_table_names.append(tmp_table_names[index])
                        filtered_dynamic_tables.append(tmp_tables[index])
            else:
                filtered_table_names = tmp_table_names
                filtered_dynamic_tables = tmp_tables

            if len(filtered_table_names) == 0:
                continue

            if optim:
                optimizer = dynamic_emb_module.optimizer

            tmp_tables_dict: Dict[str, DynamicEmbTable] = {
                name: table
                for name, table in zip(filtered_table_names, filtered_dynamic_tables)
            }

            for k, load_name in enumerate(filtered_table_names):
                # load optimizer args firstly then can check if has already dumped the optimizer states.
                if optim:
                    args_filename = load_name + "_opt_args.json"
                    args_path = os.path.join(full_collection_path, args_filename)
                    opt_args = load_from_json(args_path)
                    # TODO: A single set of optimizer arguments is sufficient for a dynamic module set.
                    optimizer.set_opt_args(opt_args)

                load_table(
                    filtered_dynamic_tables[k],
                    full_collection_path,
                    load_name,
                    pg=pg,
                    debug_mode=debug_mode,
                    optim=optim,
                )

                if rank == 0:
                    print(
                        f"DynamicEmb load table {load_name} from module {tmp_collection_name} success!"
                    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if debug_mode:
        debug_load(collections_list, path, table_names, optim, pg)
    return
