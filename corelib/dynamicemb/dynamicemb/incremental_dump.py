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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module
from torch import nn


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
) -> Union[
    Tuple[
        Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        Dict[str, Dict[str, int]],
    ],
    None,
]:
    """Dump the model's embedding tables incrementally based on the score threshold. The index-embedding pair whose score is not less than the threshold will be returned.

    Args:
        model(nn.Module):The model containing dynamic embedding tables.
         score_threshold(Uinon[int, Dict[str, Dict[str, int]]]):
            int: All embedding table's score threshold will be this integer. It will dump matched results for all tables in the model.
            Dict[str, Dict[str, int]]: the first `str` is the name of embedding collection in the model. 'str' in Dict[str, int] is the name of dynamic embedding table, and `int` in Dict[str, int] is the table's score threshold. It will dump for only tables whose names present in this Dict.
        pg(Optional[dist.ProcessGroup]): optional. The process group used to control the communication scope in the dump. Defaults to None.

    Returns
    -------
    Tuple:
        Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
            The first 'str' is the name of embedding collection.
            The second 'str' is the name of embedding table.
            The first tensor in the Tuple is matched keys on hosts.
            The second tensor in the Tuple is matched values on hosts.
        Dict[str, Dict[str, int]]:
            The first 'str' is the name of embedding collection.
            The second 'str' is the name of embedding table.
            `int` is the current score after finishing the dumping process, which will be used as the score for the next forward pass, and can also be used as the input of the next incremental_dump. If input score_threshold is `int`, the Dict will contain all dynamic embedding tables' current score, otherwise only dumped tables' current score will be returned.
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
            "Input model don't have any Dynamic embedding tables, can't incremental dump!",
            UserWarning,
        )
        return
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

    ret_tensors: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    ret_scores: Dict[str, Dict[str, int]] = {}
    for i, tmp_collection in enumerate(collections_list):
        collection_path, tmp_collection_name, tmp_collection_module = tmp_collection
        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)

        collection_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        collection_scores: Dict[str, int] = {}

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
            # do incremental dump
            tensors, scores = dynamic_emb_module.incremental_dump(
                dict(zip(filtered_table_names, filtered_thresholds)), pg
            )
            collection_tensors.update(tensors)
            collection_scores.update(scores)

        ret_tensors[collection_path] = collection_tensors
        ret_scores[collection_path] = collection_scores

    return ret_tensors, ret_scores
