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

import logging
import os
import random
import shutil
from typing import Dict

import click
import torch
import torch.distributed as dist
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScoredHashTable,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb_extensions import ScorePolicy
from ordered_set import OrderedSet


def get_score_policy(score_policy: str) -> ScorePolicy:
    if score_policy == "assign":
        return ScorePolicy.ASSIGN
    elif score_policy == "accumulate":
        return ScorePolicy.ACCUMULATE
    elif score_policy == "global_timer":
        return ScorePolicy.GLOBAL_TIMER
    else:
        raise ValueError(f"Invalid score policy: {score_policy}")


score_step = 0


def get_scores(score_policy, keys):
    batch = keys.numel()
    device = keys.device

    global score_step

    score_step += 1

    if score_policy == ScorePolicy.ASSIGN:
        return torch.empty(batch, dtype=torch.uint64, device=device).fill_(score_step)
    elif score_policy == ScorePolicy.ACCUMULATE:
        return torch.ones(batch, dtype=torch.uint64, device=device)
    else:
        return torch.zeros(batch, dtype=torch.uint64, device=device)


def update_scores(
    score_policy: ScorePolicy,
    expect_scores: Dict[int, int],
    key: int,
    step: int,
):
    if score_policy == ScorePolicy.ASSIGN:
        expect_scores[key] = step
    elif score_policy == ScorePolicy.ACCUMULATE:
        if key not in expect_scores:
            expect_scores[key] = 1
        else:
            expect_scores[key] = expect_scores[key] + 1
    else:
        return  # not to check GLOBAL_TIMER by expect_scores


def generate_sparse_feature(
    capacity: int,
    rank: int,
    world_size: int,
    batch_size: int,
    num_iterations: int,
    score_policy: ScorePolicy,
    expect_scores: Dict[int, int],
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size // world_size
    local_keys = []
    global_keys = []

    step = 0
    for _ in range(num_iterations):
        result = OrderedSet()
        while len(result) < batch_size:
            result.add(random.randint(0, (1 << 63) - 1))

        global_indices = list(result)
        # input distributor
        local_indices = []

        step += 1
        for index in global_indices:
            update_scores(score_policy, expect_scores, index, step)

            if index % world_size == rank:
                local_indices.append(index)

        local_keys.append(torch.tensor(local_indices, dtype=torch.int64).cuda())
        global_keys.append(torch.tensor(global_indices, dtype=torch.int64).cuda())
    return local_keys, global_keys


def distributed_dump_table(
    table: ScoredHashTable,
    score_name: str,
    path: str,
    rank: int,
    world_size: int,
    pg: dist.ProcessGroup = dist.group.WORLD,
):
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
                logging.warning(f"Overwriting existing files in {path}")
        else:
            raise Exception(f"The path '{path}' exists and is not a directory.")
    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    key_path = os.path.join(path, f"keys.rank_{rank}.world_size_{world_size}")
    score_path = os.path.join(path, f"scores.rank_{rank}.world_size_{world_size}")

    table.dump(key_path, {score_name: score_path})

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])


def distributed_load_table(
    table: ScoredHashTable,
    score_name: str,
    path: str,
    rank: int,
    world_size: int,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if not os.path.exists(path):
        raise Exception("can't find path to load, path:", path)

    import glob

    key_files = glob.glob(os.path.join(path, "keys*"))
    score_files = glob.glob(os.path.join(path, "scores*"))

    key_files = sorted(key_files)
    score_files = sorted(score_files)

    for key_file, score_file in zip(key_files, score_files):
        table.load(key_file, {score_name: score_file})

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@click.command()
@click.option("--capacity", type=int, required=True)
@click.option("--save-path", type=str, required=True)
@click.option("--mode", type=click.Choice(["load", "dump"]), required=True)
@click.option(
    "--score_policy",
    type=click.Choice(["assign", "accumulate", "global_timer"]),
    required=True,
)
def test_table_load_dump(
    capacity: int,
    save_path: str,
    mode: str,
    score_policy: str,
    batch_size: int = 128,
    num_iterations: int = 10,
):
    if batch_size * num_iterations > capacity:
        raise ValueError(
            "batch_size * num_iterations > capacity, this may lead to eviction of table and cause test fail."
        )

    score_policy_ = get_score_policy(score_policy)
    ref_table = get_scored_table(
        capacity=capacity,
        bucket_capacity=capacity,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name="score1", policy=score_policy_)],
    )

    expect_scores: Dict[int, int] = {}
    local_keys, global_keys = generate_sparse_feature(
        capacity=capacity,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        batch_size=batch_size,
        num_iterations=num_iterations,
        score_policy=score_policy_,
        expect_scores=expect_scores,
    )

    pg: dist.ProcessGroup = dist.group.WORLD
    for keys in local_keys:
        dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])
        score_args = [
            ScoreArg(
                name="score1", value=get_scores(score_policy_, keys), is_return=True
            )
        ]
        ref_table.insert(keys, score_args)

    if mode == "dump":
        shutil.rmtree(save_path, ignore_errors=True)
        distributed_dump_table(
            ref_table,
            "score1",
            save_path,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )

    if mode == "load":
        table = get_scored_table(
            capacity=capacity,
            bucket_capacity=capacity,
            key_type=torch.int64,
            score_specs=[ScoreSpec(name="score1", policy=score_policy_)],
        )

        world_size = dist.get_world_size()
        rank = dist.get_world_size()
        distributed_load_table(
            table,
            "score1",
            save_path,
            rank=rank,
            world_size=world_size,
        )

        table_scores = {}
        for batched_key, named_batched_score in table._batched_export_keys_scores(
            ["score1"], torch.device(f"cpu")
        ):
            for key, score in zip(
                batched_key.tolist(), named_batched_score["score1"].tolist()
            ):
                table_scores[key] = score

        if (
            score_policy_ == ScorePolicy.ASSIGN
            or score_policy_ == ScorePolicy.ACCUMULATE
        ):
            for keys in global_keys:
                keys = keys.tolist()
                for key in keys:
                    if key % world_size == rank:
                        assert (
                            key in table_scores
                        ), f"Key {key} must exist in table of rank {rank}."
                        assert table_scores[key] == expect_scores[key]

        elif score_policy_ == ScorePolicy.GLOBAL_TIMER:
            visited_keys = set({})
            min_score = float("inf")
            lasted_min_score = float("inf")
            for keys in reversed(global_keys):
                keys = keys.tolist()
                for key in keys:
                    if key % world_size == rank:
                        assert (
                            key in table_scores
                        ), f"Key {key} must exist in table of rank {rank}."
                    else:
                        continue

                    if key not in visited_keys:
                        assert (
                            table_scores[key] < min_score
                        ), f"key {key} score {table_scores[key]} should be < min_score {min_score}"
                        lasted_min_score = min(lasted_min_score, table_scores[key])
                        visited_keys.add(key)

                min_score = lasted_min_score
                lasted_min_score = min_score

        else:
            raise RuntimeError("Not supported score policy.")


if __name__ == "__main__":
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(backend="nccl")
    test_table_load_dump()
    dist.destroy_process_group()
