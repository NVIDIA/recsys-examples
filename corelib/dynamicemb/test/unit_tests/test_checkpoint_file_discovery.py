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

import os

import pytest
from dynamicemb.batched_dynamicemb_tables import (
    encode_checkpoint_file_path,
    find_files,
    get_loading_files,
    list_checkpoint_files,
)

WORLD_SIZE = 2


def _write_table(
    root_path: str, table_name: str, items=("keys", "values", "scores")
) -> None:
    for item in items:
        for rank in range(WORLD_SIZE):
            path = encode_checkpoint_file_path(
                root_path, table_name, rank, WORLD_SIZE, item
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"\0")


@pytest.mark.parametrize(
    "table_name", ["flat_table", "scope/scoped_table", "a/b/deep_table"]
)
def test_find_files_finds_every_shard_of_a_table(tmp_path, table_name):
    root_path = str(tmp_path)
    _write_table(root_path, table_name)
    listing = list_checkpoint_files(root_path, [table_name])

    files, num_files = find_files(root_path, table_name, "emb_keys", listing)

    assert num_files == WORLD_SIZE
    assert files == sorted(
        encode_checkpoint_file_path(root_path, table_name, rank, WORLD_SIZE, "keys")
        for rank in range(WORLD_SIZE)
    )
    assert all(os.path.exists(path) for path in files)


def test_find_files_keeps_scoped_and_flat_tables_apart(tmp_path):
    root_path = str(tmp_path)
    _write_table(root_path, "table")
    _write_table(root_path, "scope/table")
    listing = list_checkpoint_files(root_path, ["table", "scope/table"])

    flat, _ = find_files(root_path, "table", "emb_keys", listing)
    scoped, _ = find_files(root_path, "scope/table", "emb_keys", listing)

    assert all(os.sep + "scope" + os.sep not in path for path in flat)
    assert all(os.sep + "scope" + os.sep in path for path in scoped)


def test_find_files_returns_nothing_for_a_missing_scoped_table(tmp_path):
    root_path = str(tmp_path)
    _write_table(root_path, "table")

    files, num_files = find_files(
        root_path,
        "missing_scope/table",
        "emb_keys",
        list_checkpoint_files(root_path, ["missing_scope/table"]),
    )

    assert files == [] and num_files == 0


def test_get_loading_files_resolves_a_scoped_table(tmp_path):
    root_path = str(tmp_path)
    _write_table(root_path, "scope/table")

    (
        key_files,
        value_files,
        score_files,
        opt_files,
        counter_keys,
        counter_freqs,
    ) = get_loading_files(
        root_path,
        "scope/table",
        rank=1,
        world_size=WORLD_SIZE,
        sorted_file_names=list_checkpoint_files(root_path, ["scope/table"]),
    )

    assert key_files == [
        encode_checkpoint_file_path(root_path, "scope/table", 1, WORLD_SIZE, "keys")
    ]
    assert value_files == [
        encode_checkpoint_file_path(root_path, "scope/table", 1, WORLD_SIZE, "values")
    ]
    assert score_files == [
        encode_checkpoint_file_path(root_path, "scope/table", 1, WORLD_SIZE, "scores")
    ]
    assert opt_files == [] and counter_keys == [] and counter_freqs == []
    assert all(os.path.exists(path) for path in key_files + value_files + score_files)


def test_list_checkpoint_files_lists_each_directory_once(tmp_path, monkeypatch):
    root_path = str(tmp_path)
    table_names = ["flat"] + [f"scope/table_{i}" for i in range(8)]
    for table_name in table_names:
        _write_table(root_path, table_name)

    listed = []
    listdir = os.listdir

    def counting_listdir(path):
        listed.append(path)
        return listdir(path)

    monkeypatch.setattr(os, "listdir", counting_listdir)
    listing = list_checkpoint_files(root_path, table_names)

    assert sorted(listed) == sorted([root_path, os.path.join(root_path, "scope")])
    assert listing == sorted(listing)


def test_lookups_do_not_list_the_checkpoint_directory(tmp_path, monkeypatch):
    root_path = str(tmp_path)
    table_names = ["flat", "scope/table", "a/b/deep_table"]
    for table_name in table_names:
        _write_table(root_path, table_name)
    listing = list_checkpoint_files(root_path, table_names)

    def no_listdir(path):
        raise AssertionError(f"listed {path} during a lookup")

    monkeypatch.setattr(os, "listdir", no_listdir)
    for table_name in table_names:
        files, num_files = find_files(root_path, table_name, "emb_keys", listing)
        assert num_files == WORLD_SIZE
        key_files, *_ = get_loading_files(
            root_path,
            table_name,
            rank=0,
            world_size=WORLD_SIZE,
            sorted_file_names=listing,
        )
        assert key_files == [
            encode_checkpoint_file_path(root_path, table_name, 0, WORLD_SIZE, "keys")
        ]
