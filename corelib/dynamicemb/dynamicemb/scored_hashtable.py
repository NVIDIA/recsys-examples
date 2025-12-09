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

import abc
import enum
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from dynamicemb.dynamicemb_config import dtype_to_bytes
from dynamicemb.types import (
    COUNTER_TYPE,
    KEY_TYPE,
    SCORE_TYPE,
    MemoryType,
    torch_dtype_to_np_dtype,
)
from dynamicemb_extensions import (
    InsertResult,
    ScorePolicy,
    device_timestamp,
    table_count_matched,
    table_erase,
    table_export_batch,
    table_insert,
    table_insert_and_evict,
    table_lookup,
    table_partition,
)


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    policy: ScorePolicy  # How to set the new score, this is the default behavior.
    dtype: torch.dtype = torch.uint64
    priority: int = 0  # If multiple scores exist, the one with lower priority will be reduced first.
    is_reduction: bool = True  # Whether it is reduced


@dataclass
class ScoreArg:
    name: str
    value: Optional[torch.Tensor] = None
    is_return: bool = (
        False  # Whether return the new score, if true will overwrite the `value`
    )
    policy: Optional[
        ScorePolicy
    ] = None  # How to set the new score, and providing this will override the default.


@enum.unique
class ProbingType(enum.Enum):
    LINEAR = "linear"
    CHAINED = "separate_chain"


@enum.unique
class ReductionType(enum.Enum):
    LINEAR = "linear"
    DOUBLY_LINKED = "doubly_linked"


class ScoredHashTable(abc.ABC):
    """
    Multiple scores are supported.
    If a hash collision cannot be resolved during insertion, the key with the lower score will be evicted.
    The value of the table is the index/ID of each key in the table， which is read-only.
    """

    @property
    @abc.abstractmethod
    def key_type(self) -> torch.dtype:
        """
        Return the key type.
        """

    @property
    def index_type(self) -> torch.dtype:
        """
        Return the index type.
        """
        return torch.int64

    @property
    @abc.abstractmethod
    def score_specs(
        self,
        score_names: List[str] = None,
    ) -> List[ScoreSpec]:
        """
        Return the score specifics.
        """

    @property
    def result_type(self) -> torch.dtype:
        """
        Return the insert-result type.
        """
        return torch.uint8

    @abc.abstractmethod
    def lookup(
        self,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        founds: Optional[torch.Tensor],
        indices: torch.Tensor = None,
    ) -> None:
        """
        TODO: kernel fusion
        Argument::
            missing_keys: torch.Tensor=None
            missing_indices: torch.Tensor=None
            missing_scores: List[ScoreArg]=None
        Returns:
            num_missing: int
        """

    @abc.abstractmethod
    def insert(
        self,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        indices: Optional[torch.Tensor] = None,
        insert_results: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Keys have to be unique.
        Indices is output buffer if provided.
        """

    @abc.abstractmethod
    def insert_and_evict(
        self,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        indices: Optional[torch.Tensor] = None,
        insert_results: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Keys have to be unique.
        Indices is output buffer if provided.
        """

        num_evicted: int
        evicted_keys: torch.Tensor
        evicted_indices: torch.Tensor
        evicted_scores: List[torch.Tensor]
        return num_evicted, evicted_keys, evicted_indices, evicted_scores

    @abc.abstractmethod
    def erase(
        self,
        keys: torch.Tensor,
    ) -> None:
        """
        Erase Keys
        """

    @abc.abstractmethod
    def load(
        self,
        key_file: str,
        score_files: Dict[str, str],
    ) -> None:
        """
        Load keys and scores from input file path.

        Args:
            key_file (str): the file path of keys.
            score_files: Dict[str, str]: Dict from score name to score file path.
        """

    @abc.abstractmethod
    def dump(
        self,
        key_file: str,
        score_files: Dict[str, str],
    ) -> None:
        """
        Dump keys and scores to output file path.

        Args:
            key_file (str): the file path of keys.
            score_files: Dict[str, str]: Dict from score name to score file path.
        """

    @abc.abstractmethod
    def incremental_dump(
        self,
        score_threshold: Dict[str, int],
        batch_size: int = 65536,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Dump incremental keys and scores into cpu tensors.

        Args:
            score_threshold (Dict[str, int]): input threshold of each score.
            batch_size (int): the batch size when scan the table.
            pg (Optional[dist.ProcessGroup]): process group.

        Returns:
            out_key (torch.Tensor): output tensor of keys
            out_scores (Dict[str, torch.Tensor]): output tensors of scores.
        """

    @abc.abstractmethod
    def capacity(self) -> int:
        """
        Return the capacity of the table.
        """

    @abc.abstractmethod
    def size(self) -> int:
        """
        Return the size of the table.
        """

    @abc.abstractmethod
    def load_factor(self) -> float:
        """
        Return the load factor of the table.
        """

    @abc.abstractmethod
    def reserve(
        self,
        target_capacity,
    ):
        """
        Table's growth is controlled outside.
        """

    @abc.abstractmethod
    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """


class GroupedScoredHashTable(abc.ABC):
    """
    Multiple scores are supported.
    If a hash collision cannot be resolved during insertion, the key with the lower score will be evicted.
    The value of the table is the index/ID of each key in the table， which is read-only.

    key_type, index_type, offset_type, score_specs, result_type are the same for tables in the same group.
    """

    @property
    @abc.abstractmethod
    def key_type(self) -> torch.dtype:
        """
        Return the key type.
        """

    @property
    def index_type(self) -> torch.dtype:
        """
        Return the index type.
        """
        return torch.int64

    @property
    @abc.abstractmethod
    def score_specs(
        self,
        score_names: List[str] = None,
    ) -> List[ScoreSpec]:
        """
        Return the score specifics.
        """

    @property
    def result_type(self) -> torch.dtype:
        """
        Return the insert-result type.
        """
        return torch.uint8

    @property
    def offset_type(self) -> torch.dtype:
        """
        Return the offset type, used for e.g. table range.
        """
        return torch.int64

    @property
    @abc.abstractmethod
    def table_names(
        self,
        table_names: List[str] = None,
    ) -> List[str]:
        """
        Return the table names in the group.
        """

    @abc.abstractmethod
    def lookup(
        self,
        table_range: torch.Tensor,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        founds: Optional[torch.Tensor],
        indices: torch.Tensor = None,
    ) -> None:
        """
        TODO: kernel fusion
        Argument:
            missing_table_range: torch.Tensor
            missing_keys: torch.Tensor=None
            missing_indices: torch.Tensor=None
            missing_scores: List[ScoreArg]=None
        Returns:
            num_missing: int
        """

    @abc.abstractmethod
    def insert(
        self,
        table_range: torch.Tensor,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        indices: Optional[torch.Tensor] = None,
        insert_results: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Keys have to be unique.
        Indices is output buffer if provided.
        """

    @abc.abstractmethod
    def insert_and_evict(
        self,
        table_range: torch.Tensor,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        indices: Optional[torch.Tensor] = None,
        insert_results: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Keys have to be unique.
        Indices is output buffer if provided.
        """

        num_evicted: int
        missing_table_range: torch.Tensor
        evicted_keys: torch.Tensor
        evicted_indices: torch.Tensor
        evicted_scores: List[torch.Tensor]
        return (
            num_evicted,
            missing_table_range,
            evicted_keys,
            evicted_indices,
            evicted_scores,
        )

    @abc.abstractmethod
    def erase(
        self,
        table_range: torch.Tensor,
        keys: torch.Tensor,
    ) -> None:
        """
        Erase Keys.
        """

    @abc.abstractmethod
    def load(
        self,
        table_names: List[str],
        key_files: List[str],
        score_files: List[Dict[str, str]],
    ) -> None:
        """
        Load keys and scores from input file path.

        Args:
            table_names: List[str]
            key_files: List[str],
            score_files: List[Dict[str, str]]: Dict from score name to score file path.
        """

    @abc.abstractmethod
    def dump(
        self,
        table_names: List[str],
        key_files: List[str],
        score_files: List[Dict[str, str]],
    ) -> None:
        """
        Dump keys and scores to output file path.

        Args:
            table_names: List[str]
            key_files: List[str],
            score_files: List[Dict[str, str]]: Dict from score name to score file path.
        """

    @abc.abstractmethod
    def incremental_dump(
        self,
        table_names: List[str],
        score_threshold: List[Dict[str, int]],
        batch_size: int = 65536,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Dump incremental keys and scores into cpu tensors.

        Args:
            table_names (List[str]): table names.
            score_threshold (List[Dict[str, int]]): input threshold of each score for tables.
            batch_size (int): the batch size when scan the table.
            pg (Optional[dist.ProcessGroup]): process group.

        Returns:
            out_keys (List[torch.Tensor]): output tensor of keys for tables.
            out_scores (List[Dict[str, torch.Tensor]]): output tensors of scores for tables.
        """

    @abc.abstractmethod
    def capacity(self, table_name: str) -> int:
        """
        Return the capacity of the table.
        """

    @abc.abstractmethod
    def size(self, table_name: str) -> int:
        """
        Return the size of the table.
        """

    @abc.abstractmethod
    def load_factor(self, table_name: str) -> float:
        """
        Return the load factor of the table.
        """

    @abc.abstractmethod
    def reserve(
        self,
        table_name: str,
        target_capacity: int,
    ):
        """
        Table's growth is controlled outside.
        """

    @abc.abstractmethod
    def memory_usage(self, table_name: str, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
            table_name: str,
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """


def uint64_to_int64(x):
    return x if x < (1 << 63) else x - (1 << 64)


def murmur3_hash_64bits(key: int) -> int:
    """ """
    k = key & 0xFFFFFFFFFFFFFFFF

    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF

    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF

    k ^= k >> 33

    return k


class LinearBucketTable(ScoredHashTable):
    def __init__(
        self,
        capacity: int,
        score_specs: List[ScoreSpec],
        key_type: torch.dtype = torch.int64,
        bucket_capacity: Optional[int] = None,
        device: torch.device = None,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda", torch.cuda.current_device())
        )

        # key type
        self.key_type_ = key_type
        accepted_key_types = {torch.int64, torch.uint64}
        assert (
            key_type in accepted_key_types
        ), "Only accept 64 bits integer as key's type."

        # score type
        assert (
            len(score_specs) >= 1 and len(score_specs) <= 1
        ), "Only support at least one and at most one ScoreSpec in this version."
        self.score_specs_ = sorted(
            score_specs, key=lambda x: (not x.is_reduction, x.priority)
        )
        assert self.score_specs_[0].is_reduction is True
        accepted_score_types = {torch.uint64}
        self.score_types_ = []
        self.score_names_ = []
        for score_spec in self.score_specs_:
            assert (
                score_spec.dtype in accepted_score_types
            ), "Only accept 64 bits unsigned integer as score's type."
            self.score_types_.append(score_spec.dtype)
            self.score_names_.append(score_spec.name)

        # digest type
        self.digest_type_ = torch.uint8

        # capacity & bucket capacity
        if bucket_capacity is None:
            bucket_capacity = 128

        assert capacity > 0 and bucket_capacity > 0 and capacity >= bucket_capacity
        max_load_bytes = 16
        digest_load_dim = max_load_bytes // dtype_to_bytes(self.digest_type_)
        if bucket_capacity % digest_load_dim == 0:
            self.bucket_capacity_ = bucket_capacity
        else:
            self.bucket_capacity_ = (
                (bucket_capacity + digest_load_dim - 1) // digest_load_dim
            ) * digest_load_dim
        # self.bucket_capacity_ = _next_power_of_2(self.bucket_capacity_)

        if self.bucket_capacity_ != bucket_capacity:
            warnings.warn(
                f"Bucket capacity is rounded from {bucket_capacity} to {self.bucket_capacity_}.",
                UserWarning,
            )

        # storage
        self.fileds_type_ = [self.key_type_, self.digest_type_] + self.score_types_
        self.fields_byte_ = [dtype_to_bytes(x) for x in self.fileds_type_]

        # variable part when reserve.
        self.num_buckets_ = (
            capacity + self.bucket_capacity_ - 1
        ) // self.bucket_capacity_
        self.capacity_ = self.num_buckets_ * self.bucket_capacity_
        if self.capacity_ != capacity:
            warnings.warn(
                f"Table capacity is rounded from {capacity} to {self.capacity_}.",
                UserWarning,
            )

        self.storage_bytes_ = (
            sum(self.fields_byte_) * self.bucket_capacity_ * self.num_buckets_
        )
        self.table_storage_ = torch.empty(
            self.storage_bytes_, dtype=torch.uint8, device=self.device
        )

        self.keys_, self.digests_, *self.scores_list = table_partition(
            self.table_storage_,
            self.fileds_type_,
            self.bucket_capacity_,
            self.num_buckets_,
        )
        self._init_table(self.keys_, self.scores_list, self.digests_)

        self.bucket_sizes = torch.zeros(
            self.num_buckets_, dtype=torch.int32, device=self.device
        )

    def _init_table(
        self,
        keys,
        scores_list,
        digests,
    ):
        # init keys
        empty_key = 0xFFFFFFFFFFFFFFFF
        if self.key_type_ == torch.int64:
            empty_key = uint64_to_int64(empty_key)
        keys.fill_(empty_key)

        # init scores
        empty_score = 0
        for scores in scores_list:
            scores.fill_(empty_score)

        # init digest
        empty_digest = (murmur3_hash_64bits(empty_key) >> 32) & 0xFF
        digests.fill_(empty_digest)

    @property
    def key_type(self) -> torch.dtype:
        """
        Return the key type.
        """
        return self.key_type_

    @property
    def score_specs(
        self,
        score_names: List[str] = None,
    ) -> List[ScoreSpec]:
        """
        Return the score specifics.
        """
        return self.score_specs_

    def _parse_scores(
        self,
        scores: List[ScoreArg],
    ) -> Tuple[List[torch.Tensor], List[ScorePolicy], List[bool]]:
        scores_ = [None for _ in self.score_names_]
        policies = [ScorePolicy.CONST for _ in self.score_names_]
        is_returns = [False for _ in self.score_names_]

        for score in scores:
            index = self.score_names_.index(score.name)
            if score.is_return:
                assert score.value is not None
            scores_[index] = score.value
            policies[index] = (
                score.policy
                if score.policy is not None
                else self.score_specs_[index].policy
            )
            is_returns[index] = score.is_return

            if score.policy == ScorePolicy.GLOBAL_TIMER:
                assert (
                    self.score_specs_[index].dtype == torch.uint64
                ), "Global timer can only work for torch.uint64"

        return scores_, policies, is_returns

    def lookup(
        self,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        founds: Optional[torch.Tensor],
        indices: torch.Tensor = None,
    ) -> None:
        """
        TODO: kernel fusion
        Argument::
            missing_keys: torch.Tensor=None
            missing_indices: torch.Tensor=None
            missing_scores: List[ScoreArg]=None
        Returns:
            num_missing: int
        """
        scores_, policies, is_returns = self._parse_scores(scores)

        table_lookup(
            self.table_storage_,
            self.fileds_type_,
            self.bucket_capacity_,
            keys,
            scores_,
            policies,
            is_returns,
            founds,
            indices,
        )

    def insert(
        self,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        indices: Optional[torch.Tensor] = None,
        insert_results: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Keys have to be unique.
        Indices is output buffer if provided.
        """

        scores_, policies, is_returns = self._parse_scores(scores)

        table_insert(
            self.table_storage_,
            self.fileds_type_,
            self.bucket_capacity_,
            self.bucket_sizes,
            keys,
            scores_,
            policies,
            is_returns,
            indices,
            insert_results,
        )

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        scores: List[ScoreArg],
        indices: Optional[torch.Tensor] = None,
        insert_results: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Keys have to be unique.
        Indices is output buffer if provided.
        """

        scores_, policies, is_returns = self._parse_scores(scores)

        batch = keys.numel()
        num_evicted = torch.zeros(1, dtype=COUNTER_TYPE, device=keys.device)
        evicted_keys = torch.empty(batch, dtype=self.key_type_, device=keys.device)
        evicted_indices = torch.empty(batch, dtype=self.index_type, device=keys.device)
        evicted_scores_list = [
            torch.empty(batch, dtype=dtype, device=keys.device)
            for dtype in self.score_types_
        ]

        table_insert_and_evict(
            self.table_storage_,
            self.fileds_type_,
            self.bucket_capacity_,
            self.bucket_sizes,
            keys,
            scores_,
            policies,
            is_returns,
            insert_results,
            indices,
            num_evicted,
            evicted_keys,
            evicted_indices,
            evicted_scores_list,
        )

        h_num_evicted = num_evicted.cpu().item()
        return (
            h_num_evicted,
            evicted_keys[:h_num_evicted],
            evicted_indices[:h_num_evicted],
            [evicted_scores[:h_num_evicted] for evicted_scores in evicted_scores_list],
        )

    def erase(
        self,
        keys: torch.Tensor,
    ) -> None:
        """
        Erase Keys
        """
        table_erase(
            self.table_storage_,
            self.fileds_type_,
            self.bucket_capacity_,
            self.bucket_sizes,
            keys,
        )

    def load(
        self,
        key_file: str,
        score_files: Dict[str, str],
    ) -> None:
        """
        Load keys and scores from input file path.

        Args:
            key_file (str): the file path of keys.
            score_files: Dict[str, str]: Dict from score name to score file path.
        """

        for score_name in self.score_names_:
            if score_name not in score_files or not os.path.exists(
                score_files[score_name]
            ):
                print(
                    f"Will not load scores for {score_name}, as not provide the file path or file path not existed."
                )

        fkey = open(key_file, "rb")

        fscores: Dict[str, Any] = {}
        for score_name, score_path in score_files.items():
            if score_name not in self.score_names_:
                print(
                    f"Score name {score_name} not existed, will not load from {score_path}."
                )
            elif os.path.exists(score_path):
                fscores[score_name] = open(score_path, "rb")

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        num_keys = os.path.getsize(key_file) // KEY_TYPE.itemsize

        for score_name in fscores.keys():
            num_scores = os.path.getsize(score_files[score_name]) // SCORE_TYPE.itemsize

            if num_keys != num_scores:
                raise ValueError(
                    f"The number of keys({num_keys}) in {key_file} does not match with number of scores({num_keys}) in {score_files[score_name]}."
                )

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        dump_timestamp = device_timestamp()

        batch_size = 65536
        for start in range(0, num_keys, batch_size):
            num_keys_to_read = min(num_keys - start, batch_size)
            keys_bytes = fkey.read(KEY_TYPE.itemsize * num_keys_to_read)

            score_bytes_dict: Dict[str, Any] = {}
            for score_name in fscores.keys():
                score_bytes_dict[score_name] = fscores[score_name].read(
                    SCORE_TYPE.itemsize * num_keys_to_read
                )

            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )
            scores_dict: Dict[str, torch.Tensor] = {}
            for score_name, score_bytes in score_bytes_dict.items():
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )
                index = self.score_names_.index(score_name)
                if self.score_specs_[index].policy == ScorePolicy.GLOBAL_TIMER:
                    scores = torch.clamp(dump_timestamp - scores, min=0)
                scores_dict[score_name] = scores

            if world_size > 1:
                masks = keys % world_size == rank
                keys = keys[masks]
                for score_name in scores_dict:
                    scores_dict[score_name] = scores_dict[score_name][masks]

            score_args = []
            for score_name, scores in scores_dict.items():
                score_args.append(
                    ScoreArg(name=score_name, value=scores, policy=ScorePolicy.ASSIGN)
                )
            self.insert(keys, score_args)

        fkey.close()
        for name in fscores.keys():
            fscores[name].close()

    def _batched_export_keys_scores(
        self,
        score_names: List[str],
        target_device: torch.device,
        thresholds: Optional[List[int]] = None,
        batch_size: int = 65536,
    ) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        export keys, {score_name: scores}

        Args:
            score_names (List[str]): list of score names
            target_device (torch.device): the device where to put the dumped keys, scores.
            thresholds (Optional[List[int]]): maps to score_names, the threshold to determine whether dump a key or not:
                only dump a key when all its scores which in score_names are not less than thresholds.
                only dump scores for score_names.
            batch_size (int): the batch size when scan the table.

        Returns:
            out_keys (torch.Tensor): output tensor of keys
            out_scores (Dict[str, torch.Tensor]): output tensors of scores.
        """

        search_capacity = self.capacity_

        offset = 0

        device = self.device

        key_dtype = self.key_type_
        score_dtype = torch.uint64

        thresholds_ = None

        if thresholds is not None:
            assert len(score_names) == len(
                thresholds
            ), "Thresholds' length have to consistent with score names."

            thresholds_ = [0 for _ in self.score_names_]

            for score_name, threshold in zip(score_names, thresholds):
                if score_name in self.score_names_:
                    index = self.score_names_.index(score_name)
                    thresholds_[index] = threshold

        while offset < search_capacity:
            batch_ = min(batch_size, search_capacity - offset)

            keys = torch.empty(batch_, dtype=key_dtype, device=device)
            scores_list = []
            for score_name in self.score_names_:
                if score_name in score_names:
                    scores_list.append(
                        torch.zeros(batch_, dtype=score_dtype, device=device)
                    )
                else:
                    scores_list.append(None)
            d_counter = torch.zeros(1, dtype=COUNTER_TYPE, device=device)

            table_export_batch(
                self.table_storage_,
                self.fileds_type_,
                self.bucket_capacity_,
                batch_,
                offset,
                d_counter,
                keys,
                scores_list,
                thresholds_,
            )

            actual_length = d_counter.item()
            if actual_length > 0:
                named_scores: Dict[str, torch.Tensor] = {}
                for score_name in score_names:
                    index = self.score_names_.index(score_name)
                    scores_ = scores_list[index]
                    named_scores[score_name] = (
                        scores_[:actual_length].to(SCORE_TYPE).to(target_device)
                    )

                yield (
                    keys[:actual_length].to(KEY_TYPE).to(target_device),
                    named_scores,
                )
            offset += batch_size

    def dump(
        self,
        key_file: str,
        score_files: Dict[str, str],
    ) -> None:
        """
        Dump keys and scores to output file path.

        Args:
            key_file (str): the file path of keys.
            score_files: Dict[str, str]: Dict from score name to score file path.
        """

        fkey = open(key_file, "wb")
        fscores: Dict[str, Any] = {}
        for score_name, score_path in score_files.items():
            if score_name not in self.score_names_:
                print(
                    f"Score name {score_name} not existed, will not dump to {score_path}."
                )
            else:
                fscores[score_name] = open(score_path, "wb")

        dump_timestamp = device_timestamp()

        for keys, named_scores in self._batched_export_keys_scores(
            fscores.keys(), self.device
        ):
            fkey.write(keys.cpu().numpy().tobytes())
            for name, scores in named_scores.items():
                index = self.score_names_.index(name)
                if self.score_specs_[index].policy == ScorePolicy.GLOBAL_TIMER:
                    scores = dump_timestamp - scores
                fscores[name].write(scores.cpu().numpy().tobytes())

        fkey.close()
        for fscore in fscores.values():
            fscore.close()

        return

    def incremental_dump(
        self,
        score_threshold: Dict[str, int],
        batch_size: int = 65536,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Dump incremental keys and scores into cpu tensors.

        Args:
            score_threshold (Dict[str, int]): input threshold of each score.
            batch_size (int): the batch size when scan the table.
            pg (Optional[dist.ProcessGroup]): process group.

        Returns:
            out_keys (torch.Tensor): output tensor of keys
            out_scores (Dict[str, torch.Tensor]): output tensors of scores.
        """

        out_keys: torch.Tensor
        out_scores: Dict[str, torch.Tensor] = {}

        scores = []
        thresholds = []
        thresholds_total = [0 for _ in self.score_names_]
        for score_name, threshold in score_threshold.items():
            if score_name not in self.score_names_:
                print(f"Score name {score_name} not existed, will not dump it.")
            else:
                scores.append(score_name)
                thresholds.append(threshold)

                out_scores[score_name] = None

                index = self.score_names_.index(score_name)
                thresholds_total[index] = threshold

        d_num_matched = torch.zeros(1, dtype=COUNTER_TYPE, device=self.device)
        table_count_matched(
            self.table_storage_,
            self.fileds_type_,
            self.bucket_capacity_,
            thresholds_total,
            d_num_matched,
        )

        if not dist.is_initialized() or dist.get_world_size(group=pg) == 1:
            total_matched = d_num_matched.cpu().item()

            out_keys = torch.empty(total_matched, dtype=KEY_TYPE, device="cpu")
            for score_name in out_scores.keys():
                out_scores[score_name] = torch.empty(
                    total_matched, dtype=SCORE_TYPE, device="cpu"
                )

            out_offset = 0
            for keys, named_scores in self._batched_export_keys_scores(
                scores, self.device, thresholds, batch_size
            ):
                h_count = keys.numel()
                out_keys[out_offset : out_offset + h_count].copy_(
                    keys, non_blocking=True
                )
                for score_name in out_scores.keys():
                    out_scores[score_name][out_offset : out_offset + h_count].copy_(
                        named_scores[score_name], non_blocking=True
                    )

                out_offset += h_count

            assert (
                total_matched == out_offset
            ), "Dumped keys number mismatched with the expected count."

        else:
            # Get the rank of the current process
            world_size = dist.get_world_size(group=pg)

            gathered_num_matched = [
                torch.tensor(0, dtype=COUNTER_TYPE, device=self.device)
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_num_matched, d_num_matched, group=pg)

            total_matched = sum([t.item() for t in gathered_num_matched])

            out_keys = torch.empty(total_matched, dtype=KEY_TYPE, device="cpu")
            for score_name in out_scores.keys():
                out_scores[score_name] = torch.empty(
                    total_matched, dtype=SCORE_TYPE, device="cpu"
                )

            d_keys = torch.empty(batch_size, dtype=KEY_TYPE, device=self.device)
            d_scores: List[torch.Tensor] = [None for _ in self.score_names_]
            for score_name in out_scores.keys():
                index = self.score_names_.index(score_name)
                d_scores[index] = torch.empty(
                    batch_size, dtype=SCORE_TYPE, device=self.device
                )
            d_count = torch.zeros(1, dtype=COUNTER_TYPE, device=self.device)

            # Gather keys and scores for all ranks
            gathered_keys = [torch.empty_like(d_keys) for _ in range(world_size)]
            gathered_scores: Dict[str, List[torch.Tensor]] = {}

            for score_name in out_scores.keys():
                index = self.score_names_.index(score_name)
                gathered_scores[score_name] = [
                    torch.empty_like(d_scores[index]) for _ in range(world_size)
                ]

            gathered_counts = [
                torch.empty_like(d_count, dtype=COUNTER_TYPE) for _ in range(world_size)
            ]

            out_offset = 0
            search_offset = 0
            search_capacity = self.capacity_

            while search_offset < search_capacity:
                batch_ = min(batch_size, search_capacity - search_offset)
                table_export_batch(
                    self.table_storage_,
                    self.fileds_type_,
                    self.bucket_capacity_,
                    batch_,
                    search_offset,
                    d_count,
                    d_keys,
                    d_scores,
                    thresholds_total,
                )

                dist.all_gather(gathered_keys, d_keys, group=pg)
                for score_name in out_scores.keys():
                    index = self.score_names_.index(score_name)
                    dist.all_gather(
                        gathered_scores[score_name], d_scores[index], group=pg
                    )

                dist.all_gather(gathered_counts, d_count, group=pg)

                for i in range(world_size):
                    d_keys_ = gathered_keys[i]
                    d_count_ = gathered_counts[i]

                    h_count = d_count_.cpu().item()
                    out_keys[out_offset : out_offset + h_count].copy_(
                        d_keys_[:h_count], non_blocking=True
                    )
                    for score_name in out_scores.keys():
                        out_scores[score_name][out_offset : out_offset + h_count].copy_(
                            gathered_scores[score_name][i][:h_count], non_blocking=True
                        )

                    out_offset += h_count

                search_offset += batch_
                d_count.fill_(0)

            assert out_offset == total_matched

        return out_keys, out_scores

    def capacity(self) -> int:
        """
        Return the capacity of the table.
        """
        return self.capacity_

    def size(self) -> int:
        """
        Return the size of the table.
        """
        return self.bucket_sizes.sum()

    def load_factor(self) -> float:
        """
        Return the load factor of the table.
        """
        return self.bucket_sizes.sum() / self.capacity_

    def reserve(
        self,
        target_capacity,
    ):
        """
        Table's growth is controlled outside.
        """

        if target_capacity <= self.capacity_:
            return

        num_buckets = (
            target_capacity + self.bucket_capacity_ - 1
        ) // self.bucket_capacity_
        capacity = num_buckets * self.bucket_capacity_
        if capacity != target_capacity:
            warnings.warn(
                f"Table capacity is rounded from {target_capacity} to {capacity}.",
                UserWarning,
            )

        # Apply for resources
        storage_bytes = sum(self.fields_byte_) * self.bucket_capacity_ * num_buckets
        table_storage = torch.empty(
            storage_bytes, dtype=torch.uint8, device=self.device
        )

        keys, digests, scores_list = table_partition(
            table_storage,
            self.fileds_type_,
            self.bucket_capacity_,
            num_buckets,
        )
        self._init_table(keys, scores_list, digests)

        bucket_sizes = torch.zeros(num_buckets, dtype=torch.int32, device=self.device)

        # move existed data to new table
        for keys_, named_scores in self._batched_export_keys_scores(
            self.score_names_, self.device
        ):
            score_args = []
            for name, scores in named_scores.items():
                score_args.append(
                    ScoreArg(name=name, value=scores, policy=ScorePolicy.ASSIGN)
                )
            scores_, policies, is_returns = self._parse_scores(score_args)

            insert_results = torch.empty(
                keys_.numel(), dtype=self.result_type, device=self.device
            )

            table_insert(
                table_storage,
                self.fileds_type_,
                self.bucket_capacity_,
                bucket_sizes,
                keys_,
                scores_,
                policies,
                is_returns,
                None,
                insert_results,
            )

            evicted_cnt = insert_results.eq(InsertResult.EVICT.value).sum()
            if evicted_cnt != 0:
                warnings.warn(
                    f"There are {evicted_cnt} keys were evicted during reserve, try a larger target capacity."
                )

        # replace and release resources
        self.table_storage_ = table_storage
        self.keys_ = keys
        self.scores_list = scores_list
        self.digests_ = digests
        self.bucket_sizes = bucket_sizes
        self.num_buckets_ = num_buckets
        self.capacity_ = capacity
        self.storage_bytes_ = storage_bytes

    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """
        return (
            self.storage_bytes_
            + self.bucket_sizes.numel() * self.bucket_sizes.element_size()
        )


def get_scored_table(
    capacity: int,
    bucket_capacity: Optional[int] = None,
    key_type: Optional[torch.dtype] = torch.int64,
    score_specs: List[ScoreSpec] = [
        ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    ],
    device: torch.device = None,
    probing_type=ProbingType.LINEAR,
    reduction_type=ReductionType.LINEAR,
    bucket_load_factor=0.5,  # used when probing_type=ProbingType.CHAINED
) -> ScoredHashTable:
    if probing_type == ProbingType.LINEAR and reduction_type == ReductionType.LINEAR:
        return LinearBucketTable(
            capacity,
            score_specs,
            key_type=key_type,
            bucket_capacity=bucket_capacity,
            device=device,
        )
    else:
        raise NotImplementedError


def get_grouped_scored_table(
    capacities: List[int],
    bucket_capacity: Optional[List[int]] = None,
    key_type: Optional[torch.dtype] = torch.int64,
    score_specs: List[ScoreSpec] = [
        ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    ],
    device: torch.device = None,
    probing_type=ProbingType.LINEAR,
    reduction_type=ReductionType.LINEAR,
    bucket_load_factor=0.5,  # used when probing_type=ProbingType.CHAINED
) -> GroupedScoredHashTable:
    raise NotImplementedError
