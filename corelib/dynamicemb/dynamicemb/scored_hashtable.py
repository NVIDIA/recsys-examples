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
from dynamicemb.types import KEY_TYPE, SCORE_TYPE, MemoryType, torch_dtype_to_np_dtype
from dynamicemb_extensions import (
    ScorePolicy,
    bucketize_keys,
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
        table_ids: torch.Tensor,
        score: ScoreArg,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lookup keys in the table.
        Args:
            keys: input keys tensor.
            table_ids: int32 tensor of same length as keys, identifying which logical table each key belongs to.
            score: score argument.
        Returns:
            (score_out, founds, indices)
        """

    @abc.abstractmethod
    def insert(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score: ScoreArg,
        insert_results: Optional[torch.Tensor] = None,
        score_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Keys have to be unique.
        Args:
            table_ids: int32 tensor of same length as keys, identifying which logical table each key belongs to.
        Returns:
            indices
        If score_out is provided (caller-allocated int64 tensor), it is filled with output scores.
        """

    @abc.abstractmethod
    def insert_and_evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score: ScoreArg,
        insert_results: Optional[torch.Tensor] = None,
        score_out: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Keys have to be unique.
        Args:
            table_ids: int32 tensor of same length as keys, identifying which logical table each key belongs to.
        Returns:
            (indices, num_evicted, evicted_keys, evicted_indices, evicted_scores, evicted_table_ids)
        If score_out is provided (caller-allocated int64 tensor), it is filled with output scores.
        """

    @abc.abstractmethod
    def erase(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """
        Erase Keys
        Args:
            table_ids: int32 tensor of same length as keys, identifying which logical table each key belongs to.
        """

    @abc.abstractmethod
    def load(
        self,
        key_file: str,
        score_files: Dict[str, str],
        table_id: Optional[int] = None,
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
        table_id: Optional[int] = None,
    ) -> None:
        """
        Dump keys and scores to output file path.

        Args:
            key_file (str): the file path of keys.
            score_files: Dict[str, str]: Dict from score name to score file path.
            table_id (Optional[int]): if provided, only dump the specified table.
        """

    @abc.abstractmethod
    def incremental_dump(
        self,
        score_threshold: Dict[str, int],
        table_id: int,
        batch_size: int = 65536,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Dump incremental keys and scores into cpu tensors.

        Args:
            score_threshold (Dict[str, int]): input threshold of each score.
            table_id (int): the logical table to dump.
            batch_size (int): the batch size when scan the table.
            pg (Optional[dist.ProcessGroup]): process group.

        Returns:
            out_key (torch.Tensor): output tensor of keys
            out_scores (Dict[str, torch.Tensor]): output tensors of scores.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the table.
        """

    @abc.abstractmethod
    def capacity(self, table_id: Optional[int] = None) -> int:
        """
        Return the capacity of the table, or a specific logical table.
        """

    @abc.abstractmethod
    def size(self, table_id: Optional[int] = None) -> int:
        """
        Return the size of the table, or a specific logical table.
        """

    @abc.abstractmethod
    def load_factor(self) -> float:
        """
        Return the load factor of the table.
        """

    @abc.abstractmethod
    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.

        Args:
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
        capacity: List[int],
        score_specs: List[ScoreSpec],
        key_type: torch.dtype = torch.int64,
        bucket_capacity: Optional[int] = None,
        device: torch.device = None,
    ):
        """
        Args:
            capacity: List of capacities, one per logical table.
            score_specs: List of ScoreSpec (currently only one is supported).
            key_type: key data type, torch.int64 or torch.uint64.
            bucket_capacity: number of slots per bucket (default 128).
            device: CUDA device.
        """
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

        assert (
            isinstance(capacity, list) and len(capacity) >= 1
        ), "capacity must be a non-empty list of ints (one per logical table)."

        max_load_bytes = 16
        digest_load_dim = max_load_bytes // dtype_to_bytes(self.digest_type_)
        if bucket_capacity % digest_load_dim == 0:
            self.bucket_capacity_ = bucket_capacity
        else:
            self.bucket_capacity_ = (
                (bucket_capacity + digest_load_dim - 1) // digest_load_dim
            ) * digest_load_dim

        if self.bucket_capacity_ != bucket_capacity:
            warnings.warn(
                f"Bucket capacity is rounded from {bucket_capacity} to {self.bucket_capacity_}.",
                UserWarning,
            )

        # storage
        self.fileds_type_ = [self.key_type_, self.digest_type_] + self.score_types_
        self.fields_byte_ = [dtype_to_bytes(x) for x in self.fileds_type_]

        # Multi-table: compute per-table bucket counts and offsets
        self.num_tables_ = len(capacity)
        self.per_table_num_buckets_: List[int] = []
        self.per_table_capacity_: List[int] = []
        bucket_offset_list = [0]
        for cap in capacity:
            nb = (cap + self.bucket_capacity_ - 1) // self.bucket_capacity_
            self.per_table_num_buckets_.append(nb)
            self.per_table_capacity_.append(nb * self.bucket_capacity_)
            bucket_offset_list.append(bucket_offset_list[-1] + nb)

        self.num_buckets_ = bucket_offset_list[-1]
        self.capacity_ = self.num_buckets_ * self.bucket_capacity_

        self.table_bucket_offsets_ = torch.tensor(
            bucket_offset_list, dtype=torch.int64, device=self.device
        )
        self.table_bucket_offsets_cpu_ = self.table_bucket_offsets_.cpu()

        total_input_capacity = sum(capacity)
        if self.capacity_ != total_input_capacity:
            warnings.warn(
                f"Table total capacity is rounded from {total_input_capacity} to {self.capacity_}.",
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

    def _parse_score(
        self,
        score: ScoreArg,
    ) -> Tuple[Optional[torch.Tensor], ScorePolicy]:
        index = self.score_names_.index(score.name)
        policy = (
            score.policy
            if score.policy is not None
            else self.score_specs_[index].policy
        )
        if policy == ScorePolicy.GLOBAL_TIMER:
            assert (
                self.score_specs_[index].dtype == torch.uint64
            ), "Global timer can only work for torch.uint64"
        return score.value, policy

    def lookup(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score: ScoreArg,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lookup keys in the table. Score is optional input for Assign/Accumulate policies.
        Args:
            table_ids: int64 tensor of same length as keys, identifying which logical table each key belongs to.
        Returns:
            (score_out, founds, indices): score tensor (int64), found mask, indices.
        """
        score_value, policy = self._parse_score(score)

        score_out, founds, indices = table_lookup(
            self.table_storage_,
            self.table_bucket_offsets_,
            self.bucket_capacity_,
            keys,
            table_ids,
            score_value,
            policy,
        )
        return score_out, founds, indices

    def insert(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score: ScoreArg,
        insert_results: Optional[torch.Tensor] = None,
        score_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Keys have to be unique.
        Args:
            table_ids: int32 tensor of same length as keys, identifying which logical table each key belongs to.
        Returns:
            indices
        If score_out is provided (caller-allocated int64 tensor), it is filled with output scores.
        """
        score_value, policy = self._parse_score(score)

        indices = table_insert(
            self.table_storage_,
            self.table_bucket_offsets_,
            self.bucket_capacity_,
            self.bucket_sizes,
            keys,
            table_ids,
            score_value,
            policy,
            insert_results,
            score_out,
        )
        return indices

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        score: ScoreArg,
        insert_results: Optional[torch.Tensor] = None,
        score_out: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Keys have to be unique.
        Args:
            table_ids: int32 tensor of same length as keys, identifying which logical table each key belongs to.
        Returns:
            (indices, num_evicted, evicted_keys, evicted_indices, evicted_scores, evicted_table_ids)
        If score_out is provided (caller-allocated int64 tensor), it is filled with output scores.
        """

        score_value, policy = self._parse_score(score)

        (
            indices,
            num_evicted,
            evicted_keys,
            evicted_indices,
            evicted_scores,
            evicted_table_ids,
        ) = table_insert_and_evict(
            self.table_storage_,
            self.table_bucket_offsets_,
            self.bucket_capacity_,
            self.bucket_sizes,
            keys,
            table_ids,
            score_value,
            policy,
            insert_results,
            score_out,
        )

        h_num_evicted = num_evicted.cpu().item()
        return (
            indices,
            h_num_evicted,
            evicted_keys[:h_num_evicted],
            evicted_indices[:h_num_evicted],
            evicted_scores[:h_num_evicted],
            evicted_table_ids[:h_num_evicted],
        )

    def erase(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """
        Erase Keys
        Args:
            table_ids: int32 tensor of same length as keys, identifying which logical table each key belongs to.
        """
        table_erase(
            self.table_storage_,
            self.table_bucket_offsets_,
            self.bucket_capacity_,
            self.bucket_sizes,
            keys,
            table_ids,
        )

    def load(
        self,
        key_file: str,
        score_files: Dict[str, str],
        table_id: Optional[int] = None,
    ) -> None:
        """
        Load keys and scores from input file path.

        Args:
            key_file (str): the file path of keys.
            score_files: Dict[str, str]: Dict from score name to score file path.
            table_id (Optional[int]): if provided, load keys into the specified logical table.
                If None, defaults to table 0.
        """

        load_table_id = table_id if table_id is not None else 0

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

            assert len(scores_dict) == 1, "Only single score is supported."
            score_name, scores = next(iter(scores_dict.items()))
            score_arg = ScoreArg(
                name=score_name, value=scores, policy=ScorePolicy.ASSIGN
            )
            tid = torch.full(
                (keys.numel(),), load_table_id, dtype=torch.int64, device=device
            )
            self.insert(keys, tid, score_arg)

        fkey.close()
        for name in fscores.keys():
            fscores[name].close()

    def _batched_export_keys_scores(
        self,
        score_names: List[str],
        target_device: torch.device,
        table_id: int,
        thresholds: Optional[List[int]] = None,
        batch_size: int = 65536,
        return_index: bool = False,
    ) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]]:
        """
        export keys, {score_name: scores}, indices

        Args:
            score_names (List[str]): list of score names
            target_device (torch.device): the device where to put the dumped keys, scores.
            table_id (int): the logical table to export from.
            thresholds (Optional[List[int]]): maps to score_names, the threshold to determine whether dump a key or not:
                only dump a key when all its scores which in score_names are not less than thresholds.
                only dump scores for score_names.
            batch_size (int): the batch size when scan the table.
            return_index (bool) : whether export indices or not, default to False.

        Returns:
            out_keys (torch.Tensor): output tensor of keys
            out_scores (Dict[str, torch.Tensor]): output tensors of scores.
            out_indices (Optional[torch.Tensor]): output tensor of indices
        """

        begin_slot = (
            int(self.table_bucket_offsets_cpu_[table_id].item()) * self.bucket_capacity_
        )
        end_slot = (
            int(self.table_bucket_offsets_cpu_[table_id + 1].item())
            * self.bucket_capacity_
        )
        search_capacity = end_slot - begin_slot

        offset = 0

        key_dtype = self.key_type_

        # With single score, resolve the threshold to a single optional value.
        threshold_: Optional[int] = None
        if thresholds is not None:
            assert len(score_names) == len(
                thresholds
            ), "Thresholds' length have to consistent with score names."
            assert len(self.score_names_) == 1, "Only single score is supported."
            if self.score_names_[0] in score_names:
                idx = score_names.index(self.score_names_[0])
                threshold_ = thresholds[idx]

        while offset < search_capacity:
            batch_ = min(batch_size, search_capacity - offset)

            d_counter, keys, score, indices = table_export_batch(
                self.table_storage_,
                self.bucket_capacity_,
                batch_,
                begin_slot + offset,
                key_dtype,
                threshold_,
                begin_slot,
            )

            actual_length = d_counter.item()
            if actual_length > 0:
                named_scores: Dict[str, torch.Tensor] = {}
                for score_name in score_names:
                    if score_name in self.score_names_:
                        named_scores[score_name] = (
                            score[:actual_length].to(SCORE_TYPE).to(target_device)
                        )

                yield (
                    keys[:actual_length].to(KEY_TYPE).to(target_device),
                    named_scores,
                    indices[:actual_length].to(target_device) if return_index else None,
                )
            offset += batch_size

    def dump(
        self,
        key_file: str,
        score_files: Dict[str, str],
        table_id: Optional[int] = None,
    ) -> None:
        """
        Dump keys and scores to output file path.

        Args:
            key_file (str): the file path of keys.
            score_files: Dict[str, str]: Dict from score name to score file path.
            table_id (Optional[int]): if provided, only dump the specified logical table.
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

        table_ids = [table_id] if table_id is not None else range(self.num_tables_)
        for tid in table_ids:
            for keys, named_scores, _ in self._batched_export_keys_scores(
                fscores.keys(), self.device, tid
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
        table_id: int,
        batch_size: int = 65536,
        pg: Optional[dist.ProcessGroup] = None,
        return_index: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Dump incremental keys and scores into cpu tensors.

        Args:
            score_threshold (Dict[str, int]): input threshold of each score.
            table_id (int): the logical table to dump.
            batch_size (int): the batch size when scan the table.
            pg (Optional[dist.ProcessGroup]): process group.
            return_index: whether return the index or not.

        Returns:
            out_keys (torch.Tensor): output tensor of keys
            out_scores (Dict[str, torch.Tensor]): output tensors of scores.
            out_indices (Optional[torch.Tensor]): output tensors of indices.
        """

        out_keys: torch.Tensor
        out_scores: Dict[str, torch.Tensor] = {}
        out_indices: torch.Tensor

        scores = []
        thresholds = []
        threshold_val: int = 0
        for score_name, threshold in score_threshold.items():
            if score_name not in self.score_names_:
                print(f"Score name {score_name} not existed, will not dump it.")
            else:
                scores.append(score_name)
                thresholds.append(threshold)

                out_scores[score_name] = None

                threshold_val = threshold

        count_begin = (
            int(self.table_bucket_offsets_cpu_[table_id].item()) * self.bucket_capacity_
        )
        count_end = (
            int(self.table_bucket_offsets_cpu_[table_id + 1].item())
            * self.bucket_capacity_
        )

        d_num_matched = table_count_matched(
            self.table_storage_,
            self.key_type_,
            self.bucket_capacity_,
            threshold_val,
            count_begin,
            count_end,
        )

        # if not dist.is_initialized() or dist.get_world_size(group=pg) == 1:
        total_matched = d_num_matched.cpu().item()

        out_keys = torch.empty(total_matched, dtype=KEY_TYPE, device="cpu")
        out_indices = (
            torch.empty(total_matched, dtype=self.index_type, device="cpu")
            if return_index
            else None
        )
        for score_name in out_scores.keys():
            out_scores[score_name] = torch.empty(
                total_matched, dtype=SCORE_TYPE, device="cpu"
            )

        out_offset = 0
        for keys, named_scores, indices in self._batched_export_keys_scores(
            scores,
            self.device,
            table_id,
            thresholds,
            batch_size,
            return_index=return_index,
        ):
            h_count = keys.numel()
            out_keys[out_offset : out_offset + h_count].copy_(keys, non_blocking=True)
            if indices is not None:
                out_indices[out_offset : out_offset + h_count].copy_(
                    indices, non_blocking=True
                )
            for score_name in out_scores.keys():
                out_scores[score_name][out_offset : out_offset + h_count].copy_(
                    named_scores[score_name], non_blocking=True
                )

            out_offset += h_count

        assert (
            total_matched == out_offset
        ), "Dumped keys number mismatched with the expected count."

        return out_keys, out_scores, out_indices

    def reset(self) -> None:
        """
        Reset the table.
        """
        self._init_table(self.keys_, self.scores_list, self.digests_)

    def capacity(self, table_id: Optional[int] = None) -> int:
        """
        Return the capacity of the table, or a specific logical table.
        """
        if table_id is not None:
            return self.per_table_capacity_[table_id]
        return self.capacity_

    def size(self, table_id: Optional[int] = None) -> int:
        """
        Return the size of the table, or a specific logical table.
        """
        if table_id is not None:
            bkt_begin = int(self.table_bucket_offsets_cpu_[table_id].item())
            bkt_end = int(self.table_bucket_offsets_cpu_[table_id + 1].item())
            return self.bucket_sizes[bkt_begin:bkt_end].sum()
        return self.bucket_sizes.sum()

    def load_factor(self) -> float:
        """
        Return the load factor of the table.
        """
        return self.bucket_sizes.sum() / self.capacity_

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

    def bucketize_keys(
        self,
        keys,
        table_ids,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bkt_keys, offsets, inverse = bucketize_keys(
            keys, table_ids, self.table_bucket_offsets_, self.bucket_capacity_
        )
        return bkt_keys, offsets, inverse


def get_scored_table(
    capacity: List[int],
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
