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
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

import numpy as np
import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.dynamicemb_config import *
from dynamicemb.dynamicemb_config import (
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    create_dynamicemb_table,
    dtype_to_bytes,
    dyn_emb_to_torch,
    torch_to_dyn_emb,
)
from dynamicemb.initializer import *
from dynamicemb.initializer import BaseDynamicEmbInitializer
from dynamicemb.optimizer import *
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizerV2
from dynamicemb.scored_hashtable import (
    LinearBucketTable,
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.types import (
    COUNTER_TYPE,
    EMBEDDING_TYPE,
    KEY_TYPE,
    OPT_STATE_TYPE,
    SCORE_TYPE,
    AdmissionStrategy,
    Cache,
    Counter,
    Storage,
    torch_dtype_to_np_dtype,
)
from dynamicemb_extensions import (
    DynamicEmbTable,
    EvictStrategy,
    clear,
    count_matched,
    device_timestamp,
    dyn_emb_capacity,
    dyn_emb_cols,
    dyn_emb_rows,
    erase,
    export_batch,
    export_batch_matched,
    find_pointers,
    find_pointers_with_scores,
    insert_and_evict,
    insert_and_evict_with_scores,
    insert_or_assign,
    load_from_combined_table,
    load_from_pointers,
    select,
    select_index,
    store_to_combined_table,
)
from torch import Tensor, nn  # usort:skip
from torchrec import JaggedTensor


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


def batched_export_keys_values(
    dynamic_table: DynamicEmbTable,
    device: torch.device,
    batch_size: int = 65536,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    export keys, embeddings, opt_states, scores
    """

    search_capacity = dyn_emb_capacity(dynamic_table)

    offset = 0

    while offset < search_capacity:
        key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
        value_dtype = dyn_emb_to_torch(dynamic_table.value_type())
        dim = dyn_emb_cols(dynamic_table)
        optstate_dim = dynamic_table.optstate_dim()
        total_dim = dim + optstate_dim

        cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        keys = torch.empty(batch_size, dtype=key_dtype, device=cuda_device)
        values = torch.empty(
            batch_size * total_dim, dtype=value_dtype, device=cuda_device
        )
        scores = torch.zeros(batch_size, dtype=SCORE_TYPE, device=cuda_device)
        d_counter = torch.zeros(1, dtype=torch.uint64, device=cuda_device)

        export_batch(dynamic_table, batch_size, offset, d_counter, keys, values, scores)

        values = values.reshape(batch_size, total_dim)

        embeddings = values[:, :dim].contiguous()
        opt_states = values[:, dim:].contiguous()

        d_counter = d_counter.to(dtype=torch.int64)
        actual_length = d_counter.item()
        if actual_length > 0:
            yield (
                keys[:actual_length].to(KEY_TYPE).to(device),
                embeddings[:actual_length, :].to(EMBEDDING_TYPE).to(device),
                opt_states[:actual_length, :].to(OPT_STATE_TYPE).to(device),
                scores[:actual_length].to(SCORE_TYPE).to(device),
            )
        offset += batch_size


def load_key_values(
    dynamic_table: DynamicEmbTable,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    opt_states: Optional[torch.Tensor] = None,
):
    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()
    if not keys.is_cuda:
        raise RuntimeError("Keys must be on GPU")
    if not embeddings.is_cuda:
        raise RuntimeError("Embeddings must be on GPU")
    if scores is not None and not scores.is_cuda:
        raise RuntimeError("Scores must be on GPU")
    if opt_states is not None and not opt_states.is_cuda:
        raise RuntimeError("Opt states must be on GPU")

    if opt_states is None and optstate_dim > 0:
        opt_states = (
            torch.ones(
                keys.numel(),
                optstate_dim,
                dtype=dyn_emb_to_torch(dynamic_table.value_type()),
                device=embeddings.device,
            )
            * dynamic_table.get_initial_optstate()
        )

    values = (
        torch.cat([embeddings.view(-1, dim), opt_states.view(-1, optstate_dim)], dim=-1)
        if opt_states is not None
        else embeddings
    )

    key_type = dyn_emb_to_torch(dynamic_table.key_type())
    value_type = dyn_emb_to_torch(dynamic_table.value_type())

    if scores is None:
        assert (
            dynamic_table.evict_strategy() == EvictStrategy.KLru
        ), "scores is None for KLru evict strategy is allowed but will be deprecated in future."
        insert_or_assign(
            dynamic_table, keys.numel(), keys.to(key_type), values.to(value_type)
        )
        return

    insert_or_assign(
        dynamic_table,
        keys.numel(),
        keys.to(key_type),
        values.to(value_type),
        scores.to(SCORE_TYPE),
        unique_key=True,
        ignore_evict_strategy=True,
    )


def get_score_policy(score_strategy):
    if score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
        return ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    elif score_strategy == DynamicEmbScoreStrategy.STEP:
        return ScoreSpec(name="step", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
        return ScoreSpec(name="customized", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.LFU:
        return ScoreSpec(name="frequency", policy=ScorePolicy.ACCUMULATE)
    else:
        raise RuntimeError("Not supported score strategy.")


def get_uvm_tensor(dim, dtype, device, is_managed=False):
    return torch.zeros(
        dim,
        out=torch.ops.fbgemm.new_unified_tensor(
            # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
            #  for 3rd param but got `Type[Type[torch._dtype]]`.
            torch.zeros(1, device=device, dtype=dtype),
            [dim],
            #  is_host_mapped (bool = False): If True, allocate every UVM tensor
            # using `malloc` + `cudaHostRegister`. Otherwise use
            # `cudaMallocManaged`
            is_host_mapped=(not is_managed),
        ),
    )


def _export_matched_and_gather(
    dynamic_table: Any,
    threshold: int,
    pg: Optional[dist.ProcessGroup] = None,
    batch_size: int = BATCH_SIZE_PER_DUMP,
) -> Tuple[Tensor, Tensor]:
    # Get the rank of the current process
    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    d_num_matched = torch.zeros(1, dtype=torch.uint64, device=device)
    dynamic_table.count_matched(threshold, d_num_matched)

    gathered_num_matched = [
        torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)
    ]
    dist.all_gather(gathered_num_matched, d_num_matched.to(dtype=torch.int64), group=pg)

    total_matched = sum([t.item() for t in gathered_num_matched])  # t is on device.
    key_dtype = dynamic_table.key_type()
    value_dtype = dynamic_table.value_type()
    dim: int = dynamic_table.embedding_dim()
    total_dim = dynamic_table.value_dim()

    ret_keys = torch.empty(total_matched, dtype=key_dtype, device="cpu")
    ret_vals = torch.empty(total_matched * dim, dtype=value_dtype, device="cpu")
    ret_offset = 0

    search_offset = 0
    search_capacity = dynamic_table.capacity()

    d_keys = torch.empty(batch_size, dtype=key_dtype, device=device)
    d_embs = torch.empty(batch_size * dim, dtype=value_dtype, device=device)
    d_vals = torch.empty(batch_size * total_dim, dtype=value_dtype, device=device)
    d_count = torch.zeros(1, dtype=torch.uint64, device=device)

    # Gather keys and values for all ranks
    gathered_keys = [torch.empty_like(d_keys) for _ in range(world_size)]
    gathered_vals = [torch.empty_like(d_embs) for _ in range(world_size)]
    gathered_counts = [
        torch.empty_like(d_count, dtype=torch.int64) for _ in range(world_size)
    ]

    while search_offset < search_capacity:
        dynamic_table.export_batch_matched(
            threshold, batch_size, search_offset, d_count, d_keys, d_vals
        )

        d_embs = d_vals.view(batch_size, total_dim)[:, :dim].reshape(-1)
        dist.all_gather(gathered_keys, d_keys, group=pg)
        dist.all_gather(gathered_vals, d_embs, group=pg)
        dist.all_gather(gathered_counts, d_count.to(dtype=torch.int64), group=pg)

        for d_keys_, d_vals_, d_count_ in zip(
            gathered_keys, gathered_vals, gathered_counts
        ):
            h_count = d_count_.cpu().item()
            ret_keys[ret_offset : ret_offset + h_count] = d_keys_[0:h_count].cpu()
            ret_vals[ret_offset * dim : (ret_offset + h_count) * dim] = d_vals_[
                0 : h_count * dim
            ].cpu()
            ret_offset += h_count

        search_offset += batch_size
        d_count.fill_(0)

    return ret_keys, ret_vals


def _export_matched(
    dynamic_table: Any,
    threshold: int,
    batch_size: int = BATCH_SIZE_PER_DUMP,
) -> Tuple[Tensor, Tensor]:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    d_num_matched = torch.zeros(1, dtype=torch.uint64, device=device)
    dynamic_table.count_matched(threshold, d_num_matched)

    total_matched = d_num_matched.cpu().item()
    key_dtype = dynamic_table.key_type()
    value_dtype = dynamic_table.value_type()
    dim: int = dynamic_table.embedding_dim()
    total_dim = dynamic_table.value_dim()

    ret_keys = torch.empty(total_matched, dtype=key_dtype, device="cpu")
    ret_vals = torch.empty(total_matched * dim, dtype=value_dtype, device="cpu")
    ret_offset = 0

    search_offset = 0
    search_capacity = dynamic_table.capacity()
    batch_size = batch_size if batch_size < search_capacity else search_capacity

    d_keys = torch.empty(batch_size, dtype=key_dtype, device=device)
    d_vals = torch.empty(batch_size * total_dim, dtype=value_dtype, device=device)
    d_count = torch.zeros(1, dtype=torch.uint64, device=device)

    while search_offset < search_capacity:
        dynamic_table.export_batch_matched(
            threshold, batch_size, search_offset, d_count, d_keys, d_vals
        )

        h_count = d_count.cpu().item()
        ret_keys[ret_offset : ret_offset + h_count] = d_keys[0:h_count].cpu()
        ret_vals[ret_offset * dim : (ret_offset + h_count) * dim] = (
            d_vals.view(batch_size, total_dim)[:h_count, :dim].reshape(-1).cpu()
        )
        ret_offset += h_count

        search_offset += batch_size
        d_count.fill_(0)

    return ret_keys, ret_vals


class KeyValueTable(
    Cache, Storage[DynamicEmbTableOptions, BaseDynamicEmbeddingOptimizerV2]
):
    def __init__(
        self,
        options: DynamicEmbTableOptions,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        self.options = options
        self.table = create_dynamicemb_table(options)
        self._capacity = options.max_capacity
        self.optimizer = optimizer
        self.score: int = None
        self._score_update = False
        self._emb_dim = self.options.dim
        self._emb_dtype = self.options.embedding_dtype
        self._de_emb_dtype = torch_to_dyn_emb(self._emb_dtype)
        self._value_dim = self._emb_dim + optimizer.get_state_dim(self._emb_dim)
        self._initial_optim_state = optimizer.get_initial_optim_states()

        device_idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{device_idx}")
        props = torch.cuda.get_device_properties(device_idx)
        self._threads_in_wave = (
            props.multi_processor_count * props.max_threads_per_multi_processor
        )

        self._cache_metrics = torch.zeros(10, dtype=torch.long, device="cpu")
        self._record_cache_metrics = False
        self._use_score = self.table.evict_strategy() != EvictStrategy.KLru
        self._timestamp = device_timestamp()

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        if unique_keys.dtype != self.key_type():
            unique_keys = unique_keys.to(self.key_type())

        if unique_embs.dtype != self.value_type():
            raise RuntimeError(
                "Embedding dtype not match {} != {}".format(
                    unique_embs.dtype, self.value_type()
                )
            )

        batch = unique_keys.size(0)
        assert unique_embs.dim() == 2
        assert unique_embs.size(0) == batch

        load_dim = unique_embs.size(1)

        device = unique_keys.device
        if founds is None:
            founds = torch.empty(batch, dtype=torch.bool, device=device)
        pointers = torch.empty(batch, dtype=torch.long, device=device)

        scores = self.create_scores(batch, device, input_scores)

        if self._score_update:
            find_pointers_with_scores(
                self.table, batch, unique_keys, pointers, founds, scores
            )
        else:
            find_pointers(self.table, batch, unique_keys, pointers, founds)

        self.value_dim()

        if load_dim != 0:
            load_from_pointers(pointers, unique_embs)

        missing = torch.logical_not(founds)
        num_missing_0: torch.Tensor = torch.empty(1, dtype=torch.long, device=device)
        num_missing_1: torch.Tensor = torch.empty(1, dtype=torch.long, device=device)
        missing_keys: torch.Tensor = torch.empty_like(unique_keys)
        missing_indices: torch.Tensor = torch.empty(
            batch, dtype=torch.long, device=device
        )
        select(missing, unique_keys, missing_keys, num_missing_0)
        select_index(missing, missing_indices, num_missing_1)

        if self._record_cache_metrics:
            self._cache_metrics[0] = batch
            self._cache_metrics[1] = founds.sum().item()

        h_num_missing = num_missing_0.cpu().item()

        # Handle missing scores: return None if scores is None
        if scores is not None:
            missing_scores = scores[missing_indices[:h_num_missing]]
        else:
            missing_scores = None

        return (
            h_num_missing,
            missing_keys[:h_num_missing],
            missing_indices[:h_num_missing],
            missing_scores,
        )

    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check shape to prevent misuse of find_embeddings and find
        if unique_embs.dim() == 2 and unique_embs.size(1) != self.embedding_dim():
            raise ValueError(
                f"find_embeddings expects dim={self.embedding_dim()}, got {unique_embs.size(1)}. "
            )
        return self.find_impl(unique_keys, unique_embs, founds, input_scores)

    def find_missed_keys(
        self,
        unique_keys: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        # dummy tensor
        unique_embs = torch.empty(
            unique_keys.numel(), 0, device=unique_keys.device, dtype=self._emb_dtype
        )
        return self.find_impl(unique_keys, unique_embs, founds, None)

    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check shape to prevent misuse of find_embeddings and find
        if unique_vals.dim() == 2 and unique_vals.size(1) != self.value_dim():
            raise ValueError(
                f"find expects dim={self.value_dim()}, got {unique_vals.size(1)}. "
            )
        return self.find_impl(unique_keys, unique_vals, founds, input_scores)

    def create_scores(
        self,
        h_num_total: int,
        device: torch.device,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Create scores tensor for lookup operation based on eviction strategy."""
        if (
            lfu_accumulated_frequency is not None
            and self.evict_strategy() == EvictStrategy.KLfu
        ):
            return lfu_accumulated_frequency
        elif self.evict_strategy() == EvictStrategy.KLfu:
            scores = torch.ones(h_num_total, device=device, dtype=torch.long)
            return scores
        elif self.evict_strategy() == EvictStrategy.KCustomized:
            scores = torch.empty(h_num_total, device=device, dtype=torch.long)
            scores.fill_(self.score)
            return scores
        else:
            return None

    def insert(
        self,
        unique_keys: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        h_num_unique_keys = unique_keys.numel()
        if self._use_score:
            if scores is None:
                scores = torch.empty(
                    h_num_unique_keys, device=unique_keys.device, dtype=torch.uint64
                )
                scores.fill_(self.score)
        else:
            scores = None

        if self.evict_strategy() == EvictStrategy.KLfu:
            erase(self.table, h_num_unique_keys, unique_keys)

        insert_or_assign(
            self.table,
            h_num_unique_keys,
            unique_keys,
            unique_values.to(self.value_type()),
            scores,
        )

    def update(
        self,
        keys: torch.Tensor,
        grads: torch.Tensor,
        return_missing: bool = True,
    ) -> Tuple[Optional[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self._score_update == False, "update is called only in backward."

        batch = keys.size(0)

        device = keys.device
        founds = torch.empty(batch, dtype=torch.bool, device=device)
        pointers = torch.empty(batch, dtype=torch.long, device=device)
        find_pointers(self.table, batch, keys, pointers, founds)

        self.optimizer.fused_update_with_pointer(
            grads.to(self.value_type()), pointers, self._de_emb_dtype
        )

        if return_missing:
            missing = torch.logical_not(founds)
            num_missing_0: torch.Tensor = torch.empty(
                1, dtype=torch.long, device=device
            )
            num_missing_1: torch.Tensor = torch.empty(
                1, dtype=torch.long, device=device
            )
            missing_keys: torch.Tensor = torch.empty_like(keys)
            missing_indices: torch.Tensor = torch.empty(
                batch, dtype=torch.long, device=device
            )
            select(missing, keys, missing_keys, num_missing_0)
            select_index(missing, missing_indices, num_missing_1)
            h_num_missing = num_missing_0.cpu().item()
            return (
                h_num_missing,
                missing_keys[:h_num_missing],
                missing_indices[:h_num_missing],
            )
        return None, None, None

    def enable_update(self) -> bool:
        return True

    def set_score(
        self,
        score: int,
    ) -> None:
        self.score = score

    @property
    def score_update(
        self,
    ) -> None:
        return self._score_update

    @score_update.setter
    def score_update(self, value: bool):
        self._score_update = value

    def update_timestamp(self) -> None:
        self._timestamp = device_timestamp()

    def dump(
        self,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool,
        include_meta: bool,
    ) -> None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if include_meta:
            meta_data = {}
            meta_data.update(self.optimizer.get_opt_args())
            meta_data["evict_strategy"] = str(self.table.evict_strategy())
            save_to_json(meta_data, meta_json_file_path)

        fkey = open(emb_key_path, "wb")
        fembedding = open(embedding_file_path, "wb")
        fscore = open(score_file_path, "wb")
        fopt_states = open(opt_file_path, "wb") if include_optim else None

        for keys, embeddings, opt_states, scores in batched_export_keys_values(
            self.table, device
        ):
            fkey.write(keys.cpu().numpy().tobytes())
            fembedding.write(embeddings.cpu().numpy().tobytes())
            if self.table.evict_strategy() == EvictStrategy.KLru:
                scores = self._timestamp - scores
            fscore.write(scores.cpu().numpy().tobytes())
            if fopt_states:
                fopt_states.write(opt_states.cpu().numpy().tobytes())

        fkey.close()
        fembedding.close()

        if fscore:
            fscore.close()

        if fopt_states:
            fopt_states.close()

        return

    def load(
        self,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool,
    ) -> None:
        meta_data = load_from_json(meta_json_file_path)
        opt_type = meta_data.get(
            "opt_type", None
        )  # for compatibility with old format, which doesn't have opt_type
        if opt_type and self.optimizer.get_opt_args().get("opt_type", None) != opt_type:
            include_optim = False
            print(
                f"Optimizer type mismatch: {opt_type} != {self.optimizer.get_opt_args().get('opt_type')}. Will not load optimizer states."
            )

        evict_strategy = meta_data.get("evict_strategy", None)
        if evict_strategy and str(self.table.evict_strategy()) != evict_strategy:
            raise ValueError(
                f"Evict strategy mismatch: {evict_strategy} != {self.table.evict_strategy()}"
            )

        if score_file_path is None:
            print(
                f"Score file {score_file_path} does not exist. Will not load score states."
            )

        if not opt_file_path or not os.path.exists(opt_file_path):
            include_optim = False
            print(
                f"Optimizer file {opt_file_path} does not exist. Will not load optimizer states."
            )

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        dim = dyn_emb_cols(self.table)
        optstate_dim = self.table.optstate_dim()

        if optstate_dim == 0:
            include_optim = False

        if include_optim:
            self.optimizer.set_opt_args(meta_data)

        fkey = open(emb_key_path, "rb")
        fembedding = open(embedding_file_path, "rb")
        fscore = (
            open(score_file_path, "rb")
            if score_file_path and os.path.exists(score_file_path)
            else None
        )
        fopt_states = open(opt_file_path, "rb") if include_optim else None
        num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize

        num_embeddings = (
            os.path.getsize(embedding_file_path) // EMBEDDING_TYPE.itemsize // dim
        )

        if num_keys != num_embeddings:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of embeddings in {embedding_file_path}."
            )

        if fscore:
            num_scores = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
            if num_keys != num_scores:
                raise ValueError(
                    f"The number of keys in {emb_key_path} does not match with number of scores in {score_file_path}."
                )

        if fopt_states:
            num_opt_states = (
                os.path.getsize(opt_file_path)
                // OPT_STATE_TYPE.itemsize
                // optstate_dim
            )
            if num_keys != num_opt_states:
                raise ValueError(
                    f"The number of keys in {emb_key_path} does not match with number of opt_states in {opt_file_path}."
                )

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        batch_size = 65536
        for start in range(0, num_keys, batch_size):
            num_keys_to_read = min(num_keys - start, batch_size)
            keys_bytes = fkey.read(KEY_TYPE.itemsize * num_keys_to_read)

            embedding_bytes = fembedding.read(
                EMBEDDING_TYPE.itemsize * dim * num_keys_to_read
            )
            embeddings = torch.tensor(
                np.frombuffer(
                    embedding_bytes, dtype=torch_dtype_to_np_dtype[EMBEDDING_TYPE]
                ),
                dtype=EMBEDDING_TYPE,
                device=device,
            ).view(-1, dim)

            opt_states = None
            if fopt_states:
                opt_state_bytes = fopt_states.read(
                    OPT_STATE_TYPE.itemsize * optstate_dim * num_keys_to_read
                )
                opt_states = torch.tensor(
                    np.frombuffer(
                        opt_state_bytes, dtype=torch_dtype_to_np_dtype[OPT_STATE_TYPE]
                    ),
                    dtype=OPT_STATE_TYPE,
                    device=device,
                ).view(-1, optstate_dim)

            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )

            scores = None
            if fscore:
                score_bytes = fscore.read(SCORE_TYPE.itemsize * num_keys_to_read)
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )
                if self.table.evict_strategy() == EvictStrategy.KLru:
                    scores = torch.clamp(self._timestamp - scores, min=0)

            if world_size > 1:
                masks = keys % world_size == rank
                keys = keys[masks]
                embeddings = embeddings[masks, :]
                if scores is not None:
                    scores = scores[masks]
                if opt_states is not None:
                    opt_states = opt_states[masks, :]
            load_key_values(self.table, keys, embeddings, scores, opt_states)

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def embedding_dtype(
        self,
    ) -> torch.dtype:
        return self._emb_dtype

    def value_dim(
        self,
    ) -> int:
        return self._value_dim

    def embedding_dim(
        self,
    ) -> int:
        return self._emb_dim

    def init_optimizer_state(
        self,
    ) -> float:
        return self._initial_optim_state

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = keys.numel()
        num_evicted: torch.Tensor = torch.zeros(1, dtype=torch.long, device=keys.device)
        evicted_keys: torch.Tensor = torch.empty_like(keys)
        evicted_values: torch.Tensor = torch.empty_like(values)
        evicted_scores: torch.Tensor = torch.empty(
            batch, dtype=torch.uint64, device=keys.device
        )
        if scores is not None:
            insert_and_evict_with_scores(
                self.table,
                batch,
                keys,
                values,
                evicted_keys,
                evicted_values,
                evicted_scores,
                num_evicted,
                scores=scores,  # scores as keyword argument
            )
        else:
            # TODO: Fix prefetch issue when scores is not provided
            insert_and_evict(
                self.table,
                batch,
                keys,
                values,
                self.score if self._use_score else None,
                evicted_keys,
                evicted_values,
                evicted_scores,
                num_evicted,
            )
        if self._record_cache_metrics:
            self._cache_metrics[2] = batch
            self._cache_metrics[3] = num_evicted.cpu().item()
        h_num_evict = num_evicted.cpu().item()
        return (
            h_num_evict,
            evicted_keys[:h_num_evict],
            evicted_values[:h_num_evict, :],
            evicted_scores[:h_num_evict],
        )

    def flush(self, storage: Storage) -> None:
        batch_size = self._threads_in_wave
        for keys, embeddings, opt_states, scores in batched_export_keys_values(
            self.table, self.device, batch_size
        ):
            if keys.numel() != 0:
                values = torch.cat((embeddings, opt_states), dim=1).contiguous()
                # TODO: assign the score directly, otherwise the score will be accumulated for LFU.
                storage.insert(keys, values, scores)

    def reset(
        self,
    ) -> None:
        clear(self.table)

    @property
    def cache_metrics(self) -> Optional[torch.Tensor]:
        return self._cache_metrics if self._record_cache_metrics else None

    def set_record_cache_metrics(self, record: bool) -> None:
        self._record_cache_metrics = record
        return

    def count_matched(
        self,
        threshold: int,
        num_matched: torch.Tensor,
    ) -> None:
        count_matched(self.table, threshold, num_matched)

    def key_type(
        self,
    ) -> torch.dtype:
        return dyn_emb_to_torch(self.table.key_type())

    def value_type(
        self,
    ) -> torch.dtype:
        return dyn_emb_to_torch(self.table.value_type())

    def capacity(
        self,
    ) -> int:
        return self._capacity

    def export_batch_matched(
        self, threshold, batch_size, search_offset, d_count, d_keys, d_vals
    ) -> None:
        export_batch_matched(
            self.table,
            threshold,
            batch_size,
            search_offset,
            d_count,
            d_keys,
            d_vals,
        )

    def evict_strategy(self) -> EvictStrategy:
        return self.table.evict_strategy()

    def optim_state_dim(self) -> int:
        return self.value_dim() - self.embedding_dim()

    def size(self) -> int:
        return dyn_emb_rows(self.table)

    def incremental_dump(
        self,
        score_threshold: int,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dump incremental keys and embeddings and combine them for all ranks into cpu tensors.

        Args:
            score_threshold (int): input threshold.
            pg (Optional[dist.ProcessGroup]): process group.

        Returns:
            out_keys (torch.Tensor): output tensor of keys
            out_embeddings (torch.Tensor): output tensors of embeddings.
        """

        def _export_matched_per_table(pg, table, threshold):
            if not dist.is_initialized() or dist.get_world_size(group=pg) == 1:
                key, value = _export_matched(table, threshold)
            else:
                key, value = _export_matched_and_gather(table, threshold, pg)
            return key, value

        key, value = _export_matched_per_table(pg, self, score_threshold)
        return key, value.view(-1, self._emb_dim)

    def export_keys_values(
        self,
        device: torch.device,
        batch_size: int = 65536,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        export keys, embeddings, opt_states, scores
        """

        return batched_export_keys_values(
            self.table, device=device, batch_size=batch_size
        )


class DynamicEmbeddingTable(KeyValueTable):
    def __init__(
        self,
        options: DynamicEmbTableOptions,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        self.options = options
        device_idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{device_idx}")

        # assert (
        #     options.init_capacity == options.max_capacity
        # ), "Capacity growth is appending..."

        self.score_policy = get_score_policy(options.score_strategy)
        self.evict_strategy_ = options.evict_strategy.value

        self.key_index_map = get_scored_table(
            capacity=options.init_capacity,
            bucket_capacity=options.bucket_capacity,
            key_type=options.index_type,
            score_specs=[self.score_policy],
            device=self.device,
        )

        self._capacity = self.key_index_map.capacity()

        # TODO: maybe we can separate it in the future like fbgemm
        # self.embedding_weights_dev = None
        # self.embedding_weights_uvm = None
        # self.optimizer_states_dev = None
        # self.optimizer_states_uvm = None
        self.dev_table = None
        self.uvm_table = None

        self._emb_dtype = self.options.embedding_dtype
        self._emb_dim = self.options.dim
        self._optim_states_dim = optimizer.get_state_dim(self._emb_dim)
        self._value_dim = self._emb_dim + self._optim_states_dim

        total_memory_need = (
            self._capacity * self._value_dim * dtype_to_bytes(self._emb_dtype)
        )

        if options.local_hbm_for_values == 0:
            # weight_uvm_size = self._capacity * self._emb_dim
            # optim_states_uvm_size = self._capacity * self._optim_states_dim
            # self.embedding_weights_uvm = get_uvm_tensor(weight_uvm_size, self._emb_dtype, self.device)
            # self.embedding_weights_uvm = get_uvm_tensor(optim_states_uvm_size, self._emb_dtype, self.device)
            uvm_size = self._capacity * self._value_dim
            self.uvm_table = get_uvm_tensor(
                uvm_size, dtype=self._emb_dtype, device=self.device
            ).view(-1, self._value_dim)
        elif options.local_hbm_for_values >= total_memory_need:
            dev_size = self._capacity * self._value_dim
            self.dev_table = torch.empty(
                dev_size, dtype=self._emb_dtype, device=self.device
            ).view(-1, self._value_dim)
        else:
            # hybrid mode
            dev_size = (
                options.local_hbm_for_values
                // (self._value_dim * dtype_to_bytes(self._emb_dtype))
                * self._value_dim
            )
            uvm_size = self._capacity * self._value_dim - dev_size
            self.dev_table = torch.empty(
                dev_size, dtype=self._emb_dtype, device=self.device
            ).view(-1, self._value_dim)
            self.uvm_table = get_uvm_tensor(
                uvm_size, dtype=self._emb_dtype, device=self.device
            ).view(-1, self._value_dim)

        self.score: int = None
        self._score_update = False

        self.optimizer = optimizer
        self._de_emb_dtype = torch_to_dyn_emb(self._emb_dtype)

        self._initial_optim_state = optimizer.get_initial_optim_states()

        props = torch.cuda.get_device_properties(device_idx)
        self._threads_in_wave = (
            props.multi_processor_count * props.max_threads_per_multi_processor
        )

        self._cache_metrics = torch.zeros(10, dtype=torch.long, device="cpu")
        self._record_cache_metrics = False
        self._use_score = self.evict_strategy_ != EvictStrategy.KLru
        self._timestamp = device_timestamp()

    def expand(self):
        if self._capacity == self.options.max_capacity:
            return
        if self.key_index_map.load_factor() < self.options.max_load_factor:
            return

        target_capacity = min(self.options.max_capacity, self._capacity * 2)
        total_memory_need = (
            target_capacity * self._value_dim * dtype_to_bytes(self._emb_dtype)
        )

        key_index_map = get_scored_table(
            capacity=target_capacity,
            bucket_capacity=self.options.bucket_capacity,
            key_type=self.options.index_type,
            score_specs=[self.score_policy],
            device=self.device,
        )

        dev_table = None
        uvm_table = None

        if self.options.local_hbm_for_values == 0:
            uvm_size = target_capacity * self._value_dim
            uvm_table = get_uvm_tensor(
                uvm_size, dtype=self._emb_dtype, device=self.device
            ).view(-1, self._value_dim)
        elif self.options.local_hbm_for_values >= total_memory_need:
            dev_size = target_capacity * self._value_dim
            dev_table = torch.empty(
                dev_size, dtype=self._emb_dtype, device=self.device
            ).view(-1, self._value_dim)
        else:
            # hybrid mode
            dev_size = (
                self.options.local_hbm_for_values
                // (self._value_dim * dtype_to_bytes(self._emb_dtype))
                * self._value_dim
            )
            uvm_size = target_capacity * self._value_dim - dev_size
            dev_table = torch.empty(
                dev_size, dtype=self._emb_dtype, device=self.device
            ).view(-1, self._value_dim)
            uvm_table = get_uvm_tensor(
                uvm_size, dtype=self._emb_dtype, device=self.device
            ).view(-1, self._value_dim)

        for (
            keys,
            named_scores,
            indices,
        ) in self.key_index_map._batched_export_keys_scores(
            [self.score_policy.name],
            self.device,
            return_index=True,
        ):
            scores = named_scores[self.score_policy.name]

            values = torch.empty(
                (keys.numel(), self._value_dim),
                dtype=self.value_type(),
                device=self.device,
            )
            load_from_combined_table(self.dev_table, self.uvm_table, indices, values)

            # when load into the table, we always assign the scores from the file.
            score_args_insert = [
                ScoreArg(
                    name=self.score_policy.name,
                    value=scores,
                    policy=ScorePolicy.ASSIGN,
                    is_return=False,
                )
            ]
            indices = torch.zeros(
                keys.numel(), dtype=key_index_map.index_type, device=keys.device
            )

            key_index_map.insert(keys, score_args_insert, indices)
            store_to_combined_table(
                dev_table,
                uvm_table,
                indices,
                values,
            )

        self.key_index_map = key_index_map
        self.dev_table = dev_table
        self.uvm_table = uvm_table
        self._capacity = target_capacity

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        if unique_keys.dtype != self.key_type():
            unique_keys = unique_keys.to(self.key_type())

        if unique_embs.dtype != self.value_type():
            raise RuntimeError(
                "Embedding dtype not match {} != {}".format(
                    unique_embs.dtype, self.value_type()
                )
            )

        batch = unique_keys.size(0)
        assert unique_embs.dim() == 2
        assert unique_embs.size(0) == batch

        load_dim = unique_embs.size(1)

        device = unique_keys.device
        if founds is None:
            founds = torch.empty(batch, dtype=torch.bool, device=device)
        indices = torch.empty(batch, dtype=self.key_index_map.index_type, device=device)

        scores = self.create_scores(batch, device, input_scores)

        score_args_lookup = [
            ScoreArg(
                name=self.score_policy.name,
                value=scores,
                policy=self.score_policy.policy
                if self._score_update
                else ScorePolicy.CONST,
                is_return=scores is not None,
            )
        ]

        self.key_index_map.lookup(unique_keys, score_args_lookup, founds, indices)

        if load_dim != 0:
            load_from_combined_table(
                self.dev_table, self.uvm_table, indices, unique_embs
            )

        missing = torch.logical_not(founds)
        num_missing_0: torch.Tensor = torch.empty(1, dtype=torch.long, device=device)
        num_missing_1: torch.Tensor = torch.empty(1, dtype=torch.long, device=device)
        missing_keys: torch.Tensor = torch.empty_like(unique_keys)
        missing_indices: torch.Tensor = torch.empty(
            batch, dtype=torch.long, device=device
        )
        select(missing, unique_keys, missing_keys, num_missing_0)
        select_index(missing, missing_indices, num_missing_1)

        if self._record_cache_metrics:
            self._cache_metrics[0] = batch
            self._cache_metrics[1] = founds.sum().item()

        h_num_missing = num_missing_0.cpu().item()

        # Handle missing scores: return None if scores is None
        if scores is not None:
            missing_scores = scores[missing_indices[:h_num_missing]]
        else:
            missing_scores = None

        return (
            h_num_missing,
            missing_keys[:h_num_missing],
            missing_indices[:h_num_missing],
            missing_scores,
        )

    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check shape to prevent misuse of find_embeddings and find
        if unique_embs.dim() == 2 and unique_embs.size(1) != self.embedding_dim():
            raise ValueError(
                f"find_embeddings expects dim={self.embedding_dim()}, got {unique_embs.size(1)}. "
            )
        return self.find_impl(unique_keys, unique_embs, founds, input_scores)

    def find_missed_keys(
        self,
        unique_keys: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        # dummy tensor
        unique_embs = torch.empty(
            unique_keys.numel(), 0, device=unique_keys.device, dtype=self._emb_dtype
        )
        return self.find_impl(unique_keys, unique_embs, founds, None)

    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check shape to prevent misuse of find_embeddings and find
        if unique_vals.dim() == 2 and unique_vals.size(1) != self.value_dim():
            raise ValueError(
                f"find expects dim={self.value_dim()}, got {unique_vals.size(1)}. "
            )
        return self.find_impl(unique_keys, unique_vals, founds, input_scores)

    def create_scores(
        self,
        h_num_total: int,
        device: torch.device,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Create scores tensor for lookup operation based on eviction strategy."""
        if (
            lfu_accumulated_frequency is not None
            and self.evict_strategy() == EvictStrategy.KLfu
        ):
            return lfu_accumulated_frequency
        elif self.evict_strategy() == EvictStrategy.KLfu:
            scores = torch.ones(h_num_total, device=device, dtype=torch.long)
            return scores
        elif self.evict_strategy() == EvictStrategy.KCustomized:
            scores = torch.empty(h_num_total, device=device, dtype=torch.long)
            scores.fill_(self.score)
            return scores
        else:
            return None

    def insert(
        self,
        unique_keys: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        if os.environ.get("DEMB_DETERMINISM_MODE", None) is not None:
            return self.deterministic_insert(unique_keys, unique_values, scores)

        self.expand()

        h_num_unique_keys = unique_keys.numel()

        if self._use_score and scores is None:
            scores = torch.empty(
                h_num_unique_keys, device=unique_keys.device, dtype=torch.uint64
            )
            scores.fill_(self.score)

        policy = self.score_policy.policy

        if self.score_policy.policy == ScorePolicy.ACCUMULATE:
            # Case 1: table works on cache+storage mode, frequencies should be assigned but not accumulated.
            # Case 2: table works on storage mode, frequencies should be accumulated but unique_keys are not in the table.
            #           so ASSIGIN has the same result as ACCUMULATE, so we didn't distinguish them
            policy = ScorePolicy.ASSIGN

        if not self._use_score and scores is not None:
            # Case: table works on cache+storage and LRU mode, and the scores is from the cache,
            # we assign it to preserve the distribution of score values ​​in the cache.
            policy = ScorePolicy.ASSIGN

        score_args_insert = [
            ScoreArg(
                name=self.score_policy.name,
                value=scores,
                policy=policy,
                is_return=False,
            )
        ]

        indices = torch.zeros(
            h_num_unique_keys,
            dtype=self.key_index_map.index_type,
            device=unique_keys.device,
        )

        self.key_index_map.insert(unique_keys, score_args_insert, indices)
        store_to_combined_table(
            self.dev_table, self.uvm_table, indices, unique_values.to(self.value_type())
        )

    def update(
        self,
        keys: torch.Tensor,
        grads: torch.Tensor,
        return_missing: bool = True,
    ) -> Tuple[Optional[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self._score_update == False, "update is called only in backward."

        batch = keys.size(0)

        device = keys.device
        founds = torch.empty(batch, dtype=torch.bool, device=device)
        indices = torch.empty(batch, dtype=self.key_index_map.index_type, device=device)

        score_args_lookup = [
            ScoreArg(
                name=self.score_policy.name,
                value=None,
                policy=ScorePolicy.CONST,
                is_return=False,
            )
        ]

        self.key_index_map.lookup(keys, score_args_lookup, founds, indices)

        self.optimizer.fused_update_with_index(
            grads.to(self.value_type()), indices, self.dev_table, self.uvm_table
        )

        if return_missing:
            missing = torch.logical_not(founds)
            num_missing_0: torch.Tensor = torch.empty(
                1, dtype=torch.long, device=device
            )
            num_missing_1: torch.Tensor = torch.empty(
                1, dtype=torch.long, device=device
            )
            missing_keys: torch.Tensor = torch.empty_like(keys)
            missing_indices: torch.Tensor = torch.empty(
                batch, dtype=torch.long, device=device
            )
            select(missing, keys, missing_keys, num_missing_0)
            select_index(missing, missing_indices, num_missing_1)
            h_num_missing = num_missing_0.cpu().item()
            return (
                h_num_missing,
                missing_keys[:h_num_missing],
                missing_indices[:h_num_missing],
            )
        return None, None, None

    def enable_update(self) -> bool:
        return True

    def set_score(
        self,
        score: int,
    ) -> None:
        self.score = score

    @property
    def score_update(
        self,
    ) -> None:
        return self._score_update

    @score_update.setter
    def score_update(self, value: bool):
        self._score_update = value

    def update_timestamp(self) -> None:
        self._timestamp = device_timestamp()

    def dump(
        self,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool,
        include_meta: bool,
    ) -> None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if include_meta:
            meta_data = {}
            meta_data.update(self.optimizer.get_opt_args())
            meta_data["evict_strategy"] = str(self.evict_strategy_)
            save_to_json(meta_data, meta_json_file_path)

        fkey = open(emb_key_path, "wb")
        fembedding = open(embedding_file_path, "wb")
        fscore = open(score_file_path, "wb")
        fopt_states = open(opt_file_path, "wb") if include_optim else None

        for keys, embeddings, opt_states, scores in self.export_keys_values(
            device=device
        ):
            fkey.write(keys.cpu().numpy().tobytes())

            if self.evict_strategy_ == EvictStrategy.KLru:
                scores = self._timestamp - scores
            fscore.write(scores.cpu().numpy().tobytes())

            fembedding.write(embeddings.cpu().numpy().tobytes())

            if fopt_states and opt_states is not None:
                fopt_states.write(opt_states.cpu().numpy().tobytes())

        fkey.close()
        fembedding.close()

        if fscore:
            fscore.close()

        if fopt_states:
            fopt_states.close()

        return

    def load(
        self,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool,
    ) -> None:
        meta_data = load_from_json(meta_json_file_path)
        opt_type = meta_data.get(
            "opt_type", None
        )  # for compatibility with old format, which doesn't have opt_type
        if opt_type and self.optimizer.get_opt_args().get("opt_type", None) != opt_type:
            include_optim = False
            print(
                f"Optimizer type mismatch: {opt_type} != {self.optimizer.get_opt_args().get('opt_type')}. Will not load optimizer states."
            )

        evict_strategy = meta_data.get("evict_strategy", None)
        if evict_strategy and str(self.evict_strategy_) != evict_strategy:
            raise ValueError(
                f"Evict strategy mismatch: {evict_strategy} != {self.evict_strategy_}"
            )

        if score_file_path is None:
            print(
                f"Score file {score_file_path} does not exist. Will not load score states."
            )

        if not opt_file_path or not os.path.exists(opt_file_path):
            include_optim = False
            print(
                f"Optimizer file {opt_file_path} does not exist. Will not load optimizer states."
            )

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        dim = self._emb_dim
        optstate_dim = self.optim_state_dim()

        if optstate_dim == 0:
            include_optim = False

        if include_optim:
            self.optimizer.set_opt_args(meta_data)

        fkey = open(emb_key_path, "rb")
        fembedding = open(embedding_file_path, "rb")
        fscore = (
            open(score_file_path, "rb")
            if score_file_path and os.path.exists(score_file_path)
            else None
        )
        fopt_states = open(opt_file_path, "rb") if include_optim else None
        num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize

        num_embeddings = (
            os.path.getsize(embedding_file_path) // EMBEDDING_TYPE.itemsize // dim
        )

        if num_keys != num_embeddings:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of embeddings in {embedding_file_path}."
            )

        if fscore:
            num_scores = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
            if num_keys != num_scores:
                raise ValueError(
                    f"The number of keys in {emb_key_path} does not match with number of scores in {score_file_path}."
                )

        if fopt_states:
            num_opt_states = (
                os.path.getsize(opt_file_path)
                // OPT_STATE_TYPE.itemsize
                // optstate_dim
            )
            if num_keys != num_opt_states:
                raise ValueError(
                    f"The number of keys in {emb_key_path} does not match with number of opt_states in {opt_file_path}."
                )

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        batch_size = 65536
        for start in range(0, num_keys, batch_size):
            num_keys_to_read = min(num_keys - start, batch_size)
            keys_bytes = fkey.read(KEY_TYPE.itemsize * num_keys_to_read)

            embedding_bytes = fembedding.read(
                EMBEDDING_TYPE.itemsize * dim * num_keys_to_read
            )
            embeddings = torch.tensor(
                np.frombuffer(
                    embedding_bytes, dtype=torch_dtype_to_np_dtype[EMBEDDING_TYPE]
                ),
                dtype=EMBEDDING_TYPE,
                device=device,
            ).view(-1, dim)

            opt_states = None
            if fopt_states:
                opt_state_bytes = fopt_states.read(
                    OPT_STATE_TYPE.itemsize * optstate_dim * num_keys_to_read
                )
                opt_states = torch.tensor(
                    np.frombuffer(
                        opt_state_bytes, dtype=torch_dtype_to_np_dtype[OPT_STATE_TYPE]
                    ),
                    dtype=OPT_STATE_TYPE,
                    device=device,
                ).view(-1, optstate_dim)

            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )

            scores = None
            if fscore:
                score_bytes = fscore.read(SCORE_TYPE.itemsize * num_keys_to_read)
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )
                if self.evict_strategy_ == EvictStrategy.KLru:
                    scores = torch.clamp(self._timestamp - scores, min=0)

            if world_size > 1:
                masks = keys % world_size == rank
                keys = keys[masks]
                embeddings = embeddings[masks, :]
                if scores is not None:
                    scores = scores[masks]
                if opt_states is not None:
                    opt_states = opt_states[masks, :]
            self.load_key_values(
                keys,
                embeddings,
                scores,
                opt_states,
            )

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def load_key_values(
        self,
        keys: torch.Tensor,
        embeddings: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        opt_states: Optional[torch.Tensor] = None,
    ):
        self.expand()

        dim = embeddings.size(1)
        optstate_dim = self.optim_state_dim()
        if not keys.is_cuda:
            raise RuntimeError("Keys must be on GPU")
        if not embeddings.is_cuda:
            raise RuntimeError("Embeddings must be on GPU")
        if scores is not None and not scores.is_cuda:
            raise RuntimeError("Scores must be on GPU")
        if opt_states is not None and not opt_states.is_cuda:
            raise RuntimeError("Opt states must be on GPU")

        if opt_states is None and optstate_dim > 0:
            opt_states = (
                torch.ones(
                    keys.numel(),
                    optstate_dim,
                    dtype=self.value_type(),
                    device=embeddings.device,
                )
                * self.init_optimizer_state()
            )

        values = (
            torch.cat(
                [embeddings.view(-1, dim), opt_states.view(-1, optstate_dim)], dim=-1
            )
            if opt_states is not None
            else embeddings
        )

        self.key_index_map.key_type
        value_type = self.value_type()

        # when load into the table, we always assign the scores from the file.
        policy = ScorePolicy.ASSIGN

        if scores is None:
            assert (
                self.evict_strategy() == EvictStrategy.KLru
            ), "scores is None for KLru evict strategy is allowed but will be deprecated in future."
            policy = ScorePolicy.GLOBAL_TIMER
        else:
            scores = scores.to(SCORE_TYPE)

        # self.insert(keys.to(key_type), values.to(value_type), scores)

        score_args_insert = [
            ScoreArg(
                name=self.score_policy.name,
                value=scores,
                policy=policy,
                is_return=False,
            )
        ]
        indices = torch.zeros(
            keys.numel(), dtype=self.key_index_map.index_type, device=keys.device
        )

        self.key_index_map.insert(keys, score_args_insert, indices)
        store_to_combined_table(
            self.dev_table,
            self.uvm_table,
            indices,
            values.to(value_type),
        )

    def export_keys_values(
        self,
        device: torch.device,
        batch_size: int = 65536,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        export keys, embeddings, opt_states, scores
        """

        for (
            keys,
            named_scores,
            indices,
        ) in self.key_index_map._batched_export_keys_scores(
            [self.score_policy.name],
            self.device,
            batch_size=batch_size,
            return_index=True,
        ):
            scores = named_scores[self.score_policy.name]

            values = torch.empty(
                (keys.numel(), self._value_dim),
                dtype=self.value_type(),
                device=self.device,
            )
            load_from_combined_table(self.dev_table, self.uvm_table, indices, values)
            embeddings = (
                values[:, : self._emb_dim].to(dtype=EMBEDDING_TYPE).contiguous()
            )

            if self.optim_state_dim() != 0:
                opt_states = (
                    values[:, -self.optim_state_dim() :]
                    .to(dtype=OPT_STATE_TYPE)
                    .contiguous()
                ).to(device)
            else:
                opt_states = None

            yield (
                keys.to(device),
                embeddings.to(device),
                opt_states,
                scores.to(SCORE_TYPE).to(device),
            )

    def embedding_dtype(
        self,
    ) -> torch.dtype:
        return self._emb_dtype

    def value_dim(
        self,
    ) -> int:
        return self._value_dim

    def embedding_dim(
        self,
    ) -> int:
        return self._emb_dim

    def init_optimizer_state(
        self,
    ) -> float:
        return self._initial_optim_state

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        if os.environ.get("DEMB_DETERMINISM_MODE", None) is not None:
            return self.deterministic_insert_and_evict(keys, values, scores)

        self.expand()

        batch = keys.numel()
        evicted_values: torch.Tensor = torch.empty_like(values)

        if self._use_score and scores is None:
            scores = torch.empty(batch, device=keys.device, dtype=torch.uint64)
            scores.fill_(self.score)

        score_args_insert = [
            ScoreArg(
                name=self.score_policy.name,
                value=scores,
                policy=self.score_policy.policy,
                is_return=False,
            )
        ]

        indices = torch.zeros(
            batch, dtype=self.key_index_map.index_type, device=keys.device
        )

        (
            num_evicted,
            evicted_keys,
            evicted_indices,
            evicted_scores,
        ) = self.key_index_map.insert_and_evict(keys, score_args_insert, indices)
        evicted_scores = evicted_scores[0]

        load_from_combined_table(
            self.dev_table,
            self.uvm_table,
            evicted_indices,
            evicted_values[:num_evicted, :],
        )

        store_to_combined_table(
            self.dev_table, self.uvm_table, indices, values.to(self.value_type())
        )

        if self._record_cache_metrics:
            self._cache_metrics[2] = batch
            self._cache_metrics[3] = num_evicted
        return (
            num_evicted,
            evicted_keys,
            evicted_values[:num_evicted, :],
            evicted_scores,
        )

    def flush(self, storage: Storage) -> None:
        batch_size = self._threads_in_wave

        for (
            keys,
            named_scores,
            indices,
        ) in self.key_index_map._batched_export_keys_scores(
            [self.score_policy.name],
            self.device,
            batch_size=batch_size,
            return_index=True,
        ):
            scores = named_scores[self.score_policy.name]
            values = torch.empty(
                keys.numel(),
                self._value_dim,
                dtype=self.value_type(),
                device=self.device,
            )

            load_from_combined_table(self.dev_table, self.uvm_table, indices, values)
            storage.insert(keys, values, scores)

    def reset(
        self,
    ) -> None:
        self.key_index_map.reset()

    @property
    def cache_metrics(self) -> Optional[torch.Tensor]:
        return self._cache_metrics if self._record_cache_metrics else None

    def set_record_cache_metrics(self, record: bool) -> None:
        self._record_cache_metrics = record
        return

    def key_type(
        self,
    ) -> torch.dtype:
        return self.key_index_map.key_type

    def value_type(
        self,
    ) -> torch.dtype:
        return self._emb_dtype

    def capacity(
        self,
    ) -> int:
        return self._capacity

    def incremental_dump(
        self,
        score_threshold: int,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dump incremental keys and embeddings and combine them for all ranks into cpu tensors.

        Args:
            score_threshold (int): input threshold.
            pg (Optional[dist.ProcessGroup]): process group.

        Returns:
            out_keys (torch.Tensor): output tensor of keys
            out_embeddings (torch.Tensor): output tensors of embeddings.
        """
        batch_size = self._threads_in_wave
        keys, _, indices = self.key_index_map.incremental_dump(
            {self.score_policy.name: score_threshold}, batch_size, pg, return_index=True
        )
        # TODO: pay attention to OOM
        embs = torch.empty(
            keys.numel(), self._emb_dim, dtype=self._emb_dtype, device=self.device
        )
        if keys.numel() != 0:
            load_from_combined_table(
                self.dev_table, self.uvm_table, indices.to(self.device), embs
            )

        if not dist.is_initialized() or dist.get_world_size(group=pg) == 1:
            return keys, embs.cpu()

        # Get the rank of the current process
        world_size = dist.get_world_size(group=pg)
        d_num_matched = torch.tensor(
            keys.numel(), dtype=COUNTER_TYPE, device=self.device
        )
        gathered_num_matched = [
            torch.tensor(0, dtype=COUNTER_TYPE, device=self.device)
            for _ in range(world_size)
        ]

        dist.all_gather(gathered_num_matched, d_num_matched, group=pg)

        total_matched = sum([t.item() for t in gathered_num_matched])
        max_num_matched = max([t.item() for t in gathered_num_matched])

        out_keys = torch.empty(total_matched, dtype=KEY_TYPE, device="cpu")
        out_embs = torch.empty(
            total_matched, self._emb_dim, dtype=self._emb_dtype, device="cpu"
        )

        d_keys = torch.empty(max_num_matched, dtype=KEY_TYPE, device=self.device)
        d_embs = torch.empty(
            max_num_matched, self._emb_dim, dtype=self._emb_dtype, device=self.device
        )
        # Gather keys and scores for all ranks
        gathered_keys = [torch.empty_like(d_keys) for _ in range(world_size)]
        gathered_embs = [torch.empty_like(d_embs) for _ in range(world_size)]

        d_keys[: keys.numel()].copy_(keys[: keys.numel()], non_blocking=True)
        d_embs[: keys.numel(), :].copy_(embs[: keys.numel(), :], non_blocking=True)
        dist.all_gather(gathered_keys, d_keys, group=pg)
        dist.all_gather(gathered_embs, d_embs, group=pg)

        out_offset = 0

        for i in range(world_size):
            d_keys_ = gathered_keys[i]
            d_embs_ = gathered_embs[i]
            d_count_ = gathered_num_matched[i]

            h_count = d_count_.cpu().item()
            out_keys[out_offset : out_offset + h_count].copy_(
                d_keys_[:h_count], non_blocking=True
            )
            out_embs[out_offset : out_offset + h_count, :].copy_(
                d_embs_[:h_count, :], non_blocking=True
            )

            out_offset += h_count

        assert out_offset == total_matched

        return out_keys, out_embs

    def evict_strategy(self) -> EvictStrategy:
        return self.evict_strategy_

    def optim_state_dim(self) -> int:
        return self.value_dim() - self.embedding_dim()

    def size(self) -> int:
        return self.key_index_map.size()

    def deterministic_insert(
        self,
        unique_keys: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        self.expand()

        # 1.Preprocess
        h_num_unique_keys = unique_keys.numel()

        if h_num_unique_keys == 0:
            return

        if self._use_score and scores is None:
            scores = torch.empty(
                h_num_unique_keys, device=unique_keys.device, dtype=torch.uint64
            )
            scores.fill_(self.score)

        policy = self.score_policy.policy

        if self.score_policy.policy == ScorePolicy.ACCUMULATE:
            # Case 1: table works on cache+storage mode, frequencies should be assigned but not accumulated.
            # Case 2: table works on storage mode, frequencies should be accumulated but unique_keys are not in the table.
            #           so ASSIGIN has the same result as ACCUMULATE, so we didn't distinguish them
            policy = ScorePolicy.ASSIGN

        if not self._use_score and scores is not None:
            # Case: table works on cache+storage and LRU mode, and the scores is from the cache,
            # we assign it to preserve the distribution of score values ​​in the cache.
            policy = ScorePolicy.ASSIGN

        # 2. Bucketize keys: bkt_keys=unique_keys[inverse]
        assert isinstance(
            self.key_index_map, LinearBucketTable
        ), "deterministic insert implementation only supports LinearBucketTable as key-index-map."
        bkt_keys, offsets, inverse = cast(
            LinearBucketTable, self.key_index_map
        ).bucketize_keys(unique_keys)

        jagged_keys = JaggedTensor(
            values=bkt_keys.to(torch.int64),
            offsets=offsets,
            weights=scores.to(torch.int64)[inverse] if scores is not None else None,
        )

        # static_cast<int64_t>(double(-1)) will get 0xFFFFFFFFFFFFFFFF, which is EmptyKey for table.
        pad_keys = jagged_keys.to_padded_dense(padding_value=-1.0)
        pad_scores = jagged_keys.to_padded_dense_weights(padding_value=0.0)

        keys_t = pad_keys.transpose(0, 1).to(self.key_type()).contiguous()
        score_t = (
            pad_scores.transpose(0, 1).contiguous() if pad_scores is not None else None
        )

        # 3. Insert iteratively
        num_iter = keys_t.size(0)
        for i in range(num_iter):
            valid_mask = keys_t[i] != -1
            valid_batch = valid_mask.sum().item()
            valid_keys = keys_t[i][valid_mask].contiguous()
            insert_results = torch.empty(
                valid_batch, dtype=torch.uint8, device=keys_t[i].device
            )
            score_args_insert = [
                ScoreArg(
                    name=self.score_policy.name,
                    # value=score_t[i] if score_t is not None else None,
                    value=score_t[i][valid_mask].contiguous().to(torch.uint64)
                    if score_t is not None
                    else None,
                    policy=policy,
                    is_return=False,
                )
            ]

            # self.key_index_map.insert(keys_t[i], score_args_insert)
            self.key_index_map.insert(
                valid_keys, score_args_insert, None, insert_results
            )

        # 4. lookup the indices in unique_keys' order and store the values.
        score_args_lookup = [
            ScoreArg(
                name=self.score_policy.name,
                policy=ScorePolicy.CONST,
                is_return=False,
            )
        ]
        indices = torch.zeros(
            h_num_unique_keys,
            dtype=self.key_index_map.index_type,
            device=unique_keys.device,
        )
        founds = torch.zeros(
            h_num_unique_keys, dtype=torch.bool, device=unique_keys.device
        )
        self.key_index_map.lookup(unique_keys, score_args_lookup, founds, indices)
        store_to_combined_table(
            self.dev_table, self.uvm_table, indices, unique_values.to(self.value_type())
        )
        return

    def deterministic_insert_and_evict(
        self,
        unique_keys: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        self.expand()

        # 1.Preprocess
        h_num_unique_keys = unique_keys.numel()

        if h_num_unique_keys == 0:
            return 0, None, None, None

        num_evicted_accum = 0
        evicted_keys_accum = torch.empty_like(unique_keys)
        evicted_indices_accum = torch.zeros(
            h_num_unique_keys,
            dtype=self.key_index_map.index_type,
            device=unique_keys.device,
        )
        evicted_scores_accum = torch.empty(
            h_num_unique_keys, dtype=torch.uint64, device=unique_keys.device
        )
        evicted_values_accum: torch.Tensor = torch.empty_like(unique_values)

        if self._use_score and scores is None:
            scores = torch.empty(
                h_num_unique_keys, device=unique_keys.device, dtype=torch.uint64
            )
            scores.fill_(self.score)

        # 2. Bucketize keys: bkt_keys=unique_keys[inverse]
        assert isinstance(
            self.key_index_map, LinearBucketTable
        ), "deterministic insert_and_evict implementation only supports LinearBucketTable as key-index-map."
        bkt_keys, offsets, inverse = cast(
            LinearBucketTable, self.key_index_map
        ).bucketize_keys(unique_keys)

        jagged_keys = JaggedTensor(
            values=bkt_keys.to(torch.int64),
            offsets=offsets,
            weights=scores.to(torch.int64)[inverse] if scores is not None else None,
        )

        # static_cast<int64_t>(double(-1)) will get 0xFFFFFFFFFFFFFFFF, which is EmptyKey for table.
        pad_keys = jagged_keys.to_padded_dense(padding_value=-1.0)
        pad_scores = jagged_keys.to_padded_dense_weights(padding_value=0.0)

        keys_t = pad_keys.transpose(0, 1).to(self.key_type()).contiguous()
        score_t = (
            pad_scores.transpose(0, 1).contiguous() if pad_scores is not None else None
        )

        # 3. Insert iteratively
        num_iter = keys_t.size(0)
        for i in range(num_iter):
            valid_mask = keys_t[i] != -1
            valid_batch = valid_mask.sum().item()
            valid_keys = keys_t[i][valid_mask].contiguous()
            insert_results = torch.empty(
                valid_batch, dtype=torch.uint8, device=keys_t[i].device
            )
            score_args_insert = [
                ScoreArg(
                    name=self.score_policy.name,
                    # value=score_t[i] if score_t is not None else None,
                    value=score_t[i][valid_mask].contiguous().to(torch.uint64)
                    if score_t is not None
                    else None,
                    policy=self.score_policy.policy,
                    is_return=False,
                )
            ]

            (
                num_evicted,
                evicted_keys,
                evicted_indices,
                evicted_scores,
            ) = self.key_index_map.insert_and_evict(
                valid_keys, score_args_insert, None, insert_results
            )
            evicted_scores = evicted_scores[0]

            if num_evicted != 0:
                evicted_keys_accum[
                    num_evicted_accum : num_evicted_accum + num_evicted
                ].copy_(evicted_keys, non_blocking=True)
                evicted_indices_accum[
                    num_evicted_accum : num_evicted_accum + num_evicted
                ].copy_(evicted_indices, non_blocking=True)
                evicted_scores_accum[
                    num_evicted_accum : num_evicted_accum + num_evicted
                ].copy_(evicted_scores, non_blocking=True)

                num_evicted_accum += num_evicted

        # check there is no duplicated eviction
        evicted_keys_accum = evicted_keys_accum[:num_evicted_accum]
        evicted_indices_accum = evicted_indices_accum[:num_evicted_accum]
        evicted_scores_accum = evicted_scores_accum[:num_evicted_accum]
        evicted_values_accum = evicted_values_accum[:num_evicted_accum, :]

        assert len(set(evicted_indices_accum.tolist())) == num_evicted_accum

        load_from_combined_table(
            self.dev_table,
            self.uvm_table,
            evicted_indices_accum,
            evicted_values_accum,
        )

        # 4. lookup the indices in unique_keys' order and store the values.
        score_args_lookup = [
            ScoreArg(
                name=self.score_policy.name,
                policy=ScorePolicy.CONST,
                is_return=False,
            )
        ]
        indices = torch.zeros(
            h_num_unique_keys,
            dtype=self.key_index_map.index_type,
            device=unique_keys.device,
        )
        founds = torch.zeros(
            h_num_unique_keys, dtype=torch.bool, device=unique_keys.device
        )
        self.key_index_map.lookup(unique_keys, score_args_lookup, founds, indices)
        store_to_combined_table(
            self.dev_table, self.uvm_table, indices, unique_values.to(self.value_type())
        )

        if self._record_cache_metrics:
            self._cache_metrics[2] = h_num_unique_keys
            self._cache_metrics[3] = num_evicted_accum
        return (
            num_evicted_accum,
            evicted_keys_accum,
            evicted_values_accum,
            evicted_scores_accum,
        )


def update_cache(
    cache: Cache,
    storage: Storage,
    missing_keys: torch.Tensor,
    missing_values: torch.Tensor,
    missing_scores: Optional[torch.Tensor] = None,
):
    # need to update score.
    num_evicted, evicted_keys, evicted_values, evicted_scores = cache.insert_and_evict(
        missing_keys,
        missing_values,
        missing_scores,
    )

    if num_evicted != 0:
        storage.insert(
            evicted_keys,
            evicted_values,
            evicted_scores,
        )


def admission(
    keys: torch.Tensor,
    freqs: torch.Tensor,
    admit_strategy: AdmissionStrategy,
    admission_counter: Counter,
) -> torch.Tensor:
    freq_for_missing_keys = admission_counter.add(keys, freqs, inplace=True)
    admit_mask = admit_strategy.admit(
        keys,
        freq_for_missing_keys,
    )
    admitted_keys = keys[admit_mask]
    admission_counter.erase(admitted_keys)

    return admit_mask


class KeyValueTableFunction:
    @staticmethod
    def lookup(
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        initializer: Callable,
        training: bool,
        evict_strategy: EvictStrategy,
        accumulated_frequency: Optional[torch.Tensor] = None,
        admit_strategy: Optional[AdmissionStrategy] = None,
        admission_counter: Optional[Counter] = None,
    ) -> None:
        assert unique_keys.dim() == 1
        h_num_toatl = unique_keys.numel()
        emb_dim = storage.embedding_dim()
        emb_dtype = storage.embedding_dtype()
        val_dim = storage.value_dim()

        is_lfu_enabled = evict_strategy == EvictStrategy.KLfu

        if h_num_toatl == 0:
            return

        # 1. find in storage
        founds = torch.empty(h_num_toatl, device=unique_keys.device, dtype=torch.bool)
        (
            h_num_missing_in_storage,
            missing_keys_in_storage,
            missing_indices_in_storage,
            missing_scores_in_storage,
        ) = storage.find_embeddings(
            unique_keys,
            unique_embs,
            founds=founds,
            input_scores=accumulated_frequency if is_lfu_enabled else None,
        )

        if h_num_missing_in_storage == 0:
            return

        # if training and admit_strategy is not None:

        admit_mask = None
        indices_to_init = missing_indices_in_storage
        if training and admit_strategy is not None:
            # do admission first
            if accumulated_frequency is not None:
                counters_for_admission = accumulated_frequency[
                    missing_indices_in_storage
                ]
            else:
                counters_for_admission = torch.ones(
                    missing_keys_in_storage.shape[0],
                    dtype=torch.int64,
                    device=unique_keys.device,
                )

            admit_mask = admission(
                missing_keys_in_storage,
                counters_for_admission,
                admit_strategy,
                admission_counter,
            )

            non_admitted_mask = ~admit_mask
            non_admitted_indices = missing_indices_in_storage[non_admitted_mask]
            initiailized_non_admitted_indices = False
            if non_admitted_indices.numel() > 0:
                initiailized_non_admitted_indices = (
                    admit_strategy.initialize_non_admitted_embeddings(
                        unique_embs[:, :emb_dim],
                        non_admitted_indices,
                    )
                )

            # Only initialize admitted embeddings with the regular initializer
            if not initiailized_non_admitted_indices:
                indices_to_init = missing_indices_in_storage[admit_mask]

        # 2. initialize missing embeddings (admitted or all if no admission)
        if indices_to_init.numel() > 0:
            initializer(
                unique_embs,
                indices_to_init,
                unique_keys,
            )

        if training:
            # insert missing values
            missing_values_in_storage = torch.empty(
                h_num_missing_in_storage,
                val_dim,
                device=unique_keys.device,
                dtype=emb_dtype,
            )
            missing_values_in_storage[:, :emb_dim] = unique_embs[
                missing_indices_in_storage, :
            ]
            if val_dim != emb_dim:
                missing_values_in_storage[
                    :, emb_dim - val_dim :
                ] = storage.init_optimizer_state()
            keys_to_insert = missing_keys_in_storage
            values_to_insert = missing_values_in_storage
            scores_to_insert = missing_scores_in_storage
            if training and admit_strategy is not None:
                keys_to_insert = keys_to_insert[admit_mask]
                values_to_insert = values_to_insert[admit_mask]
                scores_to_insert = (
                    scores_to_insert[admit_mask]
                    if scores_to_insert is not None
                    else None
                )

            # 3. insert missing values into table.
            storage.insert(
                keys_to_insert,
                values_to_insert,
                scores_to_insert,
            )
        # ignore the storage missed in eval mode

    @staticmethod
    def update(
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_grads: torch.Tensor,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        if storage.enable_update():
            storage.update(unique_keys, unique_grads, return_missing=False)
            return

        emb_dtype = storage.embedding_dtype()
        val_dim = storage.value_dim()
        h_num_toatl = unique_keys.numel()
        unique_values = torch.empty(
            h_num_toatl, val_dim, device=unique_keys.device, dtype=emb_dtype
        )
        founds = torch.empty(h_num_toatl, device=unique_keys.device, dtype=torch.bool)
        _, _, _, _ = storage.find(unique_keys, unique_values, founds=founds)

        keys_for_storage = unique_keys[founds].contiguous()
        values_for_storage = unique_values[founds, :].contiguous()
        grads_for_storage = unique_grads[founds, :].contiguous()
        optimizer.fused_update(
            grads_for_storage,
            values_for_storage,
        )

        storage.insert(keys_for_storage, values_for_storage)

        return


class KeyValueTableCachingFunction:
    @staticmethod
    def lookup(
        cache: Cache,  # partial emb + optimizer state
        storage: Storage,  # full emb + optimizer state
        unique_keys: torch.Tensor,  # input
        unique_embs: torch.Tensor,  # output
        initializer: Callable,
        enable_prefetch: bool,
        training: bool,
        evict_strategy: EvictStrategy,
        accumulated_frequency: Optional[torch.Tensor] = None,
        admit_strategy: Optional[AdmissionStrategy] = None,
        admission_counter: Optional[Counter] = None,
    ) -> None:
        assert unique_keys.dim() == 1
        unique_keys.numel()
        emb_dim = storage.embedding_dim()
        emb_dtype = storage.embedding_dtype()
        val_dim = (
            storage.value_dim()
        )  # value is generally composed of embedding and optimizer state

        is_lfu_enabled = evict_strategy == EvictStrategy.KLfu

        (
            h_num_keys_for_storage,
            missing_keys,
            missing_indices,
            missing_scores,
        ) = cache.find_embeddings(
            unique_keys,
            unique_embs,
            input_scores=accumulated_frequency if is_lfu_enabled else None,
        )
        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys

        scores_for_storage = missing_scores

        founds = torch.empty(
            h_num_keys_for_storage, device=unique_keys.device, dtype=torch.bool
        )

        # 2. find in storage
        values_for_storage = torch.empty(
            h_num_keys_for_storage,
            val_dim,
            device=unique_keys.device,
            dtype=emb_dtype,
        )
        (
            h_num_missing_in_storage,
            missing_keys_in_storage,
            missing_indices_in_storage,
            missing_scores_in_storage,
        ) = storage.find(
            keys_for_storage,
            values_for_storage,
            founds=founds,
            input_scores=scores_for_storage,
        )

        admit_mask_for_missing_keys = None
        indices_to_init = missing_indices_in_storage
        if training and admit_strategy is not None:
            # Get frequency counters for admission:
            if accumulated_frequency is not None:
                # missing_indices_in_storage is index in keys_for_storage, Need to convert to index in unique_keys via missing_indices
                indices_in_unique_keys = missing_indices[missing_indices_in_storage]
                counters_for_admission = accumulated_frequency[indices_in_unique_keys]
            else:
                counters_for_admission = torch.ones(
                    missing_keys_in_storage.shape[0],
                    dtype=torch.int64,
                    device=unique_keys.device,
                )

            admit_mask_for_missing_keys = admission(
                missing_keys_in_storage,
                counters_for_admission,
                admit_strategy,
                admission_counter,
            )

            non_admitted_mask = ~admit_mask_for_missing_keys
            non_admitted_indices = missing_indices_in_storage[non_admitted_mask]
            initiailized_non_admitted_indices = False
            if non_admitted_indices.numel() > 0:
                initiailized_non_admitted_indices = (
                    admit_strategy.initialize_non_admitted_embeddings(
                        values_for_storage[:, :emb_dim],
                        non_admitted_indices,
                    )
                )

            # Only initialize admitted embeddings with the regular initializer
            if not initiailized_non_admitted_indices:
                indices_to_init = missing_indices_in_storage[
                    admit_mask_for_missing_keys
                ]

        # 3. initialize missing embeddings (admitted or all if no admission)
        if indices_to_init.numel() > 0:
            initializer(
                values_for_storage[:, :emb_dim],
                indices_to_init,
                keys_for_storage,
            )

        # 4. copy embeddings to unique_embs
        unique_embs[missing_indices, :] = values_for_storage[:, :emb_dim]

        keys_to_update = None
        values_to_update = None
        scores_to_update = None

        if training:
            if emb_dim != val_dim:
                values_for_storage[
                    missing_indices_in_storage, emb_dim - val_dim :
                ] = storage.init_optimizer_state()
            # 5.Optional Admission part
            keys_to_update = keys_for_storage
            values_to_update = values_for_storage
            scores_to_update = scores_for_storage

            if admit_strategy is not None:
                # build mask: including storage hit keys + keys that are both miss and admitted
                mask_to_cache = founds
                admitted_indices = missing_indices_in_storage[
                    admit_mask_for_missing_keys
                ]
                mask_to_cache[admitted_indices] = True

                keys_to_update = keys_for_storage[mask_to_cache]
                values_to_update = values_for_storage[mask_to_cache]
                scores_to_update = (
                    scores_for_storage[mask_to_cache]
                    if scores_for_storage is not None
                    else None
                )
        else:  # only update those found in the storage to cache.
            found_keys_in_storage = keys_for_storage[founds].contiguous()
            found_values_in_storage = values_for_storage[founds, :].contiguous()
            found_scores_in_storage = (
                scores_for_storage[founds].contiguous()
                if scores_for_storage is not None
                else None
            )
            keys_to_update = found_keys_in_storage
            values_to_update = found_values_in_storage
            scores_to_update = found_scores_in_storage

        update_cache(cache, storage, keys_to_update, values_to_update, scores_to_update)
        return

    @staticmethod
    def update(
        cache: Cache,
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_grads: torch.Tensor,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        h_num_keys_for_storage, missing_keys, missing_indices = cache.update(
            unique_keys, unique_grads
        )
        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys
        grads_for_storage = unique_grads[missing_indices, :].contiguous()

        if storage.enable_update():
            storage.update(keys_for_storage, grads_for_storage, return_missing=False)
            return

        emb_dtype = storage.embedding_dtype()
        val_dim = storage.value_dim()
        values_for_storage = torch.empty(
            h_num_keys_for_storage, val_dim, device=unique_keys.device, dtype=emb_dtype
        )
        founds = torch.empty(
            h_num_keys_for_storage, device=unique_keys.device, dtype=torch.bool
        )
        _, _, _, _ = storage.find(keys_for_storage, values_for_storage, founds=founds)
        keys_for_storage = keys_for_storage[founds].contiguous()
        values_for_storage = values_for_storage[founds, :].contiguous()
        grads_for_storage = grads_for_storage[founds, :].contiguous()
        optimizer.fused_update(
            grads_for_storage,
            values_for_storage,
        )

        storage.insert(keys_for_storage, values_for_storage)
        return

    @staticmethod
    def prefetch(
        cache: Cache,
        storage: Storage,
        unique_keys: torch.Tensor,
        initializer: BaseDynamicEmbInitializer,
        training: bool = True,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        assert cache is not None, "prefetch is available only when caching is enabled."
        emb_dtype = storage.embedding_dtype()
        h_num_keys_for_storage, missing_keys, _, _ = cache.find_missed_keys(unique_keys)

        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys

        val_dim = storage.value_dim()
        emb_dim = storage.embedding_dim()
        values_for_storage = torch.empty(
            h_num_keys_for_storage, val_dim, device=unique_keys.device, dtype=emb_dtype
        )
        founds = torch.empty(
            h_num_keys_for_storage, device=unique_keys.device, dtype=torch.bool
        )
        (
            num_missing_in_storage,
            missing_keys_in_storage,
            missing_indices_in_storage,
            _,
        ) = storage.find(keys_for_storage, values_for_storage, founds=founds)

        if num_missing_in_storage != 0:
            if training:
                embs_for_storage = values_for_storage[:, :emb_dim]
                initializer(
                    embs_for_storage,
                    missing_indices_in_storage,
                    keys_for_storage,
                )
                if val_dim != emb_dim:
                    values_for_storage[
                        missing_indices_in_storage, emb_dim - val_dim :
                    ] = storage.init_optimizer_state()
            else:
                keys_for_storage = keys_for_storage[founds].contiguous()
                values_for_storage = values_for_storage[founds, :].contiguous()

        update_cache(
            cache,
            storage,
            keys_for_storage,
            values_for_storage,
            None,  # prefetch does not update scores
        )
