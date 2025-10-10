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
from typing import Callable, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import (
    DynamicEmbTableOptions,
    create_dynamicemb_table,
    dyn_emb_to_torch,
    torch_to_dyn_emb,
)
from dynamicemb.initializer import BaseDynamicEmbInitializer
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizerV2
from dynamicemb_extensions import (
    EvictStrategy,
    clear,
    count_matched,
    dyn_emb_rows,
    export_batch,
    export_batch_matched,
    find_pointers,
    insert_and_evict,
    insert_or_assign,
    load_from_pointers,
    select,
    select_index,
)


class Storage(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        options: DynamicEmbTableOptions,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        pass

    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def insert(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def update(
        self, keys: torch.Tensor, grads: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def enable_update(self) -> bool:
        ...

    @abc.abstractmethod
    def dump(
        self,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_dumped: torch.Tensor
        dumped_keys: torch.Tensor
        dumped_values: torch.Tensor
        dumped_scores: torch.Tensor
        return num_dumped, dumped_keys, dumped_values, dumped_scores

    @abc.abstractmethod
    def load(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        pass

    @abc.abstractmethod
    def embedding_dtype(
        self,
    ) -> torch.dtype:
        pass

    @abc.abstractmethod
    def embedding_dim(
        self,
    ) -> int:
        pass

    @abc.abstractmethod
    def value_dim(
        self,
    ) -> int:
        pass

    @abc.abstractmethod
    def init_optimizer_state(
        self,
    ) -> float:
        pass


class Cache(abc.ABC):
    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_missed_keys(
        self,
        unique_keys: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def insert_and_evict(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_evicted: torch.Tensor
        evicted_keys: torch.Tensor
        evicted_values: torch.Tensor
        evicted_scores: torch.Tensor
        return num_evicted, evicted_keys, evicted_values, evicted_scores

    @abc.abstractmethod
    def update(
        self, keys: torch.Tensor, grads: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def flush(self, storage: Storage) -> None:
        pass

    @abc.abstractmethod
    def reset(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def cache_metrics(
        self,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def set_record_cache_metrics(self, record: bool) -> None:
        pass


class KeyValueTable(Cache, Storage):
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

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = unique_keys.size(0)
        assert unique_embs.dim() == 2
        assert unique_embs.size(0) == batch

        load_dim = unique_embs.size(1)

        device = unique_keys.device
        if founds is None:
            founds = torch.empty(batch, dtype=torch.bool, device=device)
        pointers = torch.empty(batch, dtype=torch.long, device=device)

        if self._score_update:
            # TODO: support score
            find_pointers(self.table, batch, unique_keys, pointers, founds, self.score)
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

        return num_missing_0, missing_keys, missing_indices

    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.find_impl(unique_keys, unique_embs, founds)

    def find_missed_keys(
        self,
        unique_keys: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # dummy tensor
        unique_embs = torch.empty(
            unique_keys.numel(), 0, device=unique_keys.device, dtype=self._emb_dtype
        )
        return self.find_impl(unique_keys, unique_embs, founds)

    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.find_impl(unique_keys, unique_vals, founds)

    def insert(
        self,
        unique_keys: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        h_num_unique_keys = unique_keys.size(0)
        if self._use_score:
            if scores is None:
                scores = torch.empty(
                    h_num_unique_keys, device=unique_keys.device, dtype=torch.uint64
                )
                scores.fill_(self.score)
        else:
            scores = None

        insert_or_assign(
            self.table, h_num_unique_keys, unique_keys, unique_values, scores
        )

    def update(
        self,
        keys: torch.Tensor,
        grads: torch.Tensor,
        return_missing: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self._score_update == False, "update is called only in backward."

        batch = keys.size(0)

        device = keys.device
        founds = torch.empty(batch, dtype=torch.bool, device=device)
        pointers = torch.empty(batch, dtype=torch.long, device=device)
        find_pointers(self.table, batch, keys, pointers, founds)

        self.optimizer.fused_update_with_pointer(grads, pointers, self._de_emb_dtype)

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
            return num_missing_0, missing_keys, missing_indices
        else:
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

    def dump(
        self,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = end - start
        device = self.device
        key_dtype = self.options.index_type
        value_dtype = self._emb_dtype
        dim: int = self._value_dim

        num_dumped: torch.Tensor = torch.zeros(1, dtype=torch.uint64, device=device)
        dumped_keys: torch.Tensor = torch.empty(batch, dtype=key_dtype, device=device)
        dumped_values: torch.Tensor = torch.empty(
            batch, dim, dtype=value_dtype, device=device
        )
        dumped_scores: torch.Tensor = torch.empty(
            batch, dtype=torch.uint64, device=device
        )

        export_batch(
            self.table,
            batch,
            start,
            num_dumped,
            dumped_keys,
            dumped_values,
            dumped_scores,
        )

        return num_dumped, dumped_keys, dumped_values, dumped_scores

    def load(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        self.insert(keys, values, scores)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = keys.numel()
        num_evicted: torch.Tensor = torch.zeros(1, dtype=torch.long, device=keys.device)
        evicted_keys: torch.Tensor = torch.empty_like(keys)
        evicted_values: torch.Tensor = torch.empty_like(values)
        evicted_scores: torch.Tensor = torch.empty(
            batch, dtype=torch.uint64, device=keys.device
        )
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
        return num_evicted, evicted_keys, evicted_values, evicted_scores

    def flush(self, storage: Storage) -> None:
        batch_size = self._threads_in_wave
        for start in range(0, self._capacity, batch_size):
            end = min(start + batch_size, self._capacity)
            num_dumped, dumped_keys, dumped_values, dumped_scores = self.dump(
                start, end
            )
            h_num_dumped = num_dumped.cpu().item()
            dumped_keys = dumped_keys[:h_num_dumped]
            dumped_values = dumped_values[:h_num_dumped, :]
            dumped_scores = dumped_scores[:h_num_dumped]
            storage.insert(dumped_keys, dumped_values, dumped_scores)

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

    def export_batch(
        self, batch_size, search_offset, d_count, d_keys, d_vals, d_scores
    ) -> None:
        export_batch(
            self.table,
            batch_size,
            search_offset,
            d_count,
            d_keys,
            d_vals,
            d_scores,
        )

    def evict_strategy(self) -> EvictStrategy:
        return self.table.evict_strategy()

    def optim_state_dim(self) -> int:
        return self.value_dim() - self.embedding_dim()

    def size(self) -> int:
        return dyn_emb_rows(self.table)


def update_cache(
    cache: Cache,
    storage: Storage,
    missing_keys: torch.Tensor,
    missing_values: torch.Tensor,
):
    # need to update score.
    num_evicted, evicted_keys, evicted_values, evicted_scores = cache.insert_and_evict(
        missing_keys, missing_values
    )
    h_num_evicted = num_evicted.cpu().item()
    if h_num_evicted != 0:
        storage.insert(
            evicted_keys[:h_num_evicted],
            evicted_values[:h_num_evicted, :],
            evicted_scores[:h_num_evicted],
        )


class KeyValueTableFunction:
    @staticmethod
    def lookup(
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        initializer: Callable,
        training: bool,
    ) -> None:
        assert unique_keys.dim() == 1
        h_num_toatl = unique_keys.numel()
        emb_dim = storage.embedding_dim()
        emb_dtype = storage.embedding_dtype()
        val_dim = storage.value_dim()

        if h_num_toatl == 0:
            return

        # 1. find in storage
        founds = torch.empty(h_num_toatl, device=unique_keys.device, dtype=torch.bool)
        (
            num_missing_in_storage,
            missing_keys_in_storage,
            missing_indices_in_storage,
        ) = storage.find_embeddings(unique_keys, unique_embs, founds=founds)

        # 2. initialize missing embeddings
        h_num_missing_in_storage = num_missing_in_storage.cpu().item()
        missing_indices_in_storage = missing_indices_in_storage[
            :h_num_missing_in_storage
        ]
        missing_keys_in_storage = missing_keys_in_storage[:h_num_missing_in_storage]
        if h_num_missing_in_storage != 0:
            initializer(
                unique_embs,
                missing_indices_in_storage,
                unique_keys,
            )
        else:
            return

        # 3. insert missing values into table.
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
            storage.insert(missing_keys_in_storage, missing_values_in_storage)
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
        _, _, _ = storage.find(unique_keys, unique_values, founds=founds)

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
        cache: Cache,
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        initializer: Callable,
        enable_prefetch: bool,
        training: bool,
    ) -> None:
        assert unique_keys.dim() == 1
        unique_keys.numel()
        emb_dim = storage.embedding_dim()
        emb_dtype = storage.embedding_dtype()
        val_dim = storage.value_dim()

        # 1. find in cache
        num_missing, missing_keys, missing_indices = cache.find_embeddings(
            unique_keys, unique_embs
        )
        h_num_keys_for_storage = num_missing.cpu().item()
        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys[:h_num_keys_for_storage]
        missing_indices = missing_indices[:h_num_keys_for_storage]

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
            num_missing_in_storage,
            missing_keys_in_storage,
            missing_indices_in_storage,
        ) = storage.find(keys_for_storage, values_for_storage, founds=founds)

        # 3. initialize missing embeddings
        h_num_missing_in_storage = num_missing_in_storage.cpu().item()
        missing_indices_in_storage = missing_indices_in_storage[
            :h_num_missing_in_storage
        ]
        missing_keys_in_storage = missing_keys_in_storage[:h_num_missing_in_storage]
        if h_num_missing_in_storage != 0:
            initializer(
                values_for_storage[:, :emb_dim],
                missing_indices_in_storage,
                keys_for_storage,
            )

        # 4. copy embeddings to unique_embs
        unique_embs[missing_indices, :] = values_for_storage[:, :emb_dim]

        if training:
            if emb_dim != val_dim:
                values_for_storage[
                    missing_indices_in_storage, emb_dim - val_dim :
                ] = storage.init_optimizer_state()
            update_cache(cache, storage, keys_for_storage, values_for_storage)
        else:  # only update those found in the storage to cache.
            found_keys_in_storage = keys_for_storage[founds].contiguous()
            found_values_in_storage = values_for_storage[founds, :].contiguous()
            update_cache(cache, storage, found_keys_in_storage, found_values_in_storage)
        return

    @staticmethod
    def update(
        cache: Cache,
        storage: Storage,
        unique_keys: torch.Tensor,
        unique_grads: torch.Tensor,
        optimizer: BaseDynamicEmbeddingOptimizerV2,
    ):
        num_missing, missing_keys, missing_indices = cache.update(
            unique_keys, unique_grads
        )
        h_num_keys_for_storage = num_missing.cpu().item()
        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys[:h_num_keys_for_storage]
        # missing_indices = missing_indices[:h_num_keys_for_storage]
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
        _, _, _ = storage.find(keys_for_storage, values_for_storage, founds=founds)

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
        assert cache is not None
        emb_dtype = storage.embedding_dtype()
        num_missing, missing_keys, _ = cache.find_missed_keys(unique_keys)

        h_num_keys_for_storage = num_missing.cpu().item()
        if h_num_keys_for_storage == 0:
            return
        keys_for_storage = missing_keys[:h_num_keys_for_storage]

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
        ) = storage.find(keys_for_storage, values_for_storage, founds=founds)

        h_num_missing_in_storage = num_missing_in_storage.cpu().item()
        if h_num_missing_in_storage != 0:
            missing_indices_in_storage = missing_indices_in_storage[
                :h_num_missing_in_storage
            ]
            missing_keys_in_storage = missing_keys_in_storage[:h_num_missing_in_storage]

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
        )
