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
import random
from typing import Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import pytest
import torch
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import DynamicEmbStorage, HybridStorage, Storage
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)

POOLING_MODE: Dict[DynamicEmbPoolingMode, PoolingMode] = {
    DynamicEmbPoolingMode.NONE: PoolingMode.NONE,
    DynamicEmbPoolingMode.MEAN: PoolingMode.MEAN,
    DynamicEmbPoolingMode.SUM: PoolingMode.SUM,
}
OPTIM_TYPE: Dict[EmbOptimType, OptimType] = {
    EmbOptimType.SGD: OptimType.EXACT_SGD,
    EmbOptimType.ADAM: OptimType.ADAM,
    EmbOptimType.EXACT_ADAGRAD: OptimType.EXACT_ADAGRAD,
    EmbOptimType.EXACT_ROWWISE_ADAGRAD: OptimType.EXACT_ROWWISE_ADAGRAD,
}


class PyDictStorage(Storage[DynamicEmbTableOptions, BaseDynamicEmbeddingOptimizer]):
    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self.options = options
        self.dict: Dict[Tuple[int, int], torch.Tensor] = {}
        self.scores: Dict[Tuple[int, int], int] = {}
        self.capacity = max(o.max_capacity for o in options)
        self.optimizer = optimizer

        self._emb_dims = [o.dim for o in options]
        self._emb_dtype = options[0].embedding_dtype
        self._value_dims = [o.dim + optimizer.get_state_dim(o.dim) for o in options]
        self._optstate_dims = [optimizer.get_state_dim(o.dim) for o in options]
        self._max_emb_dim = max(self._emb_dims)
        self._max_value_dim = max(self._value_dims)
        self._max_optstate_dim = max(self._optstate_dims)
        self._initial_optim_state = optimizer.get_initial_optim_states()

        device_idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{device_idx}")

    def size(self) -> int:
        return len(self.dict)

    def find_impl(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_embs: torch.Tensor,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h_unique_keys = unique_keys.cpu()
        h_table_ids = table_ids.cpu()
        lookup_dim = unique_embs.size(1)
        results = []
        missing_keys = []
        missing_indices = []
        missing_scores_list = []
        founds_ = []
        for i in range(h_unique_keys.size(0)):
            key = h_unique_keys[i].item()
            tid = h_table_ids[i].item()
            composite_key = (tid, key)
            if composite_key in self.dict:
                results.append(self.dict[composite_key][0:lookup_dim])
                founds_.append(True)
            else:
                missing_keys.append(key)
                missing_indices.append(i)
                if input_scores is not None:
                    missing_scores_list.append(input_scores[i].item())
                founds_.append(False)
        founds_ = torch.tensor(founds_, dtype=torch.bool, device=self.device)
        if len(results) > 0:
            unique_embs[founds_, :] = torch.cat(
                [t.unsqueeze(0) for t in results], dim=0
            )

        num_missing = torch.tensor(
            [len(missing_keys)], dtype=torch.long, device=self.device
        )
        missing_keys = torch.tensor(
            missing_keys, dtype=unique_keys.dtype, device=self.device
        )
        missing_indices = torch.tensor(
            missing_indices, dtype=torch.long, device=self.device
        )

        if input_scores is not None and len(missing_scores_list) > 0:
            missing_scores = torch.tensor(
                missing_scores_list, dtype=input_scores.dtype, device=self.device
            )
        else:
            missing_scores = torch.empty(0, dtype=torch.uint64, device=self.device)

        # output_scores: scores for all keys (0 for missing, input_scores for found)
        output_scores = torch.zeros(
            unique_keys.size(0), dtype=torch.int64, device=self.device
        )
        if input_scores is not None:
            output_scores[founds_] = input_scores[founds_].to(torch.int64)

        return (
            num_missing,
            missing_keys,
            missing_indices,
            missing_scores,
            founds_,
            output_scores,
        )

    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_embs: torch.Tensor,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        return self.find_impl(unique_keys, table_ids, unique_embs, input_scores)

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        input_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if copy_mode == CopyMode.EMBEDDING:
            out_dim = self._max_emb_dim
        else:
            out_dim = self._max_value_dim
        unique_vals = torch.zeros(
            unique_keys.size(0), out_dim, dtype=self._emb_dtype, device=self.device
        )
        (
            num_missing,
            missing_keys,
            missing_indices,
            missing_scores,
            founds,
            output_scores,
        ) = self.find_impl(unique_keys, table_ids, unique_vals, input_scores)

        h_table_ids = table_ids.cpu()
        missing_table_ids_list = []
        for idx in missing_indices.cpu().tolist():
            missing_table_ids_list.append(h_table_ids[idx].item())
        missing_table_ids = torch.tensor(
            missing_table_ids_list, dtype=torch.int64, device=self.device
        )

        return (
            num_missing.item()
            if isinstance(num_missing, torch.Tensor)
            else num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores if missing_scores.numel() > 0 else None,
            founds,
            output_scores,
            unique_vals,
        )

    def insert(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        h_keys = keys.cpu()
        h_table_ids = table_ids.cpu()
        h_scores = scores.cpu() if scores is not None and scores.numel() > 0 else None
        for i in range(h_keys.size(0)):
            key = h_keys[i].item()
            tid = h_table_ids[i].item()
            composite_key = (tid, key)
            self.dict[composite_key] = values[i, :].clone()
            if h_scores is not None:
                self.scores[composite_key] = h_scores[i].item()

    def update(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        grads: torch.Tensor,
        return_missing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise ValueError("Can't call update of PyDictStorage")

    def enable_update(self) -> bool:
        return False

    def dump(
        self,
        table_id: int,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = False,
        include_meta: bool = False,
    ) -> None:
        if include_meta:
            meta_data = {}
            meta_data.update(self.optimizer.get_opt_args())
            with open(meta_file_path, "w") as f:
                json.dump(meta_data, f)

        fkey = open(emb_key_path, "wb")
        fembedding = open(embedding_file_path, "wb")
        fscore = open(score_file_path, "wb") if score_file_path else None
        fopt_states = open(opt_file_path, "wb") if include_optim else None

        for keys, embeddings, opt_states, scores_out in self.export_keys_values(
            table_id, self.device
        ):
            fkey.write(keys.cpu().numpy().tobytes())
            if fscore is not None:
                fscore.write(scores_out.cpu().numpy().tobytes())
            fembedding.write(embeddings.cpu().numpy().tobytes())
            if fopt_states is not None and opt_states is not None:
                fopt_states.write(opt_states.cpu().numpy().tobytes())

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def load(
        self,
        table_id: int,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = False,
    ) -> None:
        if meta_file_path and os.path.exists(meta_file_path):
            with open(meta_file_path, "r") as f:
                meta_data = json.load(f)
            opt_type = meta_data.get("opt_type", None)
            if (
                opt_type
                and self.optimizer.get_opt_args().get("opt_type", None) != opt_type
            ):
                include_optim = False
            if include_optim:
                self.optimizer.set_opt_args(meta_data)

        if not opt_file_path or not os.path.exists(opt_file_path):
            include_optim = False

        dim = self._emb_dims[table_id]
        optstate_dim = self._optstate_dims[table_id]
        value_dim = self._value_dims[table_id]

        num_keys = os.path.getsize(emb_key_path) // 8  # int64

        fkey = open(emb_key_path, "rb")
        fembedding = open(embedding_file_path, "rb")
        fscore = (
            open(score_file_path, "rb")
            if score_file_path and os.path.exists(score_file_path)
            else None
        )
        fopt_states = open(opt_file_path, "rb") if include_optim else None

        batch_size = 65536
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)

            keys_bytes = fkey.read(8 * n)
            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=np.int64).copy(),
                dtype=torch.int64,
                device=self.device,
            )

            emb_bytes = fembedding.read(4 * dim * n)
            embeddings = torch.tensor(
                np.frombuffer(emb_bytes, dtype=np.float32).copy(),
                dtype=torch.float32,
                device=self.device,
            ).view(-1, dim)

            opt_states = None
            if fopt_states and optstate_dim > 0:
                opt_bytes = fopt_states.read(4 * optstate_dim * n)
                opt_states = torch.tensor(
                    np.frombuffer(opt_bytes, dtype=np.float32).copy(),
                    dtype=torch.float32,
                    device=self.device,
                ).view(-1, optstate_dim)

            scores = None
            if fscore:
                score_bytes = fscore.read(8 * n)
                scores = torch.tensor(
                    np.frombuffer(score_bytes, dtype=np.int64).copy(),
                    dtype=torch.int64,
                    device=self.device,
                )

            # Build full values tensor [N, value_dim]
            if opt_states is not None:
                values = torch.cat([embeddings, opt_states], dim=1)
            else:
                if value_dim > dim:
                    values = torch.empty(
                        n, value_dim, dtype=torch.float32, device=self.device
                    )
                    values[:, :dim] = embeddings
                    values[:, dim:] = self._initial_optim_state
                else:
                    values = embeddings

            tid = torch.full(
                (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
            )
            self.insert(keys, tid, values, scores)

        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt_states:
            fopt_states.close()

    def export_keys_values(
        self,
        table_id: int,
        device: torch.device,
        batch_size: int = 65536,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        """Yield (keys, embeddings, opt_states, scores) batches for a given table."""
        emb_dim = self._emb_dims[table_id]
        optstate_dim = self._optstate_dims[table_id]
        all_keys = [ck for ck in self.dict.keys() if ck[0] == table_id]
        for start in range(0, len(all_keys), batch_size):
            batch_keys = all_keys[start : start + batch_size]

            keys_t = torch.tensor(
                [ck[1] for ck in batch_keys], dtype=torch.int64, device=device
            )
            values_t = torch.stack(
                [self.dict[ck].to(device) for ck in batch_keys], dim=0
            )
            embeddings = values_t[:, :emb_dim].contiguous()

            opt_states = None
            if optstate_dim > 0:
                opt_states = values_t[:, emb_dim : emb_dim + optstate_dim].contiguous()

            scores_list = [self.scores.get(ck, 0) for ck in batch_keys]
            scores_t = torch.tensor(scores_list, dtype=torch.int64, device=device)

            yield keys_t, embeddings, opt_states, scores_t

    def embedding_dtype(
        self,
    ) -> torch.dtype:
        return self._emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._emb_dims[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._value_dims[table_id]

    def max_embedding_dim(self) -> int:
        return self._max_emb_dim

    def max_value_dim(self) -> int:
        return self._max_value_dim

    def init_optimizer_state(
        self,
    ) -> float:
        return self._initial_optim_state


def create_split_table_batched_embedding(
    table_names,
    feature_table_map,
    optimizer_type,
    opt_params,
    dims,
    num_embs,
    pooling_mode,
    device,
):
    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                e,
                d,
                EmbeddingLocation.DEVICE,
                ComputeDevice.CUDA,
            )
            for (e, d) in zip(num_embs, dims)
        ],
        optimizer=optimizer_type,
        weights_precision=SparseType.FP32,
        stochastic_rounding=False,
        pooling_mode=pooling_mode,
        output_dtype=SparseType.FP32,
        device=device,
        table_names=table_names,
        feature_table_map=feature_table_map,
        **opt_params,
        bounds_check_mode=BoundsCheckMode.FATAL,
    ).cuda()
    return emb


def init_embedding_tables(stbe, bdet):
    stbe.init_embedding_weights_uniform(0, 1)
    storage = bdet.tables
    optimizer = bdet.optimizer
    for table_idx, split in enumerate(stbe.split_embedding_weights()):
        num_emb = split.size(0)
        emb_dim = split.size(1)
        opt_state_dim = optimizer.get_state_dim(emb_dim)
        val_dim = emb_dim + opt_state_dim
        indices = torch.arange(num_emb, device=split.device, dtype=torch.long)
        table_ids = torch.full(
            (num_emb,), table_idx, dtype=torch.int64, device=split.device
        )
        if isinstance(storage, DynamicEmbStorage):
            max_emb_dim = storage.max_embedding_dim()
            max_value_dim = storage.max_value_dim()
            values = torch.zeros(
                num_emb, max_value_dim, dtype=split.dtype, device=split.device
            )
            values[:, :emb_dim] = split
            if opt_state_dim > 0:
                values[
                    :, max_emb_dim : max_emb_dim + opt_state_dim
                ] = storage.init_optimizer_state()
            storage.set_score(1)
            storage.insert(indices, table_ids, values)
        elif isinstance(storage, PyDictStorage):
            pydict = cast(PyDictStorage, storage)
            values = torch.empty(
                num_emb, val_dim, dtype=split.dtype, device=split.device
            )
            values[:, :emb_dim] = split
            if val_dim > emb_dim:
                values[:, emb_dim:] = pydict.init_optimizer_state()
            pydict.insert(indices, table_ids, values)
        else:
            raise ValueError("Not support table type")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "deterministic",
    [True, False],
)
@pytest.mark.parametrize(
    "caching, PS, local_hbm_for_values",
    [
        (True, None, 1024),  # Caching + DynamicEmbStorage backend
        (True, PyDictStorage, 1024),  # Caching + PS backend
        (False, None, 1024**3),  # HBM-only
        (False, None, 1024),  # HybridStorage
    ],
)
@pytest.mark.parametrize(
    "pooling_mode, dims",
    [
        (DynamicEmbPoolingMode.NONE, [8, 8, 8]),
        (DynamicEmbPoolingMode.NONE, [7, 7, 7]),
        (DynamicEmbPoolingMode.SUM, [8, 8, 8]),
        (DynamicEmbPoolingMode.MEAN, [8, 8, 8]),
        (DynamicEmbPoolingMode.SUM, [8, 16, 32]),
        (DynamicEmbPoolingMode.MEAN, [8, 16, 32]),
        (DynamicEmbPoolingMode.SUM, [7, 11, 13]),
        (DynamicEmbPoolingMode.MEAN, [7, 11, 13]),
    ],
)
def test_forward_train_eval(
    opt_type,
    opt_params,
    caching,
    deterministic,
    PS,
    local_hbm_for_values,
    pooling_mode,
    dims,
):
    print(
        f"step in test_forward_train_eval , opt_type = {opt_type} opt_params = {opt_params}"
        f" caching = {caching} PS = {PS} local_hbm_for_values = {local_hbm_for_values}"
        f" pooling_mode = {pooling_mode} dims = {dims}"
    )

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    feature_table_map = [0, 0, 1, 2]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=local_hbm_for_values,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    if caching:
        assert bdebt._cache is not None, "Caching mode should create a cache"
    elif local_hbm_for_values == 1024**3:
        assert bdebt._cache is None, "HBM-only mode should have no cache"
        assert isinstance(
            bdebt._storage, DynamicEmbStorage
        ), f"Expected DynamicEmbStorage, got {type(bdebt._storage)}"
    else:
        assert bdebt._cache is None, "HybridStorage mode should have no cache"
        assert isinstance(
            bdebt._storage, HybridStorage
        ), f"Expected HybridStorage, got {type(bdebt._storage)}"

    """
    feature number = 4, batch size = 2

    f0  [0,1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    indices = torch.tensor(
        [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], dtype=key_type, device=device
    )
    offsets = torch.tensor(
        [0, 2, 3, 5, 6, 8, 10, 10, 11], dtype=key_type, device=device
    )
    batch_size = 2

    embs_train = bdebt(indices, offsets)
    torch.cuda.synchronize()

    # Verify output shape
    if pooling_mode == DynamicEmbPoolingMode.NONE:
        assert embs_train.shape == (indices.numel(), dims[0])
    else:
        total_D = sum(dims[feature_table_map[f]] for f in range(len(feature_table_map)))
        assert embs_train.shape == (batch_size, total_D)

    with torch.no_grad():
        bdebt.eval()
        embs_eval = bdebt(indices, offsets)
    torch.cuda.synchronize()

    # Train and eval should produce identical results for the same keys
    torch.testing.assert_close(embs_train, embs_eval)

    # non-exist key: replace index 0 (key=0) with key=777
    indices_ne = torch.tensor(
        [777, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
    ).to(key_type)
    offsets_ne = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(
        key_type
    )
    embs_non_exist = bdebt(indices_ne, offsets_ne)
    torch.cuda.synchronize()

    # train
    bdebt.train()
    embs_train_non_exist = bdebt(indices_ne, offsets_ne)
    torch.cuda.synchronize()

    if pooling_mode == DynamicEmbPoolingMode.NONE:
        # Sequence mode: row 0 is the embedding for key 777
        # In eval, non-exist key -> zero embedding
        torch.testing.assert_close(embs_train[1:, :], embs_non_exist[1:, :])
        assert torch.all(embs_non_exist[0, :] == 0)
        # In train, non-exist key gets initialized -> non-zero
        assert torch.all(embs_train_non_exist[0, :] != 0)
        torch.testing.assert_close(embs_train_non_exist[1:, :], embs_non_exist[1:, :])
    else:
        # Pooled mode: sample 1 is unaffected by the non-exist key (key 777
        # only appears in sample 0's f0 bag).
        torch.testing.assert_close(embs_non_exist[1, :], embs_train[1, :])
        torch.testing.assert_close(embs_train_non_exist[1, :], embs_non_exist[1, :])
        # Sample 0 should differ from the original because key 777 replaced
        # key 0 in f0's bag.  In eval the missing key contributes zero, so
        # the pooled result for sample 0 changes compared to embs_train.
        assert not torch.allclose(embs_non_exist[0, :], embs_train[0, :])

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]

    print("all check passed")


"""
For torchrec's adam optimizer, it will increment the optimizer_step in every forward,
    which will affect the weights update, pay attention to it or try to use `set_optimizer_step()`
    to control(not verified) it.
"""


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "caching, pooling_mode, dims",
    [
        (True, DynamicEmbPoolingMode.NONE, [8, 8, 8]),
        (False, DynamicEmbPoolingMode.NONE, [16, 16, 16]),
        (False, DynamicEmbPoolingMode.NONE, [17, 17, 17]),
        (False, DynamicEmbPoolingMode.SUM, [128, 32, 16]),
        (False, DynamicEmbPoolingMode.MEAN, [4, 8, 16]),
    ],
)
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None, PyDictStorage])
def test_backward(opt_type, opt_params, caching, pooling_mode, dims, deterministic, PS):
    print(f"step in test_backward , opt_type = {opt_type} opt_params = {opt_params}")

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048

    dyn_emb_table_options_list = []
    cmp_with_torchrec = True
    for dim in dims:
        if dim % 4 != 0:
            cmp_with_torchrec = False
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        **opt_params,
    )
    num_embs = [max_capacity // 2 for d in dims]

    if cmp_with_torchrec:
        stbe = create_split_table_batched_embedding(
            table_names,
            feature_table_map,
            OPTIM_TYPE[opt_type],
            opt_params,
            dims,
            num_embs,
            POOLING_MODE[pooling_mode],
            device,
        )
        init_embedding_tables(stbe, bdeb)
        """
        feature number = 4, batch size = 2

        f0  [0,1],      [12],
        f1  [64,8],     [12],
        f2  [15, 2, 7], [105],
        f3  [],         [0]
        """
        for i in range(10):
            indices = torch.tensor(
                [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
            ).to(key_type)
            offsets = torch.tensor([0, 2, 3, 5, 6, 9, 10, 10, 11], device=device).to(
                key_type
            )

            embs_bdeb = bdeb(indices, offsets)
            embs_stbe = stbe(indices, offsets)

            torch.cuda.synchronize()
            with torch.no_grad():
                torch.testing.assert_close(embs_bdeb, embs_stbe, rtol=1e-06, atol=1e-06)

            loss = embs_bdeb.mean()
            loss.backward()
            loss_stbe = embs_stbe.mean()
            loss_stbe.backward()

            torch.cuda.synchronize()
            torch.testing.assert_close(loss, loss_stbe)

            print(f"Passed iteration {i}")
    else:
        # This scenario will not test correctness, but rather test whether it functions correctly.
        for i in range(10):
            indices = torch.tensor(
                [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
            ).to(key_type)
            offsets = torch.tensor([0, 2, 3, 5, 6, 9, 10, 10, 11], device=device).to(
                key_type
            )

            embs_bdeb = bdeb(indices, offsets)
            loss = embs_bdeb.mean()
            loss.backward()

            torch.cuda.synchronize()

            print(f"Passed iteration {i}")

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
        (
            EmbOptimType.EXACT_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None, PyDictStorage])
def test_prefetch_flush_in_cache(opt_type, opt_params, deterministic, PS):
    print(
        f"step in test_prefetch_flush , opt_type = {opt_type} opt_params = {opt_params}"
    )
    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048
    dims = [8, 8, 8]

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.STEP,
            caching=True,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        enable_prefetch=False,
        **opt_params,
    )
    bdeb.enable_prefetch = True
    bdeb.set_record_cache_metrics(True)

    num_embs = [max_capacity // 2 for d in dims]
    stbe = create_split_table_batched_embedding(
        table_names,
        feature_table_map,
        OPTIM_TYPE[opt_type],
        opt_params,
        dims,
        num_embs,
        POOLING_MODE[DynamicEmbPoolingMode.NONE],
        device,
    )
    init_embedding_tables(stbe, bdeb)

    forward_stream = torch.cuda.Stream()
    pretch_stream = torch.cuda.Stream()

    # 1. Prepare input
    # Input A
    """
    feature number = 4, batch size = 2

    f0  [0, 1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    indicesA = torch.tensor([0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device).to(
        key_type
    )
    offsetsA = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(key_type)

    # Input B
    # A intersection B is not none
    """
    feature number = 4, batch size = 2

    f0  [4, 12],        [55],
    f1  [2, 17],        [1],
    f2  [],             [5, 13, 105],
    f3  [0, 23],        [42]
    """
    indicesB = torch.tensor(
        [4, 12, 55, 2, 17, 1, 5, 13, 105, 0, 23, 42], device=device
    ).to(key_type)
    offsetsB = torch.tensor([0, 2, 3, 5, 6, 6, 9, 11, 12], device=device).to(key_type)

    # stream capture will bring a cudaMalloc.
    with torch.cuda.stream(forward_stream):
        indicesB + 1
    with torch.cuda.stream(pretch_stream):
        indicesB + 1

    # 2. Test prefetch works when Cache empty
    with torch.cuda.stream(pretch_stream):
        bdeb.prefetch(indicesA, offsetsA, forward_stream)
        assert list(bdeb.get_score().values()) == [1] * len(dims)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(pretch_stream)
        embs_bdeb_A = bdeb(indicesA, offsetsA)
        loss_bdet_A = embs_bdeb_A.mean()
        loss_bdet_A.backward()

    embs_stbe_A = stbe(indicesA, offsetsA)
    loss_stbe_A = embs_stbe_A.mean()
    loss_stbe_A.backward()

    with torch.no_grad():
        torch.cuda.synchronize()
        torch.testing.assert_close(embs_bdeb_A, embs_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_A, loss_stbe_A, rtol=1e-06, atol=1e-06)

        cache = bdeb.cache
        if cache is not None:
            metrics = cache.cache_metrics
            assert metrics[0].item() == metrics[1].item()

    with torch.no_grad():
        bdeb.flush()
        bdeb.reset_cache_states()
        # bdeb.set_score({table_name:1 for table_name in table_names})

    # 3. Test prefetch works when Cache not empty
    with torch.cuda.stream(pretch_stream):
        bdeb.prefetch(indicesA, offsetsA, forward_stream)
        bdeb.prefetch(indicesB, offsetsB, forward_stream)
        assert list(bdeb.get_score().values()) == [2] * len(dims)

    with torch.cuda.stream(forward_stream):
        torch.cuda.current_stream().wait_stream(pretch_stream)
        embs_bdeb_A = bdeb(indicesA, offsetsA)
        loss_bdet_A = embs_bdeb_A.mean()
        loss_bdet_A.backward()
        embs_bdeb_B = bdeb(indicesB, offsetsB)
        loss_bdet_B = embs_bdeb_B.mean()
        loss_bdet_B.backward()

    embs_stbe_A = stbe(indicesA, offsetsA)
    loss_stbe_A = embs_stbe_A.mean()
    loss_stbe_A.backward()
    embs_stbe_B = stbe(indicesB, offsetsB)
    loss_stbe_B = embs_stbe_B.mean()
    loss_stbe_B.backward()

    with torch.no_grad():
        torch.cuda.synchronize()
        torch.testing.assert_close(embs_bdeb_A, embs_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_A, loss_stbe_A, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(embs_bdeb_B, embs_stbe_B, rtol=1e-06, atol=1e-06)
        torch.testing.assert_close(loss_bdet_B, loss_stbe_B, rtol=1e-06, atol=1e-06)

        cache = bdeb.cache
        if cache is not None:
            metrics = cache.cache_metrics
            assert metrics[0].item() == metrics[1].item()

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]


def random_indices(batch, min_index, max_index):
    result = set({})
    while len(result) < batch:
        result.add(random.randint(min_index, max_index))
    return result


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
    ],
)
@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("PS", [None])
@pytest.mark.parametrize("iteration", [16])
@pytest.mark.parametrize("batch_size", [2048, 65536])  # ,[])
def test_deterministic_insert(opt_type, opt_params, caching, PS, iteration, batch_size):
    print(
        f"step in test_deterministic_insert , opt_type = {opt_type} opt_params = {opt_params}"
    )

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [8]
    table_names = ["table0"]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = iteration * batch_size
    max_capacity = init_capacity

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=init_capacity * dim * 4,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt_x = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    bdebt_y = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )

    print(
        f"Test deterministic insert with batch={batch_size}, iteration={iteration}, capacity={init_capacity}"
    )
    os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    for i in range(iteration):
        indices = torch.tensor(
            list(random_indices(batch_size, 0, 2**63 - 1)),
            dtype=key_type,
            device=device,
        )
        offsets = torch.arange(0, batch_size + 1, dtype=key_type, device=device)

        bdebt_x(indices, offsets)
        bdebt_y(indices, offsets)

        torch.cuda.synchronize()

        storage_x = bdebt_x.tables
        storage_y = bdebt_y.tables
        map_x = storage_x.key_index_map
        map_y = storage_y.key_index_map

        assert torch.equal(map_x.keys_, map_y.keys_)

        print(
            f"Iteration {i} passed for deterministic insertion with table_x's size({map_x.size()}), table_y's size({map_y.size()}), totoal({map_x.capacity()})"
        )
        cache_x = bdebt_x.cache
        cache_y = bdebt_y.cache
        if cache_x is not None:
            map_x = cache_x.key_index_map
            map_y = cache_y.key_index_map

            assert torch.equal(map_x.keys_, map_y.keys_)

            print(
                f"Iteration {i} passed for deterministic insertion with cache_x's size({map_x.size()}), cache_y's size({map_y.size()}), totoal({map_x.capacity()})"
            )

    del os.environ["DEMB_DETERMINISM_MODE"]
    print("all check passed")


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            {
                "learning_rate": 0.3,
                "eps": 3e-5,
            },
        ),
    ],
)
@pytest.mark.parametrize("dim", [7, 8])
@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("PS", [None])
def test_empty_batch(opt_type, opt_params, dim, caching, deterministic, PS):
    print(
        f"step in test_forward_train_eval_empty_batch , opt_type = {opt_type} opt_params = {opt_params}"
    )

    if deterministic:
        os.environ["DEMB_DETERMINISM_MODE"] = "ON"

    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [dim, dim, dim]
    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = 1024
    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            caching=caching,
            local_hbm_for_values=1024**3,
            external_storage=PS,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0, 0, 1, 2],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )
    bdebt.enable_prefetch = True
    """
    feature number = 4, batch size = 1

    f0  [],     
    f1  [],  
    f2  [],  
    f3  [],       
    """
    indices = torch.tensor([], dtype=key_type, device=device)
    offsets = torch.tensor([0, 0, 0, 0, 0], dtype=key_type, device=device)

    pretch_stream = torch.cuda.Stream()
    forward_stream = torch.cuda.Stream()

    if caching:
        with torch.cuda.stream(pretch_stream):
            bdebt.prefetch(indices, offsets, forward_stream)
            torch.cuda.synchronize()

    with torch.cuda.stream(forward_stream):
        res = bdebt(indices, offsets)
        torch.cuda.synchronize()

        res.mean().backward()

        with torch.no_grad():
            bdebt.eval()
            bdebt(indices, offsets)
        torch.cuda.synchronize()

    if deterministic:
        del os.environ["DEMB_DETERMINISM_MODE"]

    print("all check passed")
