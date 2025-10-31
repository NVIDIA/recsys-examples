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

import math
import os
import random
import shutil
from itertools import product
from typing import Any, Dict, List, Tuple

import click
import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb import DynamicEmbScoreStrategy, DynamicEmbTableOptions
from dynamicemb.dump_load import (
    DynamicEmbDump,
    DynamicEmbLoad,
    find_sharded_modules,
    get_dynamic_emb_module,
)
from dynamicemb.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
from dynamicemb.get_planner import get_planner
from dynamicemb.key_value_table import batched_export_keys_values
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from dynamicemb.utils import TORCHREC_TYPES
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torchrec import DataType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def idx_to_name(embedding_collection_idx: int, embedding_idx: int) -> Tuple[str, str]:
    return (
        f"feature_{embedding_collection_idx}_{embedding_idx}",
        f"embedding_collection_{embedding_collection_idx}_{embedding_idx}",
    )


def get_optimizer_kwargs(optimizer_type: str) -> Dict[str, Any]:
    if optimizer_type == "sgd":
        return {"optimizer": EmbOptimType.SGD, "lr": 1e-3}
    elif optimizer_type == "adam":
        return {"optimizer": EmbOptimType.ADAM, "lr": 1e-3}
    elif optimizer_type == "adagrad":
        return {"optimizer": EmbOptimType.EXACT_ADAGRAD, "lr": 1e-3}
    elif optimizer_type == "rowwise_adagrad":
        return {"optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD, "lr": 1e-3}
    else:
        raise ValueError("unknown optimizer type")


def get_score_strategy(score_strategy_str: str) -> DynamicEmbScoreStrategy:
    if score_strategy_str == "timestamp":
        return DynamicEmbScoreStrategy.TIMESTAMP
    elif score_strategy_str == "step":
        return DynamicEmbScoreStrategy.STEP
    elif score_strategy_str == "lfu":
        return DynamicEmbScoreStrategy.LFU
    else:
        raise ValueError(f"Invalid score strategy: {score_strategy_str}")


def generate_sparse_feature(
    num_embedding_collections: int,
    num_embeddings: List[int],
    multi_hot_sizes: List[int],
    rank: int,
    world_size: int,
    batch_size: int,
    num_iterations: int,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size_per_rank = batch_size // world_size
    kjts = []
    all_kjts = []
    for _ in range(num_iterations):
        cur_indices, cur_lengths = [], []
        all_indices, all_lengths = [], []
        keys = []
        for embedding_collection_id in range(num_embedding_collections):
            for embedding_id, num_embedding in enumerate(num_embeddings):
                feature_name, _ = idx_to_name(embedding_collection_id, embedding_id)
                for sample_id in range(batch_size):
                    hotness = random.randint(
                        0, multi_hot_sizes[embedding_collection_id]
                    )
                    indices = [random.randint(0, (1 << 63) - 1) for _ in range(hotness)]
                    all_indices.extend(indices)
                    all_lengths.append(hotness)
                    if sample_id // batch_size_per_rank == rank:
                        cur_indices.extend(indices)
                        cur_lengths.append(hotness)
                keys.append(feature_name)
        kjts.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(cur_indices, dtype=torch.int64).cuda(),
                lengths=torch.tensor(cur_lengths, dtype=torch.int64).cuda(),
            )
        )
        all_kjts.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(all_indices, dtype=torch.int64).cuda(),
                lengths=torch.tensor(all_lengths, dtype=torch.int64).cuda(),
            )
        )
    return kjts, keys, all_kjts


class TestModel(nn.Module):
    def __init__(
        self,
        embedding_modules: List[EmbeddingCollection],
    ):
        super().__init__()
        self.embedding_modules = nn.ModuleList(embedding_modules)

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        embeddings_dict = [
            embedding_module(kjt).wait() for embedding_module in self.embedding_modules
        ]
        embeddings = []
        for embedding_dict in embeddings_dict:
            for embedding in embedding_dict.values():
                embeddings.append(embedding.values())
        return torch.cat(embeddings, dim=0)


DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
    DataType.FP32: 32,
    DataType.FP16: 16,
    DataType.BF16: 16,
}


def apply_dmp(
    model: torch.nn.Module,
    optimizer_kwargs: Dict[str, Any],
    device: torch.device,
    score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.LFU,
    use_index_dedup: bool = False,
    caching: bool = False,
    cache_capacity_ratio: float = 0.5,
):
    eb_configs = []
    dynamicemb_options_dict = {}
    for n, m in model.named_modules():
        if type(m) in TORCHREC_TYPES:
            eb_configs.extend(m.embedding_configs())
            for eb_config in eb_configs:
                dim = eb_config.embedding_dim
                tmp_type = eb_config.data_type

                embedding_type_bytes = DATA_TYPE_NUM_BITS[tmp_type] / 8
                emb_num_embeddings = (
                    eb_config.num_embeddings * cache_capacity_ratio
                    if caching
                    else eb_config.num_embeddings
                )
                emb_num_embeddings_next_power_of_2 = 2 ** math.ceil(
                    math.log2(emb_num_embeddings)
                )  # HKV need embedding vector num is power of 2

                # Calculate optimizer state dimension
                from dynamicemb.dynamicemb_config import (
                    data_type_to_dtype,
                    get_optimizer_state_dim,
                )
                from dynamicemb_extensions import OptimizerType

                # Map fbgemm EmbOptimType to dynamicemb OptimizerType
                emb_opt_type = (
                    optimizer_kwargs.get("optimizer") if optimizer_kwargs else None
                )
                opt_type_map = {
                    EmbOptimType.EXACT_ROWWISE_ADAGRAD: OptimizerType.RowWiseAdaGrad,
                    EmbOptimType.SGD: OptimizerType.SGD,
                    EmbOptimType.EXACT_SGD: OptimizerType.SGD,
                    EmbOptimType.ADAM: OptimizerType.Adam,
                    EmbOptimType.EXACT_ADAGRAD: OptimizerType.AdaGrad,
                }
                opt_type = opt_type_map.get(emb_opt_type) if emb_opt_type else None
                # Convert torchrec DataType to torch.dtype
                torch_dtype = data_type_to_dtype(tmp_type)
                optimizer_state_dim = (
                    get_optimizer_state_dim(opt_type, dim, torch_dtype)
                    if opt_type
                    else 0
                )

                # Include optimizer state in HBM calculation
                total_hbm_need = (
                    embedding_type_bytes
                    * (dim + optimizer_state_dim)
                    * emb_num_embeddings_next_power_of_2
                )

                dynamicemb_options_dict[eb_config.name] = DynamicEmbTableOptions(
                    global_hbm_for_values=total_hbm_need,
                    score_strategy=score_strategy,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.CONSTANT,
                        value=1e-1,
                    ),
                    bucket_capacity=emb_num_embeddings_next_power_of_2,
                    max_capacity=emb_num_embeddings_next_power_of_2,
                    caching=caching,
                    local_hbm_for_values=1024**3,
                )
    planner = get_planner(
        eb_configs,
        {},
        dynamicemb_options_dict,
        device,
    )

    fused_params = {}
    fused_params["output_dtype"] = SparseType.FP32
    fused_params.update(optimizer_kwargs)

    sharder = DynamicEmbeddingCollectionSharder(
        fused_params=fused_params,
        use_index_dedup=use_index_dedup,
    )
    plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)

    # Same usage of TorchREC
    dmp = DistributedModelParallel(
        module=model,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
    )
    return dmp


def create_model(
    num_embedding_collections: int,
    num_embeddings: List[int],
    embedding_dim: int,
    optimizer_kwargs: Dict[str, Any],
    score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.LFU,
    use_index_dedup: bool = False,
    caching: bool = False,
    cache_capacity_ratio: float = 0.5,
):
    ebc_list = []
    for embedding_collection_id in range(num_embedding_collections):
        eb_configs = []
        for embedding_id, num_embedding in enumerate(num_embeddings):
            feature_name, embedding_name = idx_to_name(
                embedding_collection_id, embedding_id
            )
            eb_configs.append(
                EmbeddingConfig(
                    name=embedding_name,
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embedding,
                    feature_names=[feature_name],
                    data_type=DataType.FP32,
                )
            )
        ebc_list.append(
            EmbeddingCollection(
                device=torch.device("meta"),
                tables=eb_configs,
            )
        )
    model = TestModel(
        embedding_modules=ebc_list,
    )

    model = apply_dmp(
        model,
        optimizer_kwargs,
        torch.device(f"cuda:{torch.cuda.current_device()}"),
        score_strategy=score_strategy,
        use_index_dedup=use_index_dedup,
        caching=caching,
        cache_capacity_ratio=cache_capacity_ratio,
    )
    return model


@click.command()
@click.option("--num-embedding-collections", type=int, required=True)
@click.option("--num-embeddings", type=str, required=True)
@click.option("--multi-hot-sizes", type=str, required=True)
@click.option("--embedding-dim", type=int, required=True)
@click.option("--save-path", type=str, required=True)
@click.option(
    "--optimizer-type",
    type=click.Choice(["sgd", "adam", "adagrad", "rowwise_adagrad"]),
    required=True,
)
@click.option("--mode", type=click.Choice(["load", "dump"]), required=True)
@click.option(
    "--score-strategy",
    type=click.Choice(["timestamp", "step", "lfu"]),
    required=True,
)
@click.option("--optim", type=bool, required=True)
def test_model_load_dump(
    num_embedding_collections: int,
    num_embeddings: str,
    multi_hot_sizes: str,
    embedding_dim: int,
    optimizer_type: str,
    score_strategy: str,
    mode: str,
    save_path: str,
    optim: bool,
    batch_size: int = 8,
    num_iterations: int = 2,
):
    num_embeddings = [int(v) for v in num_embeddings.split(",")]
    multi_hot_sizes = [int(v) for v in multi_hot_sizes.split(",")]

    for num_embedding, multi_hot_size in zip(num_embeddings, multi_hot_sizes):
        if batch_size * num_iterations * multi_hot_size > num_embedding:
            raise ValueError(
                "batch_size * num_iterations * multi_hot_size > num_embedding, this may lead to eviction of dynamicemb and cause test fail"
            )

    optimizer_kwargs = get_optimizer_kwargs(optimizer_type)
    score_strategy_ = get_score_strategy(score_strategy)

    ref_model = create_model(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        optimizer_kwargs=optimizer_kwargs,
        score_strategy=score_strategy_,
    )

    kjts, feature_names, all_kjts = generate_sparse_feature(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        multi_hot_sizes=multi_hot_sizes,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        batch_size=batch_size,
        num_iterations=num_iterations,
    )

    for kjt in kjts:
        ret = ref_model(kjt)
        loss = (
            ret.sum() * dist.get_world_size()
        )  # scale the loss by world size to make the gradients consistent between different gpu settings
        loss.backward()

    if mode == "dump":
        shutil.rmtree(save_path, ignore_errors=True)
        DynamicEmbDump(save_path, ref_model, optim=optim)

    if mode == "load":
        model = create_model(
            num_embedding_collections=num_embedding_collections,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            optimizer_kwargs=optimizer_kwargs,
            score_strategy=score_strategy_,
        )

        DynamicEmbLoad(save_path, model, optim=optim)

        table_name_to_key_score_dict = {}
        for _, _, sharded_module in find_sharded_modules(model):
            dynamic_emb_modules = get_dynamic_emb_module(sharded_module)
            for dynamic_emb_module in dynamic_emb_modules:
                for table_name, table in zip(
                    dynamic_emb_module.table_names, dynamic_emb_module.tables
                ):
                    key_to_score = {}
                    for batched_key, _, _, batched_score in batched_export_keys_values(
                        table.table, torch.device(f"cpu")
                    ):
                        for key, score in zip(
                            batched_key.tolist(), batched_score.tolist()
                        ):
                            key_to_score[key] = score
                    table_name_to_key_score_dict[table_name] = key_to_score

        for embedding_collection_idx, embedding_idx in product(
            range(num_embedding_collections), range(len(num_embeddings))
        ):
            feature_name, table_name = idx_to_name(
                embedding_collection_idx, embedding_idx
            )
            key_to_score_dict = table_name_to_key_score_dict[table_name].copy()

            # The idea is that the score of a newer key is greater than that of an older key. Therefore, I iterate through the input in reverse order and track the minimum score encountered. For each batch, the score should be lower than the minimum score from the previous batch. To avoid issues caused by duplicate keys, every time I access a key, I set its score to -inf. This ensures that if that key appears again, its score will be sufficiently small to remain below the minimum score.
            min_score = float("inf")
            for kjt in reversed(all_kjts):
                keys = kjt[feature_name].values().tolist()
                for key in keys:
                    if key not in key_to_score_dict and dist.world_size() > 1:
                        continue
                    score = key_to_score_dict[key]
                    assert (
                        score <= min_score
                    ), f"key {key} score {score} should be <= min_score {min_score}"

                for key in set(keys):
                    if key not in key_to_score_dict and dist.world_size() > 1:
                        continue
                    min_score = min(min_score, key_to_score_dict[key])
                    key_to_score_dict[key] = float("-inf")

        if optim:
            for kjt in kjts:
                ret = model(kjt)
                ret.sum().backward()
                ref_ret = ref_model(kjt)
                ref_ret.sum().backward()

        ref_model = ref_model.eval()
        model = model.eval()

        with torch.inference_mode():
            for kjt in kjts:
                ret = model(kjt)
                ref_ret = ref_model(kjt)
                assert torch.allclose(ret, ref_ret)


if __name__ == "__main__":
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(backend="nccl")
    test_model_load_dump()
    dist.destroy_process_group()
