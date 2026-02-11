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
import argparse
import enum
import math
import sys
import time
from typing import Dict, List, Optional

import gin
import torch
from commons.utils.stringify import stringify_dict
from configs import (
    EmbeddingBackend,
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
# from datasets import get_data_loader
# from datasets.inference_dataset import InferenceDataset
# from datasets.sequence_dataset import get_dataset
# from modules.metrics import get_multi_event_metric_module
from preprocessor import get_common_preprocessors
from modules.dynamicemb_embeddingcollection import (
    DynamicemBackendConfig,
)
from modules.inference_embedding import (
    InferenceEmbedding,
    apply_inference_embedding,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig, dtype_to_data_type
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import DatasetArgs, NetworkArgs, RankingArgs

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr


class RunningMode(enum.Enum):
    EVAL = "eval"

    def __str__(self):
        return self.value


def get_inference_dataset_and_embedding_configs(
    disable_contextual_features: bool = False,
):
    dataset_args = DatasetArgs()
    embedding_dim = NetworkArgs().hidden_size
    HASH_SIZE = 10_000_000
    if dataset_args.dataset_name == "kuairand-1k":
        embedding_configs = [
            EmbeddingConfig(
                name="user_id",
                embedding_dim=embedding_dim,
                num_embeddings=1000,
                feature_names=["user_id"],
                data_type=dtype_to_data_type(torch.float32),
            ),
            EmbeddingConfig(
                name="user_active_degree",
                embedding_dim=embedding_dim,
                num_embeddings=8,
                feature_names=["user_active_degree"],
                data_type=dtype_to_data_type(torch.float32),
            ),
            EmbeddingConfig(
                name="follow_user_num_range",
                embedding_dim=embedding_dim,
                num_embeddings=9,
                feature_names=["follow_user_num_range"],
                data_type=dtype_to_data_type(torch.float32),
            ),
            EmbeddingConfig(
                name="fans_user_num_range",
                embedding_dim=embedding_dim,
                num_embeddings=9,
                feature_names=["fans_user_num_range"],
                data_type=dtype_to_data_type(torch.float32),
            ),
            EmbeddingConfig(
                name="friend_user_num_range",
                embedding_dim=embedding_dim,
                num_embeddings=8,
                feature_names=["friend_user_num_range"],
                data_type=dtype_to_data_type(torch.float32),
            ),
            EmbeddingConfig(
                name="register_days_range",
                embedding_dim=embedding_dim,
                num_embeddings=8,
                feature_names=["register_days_range"],
                data_type=dtype_to_data_type(torch.float32),
            ),
            EmbeddingConfig(
                name="video_id",
                embedding_dim=embedding_dim,
                num_embeddings=HASH_SIZE,
                feature_names=["video_id"],
                data_type=dtype_to_data_type(torch.float32),
            ),
            EmbeddingConfig(
                name="action_weights",
                embedding_dim=embedding_dim,
                num_embeddings=233,
                feature_names=["action_weights"],
                data_type=dtype_to_data_type(torch.float32),
            ),
        ]
        dynamic_emb_tables = ["user_id", "video_id"]
        return (
            dataset_args,
            embedding_configs
            if not disable_contextual_features
            else embedding_configs[-2:],
            dynamic_emb_tables,
        )

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")


def get_inference_hstu_model(
    emb_configs,
    max_batch_size,
    num_contextual_features,
    total_max_seqlen,
    checkpoint_dir,
    use_kvcache,
):
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    # elif network_args.dtype_str == "float16":
    #     inference_dtype = torch.float16
    else:
        raise ValueError(
            f"Inference data type {network_args.dtype_str} is not supported"
        )

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
        static_max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
    )

    hstu_config = get_inference_hstu_config(
        hidden_size=network_args.hidden_size,
        num_layers=network_args.num_layers,
        num_attention_heads=network_args.num_attention_heads,
        head_dim=network_args.kv_channels,
        max_batch_size=max_batch_size,
        max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=num_contextual_features,
    )

    kvcache_args = {
        "blocks_in_primary_pool": 10240,
        "page_size": 32,
        "offload_chunksize": 1024,
    }
    kv_cache_config = get_kvcache_config(**kvcache_args)

    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=[ emb_configs ],
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )

    hstu_cudagraph_configs = {
        "batch_size": [1],
        "length_per_sequence": [128] + [i * 256 for i in range(1, 34)],
    }

    model = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config if use_kvcache else None,
        task_config=task_config,
        use_cudagraph=False,
        cudagraph_configs=hstu_cudagraph_configs,
    )
    if hstu_config.bf16:
        model.bfloat16()
    elif hstu_config.fp16:
        model.half()
    model.load_checkpoint(checkpoint_dir)
    model.eval()

    return model


def run_ranking_gr(
    checkpoint_dir: str,
    disable_contextual_features: bool = False,
    disable_kvcache: bool = False,
):
    dataset_args, emb_configs, dynamic_table_names = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = 4
    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    with torch.inference_mode():
        use_kvcache = not disable_kvcache
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            checkpoint_dir,
            use_kvcache,
        )

        print()
        for name, module in model.named_modules():
            print(name, type(module))
        
        inference_emb_configs = {
            "sparse_module": [
                DynamicemBackendConfig(
                    backend=EmbeddingBackend.DYNAMICEMB,
                    device=torch.cuda.current_device(),
                    caching=False,
                    bucket_capacity=128,
                    gpu_ratio_for_values=0.0,
                )
            ],
        }
        apply_inference_embedding(model, inference_emb_configs, dynamic_table_names)

        print()
        for name, module in model.named_modules():
            print(name, type(module))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--mode", type=RunningMode, choices=list(RunningMode)
    )
    parser.add_argument("--disable_auc", action="store_true")
    parser.add_argument("--disable_context", action="store_true")
    parser.add_argument("--disable_kvcache", action="store_true")

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    if args.disable_auc:
        print("disable_auc is ignored in Eval mode.")
    if args.disable_context:
        print("disable_context is ignored in Eval mode.")
    run_ranking_gr(
        checkpoint_dir=args.checkpoint_dir, disable_kvcache=args.disable_kvcache
    )
