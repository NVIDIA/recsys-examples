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
import os
import shutil

import gin
import torch
from commons.utils.stringify import stringify_dict
from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from dataset import get_data_loader
from dataset.inference_dataset import InferenceDataset
from dataset.sequence_dataset import get_dataset
from modules.metrics import get_multi_event_metric_module
from preprocessor import get_common_preprocessors
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import DatasetArgs, NetworkArgs, RankingArgs

sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR

import modules.paged_hstu_infer_layer as pg
from modules.paged_hstu_infer_layer import init

class RunningMode(enum.Enum):
    EVAL = "eval"
    SIMULATE = "simulate"

    def __str__(self):
        return self.value


def get_inference_dataset_and_embedding_configs():
    dataset_args = DatasetArgs()
    embedding_dim = NetworkArgs().hidden_size
    HASH_SIZE = 10_000_000
    if dataset_args.dataset_name == "kuairand-1k":
        embedding_configs = [
            InferenceEmbeddingConfig(
                feature_names=["user_id"],
                table_name="user_id",
                vocab_size=1000,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["video_id"],
                table_name="video_id",
                vocab_size=HASH_SIZE,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["action_weights"],
                table_name="action_weights",
                vocab_size=233,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
        ]
        return dataset_args, embedding_configs

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")


def get_inference_hstu_model(
    emb_configs,
    max_batch_size,
    num_contextual_features,
    total_max_seqlen,
    checkpoint_dir,
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
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=num_contextual_features,
        scaling_seqlen=network_args.scaling_seqlen,
    )

    kvcache_args = {
        "blocks_in_primary_pool": 10240,
        "page_size": 32,
        "offload_chunksize": 1024,
        "max_batch_size": max_batch_size,
        "max_seq_len": math.ceil(total_max_seqlen / 32) * 32,
    }
    kv_cache_config = get_kvcache_config(**kvcache_args)

    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
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

    model = InferenceRankingGR(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
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


def get_new_batch(
    batch, hist_lengths, ratio, num_contextuals
):
    partial_lengths = torch.ceil(hist_lengths * ratio).long() - num_contextuals
    partial_lengths = partial_lengths // 2

    kjt_dict = batch.features.to_dict()
    item_jt = kjt_dict["video_id"]
    vals = item_jt.values()
    lens = item_jt.lengths()
    num_candidates = batch.num_candidates
    split_lens = torch.stack(
        [partial_lengths + num_candidates, lens - partial_lengths - num_candidates], dim=1
    ).reshape((-1,))
    stripped_vals = torch.split(vals, split_lens.tolist())[::2]
    kjt_dict["video_id"] = JaggedTensor.from_dense(stripped_vals)

    action_jt = kjt_dict["action_weights"]
    vals = action_jt.values()
    lens = action_jt.lengths()
    split_lens = torch.stack(
        [partial_lengths, lens - partial_lengths], dim=1
    ).reshape((-1,))
    stripped_vals = torch.split(vals, split_lens.tolist())[::2]
    kjt_dict["action_weights"] = JaggedTensor.from_dense(stripped_vals)

    batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
    hist_lengths = num_contextuals + partial_lengths * 2

    return batch, hist_lengths
    


def run_kvcache_consistency_check(
    checkpoint_dir: str,
    disable_kvcache: bool = False,
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs()

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = len(dataproc._contextual_feature_names)

    max_batch_size = 1
    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    def strip_candidate_action_tokens(batch, action_feature_name):
        kjt_dict = batch.features.to_dict()
        action_jagged_tensor = kjt_dict[action_feature_name]
        values = action_jagged_tensor.values()
        lengths = action_jagged_tensor.lengths()
        num_candidates = batch.num_candidates
        split_lengths = torch.stack(
            [lengths - num_candidates, num_candidates], dim=1
        ).reshape((-1,))
        stripped_value = torch.split(values, split_lengths.tolist())[::2]
        kjt_dict[action_feature_name] = JaggedTensor.from_dense(stripped_value)
        batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        return batch

    def strip_padding_batch(batch, unpadded_batch_size):
        batch.batch_size = unpadded_batch_size
        kjt_dict = batch.features.to_dict()
        for k in kjt_dict:
            kjt_dict[k] = JaggedTensor.from_dense_lengths(
                kjt_dict[k].to_padded_dense()[: batch.batch_size],
                kjt_dict[k].lengths()[: batch.batch_size].long(),
            )
        batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        batch.num_candidates = batch.num_candidates[: batch.batch_size]
        return batch

    with torch.inference_mode():
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            checkpoint_dir,
        )

        eval_module = get_multi_event_metric_module(
            num_classes=model._task_config.prediction_head_arch[-1],
            num_tasks=model._task_config.num_tasks,
            metric_types=model._task_config.eval_metrics,
        )

        train_dataset, _ = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_sequence_length=dataset_args.max_sequence_length,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=model._task_config.num_tasks,
            batch_size=max_batch_size,
            rank=0,
            world_size=1,
            shuffle=False,
            random_seed=0,
            eval_batch_size=max_batch_size,
        )

        dataloader = get_data_loader(dataset=train_dataset)

        num_kvc_test_rounds = 2
        
        # torch.cuda.memory._record_memory_history()
        # torch.cuda.profiler.start()
        for round_id in [0, 1]:
            dataloader_iter = iter(dataloader)

            length_ratio = (round_id + 1) / num_kvc_test_rounds
            while True:
                try:
                    batch = next(dataloader_iter)
                    if model._task_config.num_tasks > 0:
                        batch = strip_candidate_action_tokens(
                            batch, dataproc._action_feature_name
                        )
                    
                    batch = batch.to(device=torch.cuda.current_device())

                    d = batch.features.to_dict()
                    user_ids = d["user_id"].values().cpu().long()
                    if user_ids.shape[0] != batch.batch_size:
                        batch = strip_padding_batch(batch, user_ids.shape[0])
                    total_history_lengths = torch.sum(batch.features.lengths().view(-1, batch.batch_size), 0).view(-1) - batch.num_candidates

                    if round_id != num_kvc_test_rounds - 1:
                        batch, total_history_lengths = get_new_batch(batch, total_history_lengths, length_ratio, num_contextual_features)

                    # if int(user_ids[0]) == 0:
                    #     pg.dmp = True
                    if not disable_kvcache:
                        logits = model.forward(batch, user_ids, total_history_lengths.cpu())
                    else:
                        logits = model.forward_nokvcache(batch)

                    if pg.dmp:
                        if disable_kvcache:
                            for lidx in range(model._hstu_config.num_layers):
                                if user_ids[0] < 10 or user_ids[0] >= 690:
                                    shutil.move(f"/tmp/in_l{lidx}.npy", f"dump/round{round_id}_user{user_ids[0]}_in_l{lidx}.npy")
                                    shutil.move(f"/tmp/key_l{lidx}.npy", f"dump/round{round_id}_user{user_ids[0]}_key_l{lidx}.npy")
                                    shutil.move(f"/tmp/value_l{lidx}.npy", f"dump/round{round_id}_user{user_ids[0]}_value_l{lidx}.npy")
                                    shutil.move(f"/tmp/attn_l{lidx}.npy", f"dump/round{round_id}_user{user_ids[0]}_attn_l{lidx}.npy")
                                    shutil.move(f"/tmp/out_l{lidx}.npy", f"dump/round{round_id}_user{user_ids[0]}_out_l{lidx}.npy")

                                else:
                                    os.remove(f"/tmp/key_l{lidx}.npy")
                                    os.remove(f"/tmp/value_l{lidx}.npy")
                        else:
                            for lidx in range(model._hstu_config.num_layers):
                                if user_ids[0] < 10 or user_ids[0] >= 690:
                                    shutil.move(f"/tmp/in_l{lidx}.npy", f"cached/round{round_id}_user{user_ids[0]}_in_l{lidx}.npy")
                                    shutil.move(f"/tmp/key_l{lidx}.npy", f"cached/round{round_id}_user{user_ids[0]}_key_l{lidx}.npy")
                                    shutil.move(f"/tmp/value_l{lidx}.npy", f"cached/round{round_id}_user{user_ids[0]}_value_l{lidx}.npy")
                                    shutil.move(f"/tmp/attn_l{lidx}.npy", f"cached/round{round_id}_user{user_ids[0]}_attn_l{lidx}.npy")
                                    shutil.move(f"/tmp/out_l{lidx}.npy", f"cached/round{round_id}_user{user_ids[0]}_out_l{lidx}.npy")
                                else:
                                    os.remove(f"/tmp/key_l{lidx}.npy")
                                    os.remove(f"/tmp/value_l{lidx}.npy")
                    pg.dmp = False

                    if round_id == num_kvc_test_rounds - 1:
                        eval_module(logits, batch.labels)
                except StopIteration:
                    break
        # torch.cuda.profiler.stop()
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

        eval_metric_dict = eval_module.compute()
        print(
            f"[eval]:\n    "
            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
        )
        # print("X")

if __name__ == "__main__":
    init()
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--disable_kvcache", action="store_true")
    # parser.add_argument("--max_bs", type=int, required=True)


    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    run_kvcache_consistency_check(
        checkpoint_dir=args.checkpoint_dir,
        disable_kvcache=args.disable_kvcache,
    )
    print("Finished.")
