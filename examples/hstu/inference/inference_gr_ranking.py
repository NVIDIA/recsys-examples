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

import gin
import torch
from commons.datasets import get_data_loader
from commons.datasets.hstu_sequence_dataset import get_dataset
from commons.datasets.inference_dataset import InferenceDataset
from commons.hstu_data_preprocessor import get_common_preprocessors
from commons.utils.stringify import stringify_dict
from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from modules.metrics import get_multi_event_metric_module
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import DatasetArgs, NetworkArgs, RankingArgs

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr


class RunningMode(enum.Enum):
    EVAL = "eval"
    SIMULATE = "simulate"

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
        return (
            dataset_args,
            embedding_configs
            if not disable_contextual_features
            else embedding_configs[-2:],
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

    model = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config if use_kvcache else None,
        task_config=task_config,
        use_cudagraph=True,
        cudagraph_configs=hstu_cudagraph_configs,
    )
    if hstu_config.bf16:
        model.bfloat16()
    elif hstu_config.fp16:
        model.half()
    model.load_checkpoint(checkpoint_dir)
    model.eval()

    return model


def run_ranking_gr_simulate(
    checkpoint_dir: str,
    check_auc: bool = False,
    disable_contextual_features: bool = False,
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = 1
    total_max_seqlen = (
        dataset_args.max_num_candidates + dataset_args.max_history_seqlen
    ) * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    with torch.inference_mode():
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            checkpoint_dir,
            True,
        )

        if check_auc:
            eval_module = get_multi_event_metric_module(
                num_classes=model.get_num_class(),
                num_tasks=model.get_num_tasks(),
                metric_types=model.get_metric_types(),
            )

        dataset = InferenceDataset(
            seq_logs_file=dataproc._inference_sequence_file,
            batch_logs_file=dataproc._inference_batch_file,
            batch_size=max_batch_size,
            max_seqlen=dataset_args.max_history_seqlen,
            item_feature_name=dataproc._item_feature_name,
            contextual_feature_names=dataproc._contextual_feature_names
            if not disable_contextual_features
            else [],
            action_feature_name=dataproc._action_feature_name,
            max_num_candidates=dataset_args.max_num_candidates,
            item_vocab_size=10_000_000,
            userid_name="user_id",
            date_name="date",
            sequence_endptr_name="interval_indptr",
            timestamp_names=["date", "interval_end_ts"],
        )

        dataloader = get_data_loader(dataset=dataset)
        dataloader_iter = iter(dataloader)

        num_batches_ctr = 0
        start_time = time.time()
        cur_date = None
        while True:
            try:
                uids, dates, seq_endptrs = next(dataloader_iter)
                if dates[0] != cur_date:
                    if cur_date is not None:
                        eval_metric_dict = eval_module.compute()
                        print(
                            f"[eval]:\n    "
                            + stringify_dict(
                                eval_metric_dict, prefix="Metrics", sep="\n    "
                            )
                        )
                    model.clear_kv_cache()
                    cur_date = dates[0]
                cached_start_pos, cached_len = model.get_user_kvdata_info(
                    uids, dbg_print=True
                )
                new_cache_start_pos = cached_start_pos + cached_len
                non_contextual_mask = new_cache_start_pos >= num_contextual_features
                contextual_mask = torch.logical_not(non_contextual_mask)
                seq_startptrs = (
                    torch.clip(new_cache_start_pos - num_contextual_features, 0) / 2
                ).int()

                batch_0 = dataset.get_input_batch(
                    uids[non_contextual_mask],
                    dates[non_contextual_mask],
                    seq_endptrs[non_contextual_mask],
                    seq_startptrs[non_contextual_mask],
                    with_contextual_features=False,
                    with_ranking_labels=True,
                )
                if batch_0 is not None:
                    logits = model.forward(
                        batch_0,
                        uids[non_contextual_mask].int(),
                        new_cache_start_pos[non_contextual_mask],
                    )
                    eval_module(logits, batch_0.labels.values())

                batch_1 = dataset.get_input_batch(
                    uids[contextual_mask],
                    dates[contextual_mask],
                    seq_endptrs[contextual_mask],
                    seq_startptrs[contextual_mask],
                    with_contextual_features=True,
                    with_ranking_labels=True,
                )
                if batch_1 is not None:
                    logits = model.forward(
                        batch_1,
                        uids[contextual_mask].int(),
                        new_cache_start_pos[contextual_mask],
                    )
                    eval_module(logits, batch_1.labels.values())

                num_batches_ctr += 1
            except StopIteration:
                break
        end_time = time.time()
        print("Total #batch:", num_batches_ctr)
        print("Total time(s):", end_time - start_time)


def run_ranking_gr_evaluate(
    checkpoint_dir: str,
    disable_contextual_features: bool = False,
    disable_kvcache: bool = False,
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = 4
    total_max_seqlen = (
        dataset_args.max_num_candidates + dataset_args.max_history_seqlen
    ) * 2 + num_contextual_features
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
        use_kvcache = not disable_kvcache
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            checkpoint_dir,
            use_kvcache,
        )

        eval_module = get_multi_event_metric_module(
            num_classes=model.get_num_class(),
            num_tasks=model.get_num_tasks(),
            metric_types=model.get_metric_types(),
        )

        _, eval_dataset = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_history_seqlen=dataset_args.max_history_seqlen,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=model.get_num_tasks(),
            batch_size=max_batch_size,
            rank=0,
            world_size=1,
            shuffle=False,
            random_seed=0,
            eval_batch_size=max_batch_size,
        )

        dataloader = get_data_loader(dataset=eval_dataset)
        dataloader_iter = iter(dataloader)

        while True:
            try:
                batch = next(dataloader_iter)
                if model.get_num_tasks() > 0:
                    batch = strip_candidate_action_tokens(
                        batch, dataproc._action_feature_name
                    )

                batch = batch.to(device=torch.cuda.current_device())
                d = batch.features.to_dict()
                user_ids = d["user_id"].values().cpu().int()
                seq_startpos = torch.zeros_like(user_ids)

                if user_ids.shape[0] != batch.batch_size:
                    batch = strip_padding_batch(batch, user_ids.shape[0])

                logits = model.forward(batch, user_ids, seq_startpos)
                eval_module(logits, batch.labels.values())
            except StopIteration:
                break

        eval_metric_dict = eval_module.compute()
        print(
            f"[eval]:\n    "
            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--mode", type=RunningMode, choices=list(RunningMode), required=True
    )
    parser.add_argument("--disable_auc", action="store_true")
    parser.add_argument("--disable_context", action="store_true")
    parser.add_argument("--disable_kvcache", action="store_true")

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    if args.mode == RunningMode.EVAL:
        if args.disable_auc:
            print("disable_auc is ignored in Eval mode.")
        if args.disable_context:
            print("disable_context is ignored in Eval mode.")
        run_ranking_gr_evaluate(
            checkpoint_dir=args.checkpoint_dir, disable_kvcache=args.disable_kvcache
        )
    elif args.mode == RunningMode.SIMULATE:
        if args.disable_kvcache:
            print("disable_kvcache is ignored in Eval mode.")
        run_ranking_gr_simulate(
            checkpoint_dir=args.checkpoint_dir,
            check_auc=not args.disable_auc,
            disable_contextual_features=args.disable_context,
        )
