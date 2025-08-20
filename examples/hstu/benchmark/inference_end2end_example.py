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
import sys
import time

from commons.utils.stringify import stringify_dict
from dataclasses import dataclass
import gin
import torch
from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from dataset import get_data_loader
from dataset.inference_dataset import InferenceDataset
from dataset.random_inference_dataset import RandomInferenceDataGenerator
from dataset.utils import FeatureConfig
from modules.metrics import get_multi_event_metric_module
from preprocessor import get_common_preprocessors
from typing import List, Tuple, cast

sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR
sys.path.append("./training/")
from gin_config_args import TrainerArgs, EmbeddingArgs, DatasetArgs, NetworkArgs, OptimizerArgs

#duplicate
@gin.configurable
@dataclass
class RankingArgs:
    prediction_head_arch: List[int] = cast(List[int], None)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch for ranking model"
        if isinstance(self.prediction_head_act_type, str):
            assert self.prediction_head_act_type.lower() in [
                "relu",
                "gelu",
            ], "prediction_head_act_type should be in ['relu', 'gelu']"

def run_ranking_gr_inference(
    check_auc: bool = False
):
    max_batch_size = 16
    max_seqlen = 4096
    max_num_candidates = 256
    max_incremental_seqlen = 128

    # context_emb_size = 1000
    item_fea_name, item_vocab_size = "video_id", 10_000_000
    action_fea_name, action_vocab_size = "action_weights", 256
    feature_configs = [
        FeatureConfig(
            feature_names=[item_fea_name, action_fea_name],
            max_item_ids=[item_vocab_size - 1, action_vocab_size - 1],
            max_sequence_length=max_seqlen,
            is_jagged=False,
        ),
    ]
    max_contextual_seqlen = 0
    total_max_seqlen = sum(
        [fc.max_sequence_length * len(fc.feature_names) for fc in feature_configs]
    )
    print("total_max_seqlen", total_max_seqlen)

    hidden_dim_size = 1024
    num_heads = 4
    head_dim = 256
    num_layers = 8
    inference_dtype = torch.bfloat16
    hstu_cudagraph_configs = {
        "batch_size": [1],
        "length_per_sequence": [i * 256 for i in range(2, 34)],
    }

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
    )

    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
    )

    _blocks_in_primary_pool = 10240
    _page_size = 32
    _offload_chunksize = 8192
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=_blocks_in_primary_pool,
        page_size=_page_size,
        offload_chunksize=_offload_chunksize,
        max_batch_size=max_batch_size,
        max_seq_len=total_max_seqlen,
    )
    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=[action_fea_name],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["context_feat", item_fea_name]
            if max_contextual_seqlen > 0
            else [item_fea_name],
            table_name="item",
            vocab_size=item_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=True,
        ),
    ]

    # TODO: need to init & setup
    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )

    with torch.inference_mode():
        model_predict = InferenceRankingGR(
            hstu_config=hstu_config,
            kvcache_config=kv_cache_config,
            task_config=task_config,
            use_cudagraph=True,
            cudagraph_configs=hstu_cudagraph_configs,
        )
        model_predict.bfloat16()
        model_predict.eval()

        if check_auc:
            eval_module = get_multi_event_metric_module(
                num_classes=task_config.prediction_head_arch[-1],
                num_tasks=task_config.num_tasks,
                metric_types=task_config.eval_metrics,
            )

        dataproc = get_common_preprocessors("")["kuairand-1k"]
        dataproc._batching_file = dataproc._output_file.replace("processed_seqs", "processed_batches")

        dataset = InferenceDataset(
            seq_logs_file=dataproc._output_file,
            batch_logs_file=dataproc._batching_file,
            batch_size=1,
            max_seqlen=max_seqlen,
            item_feature_name=dataproc._item_feature_name,
            contextual_feature_names=dataproc._contextual_feature_names,
            action_feature_name=dataproc._action_feature_name,
            max_num_candidates=max_num_candidates,

            item_vocab_size=10_000_000,
            userid_name='user_id',
            date_name='date',
            sequence_endptr_name='interval_indptr',
            timestamp_names=['date', 'interval_end_ts'],
        )

        dataloader = get_data_loader(dataset=dataset)
        dataloader_iter = iter(dataloader)

        num_batches_ctr = 0
        num_skipped_ctr = 0
        seq_lengths = set()
        seqlen_histogram = dict()

        # ts_start, ts_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
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
                            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
                        )
                    model_predict.clear_kv_cache()
                    cur_date = dates[0]
                cached_start_pos, cached_len = model_predict.get_user_kvdata_info(uids)
                truncate_start_pos = cached_start_pos + cached_len
                batch = dataset.get_input_batch(uids, dates, seq_endptrs, (truncate_start_pos / 2).int().tolist(), False, with_ranking_labels = True)
                if batch is None:
                    num_skipped_ctr += 1
                    continue
                logits = model_predict.forward(batch, torch.tensor(uids).int(), truncate_start_pos)
                eval_module(logits, batch.labels)
                num_batches_ctr += 1
                if num_batches_ctr == 100000:
                    break
            except StopIteration:
                break
        end_time = time.time()
        print("Total:", num_batches_ctr, "skipped", num_skipped_ctr)
        print("Total inference:", num_batches_ctr - num_skipped_ctr)
        print("Total time(s):", end_time-start_time)        

if __name__ == "__main__":
    # TODO: change to args (argsparser)
    gin.parse_config_file("./kuairand_1k_inference_ranking.gin")
    check_auc = True
    run_ranking_gr_inference(check_auc = check_auc)
