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
from typing import Tuple

import commons.utils.initialize as init
import gin
import torch
import utils.sid_batch_balancer  # noqa: F401 - triggers batch shuffler registration
from commons.distributed.batch_shuffler_factory import BatchShufflerFactory
from commons.distributed.sharding import make_optimizer_and_shard
from commons.optimizer import OptimizerParam
from commons.pipeline import TrainPipelineFactory
from commons.utils.logger import print_rank_0
from configs.args_to_config import create_embedding_config
from configs.gpt_config import get_gpt_config
from configs.sid_gin_config_args import (
    DatasetArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    TensorModelParallelArgs,
    TrainerArgs,
)
from model import get_sid_gr_model
from trainer.training import maybe_load_ckpts, train_with_pipeline
from trainer.utils import get_train_and_test_data_loader


def get_dataset_and_embedding_args() -> Tuple[DatasetArgs, EmbeddingArgs]:
    dataset_args = DatasetArgs()  # type: ignore[call-arg]

    codebook_sizes = dataset_args.codebook_sizes
    aggragated_codebook_size = sum(codebook_sizes)
    # embedding feature names should match the dataset batch feature names
    embedding_args = EmbeddingArgs(  # sid tuples share one embedding table
        feature_names=[
            dataset_args._history_sid_feature_name,
            dataset_args._candidate_sid_feature_name,
        ],  # sid tuples share one embedding table
        table_name="codebook",
        item_vocab_size_or_capacity=aggragated_codebook_size,
        sharding_type="data_parallel",
    )

    return dataset_args, embedding_args


def create_optimizer_params(optimizer_args: OptimizerArgs):
    return OptimizerParam(
        optimizer_str=optimizer_args.optimizer_str,
        learning_rate=optimizer_args.learning_rate,
        adam_beta1=optimizer_args.adam_beta1,
        adam_beta2=optimizer_args.adam_beta2,
        adam_eps=optimizer_args.adam_eps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SID-GR Example Arguments", allow_abbrev=False
    )
    parser.add_argument("--gin-config-file", type=str)
    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)
    trainer_args = TrainerArgs()
    (
        dataset_args,
        embedding_args,
    ) = get_dataset_and_embedding_args()  # auto-set by gin-config
    network_args = NetworkArgs()
    # this is a kinda hard code.
    # when share_lm_head_across_hierarchies is True, we must deduplicate the label across hierarchy.
    dataset_args.deduplicate_label_across_hierarchy = (
        network_args.share_lm_head_across_hierarchies
    )

    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()

    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)
    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"distributed env initialization done. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )
    gpt_config = get_gpt_config(
        network_args.hidden_size,
        network_args.kv_channels,
        network_args.num_attention_heads,
        network_args.num_layers,
        torch.bfloat16,
        hidden_dropout=network_args.hidden_dropout,
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size,
        loss_on_history=dataset_args.max_candidate_length == 0,
    )
    embedding_config = create_embedding_config(network_args.hidden_size, embedding_args)
    model = get_sid_gr_model(
        decoder_config=gpt_config,
        codebook_embedding_config=embedding_config,
        codebook_sizes=dataset_args.codebook_sizes,
        num_hierarchies=dataset_args.num_hierarchies,
        normalization="RMSNorm",
        top_k_for_generation=trainer_args.top_k_for_generation,
        eval_metrics=trainer_args.eval_metrics,
        share_lm_head_across_hierarchies=network_args.share_lm_head_across_hierarchies,
    )

    optimizer_param = create_optimizer_params(optimizer_args)
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=gpt_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict={},
        pipeline_type=trainer_args.pipeline_type,
    )
    stateful_metric_module = None
    train_dataloader, test_dataloader = get_train_and_test_data_loader(
        dataset_args, trainer_args
    )
    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"model initialization done, start training. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )

    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)

    # Create batch shuffler based on configuration
    if trainer_args.enable_balanced_shuffler:
        batch_shuffler = BatchShufflerFactory.create(
            "sid_gr",
            num_heads=gpt_config.num_attention_heads,
            head_dim=gpt_config.kv_channels,
        )
    else:
        batch_shuffler = BatchShufflerFactory.create("identity")

    # Map pipeline type string to factory registered name
    pipeline_type_map = {
        "prefetch": "jagged_prefetch_sparse_dist",
        "native": "jagged_sparse_dist",
        "none": "jagged_none",
    }
    pipeline_name = pipeline_type_map.get(trainer_args.pipeline_type, "jagged_none")

    # Create pipeline using factory
    pipeline = TrainPipelineFactory.create(
        pipeline_name,
        model=model_train,
        optimizer=dense_optimizer,
        device=torch.device("cuda", torch.cuda.current_device()),
        batch_shuffler=batch_shuffler,
    )

    train_with_pipeline(
        pipeline,
        stateful_metric_module,
        trainer_args,
        train_dataloader,
        test_dataloader,
        dense_optimizer,
    )
    init.destroy_global_state()


if __name__ == "__main__":
    main()
