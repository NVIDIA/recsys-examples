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
import warnings

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
import argparse
from dataclasses import dataclass
from functools import partial  # pylint: disable-unused-import
from typing import Tuple

import commons.utils.initialize as init
import gin
import torch  # pylint: disable-unused-import
from configs import RetrievalConfig
from distributed.sharding import make_optimizer_and_shard
from model import get_retrieval_model
from modules.metrics import RetrievalTaskMetricWithSampling
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from training import (
    NetworkArgs,
    OptimizerArgs,
    TensorModelParallelArgs,
    MixedPrecisionArgs,
    TrainerArgs,
    create_dynamic_optitons_dict,
    create_embedding_config,
    create_hstu_config,
    create_optimizer_params,
    get_data_loader,
    get_dataset_and_embedding_args,
    maybe_load_ckpts,
    train_with_pipeline,
)

# Optional Transformer Engine support (FP8)
try:
    import transformer_engine.pytorch as te  # type: ignore
    from transformer_engine.common.recipe import Format, DelayedScaling, Float8CurrentScaling, Float8BlockScaling  # type: ignore
    use_te = True
    format_map = {"e4m3": Format.E4M3, "e5m2": Format.E5M2, "hybrid": Format.HYBRID}
except:  # noqa: E722
    warnings.warn(
        "transformer_engine.pytorch is not installed, FP8 mixed precision will not be supported"
    )
    use_te = False


@gin.configurable
@dataclass
class RetrievalArgs:
    ### retrieval
    num_negatives: int = -1
    temperature = 0.05
    l2_norm_eps = 1e-6
    eval_metrics: Tuple[str, ...] = ("HR@10", "NDCG@10")


parser = argparse.ArgumentParser(
    description="Distributed GR Arguments", allow_abbrev=False
)
parser.add_argument("--gin-config-file", type=str)
args = parser.parse_args()
gin.parse_config_file(args.gin_config_file)
trainer_args = TrainerArgs()
dataset_args, embedding_args = get_dataset_and_embedding_args()
network_args = NetworkArgs()
optimizer_args = OptimizerArgs()
tp_args = TensorModelParallelArgs()
mp_args = MixedPrecisionArgs()

if mp_args.enabled and not use_te:
    assert False, "FP8 mixed precision only supported with Transformer Engine"

if mp_args.enabled:
    fp8_mp_kwargs = {
        "recipe": mp_args.linear_recipe,
        "fp8_format": format_map[mp_args.linear_scaling_precision],
    }
else:
    fp8_mp_kwargs = {}


def create_retrieval_config() -> RetrievalConfig:
    retrieval_args = RetrievalArgs()

    return RetrievalConfig(
        embedding_configs=[
            create_embedding_config(network_args.hidden_size, arg)
            for arg in embedding_args
        ],
        temperature=retrieval_args.temperature,
        l2_norm_eps=retrieval_args.l2_norm_eps,
        num_negatives=retrieval_args.num_negatives,
        eval_metrics=retrieval_args.eval_metrics,
    )


def main():
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)

    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = create_retrieval_config()
    model = get_retrieval_model(hstu_config=hstu_config, task_config=task_config)

    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args, network_args.hidden_size
    )
    optimizer_param = create_optimizer_params(optimizer_args)
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )
    stateful_metric_module = RetrievalTaskMetricWithSampling(
        metric_types=task_config.eval_metrics, MAX_K=500
    )
    train_dataloader, test_dataloader = get_data_loader(
        "retrieval", dataset_args, trainer_args, 0
    )
    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)
    if trainer_args.pipeline_type in ["prefetch", "native"]:
        pipeline_factory = (
            JaggedMegatronPrefetchTrainPipelineSparseDist
            if trainer_args.pipeline_type == "prefetch"
            else JaggedMegatronTrainPipelineSparseDist
        )
        pipeline = pipeline_factory(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
            te_mixed_precision=mp_args.enabled,
            **fp8_mp_kwargs,
        )
    else:
        pipeline = JaggedMegatronTrainNonePipeline(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
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
