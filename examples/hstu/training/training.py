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
import os
from itertools import chain, count, cycle, islice
from typing import Iterator, Optional, Union

import commons.checkpoint as checkpoint
import torch  # pylint: disable-unused-import
import torch.distributed as dist
from commons.checkpoint import get_unwrapped_module
from commons.utils.gpu_timer import GPUTimer
from commons.utils.logger import print_rank_0
from commons.utils.stringify import stringify_dict
from megatron.core import parallel_state
from model import RankingGR, RetrievalGR
from modules.metrics import RetrievalTaskMetricWithSampling
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from training.gin_config_args import TrainerArgs
from training.utils import cal_flops


def evaluate(
    pipeline: Union[
        JaggedMegatronPrefetchTrainPipelineSparseDist,
        JaggedMegatronTrainNonePipeline,
        JaggedMegatronTrainPipelineSparseDist,
    ],
    stateful_metric_module: torch.nn.Module,
    trainer_args: TrainerArgs,
    eval_loader: torch.utils.data.DataLoader,
):
    eval_iter = 0
    torch.cuda.nvtx.range_push(f"#evaluate")
    max_eval_iters = trainer_args.max_eval_iters or len(eval_loader)
    max_eval_iters = min(max_eval_iters, len(eval_loader))
    # make a copy of eval_loader to avoid modifying the original loader
    iterated_eval_loader = islice(eval_loader, len(eval_loader))
    with torch.no_grad():
        for i in range(max_eval_iters):
            eval_iter += 1
            reporting_loss, (_, logits, labels, _) = pipeline.progress(
                iterated_eval_loader
            )
            # metric module forward
            stateful_metric_module(logits, labels)
        # compute will reset the states
        if isinstance(stateful_metric_module, RetrievalTaskMetricWithSampling):
            retrieval_gr = get_unwrapped_module(pipeline._model)
            export_table_name = retrieval_gr.get_item_feature_table_name()
            eval_metric_dict, _, _ = stateful_metric_module.compute(
                *retrieval_gr._embedding_collection.export_local_embedding(
                    export_table_name
                ),
            )
        else:
            eval_metric_dict = stateful_metric_module.compute()
        dp_size = parallel_state.get_data_parallel_world_size()
    # TODO, fix the samples when there is incomplete batch
    print_rank_0(
        f"[eval] [eval {eval_iter * dp_size * trainer_args.eval_batch_size} users]:\n    "
        + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
    )
    torch.cuda.nvtx.range_pop()


def maybe_load_ckpts(
    ckpt_load_dir: str,
    model: Union[RankingGR, RetrievalGR],
    dense_optimizer: Optional[torch.optim.Optimizer] = None,
):
    if ckpt_load_dir == "":
        return

    assert os.path.exists(
        ckpt_load_dir
    ), f"ckpt_load_dir {ckpt_load_dir} does not exist"

    print_rank_0(f"Loading checkpoints from {ckpt_load_dir}")
    checkpoint.load(ckpt_load_dir, model, dense_optimizer=dense_optimizer)
    print_rank_0(f"Checkpoints loaded!!")


def save_ckpts(
    ckpt_save_dir: str,
    model: Union[RankingGR, RetrievalGR],
    dense_optimizer: Optional[torch.optim.Optimizer] = None,
):
    print_rank_0(f"Saving checkpoints to {ckpt_save_dir}")
    import shutil

    if dist.get_rank() == 0:
        if os.path.exists(ckpt_save_dir):
            shutil.rmtree(ckpt_save_dir)
        try:
            os.makedirs(ckpt_save_dir, exist_ok=True)
        except Exception as e:
            raise Exception("can't build path:", ckpt_save_dir) from e
    dist.barrier(device_ids=[torch.cuda.current_device()])
    checkpoint.save(ckpt_save_dir, model, dense_optimizer=dense_optimizer)
    print_rank_0(f"Checkpoints saved!!")


# TODO. Use itertools.batched if python version is 3.12+
def batched(it: Iterator, n: int):
    assert n >= 1
    for x in it:
        yield chain((x,), islice(it, n - 1))


def train_with_pipeline(
    pipeline: Union[
        JaggedMegatronPrefetchTrainPipelineSparseDist,
        JaggedMegatronTrainNonePipeline,
        JaggedMegatronTrainPipelineSparseDist,
    ],
    stateful_metric_module: torch.nn.Module,
    trainer_args: TrainerArgs,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    dense_optimizer: torch.optim.Optimizer,
):
    gpu_timer = GPUTimer()
    max_train_iters = trainer_args.max_train_iters or len(train_loader)
    gpu_timer.start()
    last_td = 0
    # used to compute achieved flops/s
    ddp_seqlens = []
    ddp_num_contextuals = []
    ddp_num_candidates = []
    # using a tensor on gpu to avoid d2h copy
    tokens_logged = torch.zeros(1).cuda().float()
    # limit the number of iters to max_train_iters
    # we support max_train_iters > n_batches, i.e. multiple epochs
    train_loader_iter = islice(cycle(iter(train_loader)), max_train_iters)

    # every eval iter
    n = trainer_args.eval_interval if trainer_args.eval_interval else max_train_iters
    # data loader is split into num_iters / eval_interval (iters) slices where each slice contains n batches
    iter_slices = batched(train_loader_iter, n)
    start_iter = 0
    pipeline._model.train()
    for batched_iterator in iter_slices:
        # for one slice(every eval interval)
        for train_iter in count(start_iter):
            if trainer_args.profile and train_iter == trainer_args.profile_step_start:
                dist.barrier(device_ids=[torch.cuda.current_device()])
                torch.cuda.profiler.start()
            if trainer_args.profile and train_iter == trainer_args.profile_step_end:
                torch.cuda.profiler.stop()
                dist.barrier(device_ids=[torch.cuda.current_device()])
            if (
                train_iter * trainer_args.ckpt_save_interval > 0
                and train_iter % trainer_args.ckpt_save_interval == 0
            ):
                save_path = os.path.join(
                    trainer_args.ckpt_save_dir, f"iter{train_iter}"
                )
                save_ckpts(save_path, pipeline._model, dense_optimizer)
            try:
                torch.cuda.nvtx.range_push(f"step {train_iter}")
                reporting_loss, (
                    local_loss,
                    logits,
                    labels,
                    (ddp_seqlen, ddp_num_contextual, ddp_num_candidate),
                ) = pipeline.progress(batched_iterator)
                ddp_seqlens.append(ddp_seqlen.view(-1))
                ddp_num_contextuals.append(ddp_num_contextual.view(-1))
                ddp_num_candidates.append(ddp_num_candidate.view(-1))
                tokens_logged += reporting_loss[1]
                torch.cuda.nvtx.range_pop()
            except StopIteration:
                start_iter = train_iter
                torch.cuda.nvtx.range_pop()
                break
            # log
            if train_iter > 0 and (train_iter + 1) % trainer_args.log_interval == 0:
                gpu_timer.stop()
                cur_td = gpu_timer.elapsed_time() - last_td
                flops = cal_flops(
                    get_unwrapped_module(pipeline._model)._hstu_config,
                    seqlens=ddp_seqlens,
                    num_contextuals=ddp_num_contextuals,
                    num_candidates=ddp_num_candidates,
                )
                print_rank_0(
                    f"[train] [iter {train_iter}, tokens {int(tokens_logged.item())}, elapsed_time {cur_td:.2f} ms, achieved FLOPS {flops / cur_td / 1e9:.2f} TFLOPS]: loss {reporting_loss[0] / reporting_loss[1]:.6f}"
                )
                last_td = cur_td + last_td
                tokens_logged.zero_()
                ddp_seqlens = []
                ddp_num_contextuals = []
                ddp_num_candidates = []
        # TODO CHECK if train pipeline is flushed
        if train_iter > 0 and train_iter % trainer_args.eval_interval == 0:
            pipeline._model.eval()
            evaluate(
                pipeline,
                stateful_metric_module,
                trainer_args=trainer_args,
                eval_loader=eval_loader,
            )
            pipeline._model.train()
