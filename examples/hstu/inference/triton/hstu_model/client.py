# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import time

import gin
import torch
import tritonclient.http as httpclient
from commons.datasets import get_data_loader
from commons.datasets.hstu_sequence_dataset import get_dataset
from commons.utils.stringify import stringify_dict
from modules.metrics import get_multi_event_metric_module
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from tritonclient.utils import *
from utils import DatasetArgs, RankingArgs

model_name = "hstu_model"
# This must not exceed HSTU_MAX_BATCH_SIZE in the Triton model configuration.
MAX_BATCH_SIZE = 2


def get_dataset_configs():
    dataset_args = DatasetArgs()
    if dataset_args.dataset_name == "kuairand-1k":
        return dataset_args

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")


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


def infer_batch(client, batch):
    uids = batch.features.to_dict()["user_id"].values()

    if uids.shape[0] != batch.batch_size:
        batch = strip_padding_batch(batch, uids.shape[0])

    uids = uids.detach().numpy()
    tokens = batch.features.values().detach().numpy()
    token_lens = batch.features.lengths().detach().numpy()
    num_candidates = batch.num_candidates.detach().numpy()

    request_start_time = time.perf_counter()
    inputs = [
        httpclient.InferInput("USER_IDS", uids.shape, np_to_triton_dtype(uids.dtype)),
        httpclient.InferInput(
            "TOKEN_LENGTHS",
            token_lens.shape,
            np_to_triton_dtype(token_lens.dtype),
        ),
        httpclient.InferInput("TOKENS", tokens.shape, np_to_triton_dtype(tokens.dtype)),
        httpclient.InferInput(
            "NUM_CANDIDATES",
            num_candidates.shape,
            np_to_triton_dtype(num_candidates.dtype),
        ),
    ]
    inputs[0].set_data_from_numpy(uids)
    inputs[1].set_data_from_numpy(token_lens)
    inputs[2].set_data_from_numpy(tokens)
    inputs[3].set_data_from_numpy(num_candidates)

    outputs = [httpclient.InferRequestedOutput("OUTPUT")]
    response = client.infer(model_name, inputs, request_id=str(0), outputs=outputs)
    elapsed_seconds = time.perf_counter() - request_start_time
    return batch, response, elapsed_seconds


def update_metrics(eval_module, batch, response):
    logits = response.as_numpy("OUTPUT")
    eval_module(
        torch.from_numpy(logits).to(
            dtype=torch.bfloat16, device=torch.cuda.current_device()
        ),
        batch.labels.values().cuda(),
    )


def run_ranking_gr_evaluate(
    use_train_dataset=False,
    post_warmup_sleep_seconds=1.0,
    num_runs=3,
):
    dataset_args = get_dataset_configs()

    with torch.inference_mode():
        ranking_args = RankingArgs()

        eval_module = get_multi_event_metric_module(
            num_classes=ranking_args.prediction_head_arch[-1],
            num_tasks=ranking_args.num_tasks,
            metric_types=ranking_args.eval_metrics,
        )

        train_dataset, eval_dataset = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_history_seqlen=dataset_args.max_history_seqlen,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=ranking_args.num_tasks,
            batch_size=MAX_BATCH_SIZE,
            rank=0,
            world_size=1,
            shuffle=False,
            random_seed=0,
            eval_batch_size=MAX_BATCH_SIZE,
        )

        selected_dataset = train_dataset if use_train_dataset else eval_dataset
        dataloader = get_data_loader(dataset=selected_dataset)

        with httpclient.InferenceServerClient("localhost:8000") as client:
            warmup_batch = next(iter(dataloader))
            warmup_batch, warmup_response, _ = infer_batch(client, warmup_batch)
            update_metrics(eval_module, warmup_batch, warmup_response)
            print("Warmup: sent the first batch, skipping it in all measured runs")
            if post_warmup_sleep_seconds > 0:
                time.sleep(post_warmup_sleep_seconds)
                print(f"Slept {post_warmup_sleep_seconds:.3f} seconds after warmup")

            run_elapsed_seconds = []
            for run_index in range(1, num_runs + 1):
                dataloader_iter = iter(dataloader)
                next(dataloader_iter, None)
                measured_batch_count = 0
                measured_elapsed_seconds = 0.0

                for batch in dataloader_iter:
                    batch, response, elapsed_seconds = infer_batch(client, batch)
                    measured_batch_count += 1
                    measured_elapsed_seconds += elapsed_seconds

                    if run_index == 1:
                        update_metrics(eval_module, batch, response)

                run_elapsed_seconds.append(measured_elapsed_seconds)
                average_latency_ms = (
                    measured_elapsed_seconds * 1000.0 / measured_batch_count
                )
                print(
                    f"Run {run_index} (no cache, Python backend): "
                    f"{measured_elapsed_seconds:.6f} seconds for "
                    f"{measured_batch_count} batches "
                    f"({average_latency_ms:.3f} ms/batch)"
                )

            average_run_seconds = sum(run_elapsed_seconds) / len(run_elapsed_seconds)
            print(
                f"Average (no cache, Python backend): {average_run_seconds:.6f} "
                f"seconds over {num_runs} runs"
            )

        eval_metric_dict = eval_module.compute()
        print(
            f"[eval]:\n    "
            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--train_dataset", action="store_true")
    parser.add_argument("--post_warmup_sleep_seconds", type=float, default=1.0)
    parser.add_argument("--num_runs", type=int, default=3)

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    run_ranking_gr_evaluate(
        use_train_dataset=args.train_dataset,
        post_warmup_sleep_seconds=args.post_warmup_sleep_seconds,
        num_runs=args.num_runs,
    )
