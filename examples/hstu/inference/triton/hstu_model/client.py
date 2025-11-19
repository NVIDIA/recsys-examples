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

import gin
import torch
import tritonclient.http as httpclient
from commons.utils.stringify import stringify_dict
from dataset import get_data_loader
from dataset.sequence_dataset import get_dataset
from modules.metrics import get_multi_event_metric_module
from preprocessor import get_common_preprocessors
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from tritonclient.utils import *
from utils import DatasetArgs, RankingArgs

model_name = "hstu_model"


def get_dataset_configs():
    dataset_args = DatasetArgs()
    if dataset_args.dataset_name == "kuairand-1k":
        return dataset_args

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")


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


def run_ranking_gr_evaluate():
    dataset_args = get_dataset_configs()

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = len(dataproc._contextual_feature_names)

    max_batch_size = 8
    dataset_args.max_sequence_length * 2 + num_contextual_features

    with torch.inference_mode():
        ranking_args = RankingArgs()

        eval_module = get_multi_event_metric_module(
            num_classes=ranking_args.prediction_head_arch[-1],
            num_tasks=ranking_args.num_tasks,
            metric_types=ranking_args.eval_metrics,
        )

        _, eval_dataset = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_sequence_length=dataset_args.max_sequence_length,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=ranking_args.num_tasks,
            batch_size=max_batch_size,
            rank=0,
            world_size=1,
            shuffle=False,
            random_seed=0,
            eval_batch_size=max_batch_size,
        )

        dataloader = get_data_loader(dataset=eval_dataset)
        dataloader_iter = iter(dataloader)

        with httpclient.InferenceServerClient("localhost:8000") as client:
            while True:
                try:
                    batch = next(dataloader_iter)

                    batch = strip_candidate_action_tokens(
                        batch, dataproc._action_feature_name
                    )

                    uids = batch.features.to_dict()["user_id"].values()

                    if uids.shape[0] != batch.batch_size:
                        batch = strip_padding_batch(batch, uids.shape[0])

                    uids = uids.detach().numpy()
                    tokens = batch.features.values().detach().numpy()
                    token_lens = batch.features.lengths().detach().numpy()
                    num_candidates = batch.num_candidates.detach().numpy()

                    inputs = [
                        httpclient.InferInput(
                            "USER_IDS", uids.shape, np_to_triton_dtype(uids.dtype)
                        ),
                        httpclient.InferInput(
                            "TOKEN_LENGTHS",
                            token_lens.shape,
                            np_to_triton_dtype(token_lens.dtype),
                        ),
                        httpclient.InferInput(
                            "TOKENS", tokens.shape, np_to_triton_dtype(tokens.dtype)
                        ),
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

                    outputs = [ httpclient.InferRequestedOutput("OUTPUT") ]
                    response = client.infer(
                        model_name, inputs, request_id=str(0), outputs=outputs
                    )
                    logits = response.as_numpy("OUTPUT")

                    eval_module(
                        torch.from_numpy(logits).to(
                            dtype=torch.bfloat16, device=torch.cuda.current_device()
                        ),
                        batch.labels.cuda(),
                    )
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

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    run_ranking_gr_evaluate()
