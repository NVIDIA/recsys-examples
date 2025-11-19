# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import json
import math
import os
import sys
import threading

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack, from_dlpack
import torch
from torch import nn

import gin
import numpy as np
from configs import (
    InferenceMode,
    InferenceEmbeddingConfig,
    EmbeddingBackend,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from preprocessor import get_common_preprocessors
from utils import DatasetArgs, NetworkArgs, RankingArgs
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR

TRITON_STRING_TO_NUMPY = {
    "TYPE_BOOL": torch.bool,
    "TYPE_UINT8": torch.uint8,
    "TYPE_UINT16": torch.uint16,
    "TYPE_UINT32": torch.uint32,
    "TYPE_UINT64": torch.uint64,
    "TYPE_INT8": torch.int8,
    "TYPE_INT16": torch.int16,
    "TYPE_INT32": torch.int32,
    "TYPE_INT64": torch.int64,
    "TYPE_BF16": torch.bfloat16,
    "TYPE_FP16": torch.float16,
    "TYPE_FP32": torch.float32,
    "TYPE_FP64": torch.float64,
}

def triton_string_to_torch_dtype(triton_type_string):
    return TRITON_STRING_TO_NUMPY[triton_type_string]


def get_hstu_max_batch_size_from_config(config):
    return int(config["parameters"]["HSTU_MAX_BATCH_SIZE"]["string_value"])

def get_hstu_gin_config_from_config(config):
    return config["parameters"]["HSTU_GIN_CONFIG_FILE"]["string_value"]

def get_hstu_ckpt_dir_from_config(config):
    return config["parameters"]["HSTU_CHECKPOINT_DIR"]["string_value"]


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


def get_shared_inference_hstu_sparse_model(
    emb_configs,
    embedding_backend,
    max_batch_size,
    num_contextual_features,
    total_max_seqlen,
    checkpoint_dir,
):
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Inference data type {network_args.dtype_str} is not supported"
        )

    hstu_config = get_inference_hstu_config(
        hidden_size=network_args.hidden_size,
        num_layers=network_args.num_layers,
        num_attention_heads=network_args.num_attention_heads,
        head_dim=network_args.kv_channels,
        max_batch_size=max_batch_size,
        max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
        dtype=inference_dtype,
        position_encoding_config=None,
        contextual_max_seqlen=num_contextual_features,
        embedding_backend=embedding_backend,
    )
    print("=====", embedding_backend, "=======")

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

    
    models = []
    sparse_shareables = None
    for dev_id in range(torch.cuda.device_count()):
        print("===== DEVICE ID", dev_id, "=====")
        if dev_id == 0 and embedding_backend == EmbeddingBackend.NVEMB:
            import pynve.nve as nve
            sparse_shareables = {
                config.table_name: nve.LinearMemBlock(config.dim, config.vocab_size, nve.DataType_t.Float16)
                for config in emb_configs if config.use_dynamicemb
            }

        torch.cuda.set_device(dev_id)
        model = InferenceRankingGR(
            hstu_config=hstu_config,
            kvcache_config=None,
            task_config=task_config,
            mode=InferenceMode.sparse,
            sparse_shareables=sparse_shareables,
        )
        if hstu_config.bf16:
            model.bfloat16()
        elif hstu_config.fp16:
            model.half()
        model.load_checkpoint(checkpoint_dir)
        model.eval()

        models.append(model)

        if embedding_backend != EmbeddingBackend.NVEMB:
            break

    return models


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])
        assert args["model_instance_kind"] == "MODEL"

        self.max_batch_size = get_hstu_max_batch_size_from_config(self.model_config)
        self.gin_config_file = get_hstu_gin_config_from_config(self.model_config)
        self.checkpoint_dir = get_hstu_ckpt_dir_from_config(self.model_config)

        gin.parse_config_file(self.gin_config_file)
        dataset_args, emb_configs = get_inference_dataset_and_embedding_configs()
        dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
        num_contextual_features = len(dataproc._contextual_feature_names)
        total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
        embedding_backend = NetworkArgs().embedding_backend
        self.embedding_backend = EmbeddingBackend(embedding_backend) if embedding_backend else None

        # Instantiate the PyTorch model
        with torch.inference_mode():
            self.hstu_model = get_shared_inference_hstu_sparse_model(
                emb_configs,
                self.embedding_backend,
                self.max_batch_size,
                num_contextual_features,
                total_max_seqlen,
                self.checkpoint_dir,
            )
    
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        with torch.inference_mode():
            for request in requests:
                response_sender = request.get_response_sender()
                # The response_sender is used to send response(s) associated with the
                # corresponding request.
                tokens = pb_utils.get_input_tensor_by_name(request, "TOKENS")
                tokens = from_dlpack(tokens.to_dlpack())
                token_lengths = pb_utils.get_input_tensor_by_name(request, "TOKEN_LENGTHS")
                token_lengths = from_dlpack(token_lengths.to_dlpack())
                token_offsets = pb_utils.get_input_tensor_by_name(request, "TOKEN_OFFSETS")
                token_offsets = from_dlpack(token_offsets.to_dlpack())

                device_id = tokens.device.index
                torch.cuda.set_device(tokens.device)

                features = KeyedJaggedTensor(
                    keys = self.hstu_model[0]._feature_names,
                    values = tokens,
                    lengths = token_lengths,
                    offsets = token_offsets,
                )

                embeddings = self.hstu_model[device_id]._embedding_collection(features)
                embeddings = torch.cat([
                    embeddings[feat_name].values() for feat_name in self.hstu_model[0]._feature_names
                ], dim = 0)
                embeddings = pb_utils.Tensor.from_dlpack("EMBEDDINGS", to_dlpack(embeddings))
                response = pb_utils.InferenceResponse(output_tensors=[embeddings])
                response_sender.send(response)

                # We must close the response sender to indicate to Triton that we are
                # done sending responses for the corresponding request. We can't use the
                # response sender after closing it. The response sender is closed by
                # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
                response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        return None
    
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
