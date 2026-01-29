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

import gin
import torch

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from commons.datasets.hstu_batch import HSTUBatch
from configs import (
    EmbeddingBackend,
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
)
from modules.inference_dense_module import get_inference_dense_model
from torch.utils.dlpack import from_dlpack, to_dlpack
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import DatasetArgs, NetworkArgs, RankingArgs

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


def check_hstu_sparse_dense_model_instance_config(model_config):
    assert "instance_group" in model_config, "No instance config found for HSTU model"

    embedding_backend = NetworkArgs().embedding_backend
    embedding_backend = (
        EmbeddingBackend(embedding_backend) if embedding_backend else None
    )
    if embedding_backend != EmbeddingBackend.NVEMB:
        conf_err_msg = "HSTU model only support 1 GPU instance when using DynamicEmb embedding backend"
        assert len(model_config["instance_group"]) == 1, conf_err_msg
        instance_group_conf = model_config["instance_group"][0]
        assert "gpus" in instance_group_conf, conf_err_msg
        assert len(instance_group_conf["gpus"]) == 1, conf_err_msg


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


def get_inference_dense_model_with_feature_names(
    max_batch_size,
    checkpoint_dir,
    use_kvcache=False,
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs()
    num_contextual_features = len(emb_configs) - 2
    total_max_seqlen = (
        dataset_args.max_num_candidates + dataset_args.max_history_seqlen
    ) * 2 + num_contextual_features
    feature_names = [ebc.feature_names[0] for ebc in emb_configs]

    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
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

    model = get_inference_dense_model(
        hstu_config=hstu_config,
        kvcache_config=None,
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

    return model, feature_names


def pack_batch_from_numpy_host_input(
    uids,
    tokens,
    token_lengths,
    num_candidates,
    feature_names,
    device,
):
    batch_size = uids.shape[0]
    features = KeyedJaggedTensor(
        keys=feature_names,
        values=torch.tensor(tokens),
        lengths=torch.tensor(token_lengths),
    )
    feature_to_max_seqlen = {}
    for idx, name in enumerate(feature_names):
        feature_to_max_seqlen[name] = int(
            max(token_lengths[idx * batch_size : (idx + 1) * batch_size])
        )
    max_num_candidates = int(max(num_candidates))
    batch = HSTUBatch(
        features=features,
        batch_size=batch_size,
        feature_to_max_seqlen=feature_to_max_seqlen,
        contextual_feature_names=feature_names[:-2],
        item_feature_name=feature_names[-2],
        action_feature_name=feature_names[-1],
        max_num_candidates=max_num_candidates,
        num_candidates=torch.tensor(num_candidates) if max_num_candidates > 0 else None,
    ).to(device=device)
    return batch


def pack_embeddings(batch, raw_embeddings):
    raw_embeddings.shape[1]
    embeddings = {}
    for idx, key in enumerate(batch.features.keys()):
        startpos, endpos = (idx * batch.batch_size, (idx + 1) * batch.batch_size)
        embeddings[key] = JaggedTensor(
            values=raw_embeddings[
                batch.features.offsets()[startpos] : batch.features.offsets()[endpos], :
            ],
            lengths=batch.features.lengths()[startpos:endpos],
        ).to(device=raw_embeddings.device)
    return embeddings


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

        self.model_config = model_config = json.loads(args["model_config"])
        self.gin_config_file = get_hstu_gin_config_from_config(self.model_config)
        gin.parse_config_file(self.gin_config_file)

        check_hstu_sparse_dense_model_instance_config(model_config)
        self._device = torch.device("cuda:" + args["model_instance_device_id"])
        torch.cuda.set_device(self._device)

        # Instantiate the PyTorch model
        self.max_batch_size = get_hstu_max_batch_size_from_config(self.model_config)
        self.checkpoint_dir = get_hstu_ckpt_dir_from_config(self.model_config)
        with torch.inference_mode():
            (
                self.hstu_model,
                self.feature_names,
            ) = get_inference_dense_model_with_feature_names(
                self.max_batch_size,
                self.checkpoint_dir,
                use_kvcache=False,
            )

        # Get OUTPUT configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype = triton_string_to_torch_dtype(output_config["data_type"])

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

        output_dtype = self.output_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            uids = pb_utils.get_input_tensor_by_name(request, "USER_IDS").as_numpy()
            token_lengths = pb_utils.get_input_tensor_by_name(
                request, "TOKEN_LENGTHS"
            ).as_numpy()
            tokens = pb_utils.get_input_tensor_by_name(request, "TOKENS").as_numpy()
            num_candidates = pb_utils.get_input_tensor_by_name(
                request, "NUM_CANDIDATES"
            ).as_numpy()

            hstu_batch = pack_batch_from_numpy_host_input(
                uids,
                tokens,
                token_lengths,
                num_candidates,
                self.feature_names,
                self._device,
            )

            hstu_sparse_request = pb_utils.InferenceRequest(
                model_name="hstu_sparse",
                inputs=[
                    pb_utils.Tensor.from_dlpack(
                        "TOKENS", to_dlpack(hstu_batch.features.values())
                    ),
                    pb_utils.Tensor.from_dlpack(
                        "TOKEN_LENGTHS", to_dlpack(hstu_batch.features.lengths())
                    ),
                    pb_utils.Tensor.from_dlpack(
                        "TOKEN_OFFSETS", to_dlpack(hstu_batch.features.offsets())
                    ),
                ],
                requested_output_names=["EMBEDDINGS"],
            )

            hstu_sparse_responses = hstu_sparse_request.exec(decoupled=True)

            raw_embeddings = []
            for resp in hstu_sparse_responses:
                sparse_output = pb_utils.get_output_tensor_by_name(resp, "EMBEDDINGS")
                if sparse_output is None:
                    break
                raw_embeddings.append(from_dlpack(sparse_output))
            assert len(raw_embeddings) == 1

            embeddings = pack_embeddings(hstu_batch, raw_embeddings[0])

            jagged_item_logit = self.hstu_model.forward(hstu_batch, embeddings)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "OUTPUT",
                        jagged_item_logit.detach()
                        .to(dtype=output_dtype, device=torch.device("cpu"))
                        .numpy(),
                    )
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
