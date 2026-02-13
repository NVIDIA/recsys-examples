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
import itertools
from typing import Dict, Optional, Union

import torch
from commons.datasets.hstu_batch import HSTUBatch
from commons.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
from commons.ops.length_to_offsets import length_to_complete_offsets
from commons.ops.triton_ops.triton_jagged import triton_split_2D_jagged
from commons.utils.nvtx_op import output_nvtx_hook
from configs.hstu_config import HSTUConfig
from configs.inference_config import InferenceHSTUConfig
from modules.jagged_data import JaggedData, pad_jd_values, unpad_jd_values
from modules.mlp import MLP
from modules.position_encoder import HSTUPositionalEncoder
from torchrec.sparse.jagged_tensor import JaggedTensor

try:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    SUPPORT_TRAINING = True
except ImportError:
    SUPPORT_TRAINING = False


def hstu_preprocess_embeddings(
    embeddings: Dict[str, JaggedTensor],
    batch: HSTUBatch,
    is_inference: bool,
    item_mlp: Optional[MLP] = None,
    contextual_mlp: Optional[MLP] = None,
    dtype: Optional[torch.dtype] = None,
    scaling_seqlen: int = -1,
) -> JaggedData:
    """
    Preprocesses the embeddings for use in the HSTU architecture.

    This method performs the following steps:
    1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
                         During inference, action embeddings are only for the history sequence, and
                         they will be interleaved with item embeddings of the history part, while
                         the embeddings of candidates need no interleaving.
    2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample,
                          following the order specified in the batch.
                          During inference, we concatenate three parts:
                          1) contextual embeedings,
                          2) interleaved *item & action* history embeddings, and
                          3) (item) candidates embeddings
                          for each sample, following the order specified in the batch.

    Args:
        embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
        batch (HSTUBatch): The batch of ranking data.
        is_inference (bool): Whether is for inference
        dtype (dtype, optional): The output data type of the embeddings.
    Returns:
        JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
    """
    item_jt = embeddings[batch.item_feature_name]  # history + candidate
    dtype = item_jt.values().dtype if dtype is None else dtype
    sequence_embeddings = item_jt.values().to(dtype)
    sequence_embeddings_lengths = item_jt.lengths()
    sequence_embeddings_lengths_offsets = item_jt.offsets()
    sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

    if batch.action_feature_name is not None:
        action_jt = embeddings[batch.action_feature_name]
        jagged_size = sequence_embeddings.size(0)
        embedding_dim = sequence_embeddings.size(1)

        if not is_inference:
            sequence_embeddings = torch.cat(
                [sequence_embeddings, action_jt.values().to(dtype)], dim=1
            ).view(2 * jagged_size, embedding_dim)
            sequence_embeddings_lengths = sequence_embeddings_lengths * 2
            sequence_embeddings_lengths_offsets = (
                sequence_embeddings_lengths_offsets * 2
            )
            sequence_max_seqlen = sequence_max_seqlen * 2
        else:
            # TODO@junyi: We can optimize the concat:
            # 1. use jagged split to get [history_embs, candidate_embs]
            # 2. use cat to interleave the history_embs and history_action_embs part
            # 3. use jagged concat to append the candidate_embs

            action_offsets = action_jt.offsets()
            item_offsets = item_jt.offsets()
            candidates_indptr = item_offsets[: batch.batch_size] + action_jt.lengths()

            item_embs = item_jt.values().to(dtype)
            action_embs = action_jt.values().to(dtype)
            interleaved_embeddings = [
                (
                    torch.cat(
                        [
                            item_embs[item_offsets[idx] : candidates_indptr[idx]],
                            action_embs[action_offsets[idx] : action_offsets[idx + 1]],
                        ],
                        dim=1,
                    ).view(-1, embedding_dim),
                    item_embs[candidates_indptr[idx] : item_offsets[idx + 1]],
                )
                for idx in range(batch.batch_size)
            ]
            interleaved_embeddings = list(itertools.chain(*interleaved_embeddings))
            sequence_embeddings = torch.cat(interleaved_embeddings, dim=0).view(
                -1, embedding_dim
            )
            sequence_embeddings_lengths = item_jt.lengths() + action_jt.lengths()
            sequence_embeddings_lengths_offsets = (
                item_jt.offsets() + action_jt.offsets()
            )
            sequence_max_seqlen += batch.feature_to_max_seqlen[
                batch.action_feature_name
            ]
        if item_mlp is not None:
            sequence_embeddings = item_mlp(sequence_embeddings)

    if (
        batch.num_candidates is not None
        and batch.action_feature_name is not None
        and not is_inference
    ):
        num_candidates = batch.num_candidates * 2
        max_num_candidates = batch.max_num_candidates * 2
    else:
        num_candidates = batch.num_candidates
        max_num_candidates = batch.max_num_candidates

    contextual_max_seqlen = 0
    contextual_seqlen = None
    contextual_seqlen_offsets = None
    if len(batch.contextual_feature_names) > 0:
        contextual_max_seqlens = [
            batch.feature_to_max_seqlen[name] for name in batch.contextual_feature_names
        ]
        contextual_jts = [embeddings[name] for name in batch.contextual_feature_names]
        contextual_jts_values = [jt.values().to(dtype) for jt in contextual_jts]
        contextual_jts_offsets = [jt.offsets() for jt in contextual_jts]

        (contextual_sequence_embeddings, contextual_seqlen) = jagged_2D_tensor_concat(
            contextual_jts_values,
            contextual_jts_offsets,
            contextual_max_seqlens,
        )
        if torch.sum(contextual_seqlen, dim=0).cpu().item() == 0:
            contextual_seqlen = None
        else:
            if contextual_mlp is not None:
                contextual_sequence_embeddings = contextual_mlp(
                    contextual_sequence_embeddings
                )
            contextual_seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                contextual_seqlen
            )
            contextual_max_seqlen = max(
                len(batch.contextual_feature_names), sum(contextual_max_seqlens)
            )
            (
                sequence_embeddings,
                sequence_embeddings_lengths,
            ) = jagged_2D_tensor_concat(
                [contextual_sequence_embeddings, sequence_embeddings],
                [contextual_seqlen_offsets, sequence_embeddings_lengths_offsets],
                [contextual_max_seqlen, sequence_max_seqlen],
            )

            sequence_embeddings_lengths_offsets = (
                torch.ops.fbgemm.asynchronous_complete_cumsum(
                    sequence_embeddings_lengths
                )
            )
            sequence_max_seqlen = sequence_max_seqlen + contextual_max_seqlen

    return JaggedData(
        values=sequence_embeddings,
        seqlen=sequence_embeddings_lengths.to(
            torch.int32
        ),  # contextual + history + candidate
        seqlen_offsets=sequence_embeddings_lengths_offsets.to(torch.int32),
        max_seqlen=sequence_max_seqlen,
        max_num_candidates=max_num_candidates,
        num_candidates=num_candidates.to(torch.int32)
        if num_candidates is not None
        else None,
        num_candidates_offsets=length_to_complete_offsets(num_candidates).to(
            torch.int32
        )
        if num_candidates is not None
        else None,
        contextual_max_seqlen=contextual_max_seqlen,
        contextual_seqlen=contextual_seqlen.to(torch.int32)
        if contextual_seqlen is not None
        else None,
        contextual_seqlen_offsets=contextual_seqlen_offsets.to(torch.int32)
        if contextual_seqlen_offsets is not None
        else None,
        has_interleaved_action=batch.action_feature_name is not None,
        scaling_seqlen=scaling_seqlen,
    )


class HSTUBlockPreprocessor(torch.nn.Module):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        config: Union[HSTUConfig, InferenceHSTUConfig],
        is_inference: bool,
    ):
        super().__init__()
        self.config = config
        self._training_dtype = torch.float32
        if config.bf16:
            self._training_dtype = torch.bfloat16
        if config.fp16:
            self._training_dtype = torch.float16
        if isinstance(config, HSTUConfig):
            self._sequence_parallel = config.sequence_parallel
        else:
            self._sequence_parallel = False
        self._tp_size = 1
        if is_inference:
            self._sequence_parallel = False
        if not is_inference and SUPPORT_TRAINING:
            self._tp_size = parallel_state.get_tensor_model_parallel_world_size()

        self._item_mlp = None
        self._contextual_mlp = None
        if config.hstu_preprocessing_config is not None:
            if config.hstu_preprocessing_config.item_embedding_dim > 0:
                self._item_mlp = MLP(
                    in_size=config.hstu_preprocessing_config.item_embedding_dim,
                    layer_sizes=[config.hidden_size, config.hidden_size],
                    activation="relu",
                    bias=True,
                )
            if config.hstu_preprocessing_config.contextual_embedding_dim > 0:
                self._contextual_mlp = MLP(
                    in_size=config.hstu_preprocessing_config.contextual_embedding_dim,
                    layer_sizes=[config.hidden_size, config.hidden_size],
                    activation="relu",
                    bias=True,
                )

        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if config.position_encoding_config is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                num_position_buckets=config.position_encoding_config.num_position_buckets,
                num_time_buckets=config.position_encoding_config.num_time_buckets,
                embedding_dim=config.hidden_size,
                is_inference=is_inference,
                use_time_encoding=config.position_encoding_config.use_time_encoding,
                training_dtype=self._training_dtype,
                static_max_seq_len=config.position_encoding_config.static_max_seq_len,
            )
        self._is_inference = is_inference
        self._dropout_ratio = 0.0
        if not self._is_inference:
            assert isinstance(
                config, HSTUConfig
            ), "Training config should be HSTUConfig"
            self._dropout_ratio = config.hidden_dropout
        self._scaling_seqlen = config.scaling_seqlen

    @output_nvtx_hook(nvtx_tag="HSTUBlock preprocess", hook_key_or_attr_name="values")
    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: HSTUBatch,
        seq_start_position: torch.Tensor = None,
    ) -> JaggedData:
        """
        Preprocesses the embeddings for use in the HSTU architecture.

        This method performs the following steps:
        1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
        2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample, following the order specified in the batch.
        3. **Padding**: Pads the jagged length of JaggedData to the TP size if sequence parallel is enabled.
        4. **Position Encoding**: Applies position encoding to the concatenated embeddings.

        Args:
            embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
            batch (HSTUBatch): The batch of ranking data.

        Returns:
            JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
        """
        device = torch.device("cuda", torch.cuda.current_device())
        batch = batch.to(device)
        # Interleaving & concatenation
        jd = hstu_preprocess_embeddings(
            embeddings,
            batch,
            is_inference=self._is_inference,
            item_mlp=self._item_mlp,
            contextual_mlp=self._contextual_mlp,
            dtype=self._training_dtype,
            scaling_seqlen=self._scaling_seqlen,
        )
        if self._sequence_parallel:
            jd = pad_jd_values(jd, self._tp_size)
        if self._positional_encoder is not None:
            jd.values = self._positional_encoder(
                max_seq_len=jd.max_seqlen,
                seq_lengths=jd.seqlen,
                seq_offsets=jd.seqlen_offsets,
                seq_timestamps=None,
                seq_embeddings=jd.values,
                num_targets=jd.num_candidates,
                seq_start_position=seq_start_position,
            )

        jd.values = torch.nn.functional.dropout(
            jd.values,
            p=self._dropout_ratio,
            training=self.training,
        ).to(self._training_dtype)
        # when sequence parallel is on, we need to scatter the values to the sequence parallel region
        # mcore performs the scatter in embedding: https://github.com/NVIDIA/Megatron-LM/blob/a32ff750191d04713ea1c15dcc65308324681016/megatron/core/tensor_parallel/layers.py#L286-L291
        # but we have to perform interleave and concatenation here.
        if self._sequence_parallel:
            jd.values = scatter_to_sequence_parallel_region(jd.values)
        return jd


class HSTUBlockPostprocessor(torch.nn.Module):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(self, is_inference: bool, sequence_parallel: bool = False):
        super().__init__()
        self._is_inference = is_inference
        self._sequence_parallel = sequence_parallel

        if self._is_inference:
            self._sequence_parallel = False

    @output_nvtx_hook(nvtx_tag="HSTUBlock postprocess", hook_key_or_attr_name="values")
    def forward(self, jd: JaggedData) -> JaggedData:
        """
        Postprocess the output from the HSTU architecture.
        1. If max_num_candidates > 0, split and only keep last ``num_candidates`` embeddings as candidates embedding for further processing.
        2, If sequence parallel is on, we need to gather the values back and remove the padding.
        3. Remove action embeddings if present. Only use item embedding for further processing.

        Args:
            jd (JaggedData): The jagged data output from the HSTU architecture that needs further processing.

        Returns:
            JaggedData: The postprocessed jagged data.
        """
        sequence_embeddings: torch.Tensor
        seqlen_offsets: torch.Tensor
        max_seqlen: int
        # the following compute is duplicated among TP ranks, we need to AG and remove the padding,
        # during backward, the gradients are scattered among TP ranks
        if self._sequence_parallel:
            jd.values = gather_from_sequence_parallel_region(
                jd.values, False
            )  # False -> output grad not RS but S
            jd = unpad_jd_values(jd)
        if jd.max_num_candidates > 0:
            seqlen_offsets = jd.num_candidates_offsets
            max_seqlen = jd.max_num_candidates
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
                offsets_b=seqlen_offsets,
            )
        elif jd.contextual_max_seqlen > 0:
            seqlen_offsets = jd.seqlen_offsets - jd.contextual_seqlen_offsets
            max_seqlen = jd.max_seqlen - jd.contextual_max_seqlen
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.contextual_seqlen_offsets,
                offsets_b=seqlen_offsets,
            )
        else:
            sequence_embeddings = jd.values
            seqlen_offsets = jd.seqlen_offsets
            max_seqlen = jd.max_seqlen

        if jd.has_interleaved_action and not self._is_inference:
            sequence_embeddings = sequence_embeddings[0::2, ...]
            seqlen_offsets = seqlen_offsets // 2
            max_seqlen = max_seqlen // 2

        sequence_embeddings = sequence_embeddings / torch.linalg.norm(
            sequence_embeddings, ord=2, dim=-1, keepdim=True
        ).clamp(min=1e-6)

        return JaggedData(
            values=sequence_embeddings,
            seqlen=torch.diff(seqlen_offsets).to(jd.seqlen.dtype),
            seqlen_offsets=seqlen_offsets.to(jd.seqlen_offsets.dtype),
            max_seqlen=max_seqlen,
            has_interleaved_action=False,
            scaling_seqlen=jd.scaling_seqlen,
        )
