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
import copy
import dataclasses
import pprint
from typing import Optional

import fbgemm_gpu  # pylint: disable-unused-import
import torch


@dataclasses.dataclass
class JaggedData:
    """JaggedData is a data structure that holds jagged tensor data, which is commonly used in sequence-based models.

    Args:
        values (torch.Tensor): The values tensor with shape (T, d), where T is the total sequence length across all batches and d is the dimensionality.
        seqlen (torch.Tensor): The sequence lengths tensor with shape (batch_size,), indicating the length of each sequence in the batch.
        seqlen_offsets (torch.Tensor): The sequence length offsets tensor with shape (batch_size + 1), used to align sequences in the batch.
        max_seqlen (int): The maximum sequence length across all batches.
        max_num_candidates (int): The maximum number of candidates. Defaults to 0.
        num_candidates (Optional[torch.Tensor]): Tensor containing the number of candidates for each sequence. Defaults to None.
        num_candidates_offsets (Optional[torch.Tensor]): Offsets tensor for the number of candidates, used to align candidates across sequences. Defaults to None.
        contextual_max_seqlen (int): The maximum sequence length for contextual features. Defaults to 0.
        contextual_seqlen (Optional[torch.Tensor]): The sequence lengths tensor for contextual features. Defaults to None.
        contextual_seqlen_offsets (Optional[torch.Tensor]): The sequence length offsets tensor for contextual features. Defaults to None.
        has_interleaved_action (bool): Whether action embeddings are interleaved with item embeddings. Defaults to False.
        scaling_seqlen (int): The sequence length to scale attention output. Defaults to -1 (max_seqlen will be used).

    """

    values: torch.Tensor
    seqlen: torch.Tensor  # (batch_size, )
    seqlen_offsets: torch.Tensor  # (batch_size + 1)

    max_seqlen: int

    max_num_candidates: int = 0
    num_candidates: Optional[torch.Tensor] = None
    num_candidates_offsets: Optional[torch.Tensor] = None

    contextual_max_seqlen: int = 0
    contextual_seqlen: Optional[torch.Tensor] = None
    contextual_seqlen_offsets: Optional[torch.Tensor] = None

    has_interleaved_action: bool = False
    scaling_seqlen: int = -1
    padding_length: int = (
        0  # the padded length of the values tensor, this is used when SP is on
    )

    def copy_others_but_set_values(self, values: Optional[torch.Tensor] = None):
        """
        copy all other fields but set values to the given tensor
        """
        new_instance = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key == "values":
                new_instance.values = (
                    values
                    if values is not None
                    else torch.tensor(
                        [], device=self.values.device, dtype=self.values.dtype
                    )
                )
            else:
                setattr(new_instance, key, copy.deepcopy(value))
        return new_instance

    def __post_init__(self):
        if self.max_num_candidates == 0:
            assert self.num_candidates is None
            assert self.num_candidates_offsets is None
        if self.contextual_max_seqlen == 0:
            assert self.contextual_seqlen is None
            assert self.contextual_seqlen_offsets is None

    @staticmethod
    def random(
        seqlen: torch.Tensor,
        dim,
        *,
        num_candidates: Optional[torch.Tensor] = None,
        contextual_seqlen: Optional[torch.Tensor] = None,
        has_interleaved_action: bool = False,
        scaling_seqlen: int = -1,
        device,
        dtype,
    ) -> "JaggedData":
        """
        Static method. Generates a random JaggedData object.

        Args:
            seqlen (torch.Tensor): A tensor with shape (batch_size,) containing the sequence length for each sample.
            dim (int): The dimension of the values tensor.
            num_candidates (Optional[torch.Tensor], optional): Tensor containing the number of candidates for each sequence. Defaults to None.
            contextual_seqlen (Optional[torch.Tensor], optional): The sequence lengths tensor for contextual features. Defaults to None.
            has_interleaved_action (bool, optional): Whether action embeddings are interleaved with item embeddings. Defaults to False.
            scaling_seqlen (int, optional): The sequence length to scale attention output. Defaults to -1 (max_seqlen will be used).
            device: The device on which to generate the random data.
            dtype (torch.dtype): The data type of the values tensor.

        Returns:
            JaggedData: The generated random JaggedData object.

        """
        seqlen = seqlen.to(device)
        assert seqlen.dim() == 1, "seqlen dim should equal to 1"
        max_seqlen = torch.max(seqlen).cpu().item()

        seqlen_sum = torch.sum(seqlen).item()

        seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seqlen)
        values = torch.rand(seqlen_sum, dim, dtype=dtype, device=device)

        max_num_candidates = 0
        num_candidates_offsets = None
        if num_candidates is not None:
            max_num_candidates = torch.max(num_candidates).cpu().item()
            num_candidates = num_candidates.to(device)
            num_candidates_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_candidates
            )

        contextual_max_seqlen = 0
        contextual_seqlen_offsets = None
        if contextual_seqlen is not None:
            contextual_max_seqlen = torch.max(contextual_seqlen).cpu().item()
            contextual_seqlen = contextual_seqlen.to(device)
            contextual_seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                contextual_seqlen
            )

        return JaggedData(
            values=values,
            seqlen=seqlen,
            seqlen_offsets=seqlen_offsets,
            max_seqlen=max_seqlen,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates,
            num_candidates_offsets=num_candidates_offsets,
            contextual_max_seqlen=contextual_max_seqlen,
            contextual_seqlen=contextual_seqlen,
            contextual_seqlen_offsets=contextual_seqlen_offsets,
            has_interleaved_action=has_interleaved_action,
            scaling_seqlen=scaling_seqlen,
        )

    def __eq__(self, other):
        """
        Checks equality between two JaggedData instances.

        Args:
            other (JaggedData): The other JaggedData instance to compare.

        Returns:
            bool: True if the instances are equal, False otherwise.
        """
        assert torch.allclose(self.values, other.values)
        assert torch.allclose(self.seqlen, other.seqlen)
        assert torch.allclose(self.seqlen_offsets, other.seqlen_offsets)
        assert self.max_seqlen == other.max_seqlen
        assert self.max_num_candidates == other.max_num_candidates
        assert torch.all(self.num_candidates == other.num_candidates)
        assert torch.all(self.num_candidates_offsets == other.num_candidates_offsets)
        assert self.contextual_max_seqlen == other.contextual_max_seqlen
        assert torch.all(self.contextual_seqlen == other.contextual_seqlen)
        assert torch.all(
            self.contextual_seqlen_offsets == other.contextual_seqlen_offsets
        )
        assert self.has_interleaved_action == other.has_interleaved_action
        assert self.scaling_seqlen == other.scaling_seqlen

    def __repr__(self):
        """
        Returns a string representation of the JaggedData instance.

        Returns:
            str: The string representation.
        """
        viz_dict = {
            "values": self.values,
            "seqlen": self.seqlen,
            "seqlen_offsets": self.seqlen_offsets,
            "max_seqlen": self.max_seqlen,
            "max_num_candidates": self.max_num_candidates,
            "num_candidates": self.num_candidates,
            "num_candidates_offsets": self.num_candidates_offsets,
            "contextual_max_seqlen": self.contextual_max_seqlen,
            "contextual_seqlen": self.contextual_seqlen,
            "contextual_seqlen_offsets": self.contextual_seqlen_offsets,
            "has_interleaved_action": self.has_interleaved_action,
            "scaling_seqlen": self.scaling_seqlen,
        }
        str_rep = pprint.pformat(viz_dict)
        return f"JaggedData {str_rep}"

    def detach(self, include_values=True):
        """
        Detaches the tensors in the JaggedData instance from the computation graph.

        Args:
            include_values (bool, optional): Whether to detach the values tensor. Defaults to True.
        """
        for f in dataclasses.fields(JaggedData):
            maybe_tensor = getattr(JaggedData, f.name)
            if f.name == "values" and include_values:
                maybe_tensor.detach()
            else:
                if maybe_tensor is not None:
                    maybe_tensor.detach()

    def to(self, device: torch.device, non_blocking: bool = False) -> "JaggedData":
        """
        Moves the JaggedData instance to the specified device.

        Args:
            device (torch.device): The target device.
            non_blocking (bool, optional): Whether to perform the move asynchronously. Defaults to False.

        Returns:
            JaggedData: The moved JaggedData instance.
        """
        return JaggedData(
            values=self.values.to(device=device, non_blocking=non_blocking),
            seqlen=self.seqlen.to(device=device, non_blocking=non_blocking),
            seqlen_offsets=self.seqlen_offsets.to(
                device=device, non_blocking=non_blocking
            ),
            max_seqlen=self.max_seqlen,
            max_num_candidates=self.max_num_candidates,
            num_candidates=self.num_candidates.to(
                device=device, non_blocking=non_blocking
            )
            if self.num_candidates is not None
            else self.num_candidates,
            num_candidates_offsets=self.num_candidates_offsets.to(
                device=device, non_blocking=non_blocking
            )
            if self.num_candidates_offsets is not None
            else self.num_candidates_offsets,
            contextual_max_seqlen=self.contextual_max_seqlen,
            contextual_seqlen=self.contextual_seqlen.to(
                device=device, non_blocking=non_blocking
            )
            if self.contextual_seqlen is not None
            else self.contextual_seqlen,
            contextual_seqlen_offsets=self.contextual_seqlen_offsets.to(
                device=device, non_blocking=non_blocking
            )
            if self.contextual_seqlen_offsets is not None
            else self.contextual_seqlen_offsets,
            has_interleaved_action=self.has_interleaved_action,
            scaling_seqlen=self.scaling_seqlen,
        )


def pad_jd_values(jd: JaggedData, pad_base: int, dim=0) -> JaggedData:
    """
    With sequence parallel, there will be Allgather & ReduceScatter communication among TP ranks during training.
    We need to pad the jagged length of JaggedData to the TP size.

    To make it differentiable, we need to return a new JaggedData instance.
    """
    # assert jd.padding_length == 0, "JaggedData must not be padded"
    assert dim == 0, "Only padding along the first dimension is supported"
    length = jd.seqlen_offsets[-1]
    # Check if already aligned
    if length % pad_base == 0:
        output_jd = jd.copy_others_but_set_values(values=jd.values.clone())
        return output_jd
    aligned_size = ((length + pad_base - 1) // pad_base) * pad_base
    values = jd.values
    assert values.dim() == 2, "Values must be a 2D tensor"
    assert values.size(0) == length, "Values shape & jagged length mismatch"
    output_jd = jd.copy_others_but_set_values(
        values=torch.nn.functional.pad(
            values, (0, 0, 0, aligned_size - length), "constant", 0
        )
    )
    output_jd.padding_length = aligned_size - length
    return output_jd


def unpad_jd_values(jd: JaggedData, dim=0) -> JaggedData:
    """
    Unpad the jagged length of JaggedData when SP is disabled.
    To make it differentiable, we need to return a new JaggedData instance.
    """
    assert dim == 0, "Only unpadding along the first dimension is supported"
    padding_length = jd.padding_length
    if padding_length == 0:
        output_jd = jd.copy_others_but_set_values(values=jd.values.clone())
        return output_jd
    output_jd = jd.copy_others_but_set_values(
        values=jd.values[0 : jd.seqlen_offsets[-1], ...]
    )
    output_jd.padding_length = 0
    return output_jd
