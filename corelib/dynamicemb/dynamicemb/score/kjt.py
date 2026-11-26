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

"""A ``KeyedJaggedTensor`` that carries one int64 score per key.

The scores travel beside ``weights``, not in it, so a collection can still use
``weights`` for pooling weights or frequency counters while its tables take a
score per key.

TorchRec's ``KeyedJaggedTensor`` methods build a plain ``KeyedJaggedTensor``
from their own fields, so a field added by a subclass would be dropped by the
first ``to``, ``split`` or ``permute``. The methods below that a batch goes
through on its way to a module return the subclass again, with the scores moved
the same way the values were. Anything not listed here (``concat``,
``from_jt_dict``, ...) still returns a plain ``KeyedJaggedTensor``, which a
collection that takes scores rejects rather than reads as "no scores".
"""

from typing import List, Optional

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

__all__ = ["ScoredKeyedJaggedTensor", "get_scores", "without_scores"]


def _shallow_copy_as(cls, kjt: KeyedJaggedTensor, **fields):
    """A ``cls`` sharing every field of ``kjt``, then ``fields`` overwritten.

    Copying ``__dict__`` keeps what ``kjt`` has already derived (per-key lengths
    and offsets), which rebuilding through the constructor would recompute, with
    a device sync for each.
    """
    out = cls.__new__(cls)
    out.__dict__.update(kjt.__dict__)
    out.__dict__.update(fields)
    return out


class ScoredKeyedJaggedTensor(KeyedJaggedTensor):
    """``KeyedJaggedTensor`` plus ``scores``: one int64 per value, aligned to it.

    Args:
        scores: 1-D int64 tensor with exactly as many elements as ``values``,
            on the same device. Element ``i`` is the score of ``values[i]``.
        Everything else is ``KeyedJaggedTensor``'s.
    """

    def __init__(self, *args, scores: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _check_scores(scores, self._values)
        self._scores: torch.Tensor = scores

    @classmethod
    def from_kjt(
        cls, kjt: KeyedJaggedTensor, scores: torch.Tensor
    ) -> "ScoredKeyedJaggedTensor":
        """``kjt`` with ``scores`` attached; ``kjt``'s tensors are shared."""
        _check_scores(scores, kjt.values())
        return _shallow_copy_as(cls, kjt, _scores=scores)

    def scores(self) -> torch.Tensor:
        return self._scores

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "ScoredKeyedJaggedTensor":
        # ``dtype`` is for the weights; a score is int64 whatever it is.
        moved = super().to(device, non_blocking=non_blocking, dtype=dtype)
        return _shallow_copy_as(
            ScoredKeyedJaggedTensor,
            moved,
            _scores=self._scores.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "ScoredKeyedJaggedTensor":
        return _shallow_copy_as(
            ScoredKeyedJaggedTensor,
            super().pin_memory(),
            _scores=self._scores.pin_memory(),
        )

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        super().record_stream(stream)
        self._scores.record_stream(stream)

    def split(self, segments: List[int]) -> List["ScoredKeyedJaggedTensor"]:
        pieces = super().split(segments)
        # A segment of keys is a contiguous run of values, and so of scores.
        offset_per_key = self.offset_per_key()
        scored = []
        first_key = 0
        for piece, num_keys in zip(pieces, segments):
            last_key = first_key + num_keys
            scored.append(
                _shallow_copy_as(
                    ScoredKeyedJaggedTensor,
                    piece,
                    _scores=self._scores[
                        offset_per_key[first_key] : offset_per_key[last_key]
                    ],
                )
            )
            first_key = last_key
        return scored

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "ScoredKeyedJaggedTensor":
        if self.variable_stride_per_key():
            raise NotImplementedError(
                "ScoredKeyedJaggedTensor.permute does not support a variable "
                "stride per key."
            )
        if indices_tensor is None:
            # Built here so that both permutes below share one host-to-device copy.
            indices_tensor = torch.tensor(
                indices, dtype=torch.int, device=self.device()
            )
        permuted = super().permute(indices, indices_tensor)
        # The op TorchRec runs on the values, run on the scores: it moves each
        # key's run of elements, sized by its lengths, to the key's new place.
        _, permuted_scores, _ = torch.ops.fbgemm.permute_2D_sparse_data(
            indices_tensor,
            self.lengths().view(len(self._keys), -1),
            self._scores,
            None,
            permuted.values().numel(),
        )
        return _shallow_copy_as(
            ScoredKeyedJaggedTensor, permuted, _scores=permuted_scores
        )


def _check_scores(scores: torch.Tensor, values: torch.Tensor) -> None:
    if scores.dtype != torch.int64:
        raise ValueError(f"scores must be int64, got {scores.dtype}.")
    if scores.dim() != 1:
        raise ValueError(f"scores must be 1-D, got {scores.dim()}-D.")
    if scores.numel() != values.numel():
        raise ValueError(
            f"scores.numel() ({scores.numel()}) must equal the number of values "
            f"({values.numel()})."
        )
    if scores.device != values.device:
        raise ValueError(
            f"scores must be on the values' device ({values.device}), "
            f"got {scores.device}."
        )


def get_scores(kjt: KeyedJaggedTensor) -> Optional[torch.Tensor]:
    """The scores ``kjt`` carries, or None if it is a plain ``KeyedJaggedTensor``."""
    return kjt.scores() if isinstance(kjt, ScoredKeyedJaggedTensor) else None


def without_scores(kjt: KeyedJaggedTensor) -> KeyedJaggedTensor:
    """``kjt`` as a plain ``KeyedJaggedTensor``, sharing its tensors.

    For evaluation, which reads no score: dropping them on every rank, whichever
    kind of input a rank was given, keeps all ranks on the same side of the
    distribution's collective calls.
    """
    if not isinstance(kjt, ScoredKeyedJaggedTensor):
        return kjt
    plain = _shallow_copy_as(KeyedJaggedTensor, kjt)
    del plain.__dict__["_scores"]
    return plain
