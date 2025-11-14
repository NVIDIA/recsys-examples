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

import abc
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from dynamicemb.dynamicemb_config import DynamicEmbInitializerArgs


class AdmissionStrategy(abc.ABC):

    @abc.abstractmethod
    def admit(
        self,
        keys: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:

        pass

    @abc.abstractmethod
    def get_initializer_args(self) -> Optional["DynamicEmbInitializerArgs"]:

        pass


class FrequencyAdmissionStrategy(AdmissionStrategy):
    """
    Frequency-based admission strategy.
    Only admits keys whose frequency (score) meets or exceeds a threshold.
    
    Parameters
    ----------
    threshold : int
        Minimum frequency threshold for admission. Keys with frequency >= threshold
        will be admitted into the embedding table.
    initializer_args : Optional[DynamicEmbInitializerArgs]
        Initializer arguments for keys that don't meet the threshold.
        If None, rejected keys will still be initialized with the table's default initializer.
    """

    def __init__(
        self,
        threshold: int,
        initializer_args: Optional["DynamicEmbInitializerArgs"] = None,
    ):
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        
        self.threshold = threshold
        self.initializer_args = initializer_args

    def admit(
        self,
        keys: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Admit keys with frequency >= threshold.

        Parameters
        ----------
        keys : torch.Tensor
            Keys to evaluate (shape: [N])
        scores : torch.Tensor
            Frequency counts for each key (shape: [N])

        Returns
        -------
        torch.Tensor
            Boolean mask (shape: [N]) where True indicates admission
        """
        if keys.shape[0] != scores.shape[0]:
            raise ValueError(
                f"Keys and scores must have same length, got {keys.shape[0]} and {scores.shape[0]}"
            )
        
        # Admit keys whose frequency meets or exceeds threshold
        admit_mask = scores >= self.threshold
        return admit_mask

    def get_initializer_args(self) -> Optional["DynamicEmbInitializerArgs"]:
        return self.initializer_args



