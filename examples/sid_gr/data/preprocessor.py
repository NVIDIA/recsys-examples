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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import warnings
from abc import ABC
from typing import List

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("main")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DataProcessor(ABC):
    def __init__(self):
        pass

    @property
    def history_feature_name(self):
        pass

    @property
    def candidate_feature_name(self):
        pass

    @property
    def contextual_feature_names(self):
        pass


# class AmazonPreprocessor(DataProcessor):
#   def __init__(self):
#     super().__init__()
#     pass


class AmazonBeautyPreprocessor(DataProcessor):
    def __init__(self, sequence_features_data_path: str, num_hierarchies: int):
        self.num_hierarchies = num_hierarchies
        super().__init__()

    @property
    def num_hierarchies(self) -> int:
        return self.num_hierarchies

    @property
    def history_feature_name(self):
        return "history_sequence"

    @property
    def candidate_feature_name(self):
        return "candidate_sequence"

    @property
    def contextual_feature_names(self):
        return []

    @property
    def sequence_is_sid(self) -> bool:
        return False

    @property
    def raw_sequence_feature_name(self) -> List[str]:
        """
        a raw sequence is split into history and candidate sequences.
        """
        return "item_ids"


def get_common_preprocessors(sequence_features_data_path: str):
    return {
        "amazon_beauty": AmazonBeautyPreprocessor(sequence_features_data_path),
    }
