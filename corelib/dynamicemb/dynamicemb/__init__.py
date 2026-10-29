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

"""Public training APIs, imported on demand so inference needs no training runtime."""

from importlib import import_module

_EXPORTS = {
    "DynamicEmbDump": "dump_load",
    "DynamicEmbLoad": "dump_load",
    "BATCH_SIZE_PER_DUMP": "dynamicemb_config",
    "DynamicEmbCheckMode": "dynamicemb_config",
    "DynamicEmbEvictStrategy": "dynamicemb_config",
    "DynamicEmbPoolingMode": "dynamicemb_config",
    "DynamicEmbScoreStrategy": "dynamicemb_config",
    "DynamicEmbTableOptions": "dynamicemb_config",
    "EvictedItemMode": "dynamicemb_config",
    "ReplayContent": "dynamicemb_config",
    "ScoreStrategy": "dynamicemb_config",
    "align_to_table_size": "dynamicemb_config",
    "data_type_to_dtype": "dynamicemb_config",
    "data_type_to_dyn_emb": "dynamicemb_config",
    "dyn_emb_to_torch": "dynamicemb_config",
    "get_sharded_table_capacity": "dynamicemb_config",
    "get_table_value_bytes": "dynamicemb_config",
    "string_to_evict_strategy": "dynamicemb_config",
    "FrequencyAdmissionStrategy": "embedding_admission",
    "KVCounter": "embedding_admission",
    "MultiTableFrequencyAdmitter": "embedding_admission",
    "MultiTableProbabilisticAdmitter": "embedding_admission",
    "ProbabilisticAdmissionStrategy": "embedding_admission",
    "DeltaDumpResult": "incremental_dump",
    "pop_erased_keys": "incremental_dump",
    "pop_evicted_keys": "incremental_dump",
    "replay_increment": "incremental_dump",
    "DynamicEmbOptimType": "optimizer",
    "EmbOptimType": "optimizer",
    "OptimizerArgs": "optimizer",
    "BUCKET_ALIGNMENT": "types",
    "DEMB_TABLE_ALIGN_SIZE": "types",
    "MAX_BUCKET_CAPACITY": "types",
    "AdmissionStrategy": "types",
    "Counter": "types",
    "DynamicEmbInitializerArgs": "types",
    "DynamicEmbInitializerMode": "types",
    "MultiTableAdmitter": "types",
    "ReplayStats": "types",
    "torch_to_dyn_emb": "utils",
}

__all__ = [
    "AdmissionStrategy",
    "BUCKET_ALIGNMENT",
    "DEMB_TABLE_ALIGN_SIZE",
    "MAX_BUCKET_CAPACITY",
    "align_to_table_size",
    "get_table_value_bytes",
    "get_sharded_table_capacity",
    "FrequencyAdmissionStrategy",
    "Counter",
    "KVCounter",
    "MultiTableAdmitter",
    "MultiTableFrequencyAdmitter",
    "MultiTableProbabilisticAdmitter",
    "ProbabilisticAdmissionStrategy",
    "DynamicEmbCheckMode",
    "DynamicEmbInitializerArgs",
    "DynamicEmbInitializerMode",
    "DynamicEmbTableOptions",
    "DynamicEmbPoolingMode",
    "DynamicEmbEvictStrategy",
    "DynamicEmbScoreStrategy",
    "EvictedItemMode",
    "ReplayContent",
    "ScoreStrategy",
    "BATCH_SIZE_PER_DUMP",
    "data_type_to_dyn_emb",
    "data_type_to_dtype",
    "dyn_emb_to_torch",
    "torch_to_dyn_emb",
    "string_to_evict_strategy",
    "DynamicEmbDump",
    "DynamicEmbLoad",
    "DeltaDumpResult",
    "ReplayStats",
    "pop_erased_keys",
    "pop_evicted_keys",
    "replay_increment",
    "DynamicEmbOptimType",
    "EmbOptimType",
    "OptimizerArgs",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
